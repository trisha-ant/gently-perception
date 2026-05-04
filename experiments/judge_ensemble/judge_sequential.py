"""
Sequential judge ensemble — the honest production config.

The previous judge_test.py leaked ground truth into the judge's history
(used GT stages for all frames). The ablation showed this inflated the
result by ~6pp. This version runs the judge SEQUENTIALLY per embryo,
building the ensemble's own predicted history as it goes — the judge at
frame T sees the ensemble's own outputs for T-3, T-2, T-1, which is
exactly what a production system would have.

Caveat: hybrid@4.6 and vote3_mm@4.7 predictions are taken from archived
runs where each classifier saw its OWN predicted history. In a fully
sequential system, they too would see the ensemble's history. This version
changes only the judge's history, not the classifiers' — so it's a lower
bound on what a fully-sequential ensemble could do (but an upper bound
over the parallel version with hybrid-only history).
"""

import asyncio
import json
import logging
import sys

sys.path.insert(0, ".")

from benchmark.ground_truth import GroundTruth
from benchmark.testset import OfflineTestset
from perception._base import call_claude, parse_stage_json, build_reference_content, build_history_text
from run import load_references, GROUND_TRUTH_PATH, VOLUMES_DIR as VOLUMES_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

STAGES_L = ["early","bean","comma","1.5fold","2fold","pretzel","hatching","hatched"]
SIDX = {s:i for i,s in enumerate(STAGES_L)}

JUDGE_SYSTEM = """\
You are arbitrating between two expert classifiers of C. elegans embryo \
developmental stages. Each image shows three orthogonal views (XY top-left, \
YZ top-right, XZ bottom-left) of a fluorescence light-sheet max-projection.

The stages in order are: early, bean, comma, 1.5fold, 2fold, pretzel, hatching, hatched.

Two experts looked at the same image and DISAGREED. You will be given their \
classifications and reasoning. Your job is to decide which one is correct.

Key discriminators for the fold stages:
- 1.5FOLD: body folded once, tail has NOT reached the head. J-shape or \
partial hairpin. Significant dark space in eggshell.
- 2FOLD: clear hairpin/U-shape, TWO parallel body segments, tail has reached \
the head. Moderately filled.
- PRETZEL: body folded THREE or more times, multiple overlapping coils, \
eggshell densely filled. Pattern looks complex and tangled.

Look at the image carefully and decide which expert is right. You MUST pick \
one of the two stages they proposed -- do not propose a third.

Respond with JSON:
{
  "stage": "<one of the two proposed stages>",
  "reasoning": "Brief explanation of what you see and why you chose this stage"
}"""


async def judge(image_b64, references, history, timepoint, stage_a, reason_a, stage_b, reason_b):
    content = build_reference_content(references)
    content.append({"type": "text", "text": f"\n=== ARBITRATE: T{timepoint} ==="})
    hist = build_history_text(history)
    if hist:
        content.append({"type": "text", "text": hist})
    content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}})
    content.append({
        "type": "text",
        "text": (
            f"Expert A says {stage_a.upper()}: {reason_a[:300]}\n\n"
            f"Expert B says {stage_b.upper()}: {reason_b[:300]}\n\n"
            f"Which is correct: {stage_a} or {stage_b}?"
        ),
    })
    for attempt in range(5):
        try:
            raw = await call_claude(system=JUDGE_SYSTEM, content=content, max_tokens=1024)
            break
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                await asyncio.sleep(5 * (attempt + 1))
            else:
                raise
    else:
        return None
    data = parse_stage_json(raw)
    js = data.get("stage")
    if js not in (stage_a, stage_b):
        js = stage_a if SIDX.get(stage_a,0) > SIDX.get(stage_b,0) else stage_b
    return js


async def run_embryo_sequential(embryo_id, frames, hf, vf, references, gt_map, log_prefix=""):
    """
    Process one embryo's frames in timepoint order, building ensemble history.
    frames: list of (timepoint, image_b64) sorted by timepoint.
    hf, vf: {(embryo_id, tp): prediction dict}
    Returns {(embryo_id, tp): ensemble_stage} and counts.
    """
    ens_preds = {}
    history = []  # ensemble's own history
    agree_ok = agree_n = disagree_ok = disagree_n = 0

    for tp, img in frames:
        key = (embryo_id, tp)
        gt = gt_map.get(key)
        hp_rec = hf.get(key)
        vp_rec = vf.get(key)
        if hp_rec is None or vp_rec is None:
            # non-scored frame: history uses GT (matches harness behavior for
            # target-stage filtering)
            history.append({"timepoint": tp, "stage": gt or "early"})
            continue
        hp, vp = hp_rec["predicted_stage"], vp_rec["predicted_stage"]
        if hp == vp:
            pred = hp
            agree_n += 1
            if pred == gt: agree_ok += 1
        else:
            disagree_n += 1
            pred = await judge(img, references, history, tp,
                               hp, hp_rec["reasoning"], vp, vp_rec["reasoning"])
            if pred is None:
                pred = hp if SIDX.get(hp,0) > SIDX.get(vp,0) else vp
            if pred == gt: disagree_ok += 1
        ens_preds[key] = pred
        history.append({"timepoint": tp, "stage": pred})

    return ens_preds, agree_ok, agree_n, disagree_ok, disagree_n


async def run_pair(hyb_file, vote3_file, embryo_frames, references, gt_map):
    hyb = json.load(open(hyb_file))
    h4 = json.load(open(vote3_file))
    hf = {(e["embryo_id"],p["timepoint"]):p for e in hyb["embryo_results"] for p in e["predictions"]}
    vf = {(e["embryo_id"],p["timepoint"]):p for e in h4["embryo_results"] for p in e["predictions"]}

    # embryos can run in parallel (sequential within each)
    tasks = []
    for eid, frames in embryo_frames.items():
        tasks.append(run_embryo_sequential(eid, frames, hf, vf, references, gt_map))
    results = await asyncio.gather(*tasks)

    all_preds = {}; ao=an=do=dn=0
    for ens, a_ok, a_n, d_ok, d_n in results:
        all_preds.update(ens); ao+=a_ok; an+=a_n; do+=d_ok; dn+=d_n

    n = an + dn
    by = {}
    for k, pred in all_preds.items():
        gt = gt_map[k]
        by.setdefault(gt, [0,0]); by[gt][1]+=1
        if pred == gt: by[gt][0]+=1
    stages = {k: 100*v[0]/v[1] for k,v in by.items()}

    return {"n": n, "agree_n": an, "agree_ok": ao, "disagree_n": dn, "disagree_ok": do,
            "hyb_acc": 100*sum(hf[k]["is_correct"] for k in hf)/len(hf),
            "vote3_acc": 100*sum(vf[k]["is_correct"] for k in hf)/len(hf),
            "ensemble_acc": 100*(ao+do)/n,
            "judge_acc_on_disagree": 100*do/dn if dn else 0,
            "by_stage": stages}


async def main():
    ground_truth = GroundTruth.from_json(GROUND_TRUTH_PATH)
    testset = OfflineTestset(session_path=VOLUMES_PATH, ground_truth=ground_truth, load_volumes=True)
    references = load_references()

    # build frame list per embryo + gt map
    embryo_frames = {}
    gt_map = {}
    for eid, tp_iter in testset.iter_all():
        frames = []
        for tc in tp_iter:
            frames.append((tc.timepoint, tc.image_b64))
            gt_map[(eid, tc.timepoint)] = tc.ground_truth_stage
        embryo_frames[eid] = sorted(frames)

    pairs = [("/tmp/hyb46_r1.json","/tmp/h4_r1.json"),
             ("/tmp/hyb46_r2.json","/tmp/h4_r2.json"),
             ("/tmp/hyb46_r3.json","/tmp/h4_r3.json")]
    results = []
    for i, (hf_p, vf_p) in enumerate(pairs, 1):
        logger.info(f"=== pair {i} ===")
        r = await run_pair(hf_p, vf_p, embryo_frames, references, gt_map)
        results.append(r)
        logger.info(f"  hyb={r['hyb_acc']:.1f} vote3={r['vote3_acc']:.1f} ensemble={r['ensemble_acc']:.1f} judge_on_disagree={r['judge_acc_on_disagree']:.1f} by_stage={r['by_stage']}")

    import statistics
    print()
    print("=== SEQUENTIAL (HONEST) JUDGE ENSEMBLE ===")
    for k in ["hyb_acc","vote3_acc","ensemble_acc","judge_acc_on_disagree"]:
        v = [r[k] for r in results]
        print(f"  {k}: {statistics.mean(v):.1f} +/- {statistics.stdev(v):.1f}  [{'/'.join(f'{x:.1f}' for x in v)}]")
    # per-stage
    for stage in ["1.5fold","2fold","pretzel"]:
        v = [r["by_stage"].get(stage, 0) for r in results]
        print(f"  {stage}: {statistics.mean(v):.1f} +/- {statistics.stdev(v):.1f}")

    with open("/tmp/judge_sequential.json","w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
