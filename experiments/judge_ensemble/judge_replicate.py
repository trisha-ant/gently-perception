"""Replicate the judge ensemble across 3 run pairs to estimate variance."""

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
    raw = await call_claude(system=JUDGE_SYSTEM, content=content, max_tokens=1024)
    data = parse_stage_json(raw)
    return data.get("stage")


async def run_pair(hyb_file, vote3_file, images, histories, references):
    hyb = json.load(open(hyb_file))
    h4 = json.load(open(vote3_file))
    hf = {(e["embryo_id"], p["timepoint"]): p for e in hyb["embryo_results"] for p in e["predictions"]}
    vf = {(e["embryo_id"], p["timepoint"]): p for e in h4["embryo_results"] for p in e["predictions"]}
    disagree_keys = [k for k in hf if hf[k]["predicted_stage"] != vf[k]["predicted_stage"]]

    sem = asyncio.Semaphore(5)
    judge_preds = {}
    async def run_one(k):
        async with sem:
            hp, vp = hf[k], vf[k]
            js = await judge(images[k], references, histories[k], k[1],
                             hp["predicted_stage"], hp["reasoning"],
                             vp["predicted_stage"], vp["reasoning"])
            if js not in (hp["predicted_stage"], vp["predicted_stage"]):
                js = hp["predicted_stage"] if SIDX.get(hp["predicted_stage"],0) > SIDX.get(vp["predicted_stage"],0) else vp["predicted_stage"]
            judge_preds[k] = js
    await asyncio.gather(*(run_one(k) for k in disagree_keys))

    n = len(hf); agree_ok = disagree_ok = agree_n = 0
    for k in hf:
        gt = hf[k]["ground_truth_stage"]
        hp, vp = hf[k]["predicted_stage"], vf[k]["predicted_stage"]
        if hp == vp:
            agree_n += 1; agree_ok += hp == gt
        else:
            disagree_ok += judge_preds[k] == gt
    return {"n": n, "agree_n": agree_n, "agree_ok": agree_ok,
            "disagree_n": len(disagree_keys), "disagree_ok": disagree_ok,
            "hyb_acc": 100*sum(hf[k]["is_correct"] for k in hf)/n,
            "vote3_acc": 100*sum(vf[k]["is_correct"] for k in hf)/n,
            "ensemble_acc": 100*(agree_ok+disagree_ok)/n,
            "judge_acc_on_disagree": 100*disagree_ok/len(disagree_keys)}


async def main():
    ground_truth = GroundTruth.from_json(GROUND_TRUTH_PATH)
    testset = OfflineTestset(session_path=VOLUMES_PATH, ground_truth=ground_truth, load_volumes=True)
    references = load_references()

    # get union of all disagreement keys across pairs
    pairs = [("/tmp/hyb46_r1.json","/tmp/h4_r1.json"),
             ("/tmp/hyb46_r2.json","/tmp/h4_r2.json"),
             ("/tmp/hyb46_r3.json","/tmp/h4_r3.json")]
    all_keys = set()
    for hf_p, vf_p in pairs:
        hyb = json.load(open(hf_p)); h4 = json.load(open(vf_p))
        hf = {(e["embryo_id"],p["timepoint"]):p for e in hyb["embryo_results"] for p in e["predictions"]}
        vf = {(e["embryo_id"],p["timepoint"]):p for e in h4["embryo_results"] for p in e["predictions"]}
        all_keys.update(k for k in hf if hf[k]["predicted_stage"] != vf[k]["predicted_stage"])
    logger.info(f"Total unique disagreement keys: {len(all_keys)}")

    # collect images + histories
    images = {}; histories = {}
    for embryo_id, tp_iter in testset.iter_all():
        history = []
        for tc in tp_iter:
            key = (embryo_id, tc.timepoint)
            if key in all_keys:
                images[key] = tc.image_b64
                histories[key] = list(history)
            history.append({"timepoint": tc.timepoint, "stage": tc.ground_truth_stage or "early"})

    results = []
    for i, (hf_p, vf_p) in enumerate(pairs, 1):
        logger.info(f"=== pair {i}: {hf_p} x {vf_p} ===")
        r = await run_pair(hf_p, vf_p, images, histories, references)
        results.append(r)
        logger.info(f"  hyb={r['hyb_acc']:.1f} vote3={r['vote3_acc']:.1f} ensemble={r['ensemble_acc']:.1f} judge_on_disagree={r['judge_acc_on_disagree']:.1f}")

    import statistics
    print()
    print("=== SUMMARY ===")
    for k in ["hyb_acc","vote3_acc","ensemble_acc","judge_acc_on_disagree"]:
        v = [r[k] for r in results]
        print(f"  {k}: {statistics.mean(v):.1f} +/- {statistics.stdev(v):.1f}  [{'/'.join(f'{x:.1f}' for x in v)}]")

    with open("/tmp/judge_replication.json","w") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
