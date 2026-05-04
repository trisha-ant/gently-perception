"""
Test the judge ensemble: on disagreement frames between hybrid@4.6 and
vote3_mm@4.7, call a judge with both predictions + reasoning and let it pick.

Runs only on disagreement frames (~52), so ~5 min.
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
    content.append(
        {"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}}
    )
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
    return data.get("stage"), data.get("reasoning", "")


async def main():
    # load the two runs
    hyb = json.load(open("/tmp/hyb46_r1.json"))
    h4 = json.load(open("/tmp/h4_r1.json"))
    hf = {(e["embryo_id"], p["timepoint"]): p for e in hyb["embryo_results"] for p in e["predictions"]}
    vf = {(e["embryo_id"], p["timepoint"]): p for e in h4["embryo_results"] for p in e["predictions"]}

    # identify disagreements
    disagree_keys = [k for k in hf if hf[k]["predicted_stage"] != vf[k]["predicted_stage"]]
    logger.info(f"Disagreements: {len(disagree_keys)} / {len(hf)} frames")

    # build testset and references
    ground_truth = GroundTruth.from_json(GROUND_TRUTH_PATH)
    testset = OfflineTestset(session_path=VOLUMES_PATH, ground_truth=ground_truth, load_volumes=True)
    references = load_references()

    # iterate testset, collect images + history for disagreement frames
    disagree_data = {}
    for embryo_id, tp_iter in testset.iter_all():
        history = []
        for tc in tp_iter:
            key = (embryo_id, tc.timepoint)
            if key in disagree_keys:
                disagree_data[key] = (tc.image_b64, list(history))
            history.append({"timepoint": tc.timepoint, "stage": tc.ground_truth_stage or "early"})

    logger.info(f"Collected images for {len(disagree_data)} disagreement frames")

    # run judge
    judge_results = {}
    sem = asyncio.Semaphore(5)

    async def run_one(key):
        async with sem:
            img, hist = disagree_data[key]
            hp, vp = hf[key], vf[key]
            stage, reason = await judge(
                img, references, hist, key[1],
                hp["predicted_stage"], hp["reasoning"],
                vp["predicted_stage"], vp["reasoning"],
            )
            judge_results[key] = {"stage": stage, "reasoning": reason}
            gt = hp["ground_truth_stage"]
            correct = stage == gt
            logger.info(f"[judge] {key[0]} T{key[1]}: hyb={hp['predicted_stage']} vote3={vp['predicted_stage']} judge={stage} GT={gt} {'OK' if correct else 'WRONG'}")

    await asyncio.gather(*(run_one(k) for k in disagree_keys))

    # compute combined accuracy
    n = len(hf)
    agree_ok = 0
    agree_n = 0
    disagree_ok = 0
    judge_on_target = 0  # judge picked one of the two options
    for k in hf:
        gt = hf[k]["ground_truth_stage"]
        hp, vp = hf[k]["predicted_stage"], vf[k]["predicted_stage"]
        if hp == vp:
            agree_n += 1
            if hp == gt:
                agree_ok += 1
        else:
            js = judge_results[k]["stage"]
            if js not in (hp, vp):
                # parse failure or off-target: fall back to more advanced
                STAGES_L = ["early","bean","comma","1.5fold","2fold","pretzel","hatching","hatched"]
                sidx = {s:i for i,s in enumerate(STAGES_L)}
                js = hp if sidx.get(hp,0) > sidx.get(vp,0) else vp
            else:
                judge_on_target += 1
            if js == gt:
                disagree_ok += 1

    ensemble_ok = agree_ok + disagree_ok
    print()
    print(f"=== RESULTS ===")
    print(f"hybrid@4.6 alone: {100*sum(hf[k]['is_correct'] for k in hf)/n:.1f}%")
    print(f"vote3_mm@4.7 alone: {100*sum(vf[k]['is_correct'] for k in hf)/n:.1f}%")
    print(f"agree ({agree_n}): {100*agree_ok/agree_n:.1f}%")
    print(f"disagree ({len(disagree_keys)}): judge acc {100*disagree_ok/len(disagree_keys):.1f}%")
    print(f"  judge picked one of two options: {judge_on_target}/{len(disagree_keys)}")
    print(f"ENSEMBLE: {100*ensemble_ok/n:.1f}%  (+{100*ensemble_ok/n - 100*sum(hf[k]['is_correct'] for k in hf)/n:.1f}pp vs hybrid)")

    # save for archiving
    with open("/tmp/judge_r1.json", "w") as f:
        json.dump({
            "n": n, "agree_n": agree_n, "agree_ok": agree_ok,
            "disagree_n": len(disagree_keys), "disagree_ok": disagree_ok,
            "ensemble_acc": 100*ensemble_ok/n,
            "judge_results": {f"{k[0]}_{k[1]}": v for k, v in judge_results.items()},
        }, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
