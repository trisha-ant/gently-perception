"""
Ablate the judge ensemble on pair 1 (hybrid@4.6 r1 x vote3_mm@4.7 r1):

A: baseline (GT history + expert reasoning + references)  -- what was tested
B: NO history (drops the potential GT leak entirely)
C: no expert reasoning (just "is it X or Y?")
D: no references (just image + candidates)
E: heuristic tiebreak "more advanced" (no judge call)

Each ablation runs only on the 52 disagreement frames.
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

Two experts looked at the same image and DISAGREED. Your job is to decide \
which one is correct.

Key discriminators for the fold stages:
- 1.5FOLD: body folded once, tail has NOT reached the head. J-shape or \
partial hairpin. Significant dark space in eggshell.
- 2FOLD: clear hairpin/U-shape, TWO parallel body segments, tail has reached \
the head. Moderately filled.
- PRETZEL: body folded THREE or more times, multiple overlapping coils, \
eggshell densely filled. Pattern looks complex and tangled.

Look at the image carefully and decide which expert is right. You MUST pick \
one of the two stages -- do not propose a third.

Respond with JSON:
{
  "stage": "<one of the two proposed stages>",
  "reasoning": "Brief explanation of what you see and why you chose this stage"
}"""


async def judge_call(image_b64, references, history, timepoint, stage_a, reason_a,
                     stage_b, reason_b, use_history, use_reasoning, use_refs):
    content = []
    if use_refs:
        content = build_reference_content(references)
    content.append({"type": "text", "text": f"\n=== ARBITRATE: T{timepoint} ==="})
    if use_history and history:
        hist = build_history_text(history)
        if hist:
            content.append({"type": "text", "text": hist})
    content.append({"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64}})
    if use_reasoning:
        txt = (f"Expert A says {stage_a.upper()}: {reason_a[:300]}\n\n"
               f"Expert B says {stage_b.upper()}: {reason_b[:300]}\n\n"
               f"Which is correct: {stage_a} or {stage_b}?")
    else:
        txt = f"Is this embryo {stage_a} or {stage_b}?"
    content.append({"type": "text", "text": txt})
    raw = await call_claude(system=JUDGE_SYSTEM, content=content, max_tokens=1024)
    data = parse_stage_json(raw)
    js = data.get("stage")
    if js not in (stage_a, stage_b):
        js = stage_a if SIDX.get(stage_a,0) > SIDX.get(stage_b,0) else stage_b
    return js


async def main():
    hyb = json.load(open("/tmp/hyb46_r1.json"))
    h4 = json.load(open("/tmp/h4_r1.json"))
    hf = {(e["embryo_id"],p["timepoint"]):p for e in hyb["embryo_results"] for p in e["predictions"]}
    vf = {(e["embryo_id"],p["timepoint"]):p for e in h4["embryo_results"] for p in e["predictions"]}
    disagree_keys = sorted(k for k in hf if hf[k]["predicted_stage"] != vf[k]["predicted_stage"])
    logger.info(f"Disagreements: {len(disagree_keys)}")

    ground_truth = GroundTruth.from_json(GROUND_TRUTH_PATH)
    testset = OfflineTestset(session_path=VOLUMES_PATH, ground_truth=ground_truth, load_volumes=True)
    references = load_references()

    # collect images + histories: GT history AND "hybrid's predicted history"
    images = {}; gt_hist = {}; pred_hist = {}
    hyb_pred_by_frame = {}  # (embryo, tp) -> hybrid's prediction (or GT for non-scored frames)
    for e in hyb["embryo_results"]:
        for p in e["predictions"]:
            hyb_pred_by_frame[(e["embryo_id"],p["timepoint"])] = p["predicted_stage"]

    for embryo_id, tp_iter in testset.iter_all():
        gh = []; ph = []
        for tc in tp_iter:
            key = (embryo_id, tc.timepoint)
            if key in disagree_keys:
                images[key] = tc.image_b64
                gt_hist[key] = list(gh)
                pred_hist[key] = list(ph)
            gh.append({"timepoint": tc.timepoint, "stage": tc.ground_truth_stage or "early"})
            # predicted history: hybrid's prediction if scored, else GT (same as harness)
            ps = hyb_pred_by_frame.get(key, tc.ground_truth_stage or "early")
            ph.append({"timepoint": tc.timepoint, "stage": ps})

    # compute GT vs predicted history delta
    hist_diff = 0
    for k in disagree_keys:
        gh3 = [h["stage"] for h in gt_hist[k][-3:]]
        ph3 = [h["stage"] for h in pred_hist[k][-3:]]
        if gh3 != ph3: hist_diff += 1
    logger.info(f"History differs between GT and hybrid-predicted on {hist_diff}/{len(disagree_keys)} disagreement frames")

    configs = {
        "A: GT-hist + reasoning + refs": (gt_hist, True, True),
        "B: NO history": ({k:[] for k in disagree_keys}, True, True),
        "C: GT-hist + NO reasoning + refs": (gt_hist, False, True),
        "D: GT-hist + reasoning + NO refs": (gt_hist, True, False),
        "F: PRED-hist + reasoning + refs": (pred_hist, True, True),
    }

    results = {}
    sem = asyncio.Semaphore(3)

    for name, (hist_map, use_reasoning, use_refs) in configs.items():
        preds = {}
        async def run_one(k, hist_map=hist_map, use_reasoning=use_reasoning, use_refs=use_refs):
            async with sem:
                hp, vp = hf[k], vf[k]
                for attempt in range(5):
                    try:
                        js = await judge_call(images[k], references, hist_map[k], k[1],
                                              hp["predicted_stage"], hp["reasoning"],
                                              vp["predicted_stage"], vp["reasoning"],
                                              bool(hist_map[k]), use_reasoning, use_refs)
                        preds[k] = js
                        return
                    except Exception as e:
                        if "429" in str(e) or "rate_limit" in str(e).lower():
                            await asyncio.sleep(5 * (attempt + 1))
                        else:
                            raise
                preds[k] = hp["predicted_stage"] if SIDX.get(hp["predicted_stage"],0)>SIDX.get(vp["predicted_stage"],0) else vp["predicted_stage"]
        await asyncio.gather(*(run_one(k) for k in disagree_keys))
        ok = sum(1 for k in disagree_keys if preds[k] == hf[k]["ground_truth_stage"])
        results[name] = ok
        logger.info(f"  {name}: {ok}/{len(disagree_keys)} = {100*ok/len(disagree_keys):.1f}%")
        await asyncio.sleep(10)  # pause between configs to avoid rate limiting

    # E: heuristic (no API call)
    ok_e = sum(1 for k in disagree_keys if
               (hf[k]["predicted_stage"] if SIDX[hf[k]["predicted_stage"]]>SIDX[vf[k]["predicted_stage"]]
                else vf[k]["predicted_stage"]) == hf[k]["ground_truth_stage"])
    results["E: heuristic more-advanced"] = ok_e

    n = len(hf)
    agree_ok = sum(1 for k in hf if hf[k]["predicted_stage"]==vf[k]["predicted_stage"] and hf[k]["is_correct"])
    print()
    print(f"=== ABLATION RESULTS (pair 1, n={n}, disagree={len(disagree_keys)}) ===")
    print(f"hybrid@4.6 alone: {100*sum(hf[k]['is_correct'] for k in hf)/n:.1f}%")
    print(f"agreement frames correct: {agree_ok}")
    for name, ok in results.items():
        ens = 100*(agree_ok+ok)/n
        print(f"  {name:40s} disagree_acc={100*ok/len(disagree_keys):5.1f}%  ensemble={ens:.1f}%")

    with open("/tmp/judge_ablation.json","w") as f:
        json.dump({"n":n,"agree_ok":agree_ok,"disagree_n":len(disagree_keys),"results":results}, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main())
