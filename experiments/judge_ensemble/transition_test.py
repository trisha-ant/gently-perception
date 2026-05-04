"""
Test transition refinement: take a coarse pass's predicted transitions,
show the model a strip of frames around each transition, ask "where is the
boundary?", snap all frames to the refined boundary.

Key insight from analysis: errors are contiguous blocks of 8-17 frames
around transition points. Each embryo has ~2-3 transitions. The task
reduces to estimating ~6 integers accurately.
"""

import asyncio
import json
import logging
import sys

sys.path.insert(0, ".")

from benchmark.ground_truth import GroundTruth
from benchmark.testset import OfflineTestset
from perception._base import call_claude, parse_stage_json, STAGES
from run import load_references, GROUND_TRUTH_PATH, VOLUMES_DIR as VOLUMES_PATH

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SIDX = {s: i for i, s in enumerate(STAGES)}
WINDOW = 12  # +/- around the coarse transition estimate

TRANS_SYSTEM = """\
You are determining when a C. elegans embryo transitions from one \
developmental stage to the next, by examining a sequence of images from the \
same embryo taken at consecutive timepoints.

Each image shows three orthogonal views (XY top-left, YZ top-right, XZ \
bottom-left) of a fluorescence light-sheet max-projection.

The stages in order are: early, bean, comma, 1.5fold, 2fold, pretzel, hatching, hatched.

You will be shown a sequence of frames, labeled by timepoint. The embryo \
transitions from stage A to stage B somewhere in this window. Your job is to \
find the FIRST timepoint that shows stage B.

Key discriminators:
- 1.5FOLD -> 2FOLD: the tail elongates until it reaches the head, forming \
two parallel body segments connected by a bend. The moment the hairpin is \
complete (tail at head) is the transition.
- 2FOLD -> PRETZEL: the body folds a third time, creating multiple \
overlapping coils. The moment you see 3+ passes or the pattern becomes \
clearly tangled is the transition.

Compare each frame to its neighbors. Development is gradual -- look for the \
frame where the defining feature of stage B first appears.

Respond with JSON:
{
  "first_timepoint_of_stage_b": <integer timepoint>,
  "reasoning": "What changed at that timepoint"
}"""


def find_transitions(preds):
    """Return [(timepoint, from_stage, to_stage)] where predicted stage changes."""
    preds = sorted(preds, key=lambda p: p["timepoint"])
    trans = []
    prev = None
    for p in preds:
        s = p["predicted_stage"]
        if prev and s != prev and SIDX.get(s, 0) > SIDX.get(prev, 0):
            trans.append((p["timepoint"], prev, s))
        prev = s
    return trans


async def refine_transition(frames_by_tp, references, coarse_tp, stage_a, stage_b):
    """Send a strip of frames around coarse_tp, ask for the exact boundary."""
    tps = sorted(frames_by_tp.keys())
    # select window
    lo, hi = coarse_tp - WINDOW, coarse_tp + WINDOW
    window_tps = [t for t in tps if lo <= t <= hi]
    if len(window_tps) < 4:
        return None, f"window too small ({len(window_tps)} frames)"
    # subsample to ~8 frames to keep cost down
    if len(window_tps) > 8:
        step = max(1, len(window_tps) // 8)
        window_tps = window_tps[::step][:8]

    # reference images for just the two stages involved
    content = []
    for s in (stage_a, stage_b):
        imgs = references.get(s, [])
        if imgs:
            content.append({"type": "text", "text": f"\n{s.upper()} REFERENCE:"})
            content.append({
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": imgs[0]},
            })
    content.append({
        "type": "text",
        "text": (
            f"\n=== FIND THE {stage_a.upper()} -> {stage_b.upper()} TRANSITION ===\n"
            f"Below are {len(window_tps)} frames from the same embryo at consecutive "
            f"timepoints. The embryo is {stage_a} at the start and {stage_b} at the end. "
            f"Find the FIRST timepoint that shows {stage_b}."
        ),
    })
    for t in window_tps:
        content.append({"type": "text", "text": f"\nTimepoint T{t}:"})
        content.append({
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": frames_by_tp[t]},
        })
    content.append({
        "type": "text",
        "text": (
            f"Which of T{window_tps[0]}-T{window_tps[-1]} is the FIRST frame showing "
            f"{stage_b}? Respond with the JSON format above."
        ),
    })

    raw = await call_claude(system=TRANS_SYSTEM, content=content, max_tokens=1024)
    data = parse_stage_json(raw)
    val = data.get("first_timepoint_of_stage_b")
    if isinstance(val, int):
        return val, data.get("reasoning", "")
    if isinstance(val, str) and val.lstrip("T").isdigit():
        return int(val.lstrip("T")), data.get("reasoning", "")
    return None, f"parse failed: {raw[:100]}"


async def main(coarse_file):
    coarse = json.load(open(coarse_file))
    ground_truth = GroundTruth.from_json(GROUND_TRUTH_PATH)
    testset = OfflineTestset(session_path=VOLUMES_PATH, ground_truth=ground_truth, load_volumes=True)
    references = load_references()

    # collect all frames per embryo (all stages, not just target — transitions may
    # straddle the boundary)
    all_frames = {}  # embryo_id -> {tp: image_b64}
    gt_map = {}      # (embryo_id, tp) -> ground_truth_stage
    for embryo_id, tp_iter in testset.iter_all():
        frames = {}
        for tc in tp_iter:
            frames[tc.timepoint] = tc.image_b64
            gt_map[(embryo_id, tc.timepoint)] = tc.ground_truth_stage
        all_frames[embryo_id] = frames

    # for each embryo, find coarse transitions and refine
    results = {}  # embryo_id -> [(coarse_tp, stage_a, stage_b, refined_tp)]
    refined_preds = {}  # (embryo_id, tp) -> refined stage

    for e in coarse["embryo_results"]:
        eid = e["embryo_id"]
        preds = sorted(e["predictions"], key=lambda p: p["timepoint"])
        if not preds:
            continue
        coarse_pred = {p["timepoint"]: p["predicted_stage"] for p in preds}
        transitions = find_transitions(preds)
        logger.info(f"[{eid}] coarse transitions: {transitions}")

        refined_trans = []
        for ct, sa, sb in transitions:
            rt, reason = await refine_transition(all_frames[eid], references, ct, sa, sb)
            if rt is None:
                logger.warning(f"[{eid}] {sa}->{sb} @ T{ct}: refine failed ({reason}); keeping coarse")
                rt = ct
            else:
                logger.info(f"[{eid}] {sa}->{sb}: coarse T{ct} -> refined T{rt}  (GT region check: T{ct} GT={gt_map.get((eid,ct))}, T{rt} GT={gt_map.get((eid,rt))})")
            refined_trans.append((rt, sa, sb))
        results[eid] = refined_trans

        # rebuild predictions: sort refined transitions, assign each frame the stage
        # determined by which transitions it's after
        refined_trans.sort()
        tps = sorted(coarse_pred.keys())
        # initial stage = whatever coarse pass said before the first transition
        initial_stage = preds[0]["predicted_stage"]
        for tp in tps:
            stage = initial_stage
            for rt, sa, sb in refined_trans:
                if tp >= rt:
                    stage = sb
            refined_preds[(eid, tp)] = stage

    # compute accuracy
    n = sum(1 for e in coarse["embryo_results"] for _ in e["predictions"])
    coarse_ok = sum(p["is_correct"] for e in coarse["embryo_results"] for p in e["predictions"])
    refined_ok = 0
    adj_ok = 0
    for e in coarse["embryo_results"]:
        eid = e["embryo_id"]
        for p in e["predictions"]:
            k = (eid, p["timepoint"])
            pred = refined_preds.get(k, p["predicted_stage"])
            gt = p["ground_truth_stage"]
            if pred == gt:
                refined_ok += 1
            if abs(SIDX.get(pred, 0) - SIDX.get(gt, 0)) <= 1:
                adj_ok += 1

    print()
    print(f"=== RESULTS ===")
    print(f"coarse ({coarse_file}): {100*coarse_ok/n:.1f}%")
    print(f"refined: {100*refined_ok/n:.1f}% (adj {100*adj_ok/n:.1f}%)  delta {100*(refined_ok-coarse_ok)/n:+.1f}pp")

    with open("/tmp/transition_r1.json", "w") as f:
        json.dump({"coarse_acc": 100*coarse_ok/n, "refined_acc": 100*refined_ok/n,
                   "transitions": {k: [(rt, sa, sb) for rt, sa, sb in v] for k, v in results.items()}}, f, indent=2)


if __name__ == "__main__":
    asyncio.run(main(sys.argv[1] if len(sys.argv) > 1 else "/tmp/hyb46_r1.json"))
