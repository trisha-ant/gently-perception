"""
Timepoint-based routing: calibrate by a coarse pass's predicted transition
points, then route each frame to the stage-region-appropriate model.

Routing rule (from per-stage accuracy):
  1.5fold region -> hybrid@4.6 (65.9% vs vote3's 44.7%)
  2fold region   -> vote3_mm@4.7 (97.7% vs hybrid's 68.4%)
  pretzel region -> hybrid@4.6 (92.5% vs vote3's 84.5%)

The region boundaries come from the coarse pass's *predicted transition
points* — where the coarse pass's prediction changes from 1.5fold->2fold
and 2fold->pretzel. This is a harness-level signal that doesn't require
GT and is more robust than any single frame's prediction.

Computed entirely from archived run JSONs. No new API calls.
"""

import json
import statistics
import sys
from collections import defaultdict

STAGES = ["early","bean","comma","1.5fold","2fold","pretzel","hatching","hatched"]
SIDX = {s: i for i, s in enumerate(STAGES)}

# which model to use for each stage region
ROUTE = {"1.5fold": "hybrid", "2fold": "vote3", "pretzel": "hybrid"}


def load(f):
    d = json.load(open(f))
    return {(e["embryo_id"], p["timepoint"]): p for e in d["embryo_results"] for p in e["predictions"]}


def find_transition_tps(preds_by_embryo):
    """For each embryo, find the timepoint where 1.5fold->2fold and 2fold->pretzel
    happen in the coarse pass's predictions. Returns {embryo: {boundary: tp}}."""
    out = {}
    for eid, preds in preds_by_embryo.items():
        preds.sort(key=lambda p: p["timepoint"])
        trans = {}
        prev = None
        for p in preds:
            s = p["predicted_stage"]
            if prev and SIDX.get(s, 0) > SIDX.get(prev, 0):
                # record the FIRST timepoint of the new stage
                for boundary_to in ("2fold", "pretzel"):
                    if s == boundary_to and boundary_to not in trans:
                        trans[boundary_to] = p["timepoint"]
            prev = s
        out[eid] = trans
    return out


def region_of(tp, transitions):
    """What stage region is timepoint `tp` in, given the coarse transitions?"""
    t2f = transitions.get("2fold", float("inf"))
    tpz = transitions.get("pretzel", float("inf"))
    if tp >= tpz:
        return "pretzel"
    if tp >= t2f:
        return "2fold"
    return "1.5fold"


def route_and_score(hf, vf, coarse_trans, n_keys):
    """Route each frame and compute accuracy."""
    ok = 0
    by_stage = defaultdict(lambda: [0, 0])
    used = {"hybrid": 0, "vote3": 0}
    for k in n_keys:
        eid, tp = k
        region = region_of(tp, coarse_trans.get(eid, {}))
        model = ROUTE[region]
        used[model] += 1
        pred = (hf if model == "hybrid" else vf)[k]["predicted_stage"]
        gt = hf[k]["ground_truth_stage"]
        by_stage[gt][1] += 1
        if pred == gt:
            ok += 1
            by_stage[gt][0] += 1
    return 100 * ok / len(n_keys), dict(by_stage), used


def main():
    pairs = [(f"/tmp/hyb46_r{i}.json", f"/tmp/h4_r{i}.json", f"/tmp/mm_r{i}.json") for i in [1,2,3]]
    results = {"mm_coarse": [], "hyb_coarse": [], "vote3_coarse": [], "gt_regions": [],
               "stage_pred_route": []}

    for hf_p, vf_p, mm_p in pairs:
        hf = load(hf_p); vf = load(vf_p); mf = load(mm_p)
        keys = sorted(hf.keys())
        hyb_acc = 100 * sum(hf[k]["is_correct"] for k in keys) / len(keys)
        vote3_acc = 100 * sum(vf[k]["is_correct"] for k in keys) / len(keys)

        # group predictions by embryo for transition finding
        def by_embryo(preds):
            d = defaultdict(list)
            for k, p in preds.items():
                d[k[0]].append(p)
            return dict(d)

        # variant 1: multimeasure coarse pass (independent third model)
        mm_trans = find_transition_tps(by_embryo(mf))
        acc_mm, bs_mm, used_mm = route_and_score(hf, vf, mm_trans, keys)

        # variant 2: hybrid's own transitions as coarse
        hyb_trans = find_transition_tps(by_embryo(hf))
        acc_hyb, bs_hyb, _ = route_and_score(hf, vf, hyb_trans, keys)

        # variant 3: vote3's own transitions as coarse
        v3_trans = find_transition_tps(by_embryo(vf))
        acc_v3, bs_v3, _ = route_and_score(hf, vf, v3_trans, keys)

        # ceiling: route using GT regions (for reference, LEAKY — ceiling only)
        gt_regions = {}
        for k in keys:
            gt_regions.setdefault(k[0], {}).setdefault("preds", []).append({"timepoint": k[1], "predicted_stage": hf[k]["ground_truth_stage"]})
        gt_by_e = {eid: v["preds"] for eid, v in gt_regions.items()}
        gt_trans = find_transition_tps(gt_by_e)
        acc_gt, bs_gt, _ = route_and_score(hf, vf, gt_trans, keys)

        # per-frame stage-pred routing: route based on coarse's predicted stage at that frame
        ok_sp = 0
        for k in keys:
            coarse_stage = mf[k]["predicted_stage"]
            region = coarse_stage if coarse_stage in ROUTE else "1.5fold"
            model = ROUTE.get(region, "hybrid")
            pred = (hf if model == "hybrid" else vf)[k]["predicted_stage"]
            if pred == hf[k]["ground_truth_stage"]:
                ok_sp += 1
        acc_sp = 100 * ok_sp / len(keys)

        results["mm_coarse"].append(acc_mm)
        results["hyb_coarse"].append(acc_hyb)
        results["vote3_coarse"].append(acc_v3)
        results["gt_regions"].append(acc_gt)
        results["stage_pred_route"].append(acc_sp)

        print(f"pair: hyb={hyb_acc:.1f} vote3={vote3_acc:.1f} | "
              f"route(mm)={acc_mm:.1f} route(hyb)={acc_hyb:.1f} route(vote3)={acc_v3:.1f} "
              f"route(GT-region*)={acc_gt:.1f} stagepred(mm)={acc_sp:.1f}")
        print(f"  mm transitions: {mm_trans}")
        print(f"  GT transitions: {gt_trans}")
        print(f"  router used hybrid on {used_mm['hybrid']}/{len(keys)} frames, vote3 on {used_mm['vote3']}")

    print()
    print("=== SUMMARY (N=3) ===")
    print(f"  hybrid@4.6: 81.7 +/- 2.4")
    print(f"  vote3_mm@4.7: 80.8 +/- 0.5")
    for name, vals in results.items():
        tag = " (LEAKY ceiling)" if name == "gt_regions" else ""
        print(f"  route({name}): {statistics.mean(vals):.1f} +/- {statistics.stdev(vals):.1f}  [{'/'.join(f'{v:.1f}' for v in vals)}]{tag}")


if __name__ == "__main__":
    main()
