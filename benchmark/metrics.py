"""Perception benchmark metrics.

Operates on the result-dict shape that ``run.py`` writes to
``data/results/*.json``::

    {
      "embryo_results": [
        {"embryo_id": str, "predictions": [
            {"timepoint": int, "predicted_stage": str,
             "ground_truth_stage": str, "is_correct": bool,
             "is_adjacent_correct": bool, "tool_calls": int, ...},
        ...]},
      ...],
      "overall_accuracy": float,
      "metrics": {...},
    }

Replaces the previous version, which imported a non-existent ``BenchmarkReport``.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field

STAGE_ORDER = ["early", "bean", "comma", "1.5fold", "2fold",
               "pretzel", "hatching", "hatched"]
_STAGE_IDX = {s: i for i, s in enumerate(STAGE_ORDER)}


# --------------------------------------------------------------------------- #
# Core per-frame metrics
# --------------------------------------------------------------------------- #

@dataclass
class PerceptionMetrics:
    accuracy: float = 0.0
    adjacent_accuracy: float = 0.0
    n: int = 0
    stage_accuracy: dict[str, float] = field(default_factory=dict)
    stage_counts: dict[str, int] = field(default_factory=dict)
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    backward_transitions: int = 0
    transition_mae: float | None = None


def _iter_predictions(report: dict):
    for er in report.get("embryo_results", []):
        for p in er.get("predictions", []):
            if p.get("ground_truth_stage") is not None:
                yield p


def compute_metrics(report: dict) -> PerceptionMetrics:
    """Compute all metrics from a result-dict report."""
    preds = list(_iter_predictions(report))
    m = PerceptionMetrics(n=len(preds))
    if not preds:
        return m

    m.accuracy = sum(1 for p in preds if p["is_correct"]) / len(preds)
    m.adjacent_accuracy = sum(1 for p in preds if p["is_adjacent_correct"]) / len(preds)

    stage_correct: dict[str, int] = defaultdict(int)
    stage_total: dict[str, int] = defaultdict(int)
    confusion: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for p in preds:
        gt = p["ground_truth_stage"]
        stage_total[gt] += 1
        if p["is_correct"]:
            stage_correct[gt] += 1
        confusion[gt][p["predicted_stage"]] += 1
    m.stage_counts = dict(stage_total)
    m.stage_accuracy = {s: stage_correct[s] / stage_total[s] for s in stage_total}
    m.confusion = {gt: dict(row) for gt, row in confusion.items()}

    for er in report.get("embryo_results", []):
        seq = er.get("predictions", [])
        for i in range(1, len(seq)):
            prev = _STAGE_IDX.get(seq[i - 1].get("predicted_stage"), -1)
            curr = _STAGE_IDX.get(seq[i].get("predicted_stage"), -1)
            if prev >= 0 and curr >= 0 and curr < prev:
                m.backward_transitions += 1

    m.transition_mae = transition_mae(report)
    return m


# --------------------------------------------------------------------------- #
# Transition-timing error (the metric that actually moves — see RESEARCH.md)
# --------------------------------------------------------------------------- #

def _find_transitions(seq: list[tuple[int, str]]) -> dict[str, int]:
    """Given [(timepoint, stage), ...] sorted by timepoint, return
    {stage: first_timepoint_at_that_stage} for every stage that appears."""
    out: dict[str, int] = {}
    prev: str | None = None
    for tp, st in seq:
        if st != prev and st not in out:
            out[st] = tp
        prev = st
    return out


def transition_mae(report: dict) -> float | None:
    """Mean absolute error (in frames) between predicted and GT stage-onset
    timepoints, averaged over all transitions in all embryos.

    A run that places every transition exactly right scores 0. Per RESEARCH.md,
    errors come in 8–17-frame contiguous blocks at boundaries — this metric
    surfaces that directly instead of projecting it onto per-frame accuracy.
    """
    errors: list[int] = []
    for er in report.get("embryo_results", []):
        preds = sorted(er.get("predictions", []), key=lambda p: p["timepoint"])
        if not preds:
            continue
        pred_seq = [(p["timepoint"], p["predicted_stage"]) for p in preds]
        gt_seq = [(p["timepoint"], p["ground_truth_stage"]) for p in preds]

        pred_tr = _find_transitions(pred_seq)
        gt_tr = _find_transitions(gt_seq)

        for stage, gt_tp in gt_tr.items():
            if stage in pred_tr:
                errors.append(abs(pred_tr[stage] - gt_tp))

    if not errors:
        return None
    return sum(errors) / len(errors)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #

def format_confusion_matrix(confusion: dict[str, dict[str, int]],
                            stages: list[str] | None = None) -> str:
    if stages is None:
        present = {gt for gt in confusion} | {p for row in confusion.values() for p in row}
        stages = [s for s in STAGE_ORDER if s in present]

    header = "GT \\ Pred | " + " | ".join(f"{s:>8}" for s in stages)
    lines = [header, "-" * len(header)]
    for gt in stages:
        row = [f"{gt:>9} |"]
        for pr in stages:
            count = confusion.get(gt, {}).get(pr, 0)
            cell = f"{count:>8}" if count else f"{'.':>8}"
            row.append(cell)
        lines.append(" | ".join(row))
    return "\n".join(lines)


def format_metrics_summary(m: PerceptionMetrics) -> str:
    lines = [
        "=" * 60,
        "PERCEPTION BENCHMARK METRICS",
        "=" * 60,
        f"  n:              {m.n}",
        f"  Exact:          {m.accuracy:.1%}",
        f"  Adjacent:       {m.adjacent_accuracy:.1%}",
        f"  Backward steps: {m.backward_transitions}",
    ]
    if m.transition_mae is not None:
        lines.append(f"  Transition MAE: {m.transition_mae:.1f} frames")
    lines.append("")
    lines.append("Per-stage:")
    for s in STAGE_ORDER:
        if s in m.stage_accuracy:
            lines.append(f"  {s:>10}: {m.stage_accuracy[s]:.1%} (n={m.stage_counts[s]})")
    if m.confusion:
        lines += ["", format_confusion_matrix(m.confusion)]
    return "\n".join(lines)
