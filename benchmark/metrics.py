"""
Perception Benchmark Metrics.

Computes accuracy metrics and confusion matrices.
"""

from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .runner import BenchmarkReport


# Stage order for metrics
STAGE_ORDER = ["early", "bean", "comma", "1.5fold", "2fold", "pretzel", "hatching", "hatched"]


@dataclass
class PerceptionMetrics:
    """Computed metrics for a benchmark run."""

    # Classification accuracy
    accuracy: float = 0.0  # Exact match
    adjacent_accuracy: float = 0.0  # Within 1 stage

    # Per-stage accuracy
    stage_accuracy: Dict[str, float] = field(default_factory=dict)
    stage_counts: Dict[str, int] = field(default_factory=dict)

    # Confusion matrix: confusion[gt_stage][pred_stage] = count
    confusion_matrix: Dict[str, Dict[str, int]] = field(default_factory=dict)

    # Temporal metrics
    backward_transitions: int = 0  # Errors where stage went backward
    stage_transition_delay: Dict[str, float] = field(default_factory=dict)
    # How many timepoints after GT transition until prediction caught up

    # Tool usage
    total_tool_calls: int = 0
    tool_call_rate: float = 0.0  # Avg tool calls per prediction
    tool_use_by_stage: Dict[str, float] = field(default_factory=dict)

    # When tools were used vs not
    accuracy_with_tools: float = 0.0
    accuracy_without_tools: float = 0.0

    # Transitional observations
    transitional_count: int = 0
    transitional_rate: float = 0.0
    transitional_accuracy: float = 0.0  # Accuracy when marked transitional

    def to_dict(self) -> Dict[str, Any]:
        return {
            "accuracy": self.accuracy,
            "adjacent_accuracy": self.adjacent_accuracy,
            "stage_accuracy": self.stage_accuracy,
            "stage_counts": self.stage_counts,
            "confusion_matrix": self.confusion_matrix,
            "backward_transitions": self.backward_transitions,
            "stage_transition_delay": self.stage_transition_delay,
            "total_tool_calls": self.total_tool_calls,
            "tool_call_rate": self.tool_call_rate,
            "tool_use_by_stage": self.tool_use_by_stage,
            "accuracy_with_tools": self.accuracy_with_tools,
            "accuracy_without_tools": self.accuracy_without_tools,
            "transitional_count": self.transitional_count,
            "transitional_rate": self.transitional_rate,
            "transitional_accuracy": self.transitional_accuracy,
        }


def compute_metrics(report: "BenchmarkReport") -> PerceptionMetrics:
    """
    Compute all metrics from a benchmark report.

    Parameters
    ----------
    report : BenchmarkReport
        Completed benchmark report

    Returns
    -------
    PerceptionMetrics
        Computed metrics
    """
    metrics = PerceptionMetrics()

    # Collect all predictions
    all_preds = [
        p for r in report.embryo_results for p in r.predictions
        if p.ground_truth_stage is not None
    ]

    if not all_preds:
        return metrics

    # Basic accuracy
    correct = sum(1 for p in all_preds if p.is_correct)
    adjacent_correct = sum(1 for p in all_preds if p.is_adjacent_correct)
    metrics.accuracy = correct / len(all_preds)
    metrics.adjacent_accuracy = adjacent_correct / len(all_preds)

    # Per-stage accuracy
    stage_correct = defaultdict(int)
    stage_total = defaultdict(int)

    for p in all_preds:
        gt = p.ground_truth_stage
        stage_total[gt] += 1
        if p.is_correct:
            stage_correct[gt] += 1

    for stage in stage_total:
        metrics.stage_counts[stage] = stage_total[stage]
        metrics.stage_accuracy[stage] = stage_correct[stage] / stage_total[stage]

    # Confusion matrix
    confusion = defaultdict(lambda: defaultdict(int))
    for p in all_preds:
        confusion[p.ground_truth_stage][p.predicted_stage] += 1

    metrics.confusion_matrix = {
        gt: dict(preds) for gt, preds in confusion.items()
    }

    # Backward transitions
    for embryo_result in report.embryo_results:
        preds = embryo_result.predictions
        for i in range(1, len(preds)):
            prev_stage = preds[i - 1].predicted_stage
            curr_stage = preds[i].predicted_stage

            try:
                prev_order = STAGE_ORDER.index(prev_stage)
                curr_order = STAGE_ORDER.index(curr_stage)
                if curr_order < prev_order:
                    metrics.backward_transitions += 1
            except ValueError:
                pass  # Stage not in order list

    # Tool usage
    tool_calls = [p.tool_calls for p in all_preds]
    metrics.total_tool_calls = sum(tool_calls)
    metrics.tool_call_rate = metrics.total_tool_calls / len(all_preds)

    # Tool use by stage
    stage_tool_calls = defaultdict(list)
    for p in all_preds:
        stage_tool_calls[p.ground_truth_stage].append(p.tool_calls)

    for stage, calls in stage_tool_calls.items():
        metrics.tool_use_by_stage[stage] = sum(calls) / len(calls)

    # Accuracy with vs without tools
    with_tools = [p for p in all_preds if p.tool_calls > 0]
    without_tools = [p for p in all_preds if p.tool_calls == 0]

    if with_tools:
        metrics.accuracy_with_tools = sum(1 for p in with_tools if p.is_correct) / len(with_tools)
    if without_tools:
        metrics.accuracy_without_tools = sum(1 for p in without_tools if p.is_correct) / len(without_tools)

    # Transitional observations
    transitional_preds = [p for p in all_preds if p.is_transitional]
    metrics.transitional_count = len(transitional_preds)
    metrics.transitional_rate = len(transitional_preds) / len(all_preds)

    if transitional_preds:
        metrics.transitional_accuracy = sum(
            1 for p in transitional_preds if p.is_correct
        ) / len(transitional_preds)

    return metrics


def format_confusion_matrix(
    confusion: Dict[str, Dict[str, int]],
    stages: Optional[List[str]] = None,
) -> str:
    """Format confusion matrix as ASCII table."""
    if stages is None:
        stages = STAGE_ORDER

    # Filter to stages present in data
    present_stages = set()
    for gt, preds in confusion.items():
        present_stages.add(gt)
        for pred in preds:
            present_stages.add(pred)

    stages = [s for s in stages if s in present_stages]

    # Build table
    header = "GT \\ Pred | " + " | ".join(f"{s:>8}" for s in stages)
    lines = [header, "-" * len(header)]

    for gt in stages:
        row = [f"{gt:>9} |"]
        for pred in stages:
            count = confusion.get(gt, {}).get(pred, 0)
            if gt == pred:
                row.append(f"{count:>8}")
            elif count > 0:
                row.append(f"{count:>8}*")
            else:
                row.append(f"{'.':>8}")
        lines.append(" | ".join(row))

    return "\n".join(lines)


def format_metrics_summary(metrics: PerceptionMetrics) -> str:
    """Format metrics as readable summary."""
    lines = [
        "=" * 60,
        "PERCEPTION BENCHMARK METRICS",
        "=" * 60,
        "",
        "ACCURACY",
        f"  Exact match:    {metrics.accuracy:.1%}",
        f"  Within 1 stage: {metrics.adjacent_accuracy:.1%}",
        "",
        "PER-STAGE ACCURACY",
    ]

    for stage in STAGE_ORDER:
        if stage in metrics.stage_accuracy:
            acc = metrics.stage_accuracy[stage]
            count = metrics.stage_counts[stage]
            lines.append(f"  {stage:>10}: {acc:.1%} (n={count})")

    lines.extend([
        "",
        "TOOL USAGE",
        f"  Total tool calls:        {metrics.total_tool_calls}",
        f"  Avg calls per pred:      {metrics.tool_call_rate:.2f}",
        f"  Accuracy with tools:     {metrics.accuracy_with_tools:.1%}",
        f"  Accuracy without tools:  {metrics.accuracy_without_tools:.1%}",
        "",
        "TEMPORAL",
        f"  Backward transitions:    {metrics.backward_transitions}",
        "",
        "TRANSITIONAL OBSERVATIONS",
        f"  Count:    {metrics.transitional_count}",
        f"  Rate:     {metrics.transitional_rate:.1%}",
        f"  Accuracy: {metrics.transitional_accuracy:.1%}",
    ])

    if metrics.confusion_matrix:
        lines.extend([
            "",
            "CONFUSION MATRIX",
            format_confusion_matrix(metrics.confusion_matrix),
        ])

    return "\n".join(lines)
