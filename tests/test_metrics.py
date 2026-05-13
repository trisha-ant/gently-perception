"""Verify benchmark/metrics.py against archived golden runs and synthetics."""

from __future__ import annotations

import pytest

from benchmark.metrics import (
    PerceptionMetrics,
    compute_metrics,
    format_metrics_summary,
    transition_mae,
)


def test_compute_metrics_on_hyb46_r1(hyb46):
    """hyb46_r1.json: 82.8% exact, 233 frames, known per-stage breakdown."""
    m = compute_metrics(hyb46[0])
    assert m.n == 233
    assert m.accuracy == pytest.approx(0.828, abs=0.001)
    assert m.stage_counts == {"1.5fold": 41, "2fold": 59, "pretzel": 133}
    assert m.stage_accuracy["1.5fold"] == pytest.approx(0.659, abs=0.002)
    assert m.stage_accuracy["pretzel"] == pytest.approx(0.970, abs=0.002)


def test_metrics_match_stored_overall(hyb46, h4_vote3mm):
    """compute_metrics should reproduce the overall_accuracy stored in the JSON."""
    for r in hyb46 + h4_vote3mm:
        m = compute_metrics(r)
        assert m.accuracy == pytest.approx(r["overall_accuracy"], abs=1e-9)


def test_confusion_totals_consistent(hyb46):
    m = compute_metrics(hyb46[0])
    for gt, row in m.confusion.items():
        assert sum(row.values()) == m.stage_counts[gt]


def test_transition_mae_perfect_is_zero():
    report = _synthetic_report(gt_onset=10, pred_onset=10)
    assert transition_mae(report) == pytest.approx(0.0)


def test_transition_mae_detects_late_transition():
    # GT transitions at T10, prediction transitions at T17 → MAE for that
    # boundary is 7; the initial stage (present from T0 in both) contributes 0.
    report = _synthetic_report(gt_onset=10, pred_onset=17)
    assert transition_mae(report) == pytest.approx(3.5)  # mean of [0, 7]


def test_transition_mae_on_hyb46(hyb46):
    """RESEARCH.md: hybrid's transition errors sum to ~40 over ~9 onsets
    (3 embryos × ~3 transitions each) → MAE in single digits."""
    mae = transition_mae(hyb46[0])
    assert mae is not None
    assert 0 < mae < 15


def test_backward_transitions_counted():
    report = {
        "embryo_results": [{
            "embryo_id": "e1",
            "predictions": [
                _p(0, "comma", "comma"),
                _p(1, "1.5fold", "1.5fold"),
                _p(2, "comma", "1.5fold"),  # backward step
                _p(3, "1.5fold", "1.5fold"),
            ],
        }],
    }
    m = compute_metrics(report)
    assert m.backward_transitions == 1


def test_empty_report():
    m = compute_metrics({"embryo_results": []})
    assert isinstance(m, PerceptionMetrics)
    assert m.n == 0
    assert m.transition_mae is None


def test_format_summary_runs(hyb46):
    s = format_metrics_summary(compute_metrics(hyb46[0]))
    assert "82.8%" in s
    assert "Transition MAE" in s


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _p(tp: int, pred: str, gt: str) -> dict:
    return {
        "timepoint": tp,
        "predicted_stage": pred,
        "ground_truth_stage": gt,
        "is_correct": pred == gt,
        "is_adjacent_correct": True,
        "tool_calls": 0,
    }


def _synthetic_report(*, gt_onset: int, pred_onset: int, n: int = 30) -> dict:
    preds = []
    for t in range(n):
        gt = "1.5fold" if t < gt_onset else "2fold"
        pr = "1.5fold" if t < pred_onset else "2fold"
        preds.append(_p(t, pr, gt))
    return {"embryo_results": [{"embryo_id": "e1", "predictions": preds}]}
