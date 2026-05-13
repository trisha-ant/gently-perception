"""Verify benchmark/stats.py against the hand-computed t-values in RESEARCH.md.

These tests pin the stats implementation to known-good numbers so future
harness changes can't silently shift what "significant" means.
"""

from __future__ import annotations

import math

import pytest

from benchmark.stats import RunSet, compare, welch_t


def test_runset_from_reports(hyb46):
    rs = RunSet.from_reports("hybrid46", hyb46)
    assert rs.n == 3
    # RESEARCH.md: 81.7 ± 2.4
    assert rs.mean == pytest.approx(0.817, abs=0.002)
    assert rs.std == pytest.approx(0.024, abs=0.002)
    assert rs.mean_std_pct() == "81.7 ± 2.4"


def test_welch_t_hybrid_vs_vote3mm(hyb46, h4_vote3mm):
    """RESEARCH.md / memory: hybrid@4.6 vs vote3_mm@4.7 → t≈0.61, not significant."""
    cmp = compare(hyb46, h4_vote3mm, "hybrid46", "vote3_mm")
    assert cmp.t == pytest.approx(0.61, abs=0.05)
    assert cmp.delta == pytest.approx(0.009, abs=0.002)  # +0.9pp
    assert not cmp.significant


def test_welch_t_symmetric_magnitude(hyb46, h4_vote3mm):
    fwd = compare(hyb46, h4_vote3mm)
    rev = compare(h4_vote3mm, hyb46)
    assert fwd.t == pytest.approx(-rev.t, abs=1e-9)
    assert fwd.df == pytest.approx(rev.df, abs=1e-9)


def test_welch_t_identical_sets_is_zero(hyb46):
    cmp = compare(hyb46, hyb46)
    assert cmp.t == pytest.approx(0.0, abs=1e-9)
    assert not cmp.significant


def test_welch_t_requires_two_runs():
    a = RunSet("a", (0.8,))
    b = RunSet("b", (0.7, 0.71))
    with pytest.raises(ValueError):
        welch_t(a, b)


def test_significant_threshold():
    """Manufactured separation: |t| > 2 → flagged."""
    a = RunSet("a", (0.90, 0.91, 0.89))
    b = RunSet("b", (0.70, 0.71, 0.69))
    cmp = welch_t(a, b)
    assert cmp.significant
    assert cmp.t > 2.0


def test_zero_variance_handled():
    a = RunSet("a", (0.80, 0.80, 0.80))
    b = RunSet("b", (0.70, 0.70, 0.70))
    cmp = welch_t(a, b)
    assert math.isinf(cmp.t)
    assert cmp.significant
