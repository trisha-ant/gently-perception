"""[P7] FrameInput must structurally prevent GT access; loop must assert."""

from __future__ import annotations

import dataclasses

import pytest

from gently_perception.types import FrameInput, LeakError, Obs
from experiments.harness.loop import _HistEntry, _assert_no_leak, _to_obs


def _fi() -> FrameInput:
    return FrameInput(image_b64="img", history=(Obs(0, "early"),),
                      timepoint=1, references={})


def test_frameinput_has_no_gt_field():
    fi = _fi()
    assert not hasattr(fi, "ground_truth_stage")
    assert not hasattr(fi, "gt")
    field_names = {f.name for f in dataclasses.fields(FrameInput)}
    assert "ground_truth_stage" not in field_names


def test_frameinput_is_frozen():
    fi = _fi()
    with pytest.raises(dataclasses.FrozenInstanceError):
        fi.timepoint = 99  # type: ignore[misc]


def test_history_is_immutable_tuple_of_obs():
    fi = _fi()
    assert isinstance(fi.history, tuple)
    assert all(isinstance(o, Obs) for o in fi.history)
    with pytest.raises((TypeError, AttributeError)):
        fi.history.append(Obs(2, "bean"))  # type: ignore[attr-defined]


def test_obs_has_no_gt_field():
    assert {f.name for f in dataclasses.fields(Obs)} == {"timepoint", "stage"}


def test_to_obs_strips_provenance():
    hist = [_HistEntry(0, "early", "gt"), _HistEntry(1, "bean", "pred")]
    obs = _to_obs(hist)
    assert obs == (Obs(0, "early"), Obs(1, "bean"))
    assert not hasattr(obs[0], "source")


def test_leak_guard_allows_gt_leadin_before_first_scored():
    hist = [_HistEntry(t, "early", "gt") for t in range(5)]
    _assert_no_leak(hist, first_scored_tp=5)  # should not raise


def test_leak_guard_rejects_gt_at_or_after_first_scored():
    hist = [_HistEntry(0, "early", "gt"),
            _HistEntry(5, "comma", "gt")]  # GT at a scored tp
    with pytest.raises(LeakError):
        _assert_no_leak(hist, first_scored_tp=5)


def test_leak_guard_allows_pred_after_first_scored():
    hist = [_HistEntry(0, "early", "gt"),
            _HistEntry(5, "comma", "pred"),
            _HistEntry(6, "comma", "pred")]
    _assert_no_leak(hist, first_scored_tp=5)
