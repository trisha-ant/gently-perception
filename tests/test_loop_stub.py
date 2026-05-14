"""Loop integration with a scripted stub model — no API, no volumes."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

import pytest

from gently_perception.types import FrameInput, ParseError, PerceptionOutput
from experiments.harness.config import RunConfig
from experiments.harness.events import EventLog, reduce_events
from experiments.harness.loop import run_variant


# --------------------------------------------------------------------------- #
# Minimal in-memory FrameSource conforming to the loop's protocol
# --------------------------------------------------------------------------- #

@dataclass
class _Frame:
    embryo_id: str
    timepoint: int
    image_b64: str
    ground_truth_stage: str | None


class _Source:
    def __init__(self, frames_by_embryo: dict[str, list[_Frame]]):
        self._d = frames_by_embryo

    def iter_all(self):
        for eid, frs in self._d.items():
            yield eid, iter(frs)


def _cfg(stages=("1.5fold", "2fold")) -> RunConfig:
    return RunConfig(variant="stub", model="stub-model", thinking=None,
                     seed=0, stages=stages)


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #

def test_loop_phases_ordered_and_history_cascades():
    """render→classify→verify→score per frame; history carries pred forward."""
    seen_histories: list[tuple] = []

    async def perceive(fi: FrameInput) -> PerceptionOutput:
        seen_histories.append(tuple(o.stage for o in fi.history))
        return PerceptionOutput(stage="2fold", reasoning="stub")

    src = _Source({"e1": [
        _Frame("e1", 0, "img", "early"),       # lead-in (filtered out)
        _Frame("e1", 1, "img", "1.5fold"),     # scored
        _Frame("e1", 2, "img", "2fold"),       # scored
        _Frame("e1", 3, "img", "2fold"),       # scored
    ]})
    log = EventLog()
    result = asyncio.run(run_variant(perceive, src, {}, _cfg(), event_log=log))

    # Phase ordering for the first scored frame
    kinds = [e.kind for e in log.events if e.timepoint == 1]
    assert kinds == ["render", "classify", "verify", "score"]

    # History cascade: T1 sees lead-in "early"; T2 sees pred "2fold"; T3 ditto
    assert seen_histories == [("early",), ("early", "2fold"),
                              ("early", "2fold", "2fold")]

    # Result derived from events
    assert result["total_predictions"] == 3
    assert result["overall_accuracy"] == pytest.approx(2 / 3)


def test_loop_emits_error_event_on_parse_failure_not_silent_early():
    """[P10] ParseError → explicit error event, frame not scored."""
    async def perceive(fi: FrameInput) -> PerceptionOutput:
        raise ParseError("unparseable")

    src = _Source({"e1": [_Frame("e1", 0, "img", "1.5fold")]})
    log = EventLog()
    result = asyncio.run(run_variant(perceive, src, {}, _cfg(), event_log=log))

    kinds = [e.kind for e in log.events]
    assert "error" in kinds
    assert "score" not in kinds  # not silently scored as "early"
    assert result["total_predictions"] == 0
    assert result["errors"][0]["error"] == "ParseError"


def test_reduce_events_matches_loop_output():
    """[P9] result is derived from events — re-reducing must be idempotent."""
    async def perceive(fi: FrameInput) -> PerceptionOutput:
        return PerceptionOutput(stage=("1.5fold", "2fold")[fi.timepoint % 2],
                                reasoning="r")

    src = _Source({"e1": [_Frame("e1", t, "img",
                                 ("1.5fold", "2fold")[t % 2])
                          for t in range(6)]})
    log = EventLog()
    cfg = _cfg()
    result = asyncio.run(run_variant(perceive, src, {}, cfg, event_log=log))
    rederived = reduce_events(log.events, cfg.to_dict())
    assert result == rederived


def test_verify_flags_backward_transition():
    async def perceive(fi: FrameInput) -> PerceptionOutput:
        # Regress on T2
        return PerceptionOutput(stage="comma" if fi.timepoint == 2 else "2fold",
                                reasoning="")

    src = _Source({"e1": [_Frame("e1", t, "img", "2fold") for t in range(3)]})
    log = EventLog()
    asyncio.run(run_variant(perceive, src, {}, _cfg(stages=("2fold",)),
                            event_log=log))
    verify_at_2 = [e for e in log.events
                   if e.kind == "verify" and e.timepoint == 2][0]
    assert verify_at_2.data["monotone"] is False


def test_config_provenance_captured_and_embedded():
    cfg = RunConfig.capture(variant="hybrid", model="m", thinking=None,
                            seed=0, stages=("2fold",))
    d = cfg.to_dict()
    assert d["variant"] == "hybrid"
    assert d["git_sha"]  # non-empty in a git repo
    assert "captured_at" in d
    # result_path is append-only shape and includes thinking
    p = cfg.result_path(__import__("pathlib").Path("/tmp/r"))
    assert p.parts[-4:] == ("hybrid", "m", "none", f"{cfg.git_short}_0.json")
