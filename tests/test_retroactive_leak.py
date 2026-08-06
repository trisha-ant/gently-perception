"""Retroactive bug catch: replay the judge-ensemble scenario through the new
harness and assert the GT-leak path that produced 90.6% is structurally closed.

The original ``experiments/judge_ensemble/judge_replicate.py:131`` built the
judge's history from ``tc.ground_truth_stage`` for every frame. Under the new
harness, the judge is a ``perceive(FrameInput)`` and ``FrameInput`` has no GT
field — so even a malicious judge that *tries* to read GT cannot.
"""

from __future__ import annotations

import asyncio
import dataclasses
from dataclasses import dataclass

import pytest

from gently_perception.types import FrameInput, LeakError, PerceptionOutput
from experiments.harness.config import RunConfig
from experiments.harness.events import EventLog
from experiments.harness.loop import run_variant


@dataclass
class _Frame:
    embryo_id: str
    timepoint: int
    image_b64: str
    ground_truth_stage: str | None


class _Source:
    def __init__(self, frames):
        self._frames = frames

    def iter_all(self):
        yield "e1", iter(self._frames)


def _cfg():
    return RunConfig(variant="judge_ensemble", model="stub", thinking=None,
                     seed=0, stages=("1.5fold", "2fold", "pretzel"))


def test_judge_pipeline_cannot_see_gt(monkeypatch):
    """Even when the pipeline disagrees and the judge fires, the FrameInput it
    receives has no GT — verified by introspection inside the judge call."""
    import experiments.pipelines.judge_ensemble as je

    seen_inputs: list[FrameInput] = []

    async def expert_a(fi: FrameInput) -> PerceptionOutput:
        return PerceptionOutput(stage="2fold", reasoning="a")

    async def expert_b(fi: FrameInput) -> PerceptionOutput:
        return PerceptionOutput(stage="pretzel", reasoning="b")

    async def fake_call_claude(system, content, **_kw):
        # Only the judge reaches call_claude (experts are stubbed above).
        return '{"stage": "2fold", "reasoning": "j"}'

    monkeypatch.setattr(je, "expert_a", expert_a)
    monkeypatch.setattr(je, "expert_b", expert_b)
    monkeypatch.setattr(je, "call_claude", fake_call_claude)

    orig_judge = je._judge

    async def spy_judge(fi, a, b):
        seen_inputs.append(fi)
        return await orig_judge(fi, a, b)

    monkeypatch.setattr(je, "_judge", spy_judge)

    src = _Source([
        _Frame("e1", t, "aW1n",
               # GT alternates so an oracle could hit 100%
               "2fold" if t % 2 == 0 else "pretzel")
        for t in range(6)
    ])
    log = EventLog()
    result = asyncio.run(run_variant(je.perceive, src, {}, _cfg(),
                                     event_log=log, concurrency=1))

    # Judge fired (disagreement on every frame)
    assert len(seen_inputs) == 6
    # The FrameInput the judge sees has no GT field — structural guarantee
    for fi in seen_inputs:
        assert not hasattr(fi, "ground_truth_stage")
        for o in fi.history:
            assert {f.name for f in dataclasses.fields(o)} == {"timepoint", "stage"}
    # Judge always picked "2fold"; GT alternates → 50%, not the leaky 90+%.
    assert result["overall_accuracy"] == pytest.approx(0.5)


def test_loop_raises_leakerror_if_pipeline_smuggles_gt_into_history():
    """A pipeline cannot inject GT-sourced history for scored frames — the
    loop owns history. This asserts that even the loop's OWN lead-in handling
    refuses GT entries at-or-after the first scored frame."""
    # GT lead-in is fine before scoring; this scenario interleaves a GT-stage
    # frame *after* scoring has started by using a stage filter that re-enters.
    src = _Source([
        _Frame("e1", 0, "i", "early"),     # filtered → GT lead-in
        _Frame("e1", 1, "i", "1.5fold"),   # scored (first_scored=1)
        _Frame("e1", 2, "i", "early"),     # filtered → GT entry at T2 ≥ 1 → LEAK
        _Frame("e1", 3, "i", "1.5fold"),   # would be scored
    ])

    async def perceive(fi: FrameInput) -> PerceptionOutput:
        return PerceptionOutput(stage="1.5fold", reasoning="")

    with pytest.raises(LeakError):
        asyncio.run(run_variant(perceive, src, {},
                                RunConfig(variant="x", model="m", thinking=None,
                                          seed=0, stages=("1.5fold",)),
                                concurrency=1))
