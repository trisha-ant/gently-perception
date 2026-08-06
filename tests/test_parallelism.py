"""Embryo-level concurrency: independent embryos run in parallel, results
identical to sequential."""

from __future__ import annotations

import asyncio
from dataclasses import dataclass

from gently_perception.types import FrameInput, PerceptionOutput
from experiments.harness.config import RunConfig
from experiments.harness.loop import run_variant


@dataclass
class _Frame:
    embryo_id: str
    timepoint: int
    image_b64: str
    ground_truth_stage: str | None


class _Source:
    def __init__(self, by_embryo):
        self._d = by_embryo

    def iter_all(self):
        for eid, frs in self._d.items():
            yield eid, iter(frs)


def _src(n_embryos=3, n_frames=4):
    return _Source({
        f"e{i}": [_Frame(f"e{i}", t, "aW1n", "2fold") for t in range(n_frames)]
        for i in range(n_embryos)
    })


def _cfg():
    return RunConfig(variant="p", model="m", thinking=None, seed=0,
                     stages=("2fold",))


def test_embryos_run_concurrently():
    """With a perceive that awaits, max concurrent in-flight should reach
    the embryo count when concurrency >= n_embryos."""
    in_flight = {"cur": 0, "max": 0}

    async def perceive(fi: FrameInput) -> PerceptionOutput:
        in_flight["cur"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["cur"])
        await asyncio.sleep(0)  # yield to let other embryos start
        in_flight["cur"] -= 1
        return PerceptionOutput(stage="2fold", reasoning="")

    asyncio.run(run_variant(perceive, _src(n_embryos=3), {}, _cfg(),
                            concurrency=4))
    assert in_flight["max"] == 3


def test_semaphore_caps_concurrency():
    in_flight = {"cur": 0, "max": 0}

    async def perceive(fi: FrameInput) -> PerceptionOutput:
        in_flight["cur"] += 1
        in_flight["max"] = max(in_flight["max"], in_flight["cur"])
        await asyncio.sleep(0)
        in_flight["cur"] -= 1
        return PerceptionOutput(stage="2fold", reasoning="")

    asyncio.run(run_variant(perceive, _src(n_embryos=5), {}, _cfg(),
                            concurrency=2))
    assert in_flight["max"] == 2


def test_parallel_result_equals_sequential():
    async def perceive(fi: FrameInput) -> PerceptionOutput:
        await asyncio.sleep(0)
        return PerceptionOutput(stage="2fold" if fi.timepoint % 2 == 0
                                else "1.5fold", reasoning="")

    seq = asyncio.run(run_variant(perceive, _src(4, 6), {}, _cfg(),
                                  concurrency=1))
    par = asyncio.run(run_variant(perceive, _src(4, 6), {}, _cfg(),
                                  concurrency=4))
    assert seq["overall_accuracy"] == par["overall_accuracy"]
    assert seq["total_predictions"] == par["total_predictions"] == 24
    assert seq["metrics"]["per_stage"] == par["metrics"]["per_stage"]
