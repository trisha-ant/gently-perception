"""Every module under experiments/variants/ must export
``async def perceive(fi: FrameInput) -> PerceptionOutput``."""

from __future__ import annotations

import inspect

from gently_perception.types import FrameInput
from experiments.harness.discover import discover_variants


def test_discover_finds_at_least_hybrid():
    fns = discover_variants()
    assert "hybrid" in fns


def test_every_variant_has_correct_signature():
    fns = discover_variants()
    for name, fn in fns.items():
        assert inspect.iscoroutinefunction(fn), f"{name}: not async"
        params = list(inspect.signature(fn).parameters.values())
        assert len(params) == 1, f"{name}: must take exactly one arg (FrameInput)"
        ann = params[0].annotation
        assert ann is FrameInput or ann == "FrameInput", (
            f"{name}: arg must be annotated FrameInput, got {ann!r}"
        )
