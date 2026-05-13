"""Adapt a legacy ``perceive_xxx(image_b64, references, history, timepoint, …)``
function to the new ``perceive(fi: FrameInput)`` contract.

Using the legacy function unchanged guarantees byte-identical API requests for
record-replay equivalence (Phase A). Once equivalence is proven, individual
variants can be rewritten natively against ``FrameInput`` if useful.
"""

from __future__ import annotations

import inspect
from typing import Awaitable, Callable

from gently_perception.types import FrameInput, PerceptionOutput

_OPTIONAL = ("midplane_b64", "zslices_b64", "top_image_b64", "side_image_b64")


def adapt(legacy_fn: Callable[..., Awaitable]) -> Callable[[FrameInput], Awaitable[PerceptionOutput]]:
    sig = inspect.signature(legacy_fn)
    has_var_kw = any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values())
    accepted = {k for k in _OPTIONAL if has_var_kw or k in sig.parameters}

    async def perceive(fi: FrameInput) -> PerceptionOutput:
        history = [{"timepoint": o.timepoint, "stage": o.stage} for o in fi.history]
        extras = {k: fi.extras.get(k) for k in accepted if k in fi.extras}
        out = await legacy_fn(
            image_b64=fi.image_b64,
            references=dict(fi.references),
            history=history,
            timepoint=fi.timepoint,
            **extras,
        )
        if isinstance(out, PerceptionOutput):
            return out
        return PerceptionOutput(stage=out.stage, reasoning=out.reasoning,
                                raw_response=getattr(out, "raw_response", ""))

    perceive.__legacy__ = legacy_fn  # type: ignore[attr-defined]
    return perceive
