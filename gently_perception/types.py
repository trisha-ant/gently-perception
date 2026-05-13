"""Shared types for the perception harness.

`FrameInput` is the contract between the harness loop and a perceive function.
It is frozen and carries NO ground-truth field, so a variant physically cannot
read GT — the leak surface that produced the inflated judge-ensemble result is
closed at the type level.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping


# --------------------------------------------------------------------------- #
# Errors
# --------------------------------------------------------------------------- #

class ParseError(ValueError):
    """Raised when a model response cannot be parsed into a stage.

    The old harness silently fell back to ``stage="early"`` on parse failure,
    which is indistinguishable from a real early-stage prediction. Callers
    should catch this and record an explicit error event instead.
    """


class LeakError(AssertionError):
    """Raised when ground truth would reach a perceive function for a scored
    frame. The loop's history-source assertion fires this."""


# --------------------------------------------------------------------------- #
# Harness ⇄ variant contract
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class Obs:
    """A single past observation, as seen by a variant. No GT field."""

    timepoint: int
    stage: str


@dataclass(frozen=True)
class FrameInput:
    """Everything a perceive function may read for one frame.

    Immutable. Constructed only by the harness loop. ``history`` is a tuple of
    :class:`Obs` so variants cannot mutate it; ``extras`` carries optional
    image renders (midplane, z-stack, top/side) keyed by name.
    """

    image_b64: str
    history: tuple[Obs, ...]
    timepoint: int
    references: Mapping[str, list[str]]
    extras: Mapping[str, object] = field(default_factory=dict)

    def last_stage(self, default: str = "early") -> str:
        return self.history[-1].stage if self.history else default


@dataclass
class PerceptionOutput:
    """What every perceive function returns.

    The harness derives reliability from session history (stability,
    temporal analysis) rather than VLM self-reported confidence — the
    latter is uncalibrated noise (0.867 correct vs 0.857 wrong) and
    has been removed.
    """

    stage: str
    reasoning: str
    raw_response: str = ""
