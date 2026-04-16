"""Shared types for the perception harness."""

from dataclasses import dataclass


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
