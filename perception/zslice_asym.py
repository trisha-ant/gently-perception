"""
Z-slice subagent with asymmetric (upgrade-only) override.

The original zslice variant regressed pretzel 88% -> 7% because a single
midplane slice rarely shows >=3 separable cross-sections on a pretzel embryo
(body folds sit at different z-depths). This variant keeps the segment-counting
subagent but only lets it move the prediction *forward* in the fold ordering
(1.5fold -> 2fold -> pretzel), never back.

Trigger excludes pretzel since there is nothing to upgrade to, so the worst
case on that stage is parity with `scientific`.
"""

from ._base import PerceptionOutput
from .scientific import perceive_scientific
from .zslice import _count_segments, _segment_count_to_stage

_FOLD_ORDER = {"1.5fold": 0, "2fold": 1, "pretzel": 2}
_TRIGGER_STAGES = {"1.5fold", "2fold"}


async def perceive_zslice_asym(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
    midplane_b64: str | None = None,
) -> PerceptionOutput:
    """Scientific classifier + upgrade-only z-slice override on 1.5fold/2fold."""
    primary = await perceive_scientific(image_b64, references, history, timepoint)

    if midplane_b64 is None or primary.stage not in _TRIGGER_STAGES:
        return primary

    count, sub_raw = await _count_segments(midplane_b64)
    if count is None:
        return PerceptionOutput(
            stage=primary.stage,
            reasoning=f"{primary.reasoning} | [zslice-asym] parse failed; kept primary",
            verification_triggered=True,
            phase_count=2,
            raw_response=primary.raw_response,
        )

    sub_stage = _segment_count_to_stage(count)
    if _FOLD_ORDER[sub_stage] > _FOLD_ORDER[primary.stage]:
        final = sub_stage
        note = f"{count} segment(s) -> {sub_stage} (upgraded from {primary.stage})"
    else:
        final = primary.stage
        note = f"{count} segment(s) -> {sub_stage}; not above primary, kept {primary.stage}"

    return PerceptionOutput(
        stage=final,
        reasoning=f"{primary.reasoning} | [zslice-asym] {note}",
        verification_triggered=True,
        phase_count=2,
        raw_response=sub_raw,
    )
