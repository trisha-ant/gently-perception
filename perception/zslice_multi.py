"""
Multi-plane z-slice subagent: max segment-count across several depths.

The single-midplane variants (zslice, zslice_asym) failed because one z-plane
rarely intersects >=3 body segments at separable locations -- folds sit at
different depths. This variant counts segments at five z-fractions
(0.25/0.40/0.50/0.60/0.75 of Z) and takes the *maximum*. The hypothesis is
that a pretzel embryo will show >=3 segments on at least one plane even if no
single plane does.

Override is symmetric (full FOLD_STAGES trigger) so the hypothesis is tested
cleanly; if max-count is still <=2 on pretzel, asymmetric damage control can
be layered on afterward.
"""

import asyncio

from ._base import PerceptionOutput
from .scientific import perceive_scientific
from .zslice import FOLD_STAGES, _count_segments, _segment_count_to_stage


async def _max_segment_count(zslices_b64: list[str]) -> tuple[int | None, list[int | None]]:
    """Count segments on each slice concurrently; return (max, per-slice list)."""
    results = await asyncio.gather(*(_count_segments(b64) for b64 in zslices_b64))
    counts: list[int | None] = [c for c, _ in results]
    valid = [c for c in counts if c is not None]
    return (max(valid) if valid else None), counts


async def perceive_zslice_multi(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
    zslices_b64: list[str] | None = None,
) -> PerceptionOutput:
    """Scientific classifier + max-count override across multiple z-slices."""
    primary = await perceive_scientific(image_b64, references, history, timepoint)

    if not zslices_b64 or primary.stage not in FOLD_STAGES:
        return primary

    max_count, per_slice = await _max_segment_count(zslices_b64)
    counts_str = "/".join("-" if c is None else str(c) for c in per_slice)

    if max_count is None:
        return PerceptionOutput(
            stage=primary.stage,
            reasoning=f"{primary.reasoning} | [zslice-multi] all parses failed; kept primary",
            verification_triggered=True,
            phase_count=2,
            raw_response=primary.raw_response,
        )

    sub_stage = _segment_count_to_stage(max_count)
    return PerceptionOutput(
        stage=sub_stage,
        reasoning=(
            f"{primary.reasoning} | [zslice-multi] counts {counts_str} max={max_count} -> "
            f"{sub_stage}"
            + ("" if sub_stage == primary.stage else f" (overrode {primary.stage})")
        ),
        verification_triggered=True,
        phase_count=2,
        raw_response=primary.raw_response,
    )
