"""
Z-slice subagent perception.

Max-intensity projections collapse depth: overlapping body segments fuse into
one bright band, destroying the discriminative feature for fold stages — how
many times the body folds back on itself. A biologist resolves this by
scrolling z-slices and counting body-segment cross-sections at the midplane.

This variant runs the scientific classifier on the projection, then for
fold-region timepoints sends the midplane z-slice to a subagent that counts
discrete body-segment profiles. The count is mapped to a stage and overrides
the primary call.
"""

import re

from ._base import PerceptionOutput, call_claude, parse_stage_json
from .scientific import perceive_scientific

FOLD_STAGES = {"1.5fold", "2fold", "pretzel"}

SEGMENT_SYSTEM = """\
You are analyzing a single optical z-slice through a folded C. elegans embryo \
inside its eggshell. This is NOT a max-intensity projection — it is one plane, \
so body segments that overlap in projections appear as separate cross-sections \
here.

Count the number of DISCRETE body-segment cross-sections visible inside the \
eggshell. A segment cross-section is a bright, roughly elliptical or band-like \
region of fluorescent nuclei. Segments separated by a clear dark gap are \
distinct; segments touching or merged count as one.

Typical counts by stage:
- 1 segment  → comma or early 1.5fold (body not yet folded back at this plane)
- 2 segments → 2fold (one hairpin: two parallel passes)
- 3+ segments → pretzel (multiple coils crossing this plane)

Respond with JSON:
{
  "segment_count": <integer>,
  "reasoning": "Brief description of what you counted"
}"""


def _segment_count_to_stage(count: int) -> str:
    if count >= 3:
        return "pretzel"
    if count == 2:
        return "2fold"
    return "1.5fold"


def _parse_segment_count(raw: str) -> int | None:
    data = parse_stage_json(raw)
    val = data.get("segment_count")
    if isinstance(val, int):
        return val
    if isinstance(val, str):
        m = re.search(r"\d+", val)
        if m:
            return int(m.group(0))
    return None


async def _count_segments(midplane_b64: str) -> tuple[int | None, str]:
    """Ask the subagent to count body-segment cross-sections in the midplane slice."""
    content = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": midplane_b64},
        },
        {
            "type": "text",
            "text": (
                "Count the discrete body-segment cross-sections visible in this "
                "single z-slice. Respond with the JSON format above."
            ),
        },
    ]
    raw = await call_claude(system=SEGMENT_SYSTEM, content=content, max_tokens=512)
    return _parse_segment_count(raw), raw


async def perceive_zslice(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
    midplane_b64: str | None = None,
) -> PerceptionOutput:
    """Scientific classifier with a z-slice segment-counting subagent on fold stages."""
    primary = await perceive_scientific(image_b64, references, history, timepoint)

    if midplane_b64 is None or primary.stage not in FOLD_STAGES:
        return primary

    count, sub_raw = await _count_segments(midplane_b64)
    if count is None:
        return PerceptionOutput(
            stage=primary.stage,
            reasoning=f"{primary.reasoning} | [zslice] parse failed; kept primary",
            verification_triggered=True,
            phase_count=2,
            raw_response=primary.raw_response,
        )

    sub_stage = _segment_count_to_stage(count)
    return PerceptionOutput(
        stage=sub_stage,
        reasoning=(
            f"{primary.reasoning} | [zslice] {count} segment(s) at midplane → "
            f"{sub_stage}"
            + ("" if sub_stage == primary.stage else f" (overrode {primary.stage})")
        ),
        verification_triggered=True,
        phase_count=2,
        raw_response=sub_raw,
    )
