"""
Pairwise 2fold-vs-pretzel disambiguator.

The de-anchored scientific prompt on Opus 4.7 confuses pretzel->2fold in
~40% of pretzel frames. Absolute classification ("is this dense or
moderate?") is harder for VLMs than forced-choice comparison ("which of
these two reference sets is this closer to?"). This variant runs scientific
first, and when the primary lands on 2fold or pretzel, runs a second call
that shows ONLY 2fold and pretzel references and asks for a binary choice.
"""

from ._base import (
    PerceptionOutput,
    call_claude,
    parse_stage_json,
)
from .scientific import perceive_scientific

_PAIR = {"2fold", "pretzel"}

_PAIR_SYSTEM = """\
You are comparing a C. elegans embryo fluorescence max-projection against two \
sets of reference images: one set of 2FOLD examples and one set of PRETZEL \
examples. Decide which set the target image is closer to.

This is a forced binary choice. Do not consider any other stage.

Respond with JSON:
{
  "stage": "2fold|pretzel",
  "reasoning": "Which reference set the target most resembles and why"
}"""


def _build_pair_content(
    image_b64: str, references: dict[str, list[str]]
) -> list[dict]:
    content: list[dict] = []
    for stage in ("2fold", "pretzel"):
        imgs = references.get(stage, [])
        if not imgs:
            continue
        content.append({"type": "text", "text": f"\n{stage.upper()} REFERENCE EXAMPLES:"})
        for b64 in imgs:
            content.append(
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": "image/jpeg", "data": b64},
                }
            )
    if content:
        content[-1]["cache_control"] = {"type": "ephemeral", "ttl": "1h"}
    content.append({"type": "text", "text": "\nTARGET IMAGE:"})
    content.append(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
        }
    )
    content.append(
        {
            "type": "text",
            "text": "Which reference set (2FOLD or PRETZEL) does the target most resemble?",
        }
    )
    return content


async def perceive_pairwise(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Scientific classifier + binary 2fold/pretzel forced-choice on those stages."""
    primary = await perceive_scientific(image_b64, references, history, timepoint)

    if primary.stage not in _PAIR:
        return primary

    raw = await call_claude(
        system=_PAIR_SYSTEM,
        content=_build_pair_content(image_b64, references),
        max_tokens=512,
    )
    data = parse_stage_json(raw)
    pair_stage = data.get("stage")
    if pair_stage not in _PAIR:
        return PerceptionOutput(
            stage=primary.stage,
            reasoning=f"{primary.reasoning} | [pairwise] parse failed; kept primary",
            verification_triggered=True,
            phase_count=2,
            raw_response=primary.raw_response,
        )

    note = (
        f"[pairwise] -> {pair_stage}"
        + ("" if pair_stage == primary.stage else f" (overrode {primary.stage})")
    )
    return PerceptionOutput(
        stage=pair_stage,
        reasoning=f"{primary.reasoning} | {note}",
        verification_triggered=True,
        phase_count=2,
        raw_response=raw,
    )
