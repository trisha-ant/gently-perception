"""H1: multimeasure with history dropped — tests cascade hypothesis."""

from ._base import PerceptionOutput, build_reference_content, call_claude, response_to_output
from .multimeasure import SYSTEM_PROMPT


async def perceive_nohist_mm(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    content = build_reference_content(references)
    content.append({"type": "text", "text": f"\n=== CLASSIFY EMBRYO AT T{timepoint} ==="})
    content.append(
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
        }
    )
    content.append(
        {
            "type": "text",
            "text": (
                "First record your observations (fill_pct, n_passes, bright_uniformity) "
                "from the XY view. Then classify the stage."
            ),
        }
    )
    raw = await call_claude(system=SYSTEM_PROMPT, content=content)
    return response_to_output(raw)
