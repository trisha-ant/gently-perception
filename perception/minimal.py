"""
Minimal perception function.

Single API call, stage names only in the prompt, reference images provided,
no tools.  This is the simplest baseline: "here are examples, classify."
"""

from ._base import (
    PerceptionOutput,
    build_history_text,
    build_reference_content,
    call_claude,
    response_to_output,
)

SYSTEM_PROMPT = """\
You are classifying C. elegans embryo developmental stages from light-sheet \
microscopy images. Each image shows three orthogonal max-intensity projections \
(XY top-left, YZ top-right, XZ bottom).

The developmental stages in order are: early, bean, comma, 1.5fold, 2fold, \
pretzel, hatched. Reference images for each stage are provided above.

Classify the current image. If the field of view is empty, return "no_object".

Respond with JSON:
{
  "stage": "early|bean|comma|1.5fold|2fold|pretzel|hatching|hatched|no_object",
  "reasoning": "Brief explanation"
}"""


async def perceive_minimal(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Single API call with minimal prompt text and reference images."""
    # Build user content: references (cached) + history + current image
    content = build_reference_content(references)

    # Dynamic section
    content.append({"type": "text", "text": f"\n=== ANALYZE EMBRYO AT T{timepoint} ==="})

    history_text = build_history_text(history)
    if history_text:
        content.append({"type": "text", "text": history_text})

    content.append(
        {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": "image/jpeg",
                "data": image_b64,
            },
        }
    )

    raw = await call_claude(system=SYSTEM_PROMPT, content=content)
    return response_to_output(raw)
