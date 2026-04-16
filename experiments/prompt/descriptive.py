"""
Descriptive perception function.

Single API call with projection-grounded stage descriptions (what each stage
looks like in the three-view images), reference images, no tools.
"""

from gently_perception.api import (
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

Developmental stages in order (what they look like in the projections):
- EARLY: Bright oval, uniform, symmetric
- BEAN: Oval with one end slightly narrower, or a pinch in the middle
- COMMA: One edge of the oval starts to flatten or curve inward (the other stays convex)
- 1.5FOLD: The bright mass starts to look like it has two layers, one tucking under
- 2FOLD: Two distinct parallel bright bands with a dark gap
- PRETZEL: Tangled bright mass, multiple crossing bands, compact
- HATCHED: The bright mass is gone or a thin worm shape is visible outside the shell

Reference images for each stage are provided above. Compare the current image \
to the references. If the field of view is empty, return "no_object".

Respond with JSON:
{
  "stage": "early|bean|comma|1.5fold|2fold|pretzel|hatching|hatched|no_object",
  "reasoning": "Brief explanation"
}"""


async def perceive_descriptive(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Single API call with descriptive stage definitions and reference images."""
    content = build_reference_content(references)

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
