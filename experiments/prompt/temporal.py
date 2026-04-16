"""
Temporal-anchored perception function.

Key insight: the model consistently over-advances stages (classifying 1.5fold
as 2fold, 2fold as pretzel). This variant:
1. Relies primarily on reference image comparison (not textual descriptions)
2. Uses strong temporal anchoring - stages change slowly
3. Explicitly warns against the common over-advancement error
4. Handles late pretzel stage where embryo may look different (approaching hatching)
"""

from gently_perception.api import (
    PerceptionOutput,
    build_history_text,
    build_reference_content,
    call_claude,
    response_to_output,
)

SYSTEM_PROMPT = """\
You are classifying C. elegans embryo developmental stages from fluorescence \
light-sheet microscopy max-intensity projection images. Each image shows three \
orthogonal views (XY top-left, YZ top-right, XZ bottom-left).

The stages in order are: early, bean, comma, 1.5fold, 2fold, pretzel, hatching, hatched.

## CRITICAL CLASSIFICATION RULES

1. **Compare to reference images first.** Match the overall shape, brightness \
pattern, and internal structure to the reference images provided. The references \
are your primary guide.

2. **Stages change slowly.** Each stage lasts many timepoints (typically 10-20+). \
If recent observations show stage X, the current timepoint is very likely also \
stage X unless you see a CLEAR morphological change.

3. **Never skip stages.** Development goes forward one stage at a time. If the \
last observation was "comma", the only valid classifications are "comma" or \
"1.5fold" — never "2fold" or later.

4. **COMMON ERROR — advancing too early.** The most frequent mistake is \
classifying an embryo as a MORE advanced stage than it actually is. When in \
doubt between two adjacent stages, choose the EARLIER one. Specifically:
   - Late 1.5fold can look like early 2fold — prefer 1.5fold unless you see \
TWO CLEARLY SEPARATED parallel bright bands
   - Late 2fold can look like early pretzel — prefer 2fold unless you see \
bands CROSSING OVER each other (not just getting closer)

5. **Late pretzel.** The pretzel stage is long-lasting. Near the end, the embryo \
may move within the eggshell, changing its appearance significantly. A compact \
bright mass that fills the eggshell is still pretzel even if it doesn't look \
"tangled" — it has not hatched unless you see the worm OUTSIDE the shell.

## WHAT TO LOOK FOR IN EACH VIEW

- **XY (top-left)**: Overall body shape and elongation
- **YZ (top-right)**: Cross-sectional shape (round vs elongated vs complex)
- **XZ (bottom-left)**: Layering and folding visible from the side

Respond with JSON:
{
  "stage": "early|bean|comma|1.5fold|2fold|pretzel|hatching|hatched|no_object",
  "reasoning": "Brief explanation of which reference images match best"
}"""


async def perceive_temporal(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Reference-focused classification with strong temporal anchoring."""
    content = build_reference_content(references)

    content.append({"type": "text", "text": f"\n=== CLASSIFY EMBRYO AT T{timepoint} ==="})

    history_text = build_history_text(history)
    if history_text:
        content.append({"type": "text", "text": history_text})
        # Add temporal anchoring reminder
        last_stage = history[-1].get("stage", "unknown") if history else "unknown"
        content.append({
            "type": "text",
            "text": (
                f"The most recent observation was '{last_stage}'. "
                f"Remember: stages change slowly. The current stage is most likely "
                f"'{last_stage}' unless you see a clear morphological change. "
                f"When uncertain, prefer the earlier stage."
            ),
        })

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

    content.append({
        "type": "text",
        "text": "Compare this image to the reference images above. Which stage's references does it most closely match? Classify accordingly.",
    })

    raw = await call_claude(system=SYSTEM_PROMPT, content=content)
    return response_to_output(raw)
