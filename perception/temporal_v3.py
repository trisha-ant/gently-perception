"""
Temporal-anchored perception v3.

Builds on temporal v1 (best performer) with:
1. Stronger late-pretzel guidance to prevent hatched misclassification
2. More explicit pretzel description for late pretzel morphology
"""

from ._base import (
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
pattern, and internal structure to the reference images provided.

2. **Stages change slowly.** Each stage lasts many timepoints (typically 10-60+). \
If recent observations show stage X, the current timepoint is very likely also \
stage X unless you see a CLEAR morphological change.

3. **Never skip stages.** Development goes forward one stage at a time.

4. **COMMON ERROR — advancing too early.** When in doubt between two adjacent \
stages, choose the EARLIER one:
   - Late 1.5fold can look like early 2fold — prefer 1.5fold unless you see \
TWO CLEARLY SEPARATED parallel bright bands
   - Late 2fold can look like early pretzel — prefer 2fold unless you see \
bands CROSSING OVER each other

5. **PRETZEL IS THE LONGEST STAGE.** The pretzel stage typically lasts 50-100+ \
timepoints. During this stage, the embryo is a compact, coiled worm inside the \
eggshell. The appearance changes dramatically throughout pretzel:
   - Early pretzel: tangled bright bands, complex crossing pattern
   - Mid pretzel: compact bright mass, may look smoother as body compacts
   - Late pretzel: the embryo becomes ACTIVE and MOVES inside the shell. \
The image may show the worm in different positions each timepoint. It may \
look like a bright oval, a thin elongated shape, or even appear partially \
empty — but as long as the embryo is INSIDE the eggshell, it is still pretzel.

6. **HATCHING IS RARE in this dataset.** Do NOT classify as hatching or hatched \
unless you can clearly see the worm body OUTSIDE and SEPARATE from the eggshell. \
If the worm is still contained within the shell boundary (even if the shell \
looks different), classify as pretzel. If you see a bright mass inside an \
oval boundary, it is pretzel.

Respond with JSON:
{
  "stage": "early|bean|comma|1.5fold|2fold|pretzel|hatching|hatched|no_object",
  "reasoning": "Brief explanation of which reference images match best"
}"""


async def perceive_temporal_v3(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Reference-focused classification with strong temporal anchoring v3."""
    content = build_reference_content(references)

    content.append({"type": "text", "text": f"\n=== CLASSIFY EMBRYO AT T{timepoint} ==="})

    history_text = build_history_text(history)
    if history_text:
        content.append({"type": "text", "text": history_text})
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
