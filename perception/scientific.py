"""
Scientific perception function.

Combines the temporal anchoring from v1 (best performer) with
scientifically-grounded visual criteria from C. elegans morphology
literature. Key improvements over temporal v1:
1. Uses eggshell fill fraction as primary discriminator
2. Counts parallel body segments instead of vague shape descriptions
3. References overall fluorescence intensity (increases with fold)
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

## STAGE DESCRIPTIONS (what to look for in fluorescence max-projections)

**EARLY**: Bright oval mass of nuclei. Uniform, roughly symmetric.

**BEAN**: Slight asymmetry — one end narrower or a subtle constriction.

**COMMA**: The embryo body has begun to elongate and curve into a C or comma \
shape. ONE curved band of nuclei visible. The eggshell is MOSTLY EMPTY — the \
body occupies a small fraction of the eggshell interior.

**1.5FOLD**: The embryo body has begun folding back on itself inside the \
eggshell. You see ONE main curve with a partial fold — the tail has curled \
back partway but has NOT reached the head. In the max-projection, look for a \
J-shape or partial hairpin. The eggshell is SPARSELY FILLED — significant \
dark/empty space remains within the eggshell boundary.

**2FOLD**: The body has folded to form a clear hairpin or U-shape — TWO \
PARALLEL body segments are visible, connected by a bend at one end. The tail \
has elongated to approximately reach the head. The eggshell is MODERATELY \
FILLED — some dark space remains but less than 1.5fold.

**PRETZEL**: The body has folded THREE or more times, creating MULTIPLE \
overlapping coils within the eggshell. The eggshell is DENSELY FILLED with \
fluorescent signal — very little empty/dark space inside the eggshell \
boundary. The overall brightness is higher because multiple body layers \
overlap in the projection. The pattern looks complex and tangled.

**HATCHING/HATCHED**: The worm is emerging or has left the eggshell. You see \
a thin elongated worm shape OUTSIDE the eggshell boundary, or an empty shell.

## CLASSIFICATION RULES

1. **Compare to references first**, then use the descriptions above.

2. **Stages change slowly** — each lasts many timepoints (10-60+). Default \
to the same stage as previous observation unless morphology clearly changed.

3. **When in doubt, choose the EARLIER stage.** The most common error is \
classifying too advanced.

4. **KEY DISCRIMINATOR: eggshell fill fraction.** How much of the eggshell \
interior is filled with bright signal?
   - Sparse (lots of dark space inside shell) → 1.5fold or earlier
   - Moderate (some dark space) → 2fold
   - Dense (shell mostly filled, bright) → pretzel

Respond with JSON:
{
  "stage": "early|bean|comma|1.5fold|2fold|pretzel|hatching|hatched|no_object",
  "reasoning": "Brief explanation"
}"""


async def perceive_scientific(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Scientific criteria with temporal anchoring."""
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
        "text": (
            "Analyze: (1) How much of the eggshell is filled with signal? "
            "(2) How many parallel body segments are visible? "
            "(3) Which reference images match best? Then classify."
        ),
    })

    raw = await call_claude(system=SYSTEM_PROMPT, content=content)
    return response_to_output(raw)
