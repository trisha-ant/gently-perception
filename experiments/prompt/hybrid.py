"""
Hybrid perception function.

Stage-adaptive prompt strategy: uses temporal prompt for 1.5fold/pretzel
(where it's stronger) and scientific prompt for 2fold (where it's stronger).

Based on per-stage analysis:
- temporal: 1.5fold 63%, 2fold 58%, pretzel 95%
- scientific: 1.5fold 55%, 2fold 76%, pretzel 92%
"""

from gently_perception.api import (
    PerceptionOutput,
    build_history_text,
    build_reference_content,
    call_claude,
    response_to_output,
    STAGES,
)

# Temporal prompt: better for 1.5fold (recovery behavior) and pretzel
TEMPORAL_SYSTEM = """\
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

# Scientific prompt: better for 2fold (eggshell fill criterion)
SCIENTIFIC_SYSTEM = """\
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


def _get_system_prompt(expected_stage: str) -> str:
    """Choose system prompt based on expected stage.

    Temporal prompt for 1.5fold/pretzel (better recovery + retention).
    Scientific prompt for 2fold (better eggshell fill discrimination).
    """
    if expected_stage in ("2fold", "pretzel"):
        # Use scientific for 2fold (76% vs 58%) and the 2fold→pretzel boundary.
        # Once in pretzel, the anchoring in both prompts works well,
        # but scientific's fill-fraction criterion helps at the boundary.
        return SCIENTIFIC_SYSTEM
    else:
        # Use temporal for everything else - better at 1.5fold (63% vs 55%)
        # and early stages
        return TEMPORAL_SYSTEM


async def perceive_hybrid(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Stage-adaptive classification using best prompt per expected stage."""
    # Determine expected stage from history
    last_stage = "early"
    if history:
        last_stage = history[-1].get("stage", "early")

    system_prompt = _get_system_prompt(last_stage)

    content = build_reference_content(references)

    content.append({"type": "text", "text": f"\n=== CLASSIFY EMBRYO AT T{timepoint} ==="})

    history_text = build_history_text(history)
    if history_text:
        content.append({"type": "text", "text": history_text})
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

    # Add analysis prompt based on which system we're using
    if system_prompt == SCIENTIFIC_SYSTEM:
        content.append({
            "type": "text",
            "text": (
                "Analyze: (1) How much of the eggshell is filled with signal? "
                "(2) How many parallel body segments are visible? "
                "(3) Which reference images match best? Then classify."
            ),
        })
    else:
        content.append({
            "type": "text",
            "text": "Compare this image to the reference images above. Which stage's references does it most closely match? Classify accordingly.",
        })

    raw = await call_claude(system=system_prompt, content=content)
    return response_to_output(raw)
