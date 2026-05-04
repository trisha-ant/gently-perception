"""
multimeasure: commit to three visual measurements before staging.

fillpct showed that forcing a numeric fill_pct before the stage helps. This
extends the idea: ask for fill_pct + n_passes + bright_uniformity, then
classify using the same rubric as fillpct. The model has to reconcile its
own observations (e.g. "n_passes=3" is hard to follow with "2fold").

Unlike fillpct_v2, this does NOT redefine the 2fold/pretzel boundary -- it
keeps fillpct's stage descriptions and fill-% rubric, just adds more
structured observation upfront.
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

**BEAN**: Slight asymmetry -- one end narrower or a subtle constriction.

**COMMA**: The embryo body has begun to elongate and curve into a C or comma \
shape. ONE curved band of nuclei visible. The eggshell is mostly empty -- the \
body occupies a small fraction of the eggshell interior.

**1.5FOLD**: The embryo body has begun folding back on itself inside the \
eggshell. You see ONE main curve with a partial fold -- the tail has curled \
back partway but has NOT reached the head. In the max-projection, look for a \
J-shape or partial hairpin. Significant dark/empty space remains within the \
eggshell boundary.

**2FOLD**: The body has folded to form a clear hairpin or U-shape -- TWO \
PARALLEL body segments are visible, connected by a bend at one end. The tail \
has elongated to approximately reach the head.

**PRETZEL**: The body has folded THREE or more times, creating MULTIPLE \
overlapping coils within the eggshell. The overall brightness is higher \
because multiple body layers overlap in the projection. The pattern looks \
complex and tangled.

**HATCHING/HATCHED**: The worm is emerging or has left the eggshell. You see \
a thin elongated worm shape OUTSIDE the eggshell boundary, or an empty shell.

## CLASSIFICATION RULES

1. **Compare to references first**, then use the descriptions above.

2. **Stages change slowly** -- each lasts many timepoints (10-60+), so \
consecutive frames are usually the same stage. Use the previous observation \
as context, but classify based on the morphology in THIS image.

3. **Fill percentage is the key fold-stage discriminator.** Estimate what \
percentage of the eggshell interior area is occupied by bright fluorescent \
signal in the XY view. Use the reference images to calibrate your estimate. \
Typical ranges:
   - <40%  -> comma or earlier
   - 40-60% -> 1.5fold
   - 60-75% -> 2fold
   - >75%  -> pretzel

Respond with JSON:
{
  "fill_pct": <integer 0-100, eggshell-interior fill in XY view>,
  "n_passes": <integer, how many distinct body passes you can resolve in the XY view>,
  "bright_uniformity": "uniform|layered",
    // "uniform" = brightness is roughly even across the body region
    // "layered" = some regions are noticeably brighter (multiple body layers overlapping in projection)
  "stage": "early|bean|comma|1.5fold|2fold|pretzel|hatching|hatched|no_object",
  "reasoning": "Brief explanation"
}"""


async def perceive_multimeasure(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """fillpct + n_passes + bright_uniformity observations before stage."""
    content = build_reference_content(references)
    content.append({"type": "text", "text": f"\n=== CLASSIFY EMBRYO AT T{timepoint} ==="})

    history_text = build_history_text(history)
    if history_text:
        content.append({"type": "text", "text": history_text})

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
