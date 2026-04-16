"""
Unified perception function.

Merges the best elements of temporal (recovery behavior, soft anchoring)
and scientific (eggshell fill criterion, body segment counting) into a
single prompt. No prompt switching = no transition artifacts.

Key design choices:
1. Eggshell fill fraction as primary discriminator (from scientific)
2. Body segment counting (from scientific)
3. Soft temporal anchoring with recovery ability (from temporal)
4. Analysis procedure that forces structured reasoning
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

Stages in order: early, bean, comma, 1.5fold, 2fold, pretzel, hatching, hatched.

## VISUAL CRITERIA (fluorescence max-projections)

The embryo develops inside a fixed-size oval eggshell (~50um). As the body \
elongates, it folds back on itself within the shell. The key discriminator \
is HOW MUCH of the eggshell is filled with fluorescent signal.

**EARLY/BEAN/COMMA**: A single bright mass or curved body. The eggshell \
interior is MOSTLY EMPTY (body occupies <40% of shell area).

**1.5FOLD**: The body has begun folding — you see a J-shape or partial \
hairpin. ONE main body segment with a partial fold. The eggshell is \
SPARSELY FILLED (~40-55% of shell area filled). Key: the fold is PARTIAL — \
the tail has NOT reached back to the head.

**2FOLD**: The body forms a clear hairpin — TWO PARALLEL body segments \
connected at one end. The eggshell is MODERATELY FILLED (~55-75% of shell \
area). Key: you can trace TWO DISTINCT parallel bright bands with a dark \
gap between them.

**PRETZEL**: THREE OR MORE overlapping coils. The eggshell is DENSELY FILLED \
(>75% of shell area, very little dark space inside). The overall brightness \
is higher due to overlapping body layers. The pattern looks complex/tangled.

**HATCHING/HATCHED**: Worm emerging or outside the eggshell.

## CLASSIFICATION RULES

1. **Compare to references first** — which stage's reference images does \
the current image most closely resemble?

2. **Stages change slowly.** Each stage lasts 10-60+ timepoints. If the \
last observation was stage X, the current is very likely X too. Only advance \
if morphology CLEARLY changed.

3. **When in doubt, choose the EARLIER stage.** The most common error is \
classifying too advanced. Specifically:
   - If you're unsure between 1.5fold and 2fold, choose 1.5fold
   - If you're unsure between 2fold and pretzel, choose 2fold

4. **Development only goes forward.** Never classify an earlier stage than \
a confirmed later stage.

Respond with JSON:
{
  "stage": "early|bean|comma|1.5fold|2fold|pretzel|hatching|hatched|no_object",
  "reasoning": "Brief explanation"
}"""


async def perceive_unified(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Unified classification combining temporal anchoring + scientific criteria."""
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
            "Analyze step by step: "
            "(1) What fraction of the eggshell is filled with signal — sparse, moderate, or dense? "
            "(2) How many distinct parallel body segments can you count? "
            "(3) Which reference images match best? "
            "Then classify."
        ),
    })

    raw = await call_claude(system=SYSTEM_PROMPT, content=content)
    return response_to_output(raw)
