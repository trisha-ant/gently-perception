"""
Contrastive perception function.

Uses transition-based descriptions that explicitly contrast adjacent stages,
focusing on the morphological features that distinguish confusable pairs
(comma vs 1.5fold, 1.5fold vs 2fold, 2fold vs pretzel).

Structured chain-of-thought forces the model to evaluate specific features
before making a classification.
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

The stages in developmental order are:
early → bean → comma → 1.5fold → 2fold → pretzel → hatching → hatched

## KEY DISTINGUISHING FEATURES

To classify correctly, focus on these specific transitions:

**EARLY → BEAN**: Early is a smooth, bright, uniform oval. Bean shows a slight \
asymmetry — one end narrower, or a subtle constriction/pinch beginning to form.

**BEAN → COMMA**: Bean is roughly symmetric. Comma shows a clear bend or curve \
on one side — the embryo body starts to elongate and curve like a comma or C-shape. \
Look for an asymmetric indentation or tail-like projection on one side.

**COMMA → 1.5FOLD**: This is a critical transition. In comma, the body curves \
but remains a SINGLE layer — you see one bright arc. In 1.5fold, the body has \
folded back on itself, creating TWO overlapping layers of brightness. Look for: \
a region where bright tissue OVERLAPS or doubles up, creating a brighter area \
where the fold occurs. The fold creates a hairpin turn visible as a U-shape.

**1.5FOLD → 2FOLD**: In 1.5fold, the fold is partial — you see about 1.5x the \
body length folded. In 2fold, the body has folded further, creating TWO DISTINCT \
PARALLEL bright bands separated by a dark gap. The key feature is TWO CLEAR \
PARALLEL LINES with a visible dark space between them.

**2FOLD → PRETZEL**: In 2fold, the two parallel bands are relatively straight \
and organized. In pretzel, the body has folded even further and the bands \
CROSS OVER each other, creating a tangled, complex pattern. Look for: \
MULTIPLE crossing points, bands going in different directions, a more compact \
and complex shape than the organized parallel lines of 2fold. The overall \
impression is tangled or knotted rather than parallel.

**PRETZEL → HATCHING/HATCHED**: Pretzel is a compact tangled mass within the \
eggshell. Hatching shows the worm beginning to emerge. Hatched shows a thin \
elongated worm shape outside the shell, or an empty shell.

## ANALYSIS PROCEDURE

Before classifying, analyze these features in order:
1. Is the embryo a simple oval shape? (early/bean)
2. Is there a single curved body without folding? (comma)
3. Are there overlapping bright regions suggesting folding? (1.5fold+)
4. If folded: are there exactly two parallel bands with a gap? (2fold)
5. If folded: do bands cross over each other in a complex pattern? (pretzel)
6. Is there a thin worm shape or empty shell? (hatching/hatched)

Respond with JSON:
{
  "stage": "early|bean|comma|1.5fold|2fold|pretzel|hatching|hatched|no_object",
  "reasoning": "Describe what specific features you see and which transition criteria they match"
}"""


async def perceive_contrastive(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Single API call with contrastive stage descriptions."""
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

    content.append({
        "type": "text",
        "text": (
            "Analyze the image above following the analysis procedure. "
            "Focus on whether you see: (a) simple oval, (b) single curved body, "
            "(c) overlapping/folded regions, (d) parallel bands with gap, "
            "(e) crossing/tangled bands, or (f) thin worm/empty shell. "
            "Then classify."
        ),
    })

    raw = await call_claude(system=SYSTEM_PROMPT, content=content)
    return response_to_output(raw)
