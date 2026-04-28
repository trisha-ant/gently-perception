"""
Hybrid + fillpct, de-anchored for Opus 4.7.

Stage-adaptive routing (from `hybrid`): picks a prompt based on the previous
frame's stage. The 2fold/pretzel path uses the fillpct prompt (numeric
fill-% commitment before staging); the earlier-stage path uses a
reference-matching prompt with the late-pretzel handling.

Both prompts are de-anchored: no "default to previous", no "prefer earlier",
no "never skip stages" hard constraint, no user-turn reinforcement block.
"""

from ._base import (
    PerceptionOutput,
    build_history_text,
    build_reference_content,
    call_claude,
    response_to_output,
)
from .fillpct import SYSTEM_PROMPT as FILLPCT_SYSTEM

TEMPORAL_SYSTEM = """\
You are classifying C. elegans embryo developmental stages from fluorescence \
light-sheet microscopy max-intensity projection images. Each image shows three \
orthogonal views (XY top-left, YZ top-right, XZ bottom-left).

The stages in order are: early, bean, comma, 1.5fold, 2fold, pretzel, hatching, hatched.

## CLASSIFICATION RULES

1. **Compare to reference images first.** Match the overall shape, brightness \
pattern, and internal structure to the reference images provided. The \
references are your primary guide.

2. **Stages change slowly.** Each lasts many timepoints (typically 10-20+), so \
consecutive frames are usually the same stage. Development progresses one \
stage at a time. Use the previous observation as context, but classify based \
on the morphology in THIS image.

3. **Late pretzel.** The pretzel stage is long-lasting. Near the end, the \
embryo may move within the eggshell, changing its appearance significantly. A \
compact bright mass that fills the eggshell is still pretzel even if it does \
not look "tangled" -- it has not hatched unless you see the worm OUTSIDE the \
shell.

## WHAT TO LOOK FOR IN EACH VIEW

- **XY (top-left)**: Overall body shape and elongation
- **YZ (top-right)**: Cross-sectional shape (round vs elongated vs complex)
- **XZ (bottom-left)**: Layering and folding visible from the side

Respond with JSON:
{
  "stage": "early|bean|comma|1.5fold|2fold|pretzel|hatching|hatched|no_object",
  "reasoning": "Brief explanation of which reference images match best"
}"""


def _get_system_prompt(last_stage: str) -> tuple[str, str]:
    """Return (system_prompt, analysis_prompt) for the expected region."""
    if last_stage in ("2fold", "pretzel"):
        return (
            FILLPCT_SYSTEM,
            "First estimate fill_pct (percentage of eggshell interior covered by "
            "bright signal in the XY view). Then classify the stage.",
        )
    return (
        TEMPORAL_SYSTEM,
        "Compare this image to the reference images above. Which stage's "
        "references does it most closely match? Classify accordingly.",
    )


async def perceive_hybrid_fillpct(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Stage-adaptive routing with de-anchored prompts; fillpct on the 2fold/pretzel path."""
    last_stage = history[-1].get("stage", "early") if history else "early"
    system_prompt, analysis_prompt = _get_system_prompt(last_stage)

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
    content.append({"type": "text", "text": analysis_prompt})

    raw = await call_claude(system=system_prompt, content=content)
    return response_to_output(raw)
