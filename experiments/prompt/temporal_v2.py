"""
Temporal-anchored perception v2.

Builds on temporal v1 with stronger stage-stickiness and explicit guidance
for the confusable boundaries. Key improvements:
1. Explicit ~90% prior toward the current stage
2. Stage-boundary-specific criteria for when to advance
3. Late pretzel handling (embryo may hatch before annotation ends)
"""

from gently_perception.api import (
    PerceptionOutput,
    build_reference_content,
    call_claude,
    response_to_output,
    STAGES,
)

SYSTEM_PROMPT = """\
You are classifying C. elegans embryo developmental stages from fluorescence \
light-sheet microscopy max-intensity projection images. Each image shows three \
orthogonal views (XY top-left, YZ top-right, XZ bottom-left).

Stages in order: early, bean, comma, 1.5fold, 2fold, pretzel, hatching, hatched.

## HOW TO CLASSIFY

**Step 1: Compare to references.** Look at the reference images above. Which \
stage's references does the current image most closely resemble in terms of \
overall shape, brightness pattern, and internal structure?

**Step 2: Apply temporal prior.** Stages last many timepoints (10-60+). \
If recent history shows stage X, there is roughly a 90% chance the current \
timepoint is also stage X. Only advance to X+1 if you are HIGHLY CONFIDENT \
the morphology has clearly changed. When in doubt, stay at the current stage.

**Step 3: Check transition criteria.** To advance from one stage to the next, \
you must see these SPECIFIC changes:

- **comma → 1.5fold**: The body must have started FOLDING BACK on itself. \
Look for a region where bright tissue OVERLAPS, creating a doubled/brighter \
area. A single curved body (even if strongly curved) is still comma.

- **1.5fold → 2fold**: You must see TWO CLEARLY SEPARATED parallel bright \
bands with a DARK GAP between them. If the bands are close together or \
partially overlapping, it is still 1.5fold. Late 1.5fold naturally looks \
like early 2fold — prefer 1.5fold unless the separation is unambiguous.

- **2fold → pretzel**: The parallel bands must be CROSSING OVER each other, \
creating a complex tangled pattern. If bands are still roughly parallel \
(even if slightly curved), it is still 2fold. Late 2fold naturally has \
some curvature — that alone does not make it pretzel.

- **pretzel → hatching**: You must see the worm EMERGING from the eggshell. \
A compact bright mass inside the eggshell is still pretzel, even if it \
looks different from earlier pretzel. Late pretzel embryos are active and \
may look very different from early pretzel — they are still pretzel until \
emergence is visible.

## OUTPUT

Respond with JSON:
{
  "stage": "early|bean|comma|1.5fold|2fold|pretzel|hatching|hatched|no_object",
  "reasoning": "Brief explanation"
}"""


def _build_history_text(history: list[dict]) -> str:
    """Format temporal context with emphasis on current stage."""
    if not history:
        return ""
    lines = ["RECENT OBSERVATIONS:"]
    for obs in history[-3:]:
        tp = obs.get("timepoint", "?")
        stage = obs.get("stage", "?")
        lines.append(f"- T{tp}: {stage}")
    return "\n".join(lines)


async def perceive_temporal_v2(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Reference-focused classification with strong temporal anchoring v2."""
    content = build_reference_content(references)

    content.append({"type": "text", "text": f"\n=== CLASSIFY EMBRYO AT T{timepoint} ==="})

    history_text = _build_history_text(history)
    if history_text:
        content.append({"type": "text", "text": history_text})

        last_stage = history[-1].get("stage", "unknown") if history else "unknown"
        # Determine valid transitions
        if last_stage in STAGES:
            idx = STAGES.index(last_stage)
            valid = [last_stage]
            if idx + 1 < len(STAGES):
                valid.append(STAGES[idx + 1])
            valid_str = " or ".join(f"'{v}'" for v in valid)
        else:
            valid_str = "the same as before"

        content.append({
            "type": "text",
            "text": (
                f"TEMPORAL CONTEXT: The current stage is most likely '{last_stage}'. "
                f"Valid classifications are {valid_str}. "
                f"Default to '{last_stage}' unless you see a CLEAR morphological "
                f"change matching the transition criteria above."
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

    raw = await call_claude(system=SYSTEM_PROMPT, content=content)
    return response_to_output(raw)
