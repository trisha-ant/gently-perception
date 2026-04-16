"""
Change-gate perception function.

Two-pass approach:
1. First pass: show previous + current image, ask "has morphology changed?" (yes/no)
2. If NO → return previous stage (skip full classification)
3. If YES → run full hybrid classification

This should dramatically reduce both errors and API cost:
- Most timepoints within a stage look the same → gate says "no change"
- Only boundary timepoints trigger full classification
"""

import json

from ._base import (
    PerceptionOutput,
    build_reference_content,
    build_history_text,
    call_claude,
    response_to_output,
    STAGES,
)

from .hybrid import _get_system_prompt, TEMPORAL_SYSTEM, SCIENTIFIC_SYSTEM

# Module-level state
_prev_image: str | None = None
_prev_timepoint: int | None = None

GATE_SYSTEM = """\
You are comparing two consecutive fluorescence microscopy images of a \
C. elegans embryo to determine if the developmental stage has changed.

Each image shows three orthogonal max-intensity projections (XY top-left, \
YZ top-right, XZ bottom-left).

Compare the two images and determine: has the embryo's morphology changed \
SIGNIFICANTLY between these timepoints? Specifically, look for:
- Changes in how much of the eggshell is filled with signal
- New parallel body segments appearing
- Body segments starting to cross over each other
- Major shape changes

Small differences in brightness, orientation, or noise are NOT significant.

Respond with JSON:
{
  "changed": true or false,
  "reasoning": "Brief explanation"
}"""


async def _check_change(prev_img: str, curr_img: str, prev_tp: int, curr_tp: int) -> bool:
    """Ask the model if morphology has changed between two timepoints."""
    content = [
        {"type": "text", "text": f"PREVIOUS (T{prev_tp}):"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": prev_img},
        },
        {"type": "text", "text": f"CURRENT (T{curr_tp}):"},
        {
            "type": "image",
            "source": {"type": "base64", "media_type": "image/jpeg", "data": curr_img},
        },
        {"type": "text", "text": "Has the morphology changed significantly? Respond with JSON."},
    ]

    raw = await call_claude(system=GATE_SYSTEM, content=content, max_tokens=512)

    try:
        # Parse the response
        data = json.loads(raw) if raw.strip().startswith("{") else {}
        if not data:
            import re
            m = re.search(r'"changed"\s*:\s*(true|false)', raw, re.IGNORECASE)
            if m:
                return m.group(1).lower() == "true"
        return data.get("changed", True)  # default to changed (safe)
    except Exception:
        return True  # default to changed (triggers full classification)


async def perceive_changegate(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Change-gated classification: only reclassify when morphology changes."""
    global _prev_image, _prev_timepoint

    # Detect new embryo
    if _prev_timepoint is not None and timepoint <= _prev_timepoint:
        _prev_image = None
        _prev_timepoint = None

    last_stage = "early"
    if history:
        last_stage = history[-1].get("stage", "early")

    # If we have a previous image, check if morphology changed
    if _prev_image is not None:
        changed = await _check_change(_prev_image, image_b64, _prev_timepoint, timepoint)

        if not changed:
            # No change detected — keep previous stage
            _prev_image = image_b64
            _prev_timepoint = timepoint
            return PerceptionOutput(
                stage=last_stage,
                reasoning=f"No significant morphological change from T{_prev_timepoint}; maintaining {last_stage}",
            )

    # Either no previous image or change detected — full classification
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

    content.append({
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
    })

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
    result = response_to_output(raw)

    # Cache current image
    _prev_image = image_b64
    _prev_timepoint = timepoint

    return result
