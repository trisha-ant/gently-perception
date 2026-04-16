"""
Duration-aware adaptive perception.

Layered on the hybrid base: adjust temporal-anchoring strength in the
prompt based on how long the model has been in the current stage vs
typical durations. Early in stage → very strong anchoring. Past typical
duration → gentle nudge toward the next stage.
"""

from ._base import (
    PerceptionOutput,
    build_history_text,
    build_reference_content,
    call_claude,
    response_to_output,
    STAGES,
)

from .hybrid import _get_system_prompt, TEMPORAL_SYSTEM, SCIENTIFIC_SYSTEM

# Typical stage durations in timepoints (min, max from ground truth)
TYPICAL_DURATIONS = {
    "early": (27, 54),
    "bean": (6, 6),
    "comma": (6, 9),
    "1.5fold": (8, 15),
    "2fold": (10, 20),
    "pretzel": (41, 60),
    "hatching": (1, 5),
    "hatched": (50, 100),
}

# Module-level state (reset per embryo)
_current_stage: str | None = None
_stage_entry_tp: int | None = None
_prev_timepoint: int | None = None
_is_first_evaluated_tp: bool = True


def _reset_state():
    """Reset state for a new embryo."""
    global _current_stage, _stage_entry_tp, _prev_timepoint, _is_first_evaluated_tp
    _current_stage = None
    _stage_entry_tp = None
    _prev_timepoint = None
    _is_first_evaluated_tp = True


def _get_duration_context(stage: str, duration: int) -> str:
    """Get duration-aware anchoring text."""
    if stage not in TYPICAL_DURATIONS:
        return ""

    min_dur, max_dur = TYPICAL_DURATIONS[stage]

    if duration < min_dur * 0.4:
        # Very early in stage — strong anchoring
        return (
            f"\nDURATION CONTEXT: '{stage}' has only been observed for "
            f"{duration} timepoints. This stage typically lasts {min_dur}-{max_dur} "
            f"timepoints. It would be very unusual to transition after only "
            f"{duration} timepoints. Only classify as the next stage if the "
            f"morphological evidence is OVERWHELMING."
        )
    elif duration > max_dur:
        # Past typical max — gentle nudge
        next_idx = STAGES.index(stage) + 1 if stage in STAGES and STAGES.index(stage) < len(STAGES) - 1 else None
        next_stage = STAGES[next_idx] if next_idx is not None else "the next stage"
        return (
            f"\nDURATION CONTEXT: '{stage}' has been observed for {duration} "
            f"timepoints, which exceeds the typical range of {min_dur}-{max_dur}. "
            f"While individual embryos vary, carefully examine whether the "
            f"morphology has progressed to '{next_stage}'."
        )
    elif duration > max_dur * 0.8:
        # Approaching typical max — soften anchoring
        return (
            f"\nDURATION CONTEXT: '{stage}' has been observed for {duration} "
            f"timepoints (typical range: {min_dur}-{max_dur}). A transition "
            f"is becoming plausible. Classify based on morphology."
        )

    return ""


async def perceive_duration_aware(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Duration-aware classification."""
    global _current_stage, _stage_entry_tp, _prev_timepoint, _is_first_evaluated_tp

    # Detect new embryo (timepoint reset)
    if _prev_timepoint is not None and timepoint <= _prev_timepoint:
        _reset_state()

    # Determine expected stage from history
    last_stage = "early"
    if history:
        last_stage = history[-1].get("stage", "early")

    # Track stage duration
    if _current_stage is None or last_stage != _current_stage:
        _current_stage = last_stage
        _stage_entry_tp = timepoint
    duration_in_stage = timepoint - _stage_entry_tp if _stage_entry_tp is not None else 0

    # Select system prompt (hybrid logic)
    system_prompt = _get_system_prompt(last_stage)

    # Build content
    content = build_reference_content(references)
    content.append({"type": "text", "text": f"\n=== CLASSIFY EMBRYO AT T{timepoint} ==="})

    history_text = build_history_text(history)
    if history_text:
        content.append({"type": "text", "text": history_text})

        # Temporal anchoring — adjusted for first evaluated timepoint
        if _is_first_evaluated_tp:
            content.append({
                "type": "text",
                "text": (
                    f"The previous stage was '{last_stage}' (confirmed by expert). "
                    f"Classify based on what you see — if the embryo has progressed "
                    f"beyond '{last_stage}', classify accordingly."
                ),
            })
            _is_first_evaluated_tp = False
        else:
            content.append({
                "type": "text",
                "text": (
                    f"The most recent observation was '{last_stage}'. "
                    f"Remember: stages change slowly. The current stage is most likely "
                    f"'{last_stage}' unless you see a clear morphological change. "
                    f"When uncertain, prefer the earlier stage."
                ),
            })

    # Add duration context
    duration_ctx = _get_duration_context(last_stage, duration_in_stage)
    if duration_ctx:
        content.append({"type": "text", "text": duration_ctx})

    content.append({
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg", "data": image_b64},
    })

    # Analysis prompt (hybrid-style)
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

    # Update state
    _prev_timepoint = timepoint
    if result.stage != _current_stage:
        _current_stage = result.stage
        _stage_entry_tp = timepoint

    return result
