"""Hybrid variant on the new FrameInput contract.

Stage-adaptive: temporal prompt for 1.5fold/early, scientific (eggshell-fill)
prompt for 2fold/pretzel boundaries. Same prompts as ``perception/hybrid.py``
— only the function shape changes (takes ``FrameInput``, no history mutation).
"""

from __future__ import annotations

from gently_perception.api import (
    build_history_text,
    build_reference_content,
    call_claude,
    response_to_output,
)
from gently_perception.types import FrameInput, PerceptionOutput

# Prompts are byte-identical to perception/hybrid.py so record-replay
# equivalence (Phase A) can succeed with P8 disabled.
from perception.hybrid import SCIENTIFIC_SYSTEM, TEMPORAL_SYSTEM


def _select_system(last_stage: str) -> str:
    return SCIENTIFIC_SYSTEM if last_stage in ("2fold", "pretzel") else TEMPORAL_SYSTEM


async def perceive(fi: FrameInput) -> PerceptionOutput:
    last_stage = fi.last_stage()
    system = _select_system(last_stage)

    content = build_reference_content(dict(fi.references))
    content.append({"type": "text",
                    "text": f"\n=== CLASSIFY EMBRYO AT T{fi.timepoint} ==="})

    hist_dicts = [{"timepoint": o.timepoint, "stage": o.stage} for o in fi.history]
    hist_text = build_history_text(hist_dicts)
    if hist_text:
        content.append({"type": "text", "text": hist_text})
        content.append({"type": "text", "text": (
            f"The most recent observation was '{last_stage}'. "
            f"Remember: stages change slowly. The current stage is most likely "
            f"'{last_stage}' unless you see a clear morphological change. "
            f"When uncertain, prefer the earlier stage."
        )})

    content.append({
        "type": "image",
        "source": {"type": "base64", "media_type": "image/jpeg",
                   "data": fi.image_b64},
    })

    if system is SCIENTIFIC_SYSTEM:
        content.append({"type": "text", "text": (
            "Analyze: (1) How much of the eggshell is filled with signal? "
            "(2) How many parallel body segments are visible? "
            "(3) Which reference images match best? Then classify."
        )})
    else:
        content.append({"type": "text", "text": (
            "Compare this image to the reference images above. Which stage's "
            "references does it most closely match? Classify accordingly."
        )})

    raw = await call_claude(system=system, content=content)
    return response_to_output(raw)
