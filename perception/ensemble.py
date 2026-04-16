"""
Ensemble perception function.

Runs the hybrid's stage-adaptive prompt 3 times with temperature=0.3
and takes majority vote. This smooths out stochastic boundary errors.

Uses hybrid approach (best at 83.2%) as the base, with temperature
sampling to reduce boundary noise.

Cost: 3x API calls per timepoint (run in parallel via asyncio.gather).
"""

import asyncio
from collections import Counter

from ._base import (
    PerceptionOutput,
    build_history_text,
    build_reference_content,
    call_claude,
    response_to_output,
    STAGES,
)

from .hybrid import _get_system_prompt, TEMPORAL_SYSTEM, SCIENTIFIC_SYSTEM

ENSEMBLE_SIZE = 3
TEMPERATURE = 0.3


async def perceive_ensemble(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Majority-vote ensemble using hybrid's stage-adaptive prompts."""
    # Determine expected stage from history (same logic as hybrid)
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

    # Run 3x in parallel with temperature > 0
    tasks = [
        call_claude(system=system_prompt, content=content, temperature=TEMPERATURE)
        for _ in range(ENSEMBLE_SIZE)
    ]
    results = await asyncio.gather(*tasks)

    # Parse each result
    outputs = [response_to_output(raw) for raw in results]

    # Majority vote on stage
    stage_counts = Counter(o.stage for o in outputs)
    majority_stage = stage_counts.most_common(1)[0][0]

    # If all 3 different, prefer the earlier stage (conservative)
    if len(stage_counts) == ENSEMBLE_SIZE:
        for s in STAGES:
            if s in stage_counts:
                majority_stage = s
                break

    votes = ", ".join(o.stage for o in outputs)

    return PerceptionOutput(
        stage=majority_stage,
        reasoning=f"Ensemble [{votes}] → {majority_stage}",
        phase_count=ENSEMBLE_SIZE,
    )
