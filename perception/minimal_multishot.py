"""
Minimal perception function with multi-shot reconsideration.

Same as minimal (single API call, stage names only, reference images),
but adds a follow-up turn asking the model to reconsider its classification.
"""

from ._base import (
    PerceptionOutput,
    build_history_text,
    build_reference_content,
    call_claude,
    call_claude_conversation,
    response_to_output,
)
from .minimal import SYSTEM_PROMPT

RECONSIDER_PROMPT = """\
You classified this as {stage}. \
Before I accept this, please carefully re-examine the image.

Look specifically at:
1. The key morphological features that distinguish this stage from adjacent stages
2. Whether your observation matches the reference examples above
3. All available views — not just the primary one

After re-examining, provide your revised (or confirmed) classification \
in the same JSON format. It's OK to change your answer."""


async def perceive_minimal_multishot(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    """Minimal prompt with one reconsideration turn."""
    # Build user content: references (cached) + history + current image
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

    # Turn 1: initial classification
    raw = await call_claude(system=SYSTEM_PROMPT, content=content)
    initial = response_to_output(raw)

    # Turn 2: reconsideration in the same conversation context
    followup = RECONSIDER_PROMPT.format(stage=initial.stage)
    messages = [
        {"role": "user", "content": content},
        {"role": "assistant", "content": [{"type": "text", "text": raw}]},
        {"role": "user", "content": [{"type": "text", "text": followup}]},
    ]

    raw2 = await call_claude_conversation(system=SYSTEM_PROMPT, messages=messages)
    revised = response_to_output(raw2)

    # Record both responses for analysis
    revised.raw_response = f"--- INITIAL ---\n{raw}\n--- REVISED ---\n{raw2}"
    return revised
