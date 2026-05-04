"""H5: multimeasure with XY and XZ views as separate images (not the composite).

Tests whether the model attends to each view better when they are separate
images vs. parsing the tiled three-view layout.
"""

from ._base import (
    PerceptionOutput,
    build_history_text,
    build_reference_content,
    call_claude,
    response_to_output,
)
from .multimeasure import SYSTEM_PROMPT as _BASE_SYSTEM

SYSTEM_PROMPT = _BASE_SYSTEM.replace(
    "Each image shows three orthogonal views (XY top-left, YZ top-right, XZ bottom-left).",
    "You will be shown two views as separate images: the XY (top-down) view first, "
    "then the XZ (side) view.",
)


async def perceive_sepview_mm(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
    top_image_b64: str | None = None,
    side_image_b64: str | None = None,
) -> PerceptionOutput:
    content = build_reference_content(references)
    content.append({"type": "text", "text": f"\n=== CLASSIFY EMBRYO AT T{timepoint} ==="})

    history_text = build_history_text(history)
    if history_text:
        content.append({"type": "text", "text": history_text})

    if top_image_b64 and side_image_b64:
        content.append({"type": "text", "text": "XY (top-down) view:"})
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": top_image_b64},
            }
        )
        content.append({"type": "text", "text": "XZ (side) view:"})
        content.append(
            {
                "type": "image",
                "source": {"type": "base64", "media_type": "image/jpeg", "data": side_image_b64},
            }
        )
    else:
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
