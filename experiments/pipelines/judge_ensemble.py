"""Judge ensemble as a first-class ``perceive(FrameInput)`` pipeline.

Two experts (hybrid + vote3_mm) classify the frame; on agreement, return that
stage; on disagreement, a judge call arbitrates. Because this is a perceive
function, it runs through ``loop.run_variant`` and inherits the leak guard:
the judge only ever sees ``FrameInput`` (no GT), and history is the loop's
own predicted history. This is the structural fix for the leak in
``experiments/judge_ensemble/judge_replicate.py:131`` that produced the
inflated 90.6% result.
"""

from __future__ import annotations

from gently_perception.types import FrameInput, PerceptionOutput
from perception._base import (
    STAGES,
    build_history_text,
    build_reference_content,
    call_claude,
    parse_stage_json,
)

from experiments.variants.hybrid import perceive as expert_a
from experiments.variants.vote3_mm import perceive as expert_b

_SIDX = {s: i for i, s in enumerate(STAGES)}

JUDGE_SYSTEM = """\
You are arbitrating between two expert classifiers of C. elegans embryo \
developmental stages. Each image shows three orthogonal views (XY top-left, \
YZ top-right, XZ bottom-left) of a fluorescence light-sheet max-projection.

The stages in order are: early, bean, comma, 1.5fold, 2fold, pretzel, hatching, hatched.

Two experts looked at the same image and DISAGREED. Decide which one is correct.
You MUST pick one of the two stages they proposed -- do not propose a third.

Respond with JSON:
{
  "stage": "<one of the two proposed stages>",
  "reasoning": "Brief explanation"
}"""


async def _judge(fi: FrameInput, a: PerceptionOutput, b: PerceptionOutput) -> str:
    content = build_reference_content(dict(fi.references))
    content.append({"type": "text", "text": f"\n=== ARBITRATE: T{fi.timepoint} ==="})
    hist = build_history_text(
        [{"timepoint": o.timepoint, "stage": o.stage} for o in fi.history]
    )
    if hist:
        content.append({"type": "text", "text": hist})
    content.append({"type": "image", "source": {"type": "base64",
                    "media_type": "image/jpeg", "data": fi.image_b64}})
    content.append({"type": "text", "text": (
        f"Expert A says {a.stage.upper()}: {a.reasoning[:300]}\n\n"
        f"Expert B says {b.stage.upper()}: {b.reasoning[:300]}\n\n"
        f"Which is correct: {a.stage} or {b.stage}?"
    )})
    raw = await call_claude(system=JUDGE_SYSTEM, content=content, max_tokens=1024)
    pick = parse_stage_json(raw).get("stage")
    if pick not in (a.stage, b.stage):
        # fall back to more-advanced (the only non-leaky tiebreak available)
        pick = a.stage if _SIDX.get(a.stage, 0) >= _SIDX.get(b.stage, 0) else b.stage
    return pick


async def perceive(fi: FrameInput) -> PerceptionOutput:
    a = await expert_a(fi)
    b = await expert_b(fi)
    if a.stage == b.stage:
        return PerceptionOutput(stage=a.stage,
                                reasoning=f"agree: {a.reasoning[:120]}")
    pick = await _judge(fi, a, b)
    chosen = a if pick == a.stage else b
    return PerceptionOutput(
        stage=pick,
        reasoning=f"judge picked {pick} over {b.stage if pick == a.stage else a.stage}: "
                  f"{chosen.reasoning[:120]}",
    )
