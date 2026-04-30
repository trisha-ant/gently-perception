"""vote3 on mm_v4: 3x self-consistency on the 4-measurement variant."""

import asyncio
from collections import Counter

from ._base import PerceptionOutput
from .mm_v4 import perceive_mm_v4


async def perceive_vote3_mm_v4(
    image_b64: str,
    references: dict[str, list[str]],
    history: list[dict],
    timepoint: int,
) -> PerceptionOutput:
    outs = await asyncio.gather(
        *(perceive_mm_v4(image_b64, references, history, timepoint) for _ in range(3))
    )
    stages = [o.stage for o in outs]
    counts = Counter(stages)
    winner, n_votes = counts.most_common(1)[0]
    if n_votes == 1:
        winner = stages[0]
    chosen = next(o for o in outs if o.stage == winner)
    note = f"[vote3] {'/'.join(stages)} -> {winner}"
    return PerceptionOutput(
        stage=winner,
        reasoning=f"{chosen.reasoning} | {note}",
        verification_triggered=True,
        phase_count=3,
        raw_response=chosen.raw_response,
    )
