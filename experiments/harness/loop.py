"""The research-harness agent loop.

Owns iteration, history construction, and event emission. Variants are pure
``perceive(FrameInput) -> PerceptionOutput`` functions that cannot see ground
truth or mutate history — the loop is the only place GT and history meet, and
it asserts (:class:`LeakError`) that GT-sourced history never crosses into a
scored frame.

Phases per frame [P6]: render → classify → verify → score → record.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Iterable, Literal, Protocol

from gently_perception.types import (
    FrameInput,
    LeakError,
    Obs,
    ParseError,
    PerceptionOutput,
)

from .config import RunConfig
from .events import EventLog, reduce_events

logger = logging.getLogger(__name__)

STAGES = ["early", "bean", "comma", "1.5fold", "2fold",
          "pretzel", "hatching", "hatched"]
_STAGE_IDX = {s: i for i, s in enumerate(STAGES)}

PerceiveFn = Callable[[FrameInput], Awaitable[PerceptionOutput]]


# --------------------------------------------------------------------------- #
# Testset protocol — keeps loop decoupled from benchmark.testset
# --------------------------------------------------------------------------- #

class Frame(Protocol):
    embryo_id: str
    timepoint: int
    image_b64: str
    ground_truth_stage: str | None


class FrameSource(Protocol):
    def iter_all(self) -> Iterable[tuple[str, Iterable[Frame]]]: ...


# --------------------------------------------------------------------------- #
# Internal history entry — tracks provenance for the leak guard
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class _HistEntry:
    timepoint: int
    stage: str
    source: Literal["gt", "pred"]


def _to_obs(hist: list[_HistEntry], window: int = 3) -> tuple[Obs, ...]:
    """Strip provenance and truncate — what the variant is allowed to see."""
    return tuple(Obs(timepoint=h.timepoint, stage=h.stage) for h in hist[-window:])


def _assert_no_leak(hist: list[_HistEntry], first_scored_tp: int) -> None:
    """[P7] GT-sourced history is allowed only as lead-in (before the first
    scored frame). Once scoring starts, every history entry at or after that
    point must be ``source="pred"``."""
    for h in hist:
        if h.source == "gt" and h.timepoint >= first_scored_tp:
            raise LeakError(
                f"GT-sourced history at T{h.timepoint} would reach a scored "
                f"frame (first_scored=T{first_scored_tp})"
            )


def _is_adjacent(pred: str, gt: str) -> bool:
    pi, gi = _STAGE_IDX.get(pred), _STAGE_IDX.get(gt)
    return pi is not None and gi is not None and abs(pi - gi) <= 1


def _verify_monotone(pred: str, hist: list[_HistEntry]) -> bool:
    """[P12] Harness-side verification: prediction must not regress."""
    if not hist:
        return True
    last = _STAGE_IDX.get(hist[-1].stage, -1)
    cur = _STAGE_IDX.get(pred, -1)
    return cur >= last if (last >= 0 and cur >= 0) else True


# --------------------------------------------------------------------------- #
# The loop
# --------------------------------------------------------------------------- #

async def _run_embryo(
    embryo_id: str,
    frames: Iterable[Frame],
    perceive: PerceiveFn,
    references: dict[str, list[str]],
    target: set[str] | None,
    log: EventLog,
    extras_fn: Callable[[Frame], dict] | None,
) -> None:
    hist: list[_HistEntry] = []
    first_scored_tp: int | None = None

    for fr in frames:
            gt = fr.ground_truth_stage

            # ---- skip / lead-in ---------------------------------------- #
            if target and gt not in target:
                hist.append(_HistEntry(fr.timepoint, gt or "early", source="gt"))
                continue

            if first_scored_tp is None:
                first_scored_tp = fr.timepoint
            _assert_no_leak(hist, first_scored_tp)

            # ---- render ------------------------------------------------- #
            extras = extras_fn(fr) if extras_fn else {}
            log.emit("render", embryo_id=embryo_id, timepoint=fr.timepoint,
                     extras=sorted(extras.keys()))

            fi = FrameInput(
                image_b64=fr.image_b64,
                history=_to_obs(hist),
                timepoint=fr.timepoint,
                references=references,
                extras=extras,
            )

            # ---- classify ----------------------------------------------- #
            try:
                out = await perceive(fi)
            except ParseError as e:
                log.emit("error", embryo_id=embryo_id, timepoint=fr.timepoint,
                         error="ParseError", detail=str(e))
                hist.append(_HistEntry(fr.timepoint,
                                       hist[-1].stage if hist else "early",
                                       source="pred"))
                continue

            log.emit("classify", embryo_id=embryo_id, timepoint=fr.timepoint,
                     predicted=out.stage, reasoning=out.reasoning[:200])

            # ---- verify ------------------------------------------------- #
            monotone = _verify_monotone(out.stage, hist)
            log.emit("verify", embryo_id=embryo_id, timepoint=fr.timepoint,
                     monotone=monotone)

            # ---- score -------------------------------------------------- #
            is_correct = (out.stage == gt) if gt else False
            is_adj = _is_adjacent(out.stage, gt) if gt else False
            log.emit("score", embryo_id=embryo_id, timepoint=fr.timepoint,
                     predicted=out.stage, gt=gt, is_correct=is_correct,
                     is_adjacent_correct=is_adj, reasoning=out.reasoning)

            # ---- record ------------------------------------------------- #
            hist.append(_HistEntry(fr.timepoint, out.stage, source="pred"))


async def run_variant(
    perceive: PerceiveFn,
    source: FrameSource,
    references: dict[str, list[str]],
    cfg: RunConfig,
    *,
    event_log: EventLog | None = None,
    extras_fn: Callable[[Frame], dict] | None = None,
    concurrency: int = 4,
) -> dict:
    """Run one seed of one variant over all embryos.

    Embryos are independent (history is per-embryo), so they run concurrently
    under a semaphore. Set ``concurrency=1`` for deterministic event ordering.
    """
    import asyncio

    log = event_log or EventLog(path=None)
    log.emit("run_start", config=cfg.to_dict())
    target = set(cfg.stages) if cfg.stages else None

    sem = asyncio.Semaphore(max(1, concurrency))

    async def one(eid: str, frs: Iterable[Frame]) -> None:
        async with sem:
            await _run_embryo(eid, frs, perceive, references, target, log,
                              extras_fn)

    await asyncio.gather(*(one(eid, frs) for eid, frs in source.iter_all()))

    log.emit("run_end")
    return reduce_events(log.events, cfg.to_dict())
