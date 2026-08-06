"""Structured event log — observability is first-class [P9].

The loop emits one JSONL line per phase event; the result JSON is *derived*
from the event log via :func:`reduce_events`, never accumulated as mutable
state. That makes transcript review and post-hoc analysis trivial.
"""

from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterator, Literal

EventKind = Literal["run_start", "render", "classify", "verify", "score",
                    "error", "run_end"]


@dataclass
class Event:
    kind: EventKind
    embryo_id: str | None = None
    timepoint: int | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(
            {"kind": self.kind, "embryo_id": self.embryo_id,
             "timepoint": self.timepoint, **self.data},
            ensure_ascii=False,
        )


class EventLog:
    """Append-only JSONL writer + in-memory buffer."""

    def __init__(self, path: Path | None = None):
        self.path = path
        self.events: list[Event] = []
        if path is not None:
            path.parent.mkdir(parents=True, exist_ok=True)
            self._fh = open(path, "w")
        else:
            self._fh = None

    def emit(self, kind: EventKind, *, embryo_id: str | None = None,
             timepoint: int | None = None, **data: Any) -> None:
        ev = Event(kind=kind, embryo_id=embryo_id, timepoint=timepoint, data=data)
        self.events.append(ev)
        if self._fh is not None:
            self._fh.write(ev.to_json() + "\n")
            self._fh.flush()

    def close(self) -> None:
        if self._fh is not None:
            self._fh.close()
            self._fh = None

    def __iter__(self) -> Iterator[Event]:
        return iter(self.events)


def reduce_events(events: list[Event], config: dict) -> dict:
    """Derive the result-dict (same shape ``run.py`` historically wrote) from
    an event stream. ``score`` events are the source of truth."""
    by_embryo: dict[str, list[dict]] = defaultdict(list)
    errors: list[dict] = []

    for ev in events:
        if ev.kind == "score":
            by_embryo[ev.embryo_id or "?"].append({
                "timepoint": ev.timepoint,
                "predicted_stage": ev.data["predicted"],
                "ground_truth_stage": ev.data["gt"],
                "reasoning": ev.data.get("reasoning", ""),
                "is_correct": ev.data["is_correct"],
                "is_adjacent_correct": ev.data["is_adjacent_correct"],
            })
        elif ev.kind == "error":
            errors.append({"embryo_id": ev.embryo_id, "timepoint": ev.timepoint,
                           **ev.data})

    embryo_results = []
    all_preds: list[dict] = []
    for eid, preds in by_embryo.items():
        n = len(preds) or 1
        embryo_results.append({
            "embryo_id": eid,
            "predictions": preds,
            "accuracy": sum(p["is_correct"] for p in preds) / n,
            "adjacent_accuracy": sum(p["is_adjacent_correct"] for p in preds) / n,
        })
        all_preds.extend(preds)

    total = len(all_preds) or 1
    exact = sum(p["is_correct"] for p in all_preds) / total
    adjacent = sum(p["is_adjacent_correct"] for p in all_preds) / total

    per_stage: dict[str, dict[str, Any]] = {}
    bucket: dict[str, list[bool]] = defaultdict(list)
    for p in all_preds:
        bucket[p["ground_truth_stage"]].append(p["is_correct"])
    for st, oks in bucket.items():
        per_stage[st] = {"accuracy": sum(oks) / len(oks), "n": len(oks)}

    return {
        "config": config,
        "embryo_results": embryo_results,
        "errors": errors,
        "total_predictions": len(all_preds),
        "overall_accuracy": exact,
        "metrics": {
            "accuracy": exact,
            "adjacent_accuracy": adjacent,
            "per_stage": per_stage,
        },
    }
