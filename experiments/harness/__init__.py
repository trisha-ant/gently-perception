"""Research harness — the new loop, config, events, and runner.

Coexists with the legacy ``perception/`` + root ``run.py`` until Phase A
recordings are captured; once replay-equivalence passes, the legacy code is
deleted and this becomes the only runner.
"""

from .config import RunConfig
from .events import Event, EventLog, reduce_events
from .loop import run_variant

__all__ = ["RunConfig", "Event", "EventLog", "reduce_events", "run_variant"]
