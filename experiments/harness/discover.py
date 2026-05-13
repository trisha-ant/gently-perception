"""Auto-discover perceive functions from ``experiments/variants/``.

A variant module exports exactly one ``async def perceive(fi: FrameInput)``.
No manual registry [P1].
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Callable

VARIANTS_DIR = Path(__file__).resolve().parent.parent / "variants"


def discover_variants() -> dict[str, Callable]:
    """Return {variant_name: perceive_fn} for every module under variants/."""
    found: dict[str, Callable] = {}
    if not VARIANTS_DIR.exists():
        return found

    for py in sorted(VARIANTS_DIR.glob("*.py")):
        if py.name.startswith("_"):
            continue
        mod = importlib.import_module(f"experiments.variants.{py.stem}")
        fn = getattr(mod, "perceive", None)
        if fn is None:
            for name in dir(mod):
                if name.startswith("perceive"):
                    fn = getattr(mod, name)
                    break
        if fn is None or not inspect.iscoroutinefunction(fn):
            raise ValueError(
                f"{py.name}: must export `async def perceive(fi: FrameInput)`"
            )
        found[py.stem] = fn
    return found
