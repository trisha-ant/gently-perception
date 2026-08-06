"""Auto-discover perceive functions from ``experiments/variants/``.

A variant module exports exactly one ``async def perceive(fi: FrameInput)``.
No manual registry [P1].
"""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path
from typing import Callable

_ROOTS = {
    "variants": Path(__file__).resolve().parent.parent / "variants",
    "pipelines": Path(__file__).resolve().parent.parent / "pipelines",
}


def _scan(root_name: str, root: Path) -> dict[str, Callable]:
    found: dict[str, Callable] = {}
    if not root.exists():
        return found
    for py in sorted(root.glob("*.py")):
        if py.name.startswith("_"):
            continue
        mod = importlib.import_module(f"experiments.{root_name}.{py.stem}")
        fn = getattr(mod, "perceive", None)
        if fn is None:
            for name in dir(mod):
                if name.startswith("perceive"):
                    fn = getattr(mod, name)
                    break
        if fn is None or not inspect.iscoroutinefunction(fn):
            raise ValueError(
                f"{root_name}/{py.name}: must export "
                f"`async def perceive(fi: FrameInput)`"
            )
        found[py.stem] = fn
    return found


def discover_variants() -> dict[str, Callable]:
    """Return {name: perceive_fn} for every module under variants/ and
    pipelines/. Pipelines are perceive functions too — they compose other
    variants but expose the same contract, so they inherit the leak guard."""
    out: dict[str, Callable] = {}
    for root_name, root in _ROOTS.items():
        out.update(_scan(root_name, root))
    return out
