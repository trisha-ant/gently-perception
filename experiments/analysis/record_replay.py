"""Record/replay shim for harness equivalence testing.

Phase A of the migration eval: capture {request → response} pairs from the
OLD harness so the NEW harness can be replayed against identical model
outputs and per-frame predictions compared deterministically.

The request key is a sha256 over the call's semantic payload (system text +
content blocks with image bytes hashed). Recordings are append-only JSONL::

    {"key": "<sha256>", "system_sha": "...", "n_images": 2,
     "history_tail": "...", "response": "<raw text>"}

Usage::

    # Record (hits live API, writes data/fixtures/recordings/<variant>.jsonl)
    python -m experiments.analysis.record_replay --record --variant hybrid

    # In tests: install replay shim, then run any code that calls call_claude
    from experiments.analysis.record_replay import Recorder, install_record, install_replay
"""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
RECORDINGS_DIR = REPO_ROOT / "data" / "fixtures" / "recordings"


class CacheMiss(RuntimeError):
    """Raised in replay mode when a request has no recorded response."""


# --------------------------------------------------------------------------- #
# Request hashing
# --------------------------------------------------------------------------- #

def _hash_content_block(block: dict[str, Any]) -> Any:
    """Replace image base64 payloads with their sha256 so the key is stable
    and the recording file stays small."""
    if block.get("type") == "image":
        src = block.get("source", {})
        data = src.get("data", "")
        return {"type": "image", "sha": hashlib.sha256(data.encode()).hexdigest()}
    out = {k: v for k, v in block.items() if k != "cache_control"}
    return out


def request_key(system: str, content: list[dict[str, Any]]) -> str:
    canon = {
        "system": system,
        "content": [_hash_content_block(b) for b in content],
    }
    blob = json.dumps(canon, sort_keys=True, ensure_ascii=False)
    return hashlib.sha256(blob.encode()).hexdigest()


# --------------------------------------------------------------------------- #
# Recorder
# --------------------------------------------------------------------------- #

class Recorder:
    """Append-only recorder. One instance per variant×model recording file."""

    def __init__(self, path: Path):
        self.path = path
        self._cache: dict[str, str] = {}
        if path.exists():
            for line in path.read_text().splitlines():
                if line.strip():
                    rec = json.loads(line)
                    self._cache[rec["key"]] = rec["response"]

    def has(self, key: str) -> bool:
        return key in self._cache

    def get(self, key: str) -> str:
        try:
            return self._cache[key]
        except KeyError as e:
            raise CacheMiss(f"no recording for key {key[:12]}… in {self.path.name}") from e

    def put(self, key: str, response: str, *, meta: dict[str, Any] | None = None) -> None:
        if key in self._cache:
            return
        self._cache[key] = response
        self.path.parent.mkdir(parents=True, exist_ok=True)
        rec = {"key": key, "response": response, **(meta or {})}
        with open(self.path, "a") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    @classmethod
    def for_variant(cls, variant: str, model: str) -> "Recorder":
        safe_model = model.replace("/", "_").replace(":", "_")
        return cls(RECORDINGS_DIR / f"{variant}__{safe_model}.jsonl")


# --------------------------------------------------------------------------- #
# Install shims (monkeypatch perception._base.call_claude)
# --------------------------------------------------------------------------- #

_CALL_NAMES = ("call_claude", "call_claude_conversation")
_OUR_PREFIXES = ("perception", "gently_perception", "experiments")


def _patch_targets() -> list[tuple[Any, str]]:
    """Return (module, attr_name) for every binding of call_claude in our
    package namespace. Variants do ``from ._base import call_claude``, so the
    name is bound in each variant module — patching only ``_base`` misses them.
    """
    # Ensure the canonical definitions are loaded so they appear in sys.modules.
    for modname in ("perception._base", "gently_perception.api"):
        try:
            __import__(modname)
        except ImportError:
            pass

    targets: list[tuple[Any, str]] = []
    for name, mod in list(sys.modules.items()):
        if mod is None or not name.startswith(_OUR_PREFIXES):
            continue
        for attr in _CALL_NAMES:
            if callable(getattr(mod, attr, None)):
                targets.append((mod, attr))
    return targets


def _extract_content(args: tuple, kwargs: dict, fn_name: str) -> list[dict]:
    """Pull the content list out of however the caller invoked call_claude.
    Handles positional, ``content=``, and ``messages=`` (conversation)."""
    if "content" in kwargs:
        return kwargs["content"]
    if "messages" in kwargs:
        return kwargs["messages"][-1]["content"]
    if len(args) >= 2:
        second = args[1]
        if fn_name == "call_claude_conversation":
            return second[-1]["content"]
        return second
    raise TypeError(f"{fn_name}: could not locate content/messages argument")


def _extract_system(args: tuple, kwargs: dict) -> str:
    return kwargs.get("system", args[0] if args else "")


@contextmanager
def install_record(recorder: Recorder):
    """Wrap call_claude so every live response is also written to ``recorder``."""
    originals: list[tuple[Any, str, Callable]] = []
    for mod, name in _patch_targets():
        orig = getattr(mod, name)
        originals.append((mod, name, orig))

        async def wrapped(*args, __orig=orig, __name=name, **kwargs) -> str:
            system = _extract_system(args, kwargs)
            content = _extract_content(args, kwargs, __name)
            key = request_key(system, content)
            resp = await __orig(*args, **kwargs)
            recorder.put(key, resp, meta={
                "system_sha": hashlib.sha256(system.encode()).hexdigest()[:12],
                "n_blocks": len(content),
            })
            return resp

        setattr(mod, name, wrapped)
    try:
        yield recorder
    finally:
        for mod, name, orig in originals:
            setattr(mod, name, orig)


@contextmanager
def install_replay(recorder: Recorder):
    """Replace call_claude with a lookup into ``recorder``. Raises CacheMiss
    on any request that wasn't recorded — that's the equivalence signal."""
    originals: list[tuple[Any, str, Callable]] = []
    for mod, name in _patch_targets():
        orig = getattr(mod, name)
        originals.append((mod, name, orig))

        async def wrapped(*args, __name=name, **kwargs) -> str:
            system = _extract_system(args, kwargs)
            content = _extract_content(args, kwargs, __name)
            return recorder.get(request_key(system, content))

        setattr(mod, name, wrapped)
    try:
        yield recorder
    finally:
        for mod, name, orig in originals:
            setattr(mod, name, orig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

async def _record_variant(variant: str, stages: list[str]) -> None:
    """Run the OLD root run.py path for one variant with recording installed."""
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

    from benchmark.ground_truth import GroundTruth
    from benchmark.testset import OfflineTestset
    from gently_perception.render import CachedFrameSource
    from perception import get_functions
    from perception._base import DEFAULT_MODEL
    import run as old_run

    fns = get_functions()
    if variant not in fns:
        raise SystemExit(f"unknown variant: {variant}. choices: {sorted(fns)}")

    gt = GroundTruth.from_json(old_run.GROUND_TRUTH_PATH)
    testset = CachedFrameSource(
        OfflineTestset(session_path=old_run.VOLUMES_DIR,
                       ground_truth=gt, load_volumes=True),
        cache_dir=REPO_ROOT / "data" / "cache" / "frames",
    )
    refs = old_run.load_references()

    rec = Recorder.for_variant(variant, DEFAULT_MODEL)
    with install_record(rec):
        await old_run.run_variant(
            variant_name=variant,
            perceive_fn=fns[variant],
            testset=testset,
            references=refs,
            max_timepoints=None,
            target_stages=set(stages) if stages else None,
        )
    print(f"recorded {len(rec._cache)} responses → {rec.path}")


def main() -> None:
    import logging
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--record", action="store_true",
                   help="Hit live API and write recordings (costs tokens).")
    p.add_argument("--variant", required=True)
    p.add_argument("--stages", nargs="+",
                   default=["1.5fold", "2fold", "pretzel"])
    args = p.parse_args()

    if not args.record:
        p.error("only --record is implemented as a CLI; replay is used "
                "programmatically via install_replay() in tests.")
    asyncio.run(_record_variant(args.variant, args.stages))


if __name__ == "__main__":
    main()
