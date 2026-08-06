"""Shared fixtures for harness eval tests.

These fixtures operate on archived result JSONs in data/results/ so the
offline suite needs no API access and no volume data.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = REPO_ROOT / "data" / "results"
FIXTURES_DIR = REPO_ROOT / "data" / "fixtures"

# Make benchmark/, perception/, gently_perception/ importable when running
# pytest from repo root without an editable install.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


def _load(path: Path) -> dict:
    with open(path) as f:
        return json.load(f)


@pytest.fixture(scope="session")
def results_dir() -> Path:
    return RESULTS_DIR


@pytest.fixture(scope="session")
def replication():
    """Load a named replication directory as a list of result dicts.

    Usage: ``runs = replication("hybrid46")`` → [r1, r2, r3]
    """
    def _load_replication(name: str) -> list[dict]:
        d = RESULTS_DIR / f"{name}_replication"
        if not d.exists():
            pytest.skip(f"replication dir not found: {d}")
        files = sorted(d.glob("*.json"))
        if not files:
            pytest.skip(f"no JSONs in {d}")
        return [_load(p) for p in files]
    return _load_replication


@pytest.fixture(scope="session")
def archived():
    """Load a single archived result JSON by stem.

    Usage: ``r = archived("judge_sequential")``
    """
    def _load_one(stem: str) -> dict:
        p = RESULTS_DIR / f"{stem}.json"
        if not p.exists():
            pytest.skip(f"archived result not found: {p}")
        return _load(p)
    return _load_one


@pytest.fixture(scope="session")
def hyb46(replication) -> list[dict]:
    """The golden hybrid@4.6 N=3 replication (81.7 ± 2.4)."""
    runs = replication("hybrid46")
    assert len(runs) == 3, f"expected 3 hybrid46 runs, got {len(runs)}"
    return runs


@pytest.fixture(scope="session")
def h4_vote3mm() -> list[dict]:
    """The golden vote3_mm@4.7 N=3 replication (80.8 ± 0.5) — h4 in h1_h5 sweep."""
    d = RESULTS_DIR / "h1_h5_replication"
    files = sorted(d.glob("h4_r*.json"))
    assert len(files) == 3, f"expected 3 h4 runs, got {len(files)}"
    return [_load(p) for p in files]


@pytest.fixture
def stub_model():
    """Return a callable that yields scripted model responses.

    Usage::

        responder = stub_model(['{"stage":"comma","reasoning":"x"}', ...])
        # responder() returns each string in order, then raises StopIteration
    """
    def _make(script: list[str]):
        it = iter(script)

        async def _respond(*_args, **_kwargs) -> str:
            return next(it)

        return _respond
    return _make
