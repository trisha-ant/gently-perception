"""Phase A: replay old-harness recordings through the new loop and assert
per-frame predictions are identical.

The recording was captured by running the legacy ``run.py`` with
``install_record`` wrapping ``call_claude``. This test installs
``install_replay`` and drives the NEW ``loop.run_variant`` over the same
testset. If the new harness emits any request the old one didn't, replay
raises ``CacheMiss`` — that's the equivalence signal.

Skipped when the recording file is absent (so CI stays green before
Phase A is run).
"""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import pytest

from experiments.analysis.record_replay import (
    CacheMiss,
    Recorder,
    install_replay,
)
from experiments.harness.config import RunConfig
from experiments.harness.discover import discover_variants
from experiments.harness.loop import run_variant

REPO_ROOT = Path(__file__).parent.parent
RECORDINGS = REPO_ROOT / "data" / "fixtures" / "recordings"
RESULTS = REPO_ROOT / "data" / "results"
STAGES = ("1.5fold", "2fold", "pretzel")


def _recording_for(variant: str) -> Path | None:
    matches = list(RECORDINGS.glob(f"{variant}__*.jsonl"))
    return matches[0] if matches else None


def _legacy_result_for(variant: str) -> Path | None:
    """The legacy run.py writes results/{variant}_{stages}.json. The recording
    CLI invokes run_variant() but doesn't persist the result, so we may not
    have this for the smoke run — fall back to recording-only check."""
    suffix = "_" + "+".join(sorted(STAGES))
    p = RESULTS / f"{variant}{suffix}.json"
    return p if p.exists() else None


@pytest.mark.slow
@pytest.mark.parametrize("variant", ["hybrid"])
def test_new_harness_replays_without_cache_miss(variant):
    """Load-bearing assertion: every request the new loop emits was emitted by
    the old loop. A CacheMiss means the harnesses diverged."""
    rec_path = _recording_for(variant)
    if rec_path is None:
        pytest.skip(f"no recording for {variant} (run record_replay --record first)")

    from benchmark.ground_truth import GroundTruth
    from benchmark.testset import OfflineTestset

    rec = Recorder(rec_path)
    perceive = discover_variants()[variant]
    gt = GroundTruth.from_json(REPO_ROOT / "data" / "ground_truth" / "59799c78.json")
    src = OfflineTestset(session_path=REPO_ROOT / "data" / "volumes",
                         ground_truth=gt, load_volumes=True)
    cfg = RunConfig(variant=variant, model="replay", thinking=None, seed=0,
                    stages=STAGES)

    try:
        with install_replay(rec):
            result = asyncio.run(run_variant(perceive, src, _refs(), cfg,
                                             concurrency=1))
    except CacheMiss as e:
        pytest.fail(f"NEW harness emitted a request the OLD harness didn't:\n  {e}")

    # Non-trivial run
    assert result["total_predictions"] > 100

    # If a legacy result JSON exists, compare per-frame predictions.
    legacy_path = _legacy_result_for(variant)
    if legacy_path:
        legacy = json.loads(legacy_path.read_text())
        new_preds = {(er["embryo_id"], p["timepoint"]): p["predicted_stage"]
                     for er in result["embryo_results"]
                     for p in er["predictions"]}
        old_preds = {(er["embryo_id"], p["timepoint"]): p["predicted_stage"]
                     for er in legacy["embryo_results"]
                     for p in er["predictions"]}
        common = set(new_preds) & set(old_preds)
        diffs = [(k, old_preds[k], new_preds[k])
                 for k in sorted(common) if old_preds[k] != new_preds[k]]
        assert not diffs, (
            f"{len(diffs)} frame(s) differ old→new:\n  "
            + "\n  ".join(f"{k}: {o}→{n}" for k, o, n in diffs[:10])
        )


def _refs():
    """Same reference-loading as run.load_references — must be byte-identical
    for replay to hit cache."""
    import base64
    from experiments.harness.loop import STAGES as ALL_STAGES
    examples = REPO_ROOT / "data" / "examples"
    refs: dict[str, list[str]] = {}
    for stage in ALL_STAGES:
        d = examples / stage
        if not d.exists():
            continue
        imgs = []
        for p in sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg")):
            imgs.append(base64.b64encode(p.read_bytes()).decode())
        if imgs:
            refs[stage] = imgs[:2]
    return refs
