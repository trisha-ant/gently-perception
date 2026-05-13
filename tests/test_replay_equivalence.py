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


VARIANTS = ["hybrid", "scientific", "temporal", "minimal", "fillpct",
            "multimeasure", "vote3_mm", "ensemble"]


@pytest.mark.slow
@pytest.mark.parametrize("variant", VARIANTS)
def test_new_harness_replays_without_cache_miss(variant):
    """Two assertions, both against the SAME recording:

    1. (request-level) The new loop never raises CacheMiss → every request it
       emits was emitted by the old loop.
    2. (prediction-level) Replaying the recording through the OLD loop and the
       NEW loop yields identical per-frame predictions.
    """
    rec_path = _recording_for(variant)
    if rec_path is None:
        pytest.skip(f"no recording for {variant} (run record_replay --record first)")

    from benchmark.ground_truth import GroundTruth
    from benchmark.testset import OfflineTestset
    from gently_perception.render import CachedFrameSource
    from perception import get_functions
    import run as legacy_run

    rec = Recorder(rec_path)
    gt = GroundTruth.from_json(REPO_ROOT / "data" / "ground_truth" / "59799c78.json")
    src = CachedFrameSource(
        OfflineTestset(session_path=REPO_ROOT / "data" / "volumes",
                       ground_truth=gt, load_volumes=True),
        cache_dir=REPO_ROOT / "data" / "cache" / "frames",
    )
    refs = _refs()

    # ---- OLD harness replay ----------------------------------------------- #
    legacy_fn = get_functions()[variant]
    with install_replay(rec):
        _, old_report = asyncio.run(legacy_run.run_variant(
            variant_name=variant, perceive_fn=legacy_fn, testset=src,
            references=refs, max_timepoints=None, target_stages=set(STAGES),
        ))

    # ---- NEW harness replay ----------------------------------------------- #
    new_fn = discover_variants()[variant]
    cfg = RunConfig(variant=variant, model="replay", thinking=None, seed=0,
                    stages=STAGES)
    try:
        with install_replay(rec):
            new_report = asyncio.run(run_variant(new_fn, src, refs, cfg,
                                                 concurrency=1))
    except CacheMiss as e:
        pytest.fail(f"NEW harness emitted a request the OLD harness didn't:\n  {e}")

    assert new_report["total_predictions"] == old_report["total_predictions"] > 100

    old_preds = _flatten(old_report)
    new_preds = _flatten(new_report)
    diffs = [(k, old_preds[k], new_preds[k])
             for k in sorted(old_preds) if old_preds[k] != new_preds.get(k)]
    assert not diffs, (
        f"{len(diffs)} frame(s) differ old→new under identical model output:\n  "
        + "\n  ".join(f"{k}: {o}→{n}" for k, o, n in diffs[:10])
    )
    # Accuracy must therefore be identical too.
    assert new_report["overall_accuracy"] == pytest.approx(
        old_report["overall_accuracy"], abs=1e-12)


def _flatten(report: dict) -> dict:
    return {(er["embryo_id"], p["timepoint"]): p["predicted_stage"]
            for er in report["embryo_results"] for p in er["predictions"]}


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
