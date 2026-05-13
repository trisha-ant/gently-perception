"""New research-harness runner: --n-runs, --model, --thinking, --baseline.

Append-only results, full provenance, built-in stats. Coexists with the legacy
root ``run.py`` until Phase A recordings are captured.

Usage::

    python -m experiments.harness.run --variant hybrid \\
        --stages 1.5fold 2fold pretzel --model claude-opus-4-6 --n-runs 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from benchmark.ground_truth import GroundTruth  # noqa: E402
from benchmark.stats import RunSet, welch_t  # noqa: E402
from benchmark.testset import OfflineTestset  # noqa: E402

from .config import RunConfig  # noqa: E402
from .discover import discover_variants  # noqa: E402
from .events import EventLog  # noqa: E402
from .loop import STAGES, run_variant  # noqa: E402

logger = logging.getLogger(__name__)

DATA_DIR = REPO_ROOT / "data"
RESULTS_DIR = DATA_DIR / "results"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth" / "59799c78.json"
EXAMPLES_DIR = DATA_DIR / "examples"


def _load_references() -> dict[str, list[str]]:
    import base64
    refs: dict[str, list[str]] = {}
    for stage in STAGES:
        d = EXAMPLES_DIR / stage
        if not d.exists():
            continue
        imgs = []
        for p in sorted(d.glob("*.jpg")) + sorted(d.glob("*.jpeg")):
            imgs.append(base64.b64encode(p.read_bytes()).decode())
        if imgs:
            refs[stage] = imgs[:2]
    return refs


async def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--variant", required=True)
    p.add_argument("--stages", nargs="+", default=["1.5fold", "2fold", "pretzel"])
    p.add_argument("--model", required=True)
    p.add_argument("--thinking", choices=["adaptive", "none"], default="none")
    p.add_argument("--n-runs", type=int, default=3)
    p.add_argument("--seed-base", type=int, default=0)
    p.add_argument("--baseline", help="variant name to compare against (uses "
                                      "stored results at same model/thinking)")
    p.add_argument("--volumes", default=str(DATA_DIR / "volumes"))
    p.add_argument("-v", "--verbose", action="store_true")
    args = p.parse_args()

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(message)s")

    for s in args.stages:
        if s not in STAGES:
            sys.exit(f"unknown stage: {s}. valid: {STAGES}")

    variants = discover_variants()
    if args.variant not in variants:
        sys.exit(f"unknown variant: {args.variant}. available: {sorted(variants)}")
    perceive = variants[args.variant]

    from gently_perception.render import CachedFrameSource

    gt = GroundTruth.from_json(GROUND_TRUTH_PATH)
    testset = CachedFrameSource(
        OfflineTestset(session_path=Path(args.volumes), ground_truth=gt,
                       load_volumes=True),
        cache_dir=DATA_DIR / "cache" / "frames",
    )
    references = _load_references()
    thinking = None if args.thinking == "none" else args.thinking

    async def one_seed(seed: int) -> dict:
        cfg = RunConfig.capture(variant=args.variant, model=args.model,
                                thinking=thinking, seed=seed,
                                stages=tuple(args.stages))
        out_path = cfg.result_path(RESULTS_DIR)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        ev_log = EventLog(out_path.with_suffix(".events.jsonl"))
        logger.info(f"[seed {seed}] → {out_path}")
        report = await run_variant(perceive, testset, references, cfg,
                                   event_log=ev_log)
        ev_log.close()
        out_path.write_text(json.dumps(report, indent=2, default=str))
        print(f"  seed {seed}: {report['overall_accuracy']:.1%}")
        return report

    reports = await asyncio.gather(
        *(one_seed(args.seed_base + i) for i in range(args.n_runs))
    )
    print(f"render cache: {testset.stats()}")

    rs = RunSet.from_reports(args.variant, reports)
    print(f"\n{args.variant} @ {args.model} ({args.thinking}): {rs.mean_std_pct()}  (N={rs.n})")

    if args.baseline:
        bl_dir = (RESULTS_DIR / args.baseline
                  / args.model.replace("/", "_").replace(":", "_"))
        bl_files = sorted(bl_dir.glob("*.json")) if bl_dir.exists() else []
        if len(bl_files) >= 2:
            bl_reports = [json.loads(f.read_text()) for f in bl_files]
            cmp = welch_t(rs, RunSet.from_reports(args.baseline, bl_reports))
            print(cmp.summary())
        else:
            print(f"(no stored baseline runs at {bl_dir})")


if __name__ == "__main__":
    asyncio.run(main())
