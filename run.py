"""
Run a perception experiment and report the score.

This is the evaluation entry point — equivalent to running train.py in autoresearch.
The agent modifies perception functions in perception/, then runs this to evaluate.

Usage:
    # Quick eval (30 timepoints per embryo, ~2 min per variant)
    python run.py --variant minimal --quick

    # Full eval (all timepoints, ~1 hour per variant)
    python run.py --variant descriptive

    # Run all registered variants
    python run.py --quick

    # Force re-run (overwrite existing results)
    python run.py --variant minimal --quick --force
"""

import argparse
import asyncio
import inspect
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path

logger = logging.getLogger(__name__)

# Paths (relative to repo root)
DATA_DIR = Path(__file__).parent / "data"
VOLUMES_DIR = DATA_DIR / "volumes"
GROUND_TRUTH_PATH = DATA_DIR / "ground_truth" / "59799c78.json"
EXAMPLES_DIR = DATA_DIR / "examples"
RESULTS_DIR = DATA_DIR / "results"


def load_references() -> dict[str, list[str]]:
    """Load reference stage images as base64 JPEG strings."""
    import base64

    refs: dict[str, list[str]] = {}
    if not EXAMPLES_DIR.exists():
        logger.warning(f"Examples directory not found: {EXAMPLES_DIR}")
        return refs

    from perception._base import STAGES

    for stage in STAGES:
        stage_dir = EXAMPLES_DIR / stage
        if not stage_dir.exists():
            continue
        images = []
        for img_path in sorted(stage_dir.glob("*.jpg")) + sorted(stage_dir.glob("*.jpeg")):
            with open(img_path, "rb") as f:
                images.append(base64.b64encode(f.read()).decode("utf-8"))
        if images:
            refs[stage] = images[:2]  # Max 2 per stage

    return refs


def make_prediction_dict(output, timepoint, ground_truth_stage) -> dict:
    """Convert PerceptionOutput to result dict."""
    from perception._base import STAGES

    predicted = output.stage
    gt = ground_truth_stage
    is_correct = (predicted == gt) if gt is not None else False

    is_adjacent_correct = False
    if gt is not None and predicted in STAGES and gt in STAGES:
        pred_idx = STAGES.index(predicted)
        gt_idx = STAGES.index(gt)
        is_adjacent_correct = abs(pred_idx - gt_idx) <= 1

    return {
        "timepoint": timepoint,
        "predicted_stage": predicted,
        "ground_truth_stage": gt,
        "is_transitional": False,
        "transition_between": None,
        "reasoning": output.reasoning,
        "reasoning_trace": None,
        "tool_calls": output.tool_calls,
        "tools_used": output.tools_used,
        "is_correct": is_correct,
        "is_adjacent_correct": is_adjacent_correct,
        "verification_triggered": output.verification_triggered,
        "phase_count": output.phase_count,
        "verification_result": None,
        "candidate_stages": None,
    }


_OPTIONAL_PERCEIVE_KWARGS = ("midplane_b64",)


def _accepted_optional_kwargs(perceive_fn) -> set[str]:
    """Subset of _OPTIONAL_PERCEIVE_KWARGS that perceive_fn's signature accepts."""
    sig = inspect.signature(perceive_fn)
    if any(p.kind is p.VAR_KEYWORD for p in sig.parameters.values()):
        return set(_OPTIONAL_PERCEIVE_KWARGS)
    return {k for k in _OPTIONAL_PERCEIVE_KWARGS if k in sig.parameters}


async def run_variant(variant_name, perceive_fn, testset, references, max_timepoints,
                      target_stages=None):
    """Run a single variant. Returns accuracy and full report dict.

    Parameters
    ----------
    target_stages : set of str, optional
        If provided, only evaluate timepoints where the ground truth is one of
        these stages. Skipped timepoints still contribute to history (temporal
        context) but are not sent to the model or scored.
    """
    started_at = datetime.now()
    all_predictions = []
    embryo_results = []
    extra_kwargs = _accepted_optional_kwargs(perceive_fn)

    for embryo_id, tp_iter in testset.iter_all():
        logger.info(f"[{variant_name}] Starting {embryo_id}")
        embryo_start = time.time()
        predictions = []
        history: list[dict] = []

        for tc in tp_iter:
            if max_timepoints is not None and tc.timepoint >= max_timepoints:
                break

            # Stage filtering: skip timepoints outside target stages
            # but still record history so temporal context is accurate
            if target_stages and tc.ground_truth_stage not in target_stages:
                history.append({
                    "timepoint": tc.timepoint,
                    "stage": tc.ground_truth_stage or "early",
                })
                continue

            try:
                output = await perceive_fn(
                    image_b64=tc.image_b64,
                    references=references,
                    history=history,
                    timepoint=tc.timepoint,
                    **{k: getattr(tc, k) for k in extra_kwargs},
                )
            except Exception as e:
                logger.error(f"[{variant_name}/{embryo_id}] T{tc.timepoint} error: {e}")
                from perception._base import PerceptionOutput
                output = PerceptionOutput(stage="early", reasoning=f"Error: {e}")

            pred = make_prediction_dict(output, tc.timepoint, tc.ground_truth_stage)
            predictions.append(pred)
            all_predictions.append(pred)

            history.append({
                "timepoint": tc.timepoint,
                "stage": output.stage,
            })

            status = "OK" if pred["is_correct"] else "WRONG"
            logger.info(
                f"[{variant_name}/{embryo_id}] T{tc.timepoint}: "
                f"{output.stage} (GT={tc.ground_truth_stage}) {status}"
            )

        n = len(predictions) or 1
        n_correct = sum(1 for p in predictions if p["is_correct"])
        n_adj = sum(1 for p in predictions if p["is_adjacent_correct"])

        embryo_results.append({
            "embryo_id": embryo_id,
            "predictions": predictions,
            "duration_seconds": time.time() - embryo_start,
            "error": None,
            "accuracy": n_correct / n,
            "adjacent_accuracy": n_adj / n,
        })

    # Compute overall metrics
    total = len(all_predictions) or 1
    exact = sum(1 for p in all_predictions if p["is_correct"]) / total
    adjacent = sum(1 for p in all_predictions if p["is_adjacent_correct"]) / total

    # Per-stage accuracy
    from collections import defaultdict
    stage_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    for p in all_predictions:
        gt = p["ground_truth_stage"]
        if gt:
            stage_stats[gt]["total"] += 1
            if p["is_correct"]:
                stage_stats[gt]["correct"] += 1

    report = {
        "config": {"description": f"Variant: {variant_name}"},
        "embryo_results": embryo_results,
        "started_at": started_at.isoformat(),
        "completed_at": datetime.now().isoformat(),
        "total_predictions": len(all_predictions),
        "overall_accuracy": exact,
        "metrics": {
            "accuracy": exact,
            "adjacent_accuracy": adjacent,
            "per_stage": {
                stage: {
                    "accuracy": s["correct"] / s["total"] if s["total"] > 0 else 0,
                    "n": s["total"],
                }
                for stage, s in stage_stats.items()
            },
        },
    }

    return exact, report


def plot_results(results: dict[str, dict], output_path: Path):
    """Generate per-stage accuracy bar chart comparing all variants."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed — skipping chart generation")
        return

    from perception._base import STAGES

    # Collect stages that have data
    stages_with_data = []
    for stage in STAGES:
        for report in results.values():
            ps = report["metrics"]["per_stage"].get(stage, {})
            if ps.get("n", 0) > 0:
                stages_with_data.append(stage)
                break

    if not stages_with_data:
        return

    variant_names = sorted(results.keys())
    n_variants = len(variant_names)
    n_stages = len(stages_with_data)

    fig, ax = plt.subplots(figsize=(max(8, n_stages * 1.5), 5))

    bar_width = 0.8 / n_variants
    x = range(n_stages)

    colors = plt.cm.Set2(range(n_variants))

    for i, name in enumerate(variant_names):
        report = results[name]
        accuracies = []
        for stage in stages_with_data:
            ps = report["metrics"]["per_stage"].get(stage, {})
            accuracies.append(ps.get("accuracy", 0))

        offsets = [xi + (i - n_variants / 2 + 0.5) * bar_width for xi in x]
        bars = ax.bar(offsets, [a * 100 for a in accuracies], bar_width,
                      label=f"{name} ({report['metrics']['accuracy']:.0%})",
                      color=colors[i], edgecolor="white", linewidth=0.5)

        # Add value labels on bars
        for bar, acc in zip(bars, accuracies):
            if acc > 0:
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                        f"{acc:.0%}", ha="center", va="bottom", fontsize=7)

    ax.set_xlabel("Stage")
    ax.set_ylabel("Accuracy (%)")
    ax.set_title("Per-Stage Classification Accuracy")
    ax.set_xticks(list(x))
    ax.set_xticklabels(stages_with_data, rotation=30, ha="right")
    ax.set_ylim(0, 105)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    print(f"\nChart saved to: {output_path}")


def print_results(results: dict[str, dict]):
    """Print results table."""
    from perception._base import STAGES

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)

    # Overall table
    header = f"{'Variant':<25} {'Exact':>8} {'Adjacent':>10} {'N':>6}"
    print(header)
    print("-" * len(header))

    for name, report in sorted(results.items()):
        m = report["metrics"]
        print(
            f"{name:<25} {m['accuracy']:>7.1%} {m['adjacent_accuracy']:>9.1%} "
            f"{report['total_predictions']:>6}"
        )

    # Per-stage breakdown
    print("\nPer-stage accuracy:")
    print(f"{'Stage':<12}", end="")
    for name in sorted(results.keys()):
        print(f"{name:>20}", end="")
    print()
    print("-" * (12 + 20 * len(results)))

    for stage in STAGES:
        print(f"{stage:<12}", end="")
        for name in sorted(results.keys()):
            ps = results[name]["metrics"]["per_stage"].get(stage, {})
            if ps.get("n", 0) > 0:
                print(f"{ps['accuracy']:>18.0%} ({ps['n']})", end="")
            else:
                print(f"{'':>20}", end="")
        print()

    print()


async def main():
    parser = argparse.ArgumentParser(
        description="Run perception experiment and report accuracy"
    )
    parser.add_argument(
        "--variant", nargs="+",
        help="Variant(s) to run (default: all registered)",
    )
    parser.add_argument(
        "--stages", nargs="+",
        help="Only evaluate these ground-truth stages (e.g., --stages pretzel 2fold 1.5fold)",
    )
    parser.add_argument(
        "--quick", action="store_true",
        help="Quick eval: 30 timepoints per embryo",
    )
    parser.add_argument(
        "--max-timepoints", type=int,
        help="Custom max timepoints per embryo",
    )
    parser.add_argument(
        "--volumes", type=str, default=str(VOLUMES_DIR),
        help="Path to volumes directory",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-run even if results exist",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Verbose logging",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    max_tp = args.max_timepoints
    if args.quick and max_tp is None:
        max_tp = 30

    # Stage filtering
    target_stages = None
    if args.stages:
        from perception._base import STAGES
        for s in args.stages:
            if s not in STAGES:
                print(f"Unknown stage: {s}. Valid: {STAGES}")
                sys.exit(1)
        target_stages = set(args.stages)
        logger.info(f"Filtering to stages: {target_stages}")

    volumes_path = Path(args.volumes)
    if not volumes_path.exists():
        print(f"Volumes not found: {volumes_path}")
        print("Symlink or copy your volume data to data/volumes/")
        sys.exit(1)

    if not GROUND_TRUTH_PATH.exists():
        print(f"Ground truth not found: {GROUND_TRUTH_PATH}")
        sys.exit(1)

    # Load testset
    from benchmark.ground_truth import GroundTruth
    from benchmark.testset import OfflineTestset

    ground_truth = GroundTruth.from_json(GROUND_TRUTH_PATH)
    testset = OfflineTestset(
        session_path=volumes_path,
        ground_truth=ground_truth,
        load_volumes=True,
    )
    logger.info(f"Testset: {len(testset.embryo_ids)} embryos")

    references = load_references()
    logger.info(f"References: {len(references)} stages")

    # Get variants
    from perception import get_functions
    all_fns = get_functions()

    if args.variant:
        for v in args.variant:
            if v not in all_fns:
                print(f"Unknown variant: {v}. Available: {list(all_fns.keys())}")
                sys.exit(1)
        variants = {v: all_fns[v] for v in args.variant}
    else:
        variants = all_fns

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    completed = {}

    # Build output filename suffix for stage-filtered runs
    stage_suffix = ""
    if target_stages:
        stage_suffix = "_" + "+".join(sorted(target_stages))

    for name, perceive_fn in variants.items():
        output_path = RESULTS_DIR / f"{name}{stage_suffix}.json"

        if output_path.exists() and not args.force:
            logger.info(f"Skipping {name} (exists). Use --force to re-run.")
            with open(output_path) as f:
                completed[name] = json.load(f)
            continue

        logger.info(f"Running: {name}")
        accuracy, report = await run_variant(
            variant_name=name,
            perceive_fn=perceive_fn,
            testset=testset,
            references=references,
            max_timepoints=max_tp,
            target_stages=target_stages,
        )

        with open(output_path, "w") as f:
            json.dump(report, f, indent=2, default=str)

        print(f"\n>>> {name}: {accuracy:.1%} exact accuracy")
        completed[name] = report

    if completed:
        print_results(completed)
        plot_results(completed, RESULTS_DIR / f"chart{stage_suffix}.png")


if __name__ == "__main__":
    asyncio.run(main())
