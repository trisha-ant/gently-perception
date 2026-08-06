"""Statistical comparison of replicated benchmark runs.

Operates on the result-dict shape produced by run.py (and archived in
data/results/). No scipy dependency — Welch's t is computed directly so the
test suite stays lightweight.
"""

from __future__ import annotations

import math
import statistics
from dataclasses import dataclass


@dataclass(frozen=True)
class RunSet:
    """A set of replicated runs of the same config."""

    name: str
    accuracies: tuple[float, ...]  # one overall_accuracy per run, in [0, 1]

    @property
    def n(self) -> int:
        return len(self.accuracies)

    @property
    def mean(self) -> float:
        return statistics.fmean(self.accuracies)

    @property
    def std(self) -> float:
        if self.n < 2:
            return 0.0
        return statistics.stdev(self.accuracies)

    def mean_std_pct(self) -> str:
        return f"{100 * self.mean:.1f} ± {100 * self.std:.1f}"

    @classmethod
    def from_reports(cls, name: str, reports: list[dict]) -> "RunSet":
        accs = tuple(r["overall_accuracy"] for r in reports)
        return cls(name=name, accuracies=accs)


@dataclass(frozen=True)
class Comparison:
    """Welch's t-test comparison between two RunSets."""

    a: RunSet
    b: RunSet
    delta: float  # a.mean - b.mean
    t: float
    df: float
    significant: bool  # |t| > 2 (rough p<0.05 for small N)

    def summary(self) -> str:
        sig = "SIGNIFICANT" if self.significant else "not significant"
        return (
            f"{self.a.name} {self.a.mean_std_pct()}  vs  "
            f"{self.b.name} {self.b.mean_std_pct()}  "
            f"Δ={100 * self.delta:+.1f}pp  t={self.t:.2f}  ({sig})"
        )


def welch_t(a: RunSet, b: RunSet) -> Comparison:
    """Welch's unequal-variance t-test on two RunSets."""
    if a.n < 2 or b.n < 2:
        raise ValueError(f"Need ≥2 runs per set (got {a.n}, {b.n})")

    var_a = a.std ** 2
    var_b = b.std ** 2
    se = math.sqrt(var_a / a.n + var_b / b.n)
    delta = a.mean - b.mean
    t = delta / se if se > 0 else math.inf

    # Welch–Satterthwaite df
    num = (var_a / a.n + var_b / b.n) ** 2
    den = (var_a / a.n) ** 2 / (a.n - 1) + (var_b / b.n) ** 2 / (b.n - 1)
    df = num / den if den > 0 else float(a.n + b.n - 2)

    return Comparison(a=a, b=b, delta=delta, t=t, df=df, significant=abs(t) > 2.0)


def compare(reports_a: list[dict], reports_b: list[dict],
            name_a: str = "A", name_b: str = "B") -> Comparison:
    """Convenience: build RunSets from result-dict lists and run Welch's t."""
    return welch_t(
        RunSet.from_reports(name_a, reports_a),
        RunSet.from_reports(name_b, reports_b),
    )
