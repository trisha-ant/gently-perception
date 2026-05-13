"""Run configuration with full provenance.

Every result file embeds a ``RunConfig.to_dict()`` so the run is reproducible
from the artifact alone. Model and thinking are parameters, not constants —
this is the [P2] fix for the silent 4.6/4.7 drift.
"""

from __future__ import annotations

import subprocess
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


def _git(*args: str) -> str:
    try:
        return subprocess.run(
            ["git", "-C", str(REPO_ROOT), *args],
            capture_output=True, text=True, check=True,
        ).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return ""


@dataclass(frozen=True)
class RunConfig:
    """Immutable run configuration captured at the start of a run."""

    variant: str
    model: str
    thinking: str | None  # "adaptive" | None
    seed: int
    stages: tuple[str, ...]
    structured_output: bool = False  # P8, default off for replay-equivalence
    n_embryos_expected: int | None = None

    # Provenance — populated by ``capture()``
    git_sha: str = ""
    git_dirty: bool = False
    captured_at: str = ""

    @classmethod
    def capture(cls, *, variant: str, model: str, thinking: str | None,
                seed: int, stages: tuple[str, ...],
                structured_output: bool = False) -> "RunConfig":
        sha = _git("rev-parse", "HEAD")
        dirty = bool(_git("status", "--porcelain"))
        return cls(
            variant=variant,
            model=model,
            thinking=thinking,
            seed=seed,
            stages=stages,
            structured_output=structured_output,
            git_sha=sha,
            git_dirty=dirty,
            captured_at=datetime.now(timezone.utc).isoformat(),
        )

    @property
    def git_short(self) -> str:
        return self.git_sha[:8] if self.git_sha else "nogit"

    def result_path(self, results_root: Path) -> Path:
        """Append-only result location: results/{variant}/{model}/{git8}_{seed}.json"""
        safe_model = self.model.replace("/", "_").replace(":", "_")
        return results_root / self.variant / safe_model / f"{self.git_short}_{self.seed}.json"

    def to_dict(self) -> dict:
        d = asdict(self)
        d["stages"] = list(self.stages)
        return d
