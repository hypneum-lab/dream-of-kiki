"""Wake-Sleep CL baseline driver (Paper 2 §7 row).

Gate ID         : Paper 2 §7 ablation row (no G-gate ; baseline
                  comparator only).
Validates       : pipeline-validation (variant c) — the baseline
                  adapter round-trips via the RunRegistry without
                  claiming new empirical results. Variants a/b
                  promote this to empirical.
Output path     : docs/milestones/wake-sleep-baseline-2026-05-03.json
                  + .md sibling.

Usage :
    uv run python scripts/baseline_wake_sleep_cl.py --seeds 42 123 7

Reference :
  docs/superpowers/plans/2026-05-02-wake-sleep-cl-ablation-baseline.md
  docs/papers/paper1/references.bib `alfarano2024wakesleep`
  docs/papers/paper2/architecture.md §5.8
  docs/papers/paper2/methodology.md §6.3
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.substrates.wake_sleep_cl_baseline import (  # noqa: E402
    WAKE_SLEEP_BASELINE_NAME,
    WAKE_SLEEP_BASELINE_VERSION,
    WakeSleepCLBaseline,
)

DEFAULT_SEEDS = (42, 123, 7)
DEFAULT_TASK_SPLIT = "split_fmnist_5tasks"
DEFAULT_PROFILE = "baseline_wake_sleep_cl"
# Hardcoded milestone date — the JSON dump path is part of the R1
# reproducibility contract and must be bit-stable across re-runs.
# Using `date.today()` would silently shift the path each calendar
# day even though the file contents are deterministic.
MILESTONE_DATE = "2026-05-03"
DEFAULT_OUT = (
    REPO_ROOT
    / "docs"
    / "milestones"
    / f"wake-sleep-baseline-{MILESTONE_DATE}.json"
)


@dataclass(frozen=True)
class BaselineRow:
    seed: int
    task_split: str
    forgetting_rate: float
    avg_accuracy: float
    n_tasks: int
    source: str
    run_id: str


def _resolve_commit_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def run(
    seeds: Iterable[int],
    out_path: Path,
    *,
    registry: RunRegistry | None = None,
    commit_sha: str | None = None,
) -> dict[str, Any]:
    """Run the wake-sleep baseline over the seed grid and dump JSON.

    The registry parameter is injectable for tests; defaults to the
    repo-level `harness/storage/runs.sqlite`.
    """
    bl = WakeSleepCLBaseline()
    sha = commit_sha if commit_sha is not None else _resolve_commit_sha()
    if registry is None:
        registry = RunRegistry(
            REPO_ROOT / "harness" / "storage" / "runs.sqlite"
        )
    rows: list[BaselineRow] = []

    for seed in seeds:
        rid = registry.register(
            c_version=WAKE_SLEEP_BASELINE_VERSION,
            profile=DEFAULT_PROFILE,
            seed=seed,
            commit_sha=sha,
        )
        out = bl.evaluate_continual(seed=seed, task_split=DEFAULT_TASK_SPLIT)
        rows.append(
            BaselineRow(
                seed=seed,
                task_split=DEFAULT_TASK_SPLIT,
                forgetting_rate=float(out["forgetting_rate"]),
                avg_accuracy=float(out["avg_accuracy"]),
                n_tasks=int(out["n_tasks"]),
                source=str(out["source"]),
                run_id=rid,
            )
        )

    dump: dict[str, Any] = {
        "baseline": WAKE_SLEEP_BASELINE_NAME,
        "version": WAKE_SLEEP_BASELINE_VERSION,
        "bibkey": "alfarano2024wakesleep",
        "task_split": DEFAULT_TASK_SPLIT,
        "commit_sha": sha,
        "rows": [asdict(r) for r in rows],
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(dump, indent=2, sort_keys=True))
    return dump


def md_companion(dump: dict[str, Any], json_path: Path) -> Path:
    """Render a Markdown companion next to the JSON dump."""
    md_path = json_path.with_suffix(".md")
    body = [
        f"# Wake-Sleep CL baseline — {dump['version']}",
        "",
        "**Source :** Alfarano et al. 2024, IEEE TNNLS, arXiv 2401.08623.",
        "**Bibkey :** `alfarano2024wakesleep`.",
        f"**Task split :** `{dump['task_split']}`.",
        f"**Commit :** `{dump['commit_sha']}`.",
        "",
        "**(synthetic placeholder — variant c, published reference values.)**",
        "",
        "| seed | run_id | forgetting_rate | avg_accuracy |",
        "|------|--------|-----------------|--------------|",
    ]
    for r in dump["rows"]:
        body.append(
            f"| {r['seed']} | `{r['run_id']}` | "
            f"{r['forgetting_rate']:.4f} | {r['avg_accuracy']:.4f} |"
        )
    md_path.write_text("\n".join(body) + "\n")
    return md_path


def cli() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS))
    p.add_argument("--out", type=Path, default=DEFAULT_OUT)
    args = p.parse_args()
    dump = run(args.seeds, args.out)
    md = md_companion(dump, args.out)
    print(f"json -> {args.out}")
    print(f"md   -> {md}")


if __name__ == "__main__":
    cli()
