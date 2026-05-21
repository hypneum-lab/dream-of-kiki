"""Wave 3b M5 milestone aggregation.

Reads per-cell records (the JSONL sidecar) and produces the
deterministic ``wave3b-bench-YYYY-MM-DD.json`` summary: per-profile
descriptive stats, H1/H2/H4 verdicts, cross-substrate scaffolding.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

_METRIC_KEYS = (
    "replay_rate", "downscale_norm", "restructure_sum",
    "recombine_rate", "delta_acc",
)


def _describe(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std,
            "min": min(values), "max": max(values)}


def aggregate_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure aggregation — deterministic given the same cell list."""
    profiles: dict[str, Any] = {}
    for profile in sorted({c["profile"] for c in cells}):
        rows = [c for c in cells if c["profile"] == profile]
        stats: dict[str, Any] = {
            key: _describe(sorted(float(r[key]) for r in rows))
            for key in _METRIC_KEYS
        }
        profiles[profile] = {"n_cells": len(rows), "stats": stats}
    return {
        "c_version": "C-v0.14.0+PARTIAL",
        "substrate": "mlx_latent_diffusion",
        "total_cells": len(cells),
        "profiles": profiles,
    }


def _hypothesis_verdicts(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """H1/H2/H4 stat-test verdicts over real per-cell metrics.

    Bonferroni alpha = 0.05 / 4 = 0.0125. Uses the same primitives
    as ablation_cycle2._run_h1_h4 but fed real arrays.

    Signatures confirmed from kiki_oniric/eval/statistics.py:
    - welch_one_sided(treatment, control, alpha) -> StatTestResult
    - tost_equivalence(treatment, control, epsilon, alpha)
      -> StatTestResult
    - one_sample_threshold(sample, threshold, alpha) -> StatTestResult
    All return .test_name, .p_value, .reject_h0 (StatTestResult
    dataclass, frozen).
    """
    from kiki_oniric.eval.statistics import (
        one_sample_threshold,
        tost_equivalence,
        welch_one_sided,
    )

    def _col(profile: str, key: str) -> list[float]:
        return [float(c[key]) for c in cells if c["profile"] == profile]

    alpha = 0.0125
    verdicts: dict[str, Any] = {}
    for profile in sorted({c["profile"] for c in cells}):
        h1 = welch_one_sided(
            treatment=_col(profile, "replay_rate"),
            control=_col("p_min", "replay_rate"),
            alpha=alpha,
        )
        h2 = tost_equivalence(
            treatment=_col(profile, "downscale_norm"),
            control=_col("p_equ", "downscale_norm"),
            epsilon=0.05,
            alpha=alpha,
        )
        h4 = one_sample_threshold(
            sample=_col(profile, "recombine_rate"),
            threshold=0.0,
            alpha=alpha,
        )
        verdicts[profile] = {
            "H1_replay": {
                "test": h1.test_name,
                "p": h1.p_value,
                "reject_h0": h1.reject_h0,
            },
            "H2_downscale": {
                "test": h2.test_name,
                "p": h2.p_value,
                "reject_h0": h2.reject_h0,
            },
            "H4_recombine": {
                "test": h4.test_name,
                "p": h4.p_value,
                "reject_h0": h4.reject_h0,
            },
        }
    return verdicts


def write_milestone(
    output_path: Path,
    registry: Any,
    commit_sha: str,
) -> dict[str, Any]:
    """Read the JSONL sidecar, aggregate, write the JSON + md.

    Deterministic: two runs over the same sidecar produce identical
    bytes (sorted keys, sorted cell order).
    """
    cells_path = Path(output_path).with_suffix(".cells.jsonl")
    cells: list[dict[str, Any]] = []
    if cells_path.exists():
        with cells_path.open(encoding="utf-8") as fh:
            cells = [json.loads(line) for line in fh if line.strip()]
    cells.sort(key=lambda c: (c["profile"], c["seed"], c["task_idx"]))

    summary = aggregate_cells(cells)
    summary["commit_sha"] = commit_sha
    summary["hypotheses"] = _hypothesis_verdicts(cells) if cells else {}

    Path(output_path).write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path = Path(output_path).with_suffix(".md")
    md_path.write_text(_render_md(summary), encoding="utf-8")
    return summary


def _render_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Wave 3b M5 bench milestone",
        "",
        f"- substrate: `{summary['substrate']}`",
        f"- c_version: `{summary['c_version']}`",
        f"- commit: `{summary.get('commit_sha', 'n/a')}`",
        f"- total cells: {summary['total_cells']}",
        "",
        "## Per-profile cell counts",
        "",
        "| profile | n_cells |",
        "|---|---|",
    ]
    for profile, body in sorted(summary["profiles"].items()):
        lines.append(f"| {profile} | {body['n_cells']} |")
    lines.append("")
    return "\n".join(lines)
