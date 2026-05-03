"""G4 pilot driver — Split-FMNIST × profile sweep on MLX substrate.

**Gate ID** : G4 — first empirical pilot
**Validates** : whether observed Hedges' g for retention on
Split-FMNIST 5-task class-incremental learning matches Hu 2020
P_equ floor (g >= 0.21) and Javadi 2024 P_min floor (|g| >= 0.13).
**Mode** : empirical claim at first-pilot scale (N=5 seeds per
arm — exploratory power, see docs/osf-prereg-g4-pilot.md §4).
**Expected output** :
    - docs/milestones/g4-pilot-2026-05-03.json (machine dump)
    - docs/milestones/g4-pilot-2026-05-03.md (human report)

Sweep : arms × seeds = 4 × 5 = 20 cells.
    arms  = [baseline, P_min, P_equ, P_max]
    seeds = [0, 1, 2, 3, 4]

Per-cell pipeline :
    1. Seeded classifier built (G4Classifier seed=cell_seed).
    2. Train task 0 (binary FMNIST classes 0+1).
    3. Snapshot acc_task1_initial = eval on task-0 test set.
    4. For task in tasks[1..4] : (optional dream_episode) +
       train_task. dream_episode() is skipped on the baseline arm.
    5. Snapshot acc_task1_final = eval on task-0 test set.
    6. retention = acc_task1_final / max(acc_task1_initial, eps).
    7. Register run_id in RunRegistry.

Verdict :
    H1 : Hedges' g on retention[P_equ] vs retention[baseline]
         compared to HU_2020_OVERALL.
    H3 : Hedges' g on retention[P_min] vs retention[baseline]
         compared to JAVADI_2024_OVERALL.
    H_DR4 : Jonckheere-Terpstra on
         [retention[P_min], retention[P_equ], retention[P_max]].

Usage ::

    # Smoke 2-seed run on synthetic fixture (used by integration test)
    uv run python experiments/g4_split_fmnist/run_g4.py --smoke

    # Full 5-seed run on real FMNIST under
    # experiments/g4_split_fmnist/data/
    uv run python experiments/g4_split_fmnist/run_g4.py
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.benchmarks.effect_size_targets import (  # noqa: E402
    HU_2020_OVERALL,
    JAVADI_2024_OVERALL,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.eval.statistics import (  # noqa: E402
    compute_hedges_g,
    jonckheere_trend,
    welch_one_sided,
)

from experiments.g4_split_fmnist.dataset import (  # noqa: E402
    SplitFMNISTTask,
    load_split_fmnist_5tasks,
)
from experiments.g4_split_fmnist.dream_wrap import (  # noqa: E402
    G4Classifier,
    build_profile,
)


# --------------------------------------------------------------------
# Typed cell shapes — mirror SplitFMNISTTask pattern from dataset.py
# --------------------------------------------------------------------


class _CellPartial(TypedDict):
    """Per-cell measurement dict produced by ``_run_cell``.

    ``run_id`` is appended later by ``run_pilot`` once the cell is
    registered against ``RunRegistry``. Together with ``run_id``,
    this becomes :class:`CellResult` — the canonical cell shape.
    """

    arm: str
    seed: int
    acc_task1_initial: float
    acc_task1_final: float
    retention: float
    excluded_underperforming_baseline: bool
    wall_time_s: float


class CellResult(_CellPartial):
    """A fully-registered pilot cell : a :class:`_CellPartial` + ``run_id``."""

    run_id: str


# --------------------------------------------------------------------
# Constants — frozen by docs/osf-prereg-g4-pilot.md
# --------------------------------------------------------------------

C_VERSION = "C-v0.12.0+PARTIAL"  # current STATUS.md version tag
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2, 3, 4)
DEFAULT_DATA_DIR = REPO_ROOT / "experiments" / "g4_split_fmnist" / "data"
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "milestones" / "g4-pilot-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "docs" / "milestones" / "g4-pilot-2026-05-03.md"
)
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
RETENTION_EPS = 1e-6


# --------------------------------------------------------------------
# Commit SHA resolution — matches scripts/ablation_g4.py convention
# --------------------------------------------------------------------


def _resolve_commit_sha() -> str:
    env_sha = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


# --------------------------------------------------------------------
# Per-cell pipeline
# --------------------------------------------------------------------


def _run_cell(
    arm: str,
    seed: int,
    tasks: list[SplitFMNISTTask],
    *,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
) -> _CellPartial:
    """Execute one (arm, seed) cell and return a :class:`_CellPartial`.

    The returned dict carries ``arm, seed, acc_task1_initial,
    acc_task1_final, retention, excluded_underperforming_baseline,
    wall_time_s``. Excludes the cell from the verdict aggregation
    when ``acc_task1_initial < 0.5`` (per pre-reg §5). The caller
    upgrades it to :class:`CellResult` by adding ``run_id``.
    """
    start = time.time()
    feat_dim = tasks[0]["x_train"].shape[1]
    clf = G4Classifier(
        in_dim=feat_dim, hidden_dim=hidden_dim, n_classes=2, seed=seed
    )

    # Stage 1 — train task 0.
    clf.train_task(
        tasks[0], epochs=epochs, batch_size=batch_size, lr=lr
    )
    acc_initial = clf.eval_accuracy(
        tasks[0]["x_test"], tasks[0]["y_test"]
    )

    # Stage 2 — train tasks 1..4 with optional dream-episode interleaving.
    profile = None
    if arm != "baseline":
        profile = build_profile(arm, seed=seed)

    for k in range(1, len(tasks)):
        if profile is not None:
            clf.dream_episode(profile, seed=seed + k)
        clf.train_task(
            tasks[k], epochs=epochs, batch_size=batch_size, lr=lr
        )

    # Stage 3 — measure final task-0 accuracy.
    acc_final = clf.eval_accuracy(
        tasks[0]["x_test"], tasks[0]["y_test"]
    )
    retention = acc_final / max(acc_initial, RETENTION_EPS)

    excluded = bool(acc_initial < 0.5)
    return {
        "arm": arm,
        "seed": seed,
        "acc_task1_initial": float(acc_initial),
        "acc_task1_final": float(acc_final),
        "retention": float(retention),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
    }


# --------------------------------------------------------------------
# Verdict aggregation — Hedges' g + Welch + Jonckheere
# --------------------------------------------------------------------


def _retention_by_arm(cells: list[CellResult]) -> dict[str, list[float]]:
    """Group retention by arm, dropping excluded cells."""
    out: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for c in cells:
        if c["excluded_underperforming_baseline"]:
            continue
        out[c["arm"]].append(c["retention"])
    return out


def _h1_verdict(retention: dict[str, list[float]]) -> dict[str, Any]:
    """H1 — P_equ retention vs baseline retention vs Hu 2020 anchor."""
    p_equ = retention["P_equ"]
    base = retention["baseline"]
    if len(p_equ) < 2 or len(base) < 2:
        return {
            "insufficient_samples": True,
            "n_p_equ": len(p_equ),
            "n_base": len(base),
        }
    g = compute_hedges_g(p_equ, base)
    welch = welch_one_sided(base, p_equ, alpha=0.05 / 3)
    return {
        "hedges_g": g,
        "is_within_hu_2020_ci": HU_2020_OVERALL.is_within_ci(g),
        "distance_from_hu_2020": HU_2020_OVERALL.distance_from_target(g),
        "above_hu_2020_lower_ci": bool(g >= HU_2020_OVERALL.ci_low),
        "welch_p": welch.p_value,
        "welch_reject_h0": welch.reject_h0,
        "alpha_per_test": 0.05 / 3,
        "n_p_equ": len(p_equ),
        "n_base": len(base),
    }


def _h3_verdict(retention: dict[str, list[float]]) -> dict[str, Any]:
    """H3 — P_min retention vs baseline retention vs Javadi 2024 anchor."""
    p_min = retention["P_min"]
    base = retention["baseline"]
    if len(p_min) < 2 or len(base) < 2:
        return {
            "insufficient_samples": True,
            "n_p_min": len(p_min),
            "n_base": len(base),
        }
    g = compute_hedges_g(p_min, base)
    welch = welch_one_sided(p_min, base, alpha=0.05 / 3)
    return {
        "hedges_g": g,
        "is_within_javadi_2024_ci": JAVADI_2024_OVERALL.is_within_ci(
            abs(g)
        ),
        "distance_from_javadi_2024": JAVADI_2024_OVERALL.distance_from_target(
            abs(g)
        ),
        "below_javadi_2024_lower_ci_decrement": bool(
            g <= -JAVADI_2024_OVERALL.ci_low
        ),
        "welch_p": welch.p_value,
        "welch_reject_h0": welch.reject_h0,
        "alpha_per_test": 0.05 / 3,
        "n_p_min": len(p_min),
        "n_base": len(base),
    }


def _h_dr4_verdict(retention: dict[str, list[float]]) -> dict[str, Any]:
    """H_DR4 — Jonckheere monotonic trend [P_min, P_equ, P_max]."""
    groups = [retention["P_min"], retention["P_equ"], retention["P_max"]]
    if any(len(g) < 2 for g in groups):
        return {
            "insufficient_samples": True,
            "n_per_arm": [len(g) for g in groups],
        }
    res = jonckheere_trend(groups, alpha=0.05)
    mean_p_min = float(sum(groups[0]) / len(groups[0]))
    mean_p_equ = float(sum(groups[1]) / len(groups[1]))
    mean_p_max = float(sum(groups[2]) / len(groups[2]))
    return {
        "j_statistic": res.statistic,
        "p_value": res.p_value,
        "reject_h0": res.reject_h0,
        "mean_p_min": mean_p_min,
        "mean_p_equ": mean_p_equ,
        "mean_p_max": mean_p_max,
        "monotonic_observed": (
            mean_p_min <= mean_p_equ <= mean_p_max
        ),
    }


def _aggregate_verdict(cells: list[CellResult]) -> dict[str, Any]:
    retention = _retention_by_arm(cells)
    return {
        "h1_p_equ_vs_baseline": _h1_verdict(retention),
        "h3_p_min_vs_baseline": _h3_verdict(retention),
        "h_dr4_jonckheere": _h_dr4_verdict(retention),
        "retention_by_arm": retention,
    }


# --------------------------------------------------------------------
# Markdown report — append-only, dated immutable
# --------------------------------------------------------------------


def _render_md_report(payload: dict[str, Any]) -> str:
    """Render the human-readable milestone report."""
    h1 = payload["verdict"]["h1_p_equ_vs_baseline"]
    h3 = payload["verdict"]["h3_p_min_vs_baseline"]
    h4 = payload["verdict"]["h_dr4_jonckheere"]
    lines: list[str] = []
    lines.append("# G4 pilot — MLX × Split-FMNIST × profile sweep")
    lines.append("")
    lines.append(f"**Date** : {payload['date']}")
    lines.append(f"**c_version** : `{payload['c_version']}`")
    lines.append(f"**commit_sha** : `{payload['commit_sha']}`")
    lines.append(
        f"**Cells** : {len(payload['cells'])} "
        f"({len(ARMS)} arms × {payload['n_seeds']} seeds)"
    )
    lines.append(f"**Wall time** : {payload['wall_time_s']:.1f}s")
    lines.append("")
    lines.append("## Pre-registered hypotheses")
    lines.append("")
    lines.append("Pre-registration : `docs/osf-prereg-g4-pilot.md`")
    lines.append("")
    lines.append("### H1 — P_equ retention vs Hu 2020 (g >= 0.21)")
    if h1.get("insufficient_samples"):
        lines.append(
            f"INSUFFICIENT SAMPLES "
            f"(n_p_equ={h1['n_p_equ']}, n_base={h1['n_base']})"
        )
    else:
        lines.append(f"- observed Hedges' g : **{h1['hedges_g']:.4f}**")
        lines.append(
            f"- within Hu 2020 95% CI [0.21, 0.38] : "
            f"{h1['is_within_hu_2020_ci']}"
        )
        lines.append(
            f"- above Hu 2020 lower CI 0.21 : "
            f"{h1['above_hu_2020_lower_ci']}"
        )
        lines.append(
            f"- distance from Hu 2020 point estimate (g=0.29) : "
            f"{h1['distance_from_hu_2020']:+.4f}"
        )
        lines.append(
            f"- Welch one-sided p (α/3 = "
            f"{h1['alpha_per_test']:.4f}) : {h1['welch_p']:.4f} → "
            f"reject_h0 = {h1['welch_reject_h0']}"
        )
    lines.append("")
    lines.append(
        "### H3 — P_min retention vs Javadi 2024 (|g| >= 0.13, decrement)"
    )
    if h3.get("insufficient_samples"):
        lines.append(
            f"INSUFFICIENT SAMPLES "
            f"(n_p_min={h3['n_p_min']}, n_base={h3['n_base']})"
        )
    else:
        lines.append(f"- observed Hedges' g : **{h3['hedges_g']:.4f}**")
        lines.append(
            f"- |g| within Javadi 2024 95% CI [0.13, 0.44] : "
            f"{h3['is_within_javadi_2024_ci']}"
        )
        lines.append(
            f"- below -Javadi lower CI -0.13 (decrement) : "
            f"{h3['below_javadi_2024_lower_ci_decrement']}"
        )
        lines.append(
            f"- distance from Javadi 2024 point estimate (g=0.29) : "
            f"{h3['distance_from_javadi_2024']:+.4f}"
        )
        lines.append(
            f"- Welch one-sided p (α/3 = "
            f"{h3['alpha_per_test']:.4f}) : {h3['welch_p']:.4f} → "
            f"reject_h0 = {h3['welch_reject_h0']}"
        )
    lines.append("")
    lines.append(
        "### H_DR4 — Jonckheere monotonic trend [P_min, P_equ, P_max]"
    )
    if h4.get("insufficient_samples"):
        lines.append(
            f"INSUFFICIENT SAMPLES (n_per_arm={h4['n_per_arm']})"
        )
    else:
        lines.append(
            f"- mean retention P_min : {h4['mean_p_min']:.4f}"
        )
        lines.append(
            f"- mean retention P_equ : {h4['mean_p_equ']:.4f}"
        )
        lines.append(
            f"- mean retention P_max : {h4['mean_p_max']:.4f}"
        )
        lines.append(
            f"- monotonic observed P_max >= P_equ >= P_min : "
            f"{h4['monotonic_observed']}"
        )
        lines.append(
            f"- Jonckheere J statistic : {h4['j_statistic']:.4f}"
        )
        lines.append(
            f"- one-sided p (α = 0.05) : {h4['p_value']:.4f} → "
            f"reject_h0 = {h4['reject_h0']}"
        )
    lines.append("")
    lines.append("## Cells (R1 traceability)")
    lines.append("")
    lines.append(
        "| arm | seed | acc_initial | acc_final | retention | "
        "excluded | run_id |"
    )
    lines.append(
        "|-----|------|-------------|-----------|-----------|"
        "----------|--------|"
    )
    for c in payload["cells"]:
        lines.append(
            f"| {c['arm']} | {c['seed']} | "
            f"{c['acc_task1_initial']:.4f} | "
            f"{c['acc_task1_final']:.4f} | "
            f"{c['retention']:.4f} | "
            f"{c['excluded_underperforming_baseline']} | "
            f"`{c['run_id']}` |"
        )
    lines.append("")
    lines.append("## Provenance")
    lines.append("")
    lines.append(
        "- Pre-registration : "
        "[docs/osf-prereg-g4-pilot.md](../osf-prereg-g4-pilot.md)"
    )
    lines.append(
        "- Driver : `experiments/g4_split_fmnist/run_g4.py`"
    )
    lines.append(
        "- Effect-size helper : "
        "`kiki_oniric.eval.statistics.compute_hedges_g`"
    )
    lines.append(
        "- Verdict anchors : "
        "`harness.benchmarks.effect_size_targets."
        "{HU_2020_OVERALL, JAVADI_2024_OVERALL}`"
    )
    lines.append(
        "- Run registry : `harness/storage/run_registry.RunRegistry` "
        "(db `.run_registry.sqlite`)"
    )
    lines.append("")
    return "\n".join(lines)


# --------------------------------------------------------------------
# Public entrypoint — function form for tests, CLI form for shell use
# --------------------------------------------------------------------


def run_pilot(
    *,
    data_dir: Path,
    seeds: tuple[int, ...],
    out_json: Path,
    out_md: Path,
    registry_db: Path,
    epochs: int,
    batch_size: int,
    hidden_dim: int,
    lr: float,
) -> dict[str, Any]:
    """Execute the pilot sweep and return the verdict payload.

    Tests invoke this directly with synthetic fixtures ; the CLI
    in :func:`main` calls it with production paths.
    """
    tasks = load_split_fmnist_5tasks(data_dir)
    if len(tasks) != 5:
        raise RuntimeError(
            f"Split-FMNIST loader returned {len(tasks)} tasks (expected 5)"
        )

    registry = RunRegistry(registry_db)
    commit_sha = _resolve_commit_sha()

    cells: list[CellResult] = []
    sweep_start = time.time()
    for arm in ARMS:
        for seed in seeds:
            cell = _run_cell(
                arm,
                seed,
                tasks,
                epochs=epochs,
                batch_size=batch_size,
                hidden_dim=hidden_dim,
                lr=lr,
            )
            run_id = registry.register(
                c_version=C_VERSION,
                profile=f"g4/{arm}",
                seed=seed,
                commit_sha=commit_sha,
            )
            cells.append(CellResult(**cell, run_id=run_id))
    wall = time.time() - sweep_start

    verdict = _aggregate_verdict(cells)
    payload = {
        "date": "2026-05-03",
        "c_version": C_VERSION,
        "commit_sha": commit_sha,
        "n_seeds": len(seeds),
        "arms": list(ARMS),
        "data_dir": str(data_dir),
        "wall_time_s": wall,
        "cells": cells,
        "verdict": verdict,
    }

    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    out_md.write_text(_render_md_report(payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G4 pilot driver")
    parser.add_argument(
        "--smoke",
        action="store_true",
        help="2-seed run on whichever data dir is given",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing FMNIST IDX gzipped files",
    )
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--hidden-dim", type=int, default=128)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument(
        "--out-json", type=Path, default=DEFAULT_OUT_JSON
    )
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--registry-db", type=Path, default=DEFAULT_REGISTRY_DB
    )
    args = parser.parse_args(argv)

    seeds = (0, 1) if args.smoke else DEFAULT_SEEDS
    payload = run_pilot(
        data_dir=args.data_dir,
        seeds=seeds,
        out_json=args.out_json,
        out_md=args.out_md,
        registry_db=args.registry_db,
        epochs=args.epochs,
        batch_size=args.batch_size,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Cells : {len(payload['cells'])}")
    print(
        f"Verdict H1.hedges_g : "
        f"{payload['verdict']['h1_p_equ_vs_baseline'].get('hedges_g')}"
    )
    print(
        f"Verdict H3.hedges_g : "
        f"{payload['verdict']['h3_p_min_vs_baseline'].get('hedges_g')}"
    )
    print(
        f"Verdict H_DR4.monotonic : "
        f"{payload['verdict']['h_dr4_jonckheere'].get('monotonic_observed')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
