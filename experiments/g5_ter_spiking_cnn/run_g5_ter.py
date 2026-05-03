"""G5-ter pilot driver - spiking-CNN cross-substrate H8 test.

Sweeps 4 arms x 10 seeds (Option B) of the
``EsnnG5TerSpikingCNN`` on Split-CIFAR-10 5-task with HP combo C5
anchor. Per-cell pipeline transposes
``experiments.g5_bis_richer_esnn.run_g5_bis._run_cell`` onto a
spiking-CNN substrate with the four-op coupling (REPLAY +
DOWNSCALE + RESTRUCTURE + RECOMBINE).

Outputs :
    docs/milestones/g5-ter-spiking-cnn-2026-05-03.{json,md}

The own-substrate ``h8a_spiking_cnn`` verdict (P_equ vs baseline)
is computed by this driver. The full cross-substrate H8-A/B/C
classification is computed by
``experiments.g5_ter_spiking_cnn.aggregator.write_aggregate_dump``.

Pre-reg : docs/osf-prereg-g5-ter-spiking-cnn.md
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

import numpy as np  # noqa: E402

from experiments.g4_quinto_test.cifar10_dataset import (  # noqa: E402
    SplitCIFAR10Task,
    load_split_cifar10_5tasks_auto,
)
from experiments.g4_ter_hp_sweep.hp_grid import (  # noqa: E402
    representative_combo,
)
from experiments.g5_ter_spiking_cnn.dream_wrap_cnn import (  # noqa: E402
    EsnnCNNBetaBuffer,
    build_esnn_cnn_profile,
    dream_episode_cnn_esnn,
)
from experiments.g5_ter_spiking_cnn.spiking_cnn import (  # noqa: E402
    EsnnG5TerSpikingCNN,
)
from harness.benchmarks.effect_size_targets import (  # noqa: E402
    HU_2020_OVERALL,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.eval.statistics import (  # noqa: E402
    compute_hedges_g,
    welch_one_sided,
)


C_VERSION = "C-v0.12.0+PARTIAL"
SUBSTRATE_NAME = "esnn_thalamocortical_spiking_cnn"
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(10))  # Option B
DEFAULT_DATA_DIR = (
    REPO_ROOT / "experiments" / "g4_quinto_test" / "data"
)
DEFAULT_OUT_JSON = (
    REPO_ROOT
    / "docs"
    / "milestones"
    / "g5-ter-spiking-cnn-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT
    / "docs"
    / "milestones"
    / "g5-ter-spiking-cnn-2026-05-03.md"
)
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
RETENTION_EPS = 1e-6
BETA_BUFFER_CAPACITY = 256
BETA_BUFFER_FILL_PER_TASK = 32
RESTRUCTURE_FACTOR = 0.05
RECOMBINE_N_SYNTHETIC = 16
RECOMBINE_LR = 0.01


class _CellPartial(TypedDict):
    arm: str
    seed: int
    hp_combo_id: str
    acc_task1_initial: float
    acc_task1_final: float
    retention: float
    excluded_underperforming_baseline: bool
    wall_time_s: float


class CellResult(_CellPartial):
    run_id: str


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


def _subsample_task(
    task: SplitCIFAR10Task, n_train_per_task: int, seed: int
) -> dict[str, Any]:
    """Return a subsampled view of ``task`` keeping NHWC layout intact.

    Deterministic per ``(seed, n_train_per_task)``. Test set is left
    intact so the retention denominator is comparable across cells.
    """
    n = task["x_train_nhwc"].shape[0]
    if n_train_per_task <= 0 or n_train_per_task >= n:
        return dict(task)
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=n_train_per_task, replace=False)
    idx.sort()
    return {
        "x_train": task["x_train"][idx],
        "x_train_nhwc": task["x_train_nhwc"][idx],
        "y_train": task["y_train"][idx],
        "x_test": task["x_test"],
        "x_test_nhwc": task["x_test_nhwc"],
        "y_test": task["y_test"],
    }


def _run_cell(
    arm: str,
    seed: int,
    tasks: list[SplitCIFAR10Task],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
    n_steps: int,
    n_train_per_task: int = 0,
) -> _CellPartial:
    start = time.time()
    combo = representative_combo()
    if n_train_per_task > 0:
        tasks = [
            _subsample_task(t, n_train_per_task, seed + 100 * idx)
            for idx, t in enumerate(tasks)
        ]
    clf = EsnnG5TerSpikingCNN(
        n_classes=2,
        seed=seed,
        n_steps=n_steps,
    )
    buffer = EsnnCNNBetaBuffer(capacity=BETA_BUFFER_CAPACITY)
    fill_rng = np.random.default_rng(seed + 5_000)

    def _push_task(task: dict[str, Any]) -> None:
        n = task["x_train_nhwc"].shape[0]
        n_take = min(BETA_BUFFER_FILL_PER_TASK, n)
        idx = fill_rng.choice(n, size=n_take, replace=False)
        # Vectorised latent computation for the buffer fill
        xs = task["x_train_nhwc"][idx].astype(np.float32)
        latents = clf.latent(xs)
        for k, i in enumerate(idx.tolist()):
            buffer.push(
                x=xs[k],
                y=int(task["y_train"][i]),
                latent=latents[k],
            )

    clf.train_task(
        {
            "x_train": tasks[0]["x_train_nhwc"],
            "y_train": tasks[0]["y_train"],
        },
        epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    acc_initial = clf.eval_accuracy(
        tasks[0]["x_test_nhwc"], tasks[0]["y_test"]
    )
    _push_task(tasks[0])

    profile = None
    if arm != "baseline":
        profile = build_esnn_cnn_profile(arm, seed=seed)

    for k in range(1, len(tasks)):
        if profile is not None:
            dream_episode_cnn_esnn(
                clf,
                profile,
                seed=seed + k,
                beta_buffer=buffer,
                replay_n_records=combo.replay_batch,
                replay_n_steps=combo.replay_n_steps,
                replay_lr=combo.replay_lr,
                downscale_factor=combo.downscale_factor,
                restructure_factor=RESTRUCTURE_FACTOR,
                recombine_n_synthetic=RECOMBINE_N_SYNTHETIC,
                recombine_lr=RECOMBINE_LR,
            )
        clf.train_task(
            {
                "x_train": tasks[k]["x_train_nhwc"],
                "y_train": tasks[k]["y_train"],
            },
            epochs=epochs,
            batch_size=batch_size,
            lr=lr,
        )
        _push_task(tasks[k])

    acc_final = clf.eval_accuracy(
        tasks[0]["x_test_nhwc"], tasks[0]["y_test"]
    )
    retention = acc_final / max(acc_initial, RETENTION_EPS)
    excluded = bool(acc_initial < 0.5)
    return {
        "arm": arm,
        "seed": seed,
        "hp_combo_id": combo.combo_id,
        "acc_task1_initial": float(acc_initial),
        "acc_task1_final": float(acc_final),
        "retention": float(retention),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
    }


def _retention_by_arm(
    cells: list[dict[str, Any]],
) -> dict[str, list[float]]:
    arms = sorted({c["arm"] for c in cells})
    out: dict[str, list[float]] = {a: [] for a in arms}
    for c in cells:
        if c["excluded_underperforming_baseline"]:
            continue
        out[c["arm"]].append(c["retention"])
    return out


def _h8a_verdict(
    retention: dict[str, list[float]],
) -> dict[str, Any]:
    """H8-A own-substrate verdict (P_equ vs baseline on E-SNN CNN)."""
    p_equ = retention.get("P_equ", [])
    base = retention.get("baseline", [])
    if len(p_equ) < 2 or len(base) < 2:
        return {
            "insufficient_samples": True,
            "n_p_equ": len(p_equ),
            "n_base": len(base),
        }
    g = compute_hedges_g(p_equ, base)
    welch = welch_one_sided(base, p_equ, alpha=0.05 / 4)
    return {
        "hedges_g": g,
        "above_zero": bool(g >= 0.0),
        "above_hu_2020_lower_ci": bool(g >= HU_2020_OVERALL.ci_low),
        "welch_p": welch.p_value,
        "welch_reject_h0": welch.reject_h0,
        "alpha_per_test": 0.05 / 4,
        "n_p_equ": len(p_equ),
        "n_base": len(base),
    }


def _aggregate_verdict(
    cells: list[dict[str, Any]],
) -> dict[str, Any]:
    retention = _retention_by_arm(cells)
    return {
        "h8a_spiking_cnn": _h8a_verdict(retention),
        "retention_by_arm": retention,
    }


def _render_md_report(payload: dict[str, Any]) -> str:
    h8 = payload["verdict"]["h8a_spiking_cnn"]
    lines: list[str] = [
        "# G5-ter pilot - spiking-CNN ported to E-SNN substrate",
        "",
        f"**Date** : {payload['date']}",
        f"**c_version** : `{payload['c_version']}`",
        f"**commit_sha** : `{payload['commit_sha']}`",
        f"**Substrate** : {payload['substrate']}",
        (
            f"**Cells** : {len(payload['cells'])} "
            f"({len(ARMS)} arms x {payload['n_seeds']} seeds x 1 HP)"
        ),
        f"**Wall time** : {payload['wall_time_s']:.1f}s",
        "",
        "## Pre-registered hypothesis (own-substrate H8-A)",
        "",
        "Pre-registration : `docs/osf-prereg-g5-ter-spiking-cnn.md`",
        "",
        "### H8-A - E-SNN spiking-CNN (P_equ vs baseline retention)",
    ]
    if h8.get("insufficient_samples"):
        lines.append(
            f"INSUFFICIENT SAMPLES (n_p_equ={h8.get('n_p_equ')}, "
            f"n_base={h8.get('n_base')})"
        )
    else:
        lines += [
            f"- observed Hedges' g_h8 : **{h8['hedges_g']:.4f}**",
            f"- above zero : {h8['above_zero']}",
            (
                "- above Hu 2020 lower CI 0.21 : "
                f"{h8['above_hu_2020_lower_ci']}"
            ),
            (
                f"- Welch one-sided p (alpha/4 = "
                f"{h8['alpha_per_test']:.4f}) : {h8['welch_p']:.4f} -> "
                f"reject_h0 = {h8['welch_reject_h0']}"
            ),
        ]
    lines += [
        "",
        "## Cells (per arm x seed)",
        "",
        "| arm | seed | hp | acc_task1_initial | acc_task1_final | "
        "retention | excluded |",
        "|-----|------|----|--------------------|------------------|"
        "-----------|----------|",
    ]
    for cell in payload["cells"]:
        lines.append(
            f"| {cell['arm']} | {cell['seed']} | {cell['hp_combo_id']} | "
            f"{cell['acc_task1_initial']:.4f} | "
            f"{cell['acc_task1_final']:.4f} | "
            f"{cell['retention']:.4f} | "
            f"{cell['excluded_underperforming_baseline']} |"
        )
    lines += [
        "",
        "## Provenance",
        "",
        "- Pre-registration : "
        "[docs/osf-prereg-g5-ter-spiking-cnn.md]"
        "(../osf-prereg-g5-ter-spiking-cnn.md)",
        "- Sister G5-bis aggregate : "
        "[g5-bis-aggregate-2026-05-03.md]"
        "(g5-bis-aggregate-2026-05-03.md)",
        "- Sister G4-quinto Step 2 milestone : "
        "[g4-quinto-step2-2026-05-03.md]"
        "(g4-quinto-step2-2026-05-03.md)",
        "- Cross-substrate aggregate : "
        "[g5-ter-aggregate-2026-05-03.md]"
        "(g5-ter-aggregate-2026-05-03.md)",
        "- Driver : `experiments/g5_ter_spiking_cnn/run_g5_ter.py`",
        "- Substrate : `experiments.g5_ter_spiking_cnn."
        "spiking_cnn.EsnnG5TerSpikingCNN`",
        "- HP combo : C5 (`representative_combo()`)",
        "- Run registry : `harness/storage/run_registry.RunRegistry` "
        "(db `.run_registry.sqlite`)",
        "",
    ]
    return "\n".join(lines)


def run_pilot(
    *,
    data_dir: Path,
    seeds: tuple[int, ...],
    out_json: Path,
    out_md: Path,
    registry_db: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    n_steps: int,
    n_train_per_task: int = 0,
) -> dict[str, Any]:
    tasks = load_split_cifar10_5tasks_auto(data_dir)
    if len(tasks) != 5:
        raise RuntimeError(
            f"Split-CIFAR-10 loader returned {len(tasks)} tasks "
            "(expected 5)"
        )
    registry = RunRegistry(registry_db)
    commit_sha = _resolve_commit_sha()

    cells: list[dict[str, Any]] = []
    sweep_start = time.time()
    for arm in ARMS:
        for seed in seeds:
            partial = _run_cell(
                arm,
                seed,
                tasks,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                n_steps=n_steps,
                n_train_per_task=n_train_per_task,
            )
            run_id = registry.register(
                c_version=C_VERSION,
                profile=f"g5-ter/spiking_cnn/{arm}",
                seed=seed,
                commit_sha=commit_sha,
            )
            cell: dict[str, Any] = dict(partial)
            cell["run_id"] = run_id
            cells.append(cell)

    wall = time.time() - sweep_start
    verdict = _aggregate_verdict(cells)

    payload = {
        "date": "2026-05-03",
        "c_version": C_VERSION,
        "commit_sha": commit_sha,
        "substrate": SUBSTRATE_NAME,
        "n_seeds": len(seeds),
        "arms": list(ARMS),
        "data_dir": str(data_dir),
        "wall_time_s": wall,
        "n_train_per_task": int(n_train_per_task),
        "cells": cells,
        "verdict": verdict,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    out_md.write_text(_render_md_report(payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G5-ter pilot driver")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument(
        "--data-dir", type=Path, default=DEFAULT_DATA_DIR
    )
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--n-steps", type=int, default=20)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument(
        "--out-json", type=Path, default=DEFAULT_OUT_JSON
    )
    parser.add_argument(
        "--out-md", type=Path, default=DEFAULT_OUT_MD
    )
    parser.add_argument(
        "--registry-db", type=Path, default=DEFAULT_REGISTRY_DB
    )
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS)
    )
    parser.add_argument(
        "--n-train-per-task",
        type=int,
        default=0,
        help=(
            "Subsample CIFAR-10 train shard to this many examples "
            "per task ; 0 (default) keeps the full shard. Documented "
            "amendment under docs/osf-deviations-g5-ter-2026-05-03.md "
            "when used."
        ),
    )
    args = parser.parse_args(argv)

    seeds = (0, 1) if args.smoke else tuple(args.seeds)
    payload = run_pilot(
        data_dir=args.data_dir,
        seeds=seeds,
        out_json=args.out_json,
        out_md=args.out_md,
        registry_db=args.registry_db,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        n_steps=args.n_steps,
        n_train_per_task=args.n_train_per_task,
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Cells : {len(payload['cells'])}")
    print(
        f"H8-A.hedges_g : "
        f"{payload['verdict']['h8a_spiking_cnn'].get('hedges_g')}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
