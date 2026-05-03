"""G4-quater Step 3 driver — H4-C test (RECOMBINE strategy + placebo).

Reuses the existing G4-ter 3-layer ``G4HierarchicalClassifier`` and
threads a ``RecombineStrategy`` switch ({mog, ae, none}) through a
local ``_dream_episode_strategy`` mirror of ``dream_episode_hier``.

3 strategies x 4 arms x N seeds = 1140 cells (N=95 default).

H4-C : Welch two-sided between (P_max with mog) and (P_max with
none), alpha = 0.05 / 3 = 0.0167. **Failing to reject** confirms
RECOMBINE is empirically empty at this scale (positive empirical
claim, mog ≈ none).

Outputs :
    docs/milestones/g4-quater-step3-2026-05-03.{json,md}

Pre-reg : docs/osf-prereg-g4-quater-pilot.md sec 2 (H4-C).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import mlx.nn as nn  # noqa: E402
import mlx.optimizers as optim  # noqa: E402
import numpy as np  # noqa: E402

from experiments.g4_quater_test.recombine_strategies import (  # noqa: E402
    RecombineStrategy,
    sample_synthetic_latents,
)
from experiments.g4_split_fmnist.dataset import (  # noqa: E402
    SplitFMNISTTask,
    load_split_fmnist_5tasks,
)
from experiments.g4_split_fmnist.dream_wrap import (  # noqa: E402
    build_profile,
)
from experiments.g4_ter_hp_sweep.dream_wrap_hier import (  # noqa: E402
    BetaBufferHierFIFO,
    G4HierarchicalClassifier,
)
from experiments.g4_ter_hp_sweep.hp_grid import (  # noqa: E402
    HPCombo,
    representative_combo,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.eval.statistics import (  # noqa: E402
    compute_hedges_g,
)

C_VERSION = "C-v0.12.0+PARTIAL"
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
STRATEGIES: tuple[RecombineStrategy, ...] = ("mog", "ae", "none")
DEFAULT_N_SEEDS = 95
DEFAULT_DATA_DIR = REPO_ROOT / "experiments" / "g4_split_fmnist" / "data"
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "milestones" / "g4-quater-step3-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "docs" / "milestones" / "g4-quater-step3-2026-05-03.md"
)
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
RETENTION_EPS = 1e-6
BETA_BUFFER_CAPACITY = 256
BETA_BUFFER_FILL_PER_TASK = 32
HIDDEN_1 = 32
HIDDEN_2 = 16
RECOMBINE_N_SYNTHETIC = 16
RECOMBINE_LR = 0.01
RESTRUCTURE_FACTOR = 0.05


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


def _strategy_aware_recombine(
    clf: G4HierarchicalClassifier,
    *,
    strategy: RecombineStrategy,
    beta_buffer: BetaBufferHierFIFO,
    n_synthetic: int,
    lr: float,
    seed: int,
) -> None:
    """RECOMBINE step dispatched by ``strategy``.

    For ``mog`` and ``ae`` strategies, synthetic latents are produced
    by ``sample_synthetic_latents`` then projected through ``_l3``
    only. For ``none``, this function is a no-op (placebo control).
    """
    snapshot = beta_buffer.snapshot()
    populated = [
        (np.asarray(r["latent"], dtype=np.float32), int(r["y"]))
        for r in snapshot
        if r["latent"] is not None
    ]
    if not populated:
        return
    latents = np.stack([p[0] for p in populated])
    labels = np.asarray([p[1] for p in populated], dtype=np.int64)

    batch = sample_synthetic_latents(
        strategy=strategy,
        latents=latents,
        labels=labels,
        n_synthetic=n_synthetic,
        seed=seed,
    )
    if batch is None:
        return

    x = mx.array(batch["x"])
    y = mx.array(batch["y"].astype(np.int32))

    opt = optim.SGD(learning_rate=lr)

    def loss_fn(layer: nn.Linear, xb: mx.array, yb: mx.array) -> mx.array:
        return nn.losses.cross_entropy(layer(xb), yb, reduction="mean")

    loss_and_grad = nn.value_and_grad(clf._l3, loss_fn)
    _loss, grads = loss_and_grad(clf._l3, x, y)
    opt.update(clf._l3, grads)
    mx.eval(clf._l3.parameters(), opt.state)


def _dream_episode_strategy(
    clf: G4HierarchicalClassifier,
    profile: object,
    seed: int,
    *,
    beta_buffer: BetaBufferHierFIFO,
    combo: HPCombo,
    restructure_factor: float,
    recombine_strategy: RecombineStrategy,
    recombine_n_synthetic: int,
    recombine_lr: float,
) -> None:
    """Mirror of ``G4HierarchicalClassifier.dream_episode_hier`` but
    swaps the RECOMBINE step with ``_strategy_aware_recombine``.

    DR-0 spectator runtime path is preserved.
    """
    from kiki_oniric.dream.episode import (
        BudgetCap,
        DreamEpisode,
        EpisodeTrigger,
        Operation,
        OutputChannel,
    )
    from kiki_oniric.profiles.p_min import PMinProfile

    if isinstance(profile, PMinProfile):
        ops: tuple[Operation, ...] = (
            Operation.REPLAY,
            Operation.DOWNSCALE,
        )
        channels: tuple[OutputChannel, ...] = (
            OutputChannel.WEIGHT_DELTA,
        )
    else:
        ops = (
            Operation.REPLAY,
            Operation.DOWNSCALE,
            Operation.RESTRUCTURE,
            Operation.RECOMBINE,
        )
        channels = (
            OutputChannel.WEIGHT_DELTA,
            OutputChannel.HIERARCHY_CHG,
            OutputChannel.LATENT_SAMPLE,
        )

    rng = np.random.default_rng(seed + 10_000)
    synthetic_records = [
        {
            "x": rng.standard_normal(4).astype(np.float32).tolist(),
            "y": rng.standard_normal(4).astype(np.float32).tolist(),
        }
        for _ in range(4)
    ]
    delta_latents = [
        rng.standard_normal(4).astype(np.float32).tolist() for _ in range(2)
    ]
    episode = DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={
            "beta_records": synthetic_records,
            "shrink_factor": 0.99,
            "topo_op": "reroute",
            "swap_indices": [0, 1],
            "delta_latents": delta_latents,
        },
        operation_set=ops,
        output_channels=channels,
        budget=BudgetCap(
            flops=10_000_000, wall_time_s=10.0, energy_j=1.0
        ),
        episode_id=(
            f"g4-quater-step3-{type(profile).__name__}-"
            f"{recombine_strategy}-seed{seed}"
        ),
    )
    profile.runtime.execute(episode)  # type: ignore[attr-defined]

    if Operation.REPLAY in ops:
        sampled = beta_buffer.sample(n=combo.replay_batch, seed=seed)
        clf._replay_optimizer_step(
            sampled, lr=combo.replay_lr, n_steps=combo.replay_n_steps
        )
    if Operation.DOWNSCALE in ops:
        clf._downscale_step(factor=combo.downscale_factor)
    if Operation.RESTRUCTURE in ops:
        clf._restructure_step(
            factor=restructure_factor, seed=seed + 20_000
        )
    if Operation.RECOMBINE in ops:
        _strategy_aware_recombine(
            clf,
            strategy=recombine_strategy,
            beta_buffer=beta_buffer,
            n_synthetic=recombine_n_synthetic,
            lr=recombine_lr,
            seed=seed + 30_000,
        )


def _run_cell(
    arm: str,
    seed: int,
    combo: HPCombo,
    strategy: RecombineStrategy,
    tasks: list[SplitFMNISTTask],
    *,
    epochs: int,
    batch_size: int,
    lr: float,
) -> dict[str, Any]:
    start = time.time()
    feat_dim = tasks[0]["x_train"].shape[1]
    clf = G4HierarchicalClassifier(
        in_dim=feat_dim,
        hidden_1=HIDDEN_1,
        hidden_2=HIDDEN_2,
        n_classes=2,
        seed=seed,
    )
    buffer = BetaBufferHierFIFO(capacity=BETA_BUFFER_CAPACITY)
    fill_rng = np.random.default_rng(seed + 5_000)

    def _push_task(task: SplitFMNISTTask) -> None:
        n = task["x_train"].shape[0]
        n_take = min(BETA_BUFFER_FILL_PER_TASK, n)
        idx = fill_rng.choice(n, size=n_take, replace=False)
        for i in idx.tolist():
            x = task["x_train"][i]
            latent = clf.latent(x[None, :])[0]
            buffer.push(x=x, y=int(task["y_train"][i]), latent=latent)

    clf.train_task(tasks[0], epochs=epochs, batch_size=batch_size, lr=lr)
    acc_initial = clf.eval_accuracy(tasks[0]["x_test"], tasks[0]["y_test"])
    _push_task(tasks[0])

    profile = None
    if arm != "baseline":
        profile = build_profile(arm, seed=seed)

    for k in range(1, len(tasks)):
        if profile is not None:
            _dream_episode_strategy(
                clf,
                profile,
                seed=seed + k,
                beta_buffer=buffer,
                combo=combo,
                restructure_factor=RESTRUCTURE_FACTOR,
                recombine_strategy=strategy,
                recombine_n_synthetic=RECOMBINE_N_SYNTHETIC,
                recombine_lr=RECOMBINE_LR,
            )
        clf.train_task(
            tasks[k], epochs=epochs, batch_size=batch_size, lr=lr
        )
        _push_task(tasks[k])

    acc_final = clf.eval_accuracy(tasks[0]["x_test"], tasks[0]["y_test"])
    retention = acc_final / max(acc_initial, RETENTION_EPS)
    excluded = bool(acc_initial < 0.5)
    return {
        "arm": arm,
        "seed": seed,
        "hp_combo_id": combo.combo_id,
        "recombine_strategy": strategy,
        "acc_task1_initial": float(acc_initial),
        "acc_task1_final": float(acc_final),
        "retention": float(retention),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
    }


def _retention_by_arm_strategy(
    cells: list[dict[str, Any]], strategy: RecombineStrategy
) -> dict[str, list[float]]:
    arms = sorted({c["arm"] for c in cells})
    out: dict[str, list[float]] = {a: [] for a in arms}
    for c in cells:
        if c["excluded_underperforming_baseline"]:
            continue
        if c["recombine_strategy"] != strategy:
            continue
        out[c["arm"]].append(c["retention"])
    return out


def _h4c_verdict(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """H4-C — Welch two-sided P_max(mog) vs P_max(none).

    Failing to reject H0 at alpha = 0.05 / 3 = 0.0167 confirms H4-C
    (RECOMBINE empirically empty at this scale).
    """
    from scipy import stats as scistats

    ret_mog = _retention_by_arm_strategy(cells, "mog")
    ret_none = _retention_by_arm_strategy(cells, "none")
    p_max_mog = ret_mog.get("P_max", [])
    p_max_none = ret_none.get("P_max", [])
    if len(p_max_mog) < 2 or len(p_max_none) < 2:
        return {
            "insufficient_samples": True,
            "n_p_max_mog": len(p_max_mog),
            "n_p_max_none": len(p_max_none),
        }
    alpha = 0.05 / 3
    welch = scistats.ttest_ind(p_max_mog, p_max_none, equal_var=False)
    fail_to_reject = bool(welch.pvalue > alpha)
    g = compute_hedges_g(p_max_mog, p_max_none)
    return {
        "n_p_max_mog": len(p_max_mog),
        "n_p_max_none": len(p_max_none),
        "mean_p_max_mog": float(np.mean(p_max_mog)),
        "mean_p_max_none": float(np.mean(p_max_none)),
        "welch_t": float(welch.statistic),
        "welch_p_two_sided": float(welch.pvalue),
        "alpha_per_test": alpha,
        "fail_to_reject_h0": fail_to_reject,
        "h4c_recombine_empty_confirmed": fail_to_reject,
        "hedges_g_mog_vs_none": g,
    }


def _h4c_ae_observation(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Secondary observation : P_max(ae) vs P_max(none) Welch.

    Reported for completeness but not part of the pre-registered H4-C
    verdict (H4-C is mog vs none).
    """
    from scipy import stats as scistats

    ret_ae = _retention_by_arm_strategy(cells, "ae")
    ret_none = _retention_by_arm_strategy(cells, "none")
    p_max_ae = ret_ae.get("P_max", [])
    p_max_none = ret_none.get("P_max", [])
    if len(p_max_ae) < 2 or len(p_max_none) < 2:
        return {"insufficient_samples": True}
    welch = scistats.ttest_ind(p_max_ae, p_max_none, equal_var=False)
    return {
        "n_p_max_ae": len(p_max_ae),
        "n_p_max_none": len(p_max_none),
        "mean_p_max_ae": float(np.mean(p_max_ae)),
        "mean_p_max_none": float(np.mean(p_max_none)),
        "welch_t": float(welch.statistic),
        "welch_p_two_sided": float(welch.pvalue),
    }


def _render_md_report(payload: dict[str, Any]) -> str:
    h = payload["verdict"]["h4c_recombine_strategy"]
    ae_obs = payload["verdict"]["ae_observation"]
    lines: list[str] = [
        "# G4-quater Step 3 — H4-C RECOMBINE strategy + placebo",
        "",
        f"**Date** : {payload['date']}",
        f"**c_version** : `{payload['c_version']}`",
        f"**commit_sha** : `{payload['commit_sha']}`",
        (
            f"**Cells** : {len(payload['cells'])} "
            f"({len(STRATEGIES)} strategies x {len(ARMS)} arms x "
            f"{payload['n_seeds']} seeds)"
        ),
        f"**Wall time** : {payload['wall_time_s']:.1f}s",
        "",
        "## Pre-registered hypothesis",
        "",
        "Pre-registration : `docs/osf-prereg-g4-quater-pilot.md`",
        "",
        "### H4-C — RECOMBINE empirical-emptiness",
        "",
        (
            "Welch two-sided test of `retention(P_max with mog)` vs "
            "`retention(P_max with none)`. **Failing** to reject H0 "
            "at alpha = 0.05 / 3 = 0.0167 confirms H4-C : RECOMBINE "
            "is empirically empty at this scale (mog ≈ none)."
        ),
        "",
    ]
    if h.get("insufficient_samples"):
        lines.append(
            f"INSUFFICIENT SAMPLES "
            f"(mog={h.get('n_p_max_mog')}, "
            f"none={h.get('n_p_max_none')})"
        )
    else:
        lines += [
            f"- mean retention P_max (mog) : {h['mean_p_max_mog']:.4f} "
            f"(N={h['n_p_max_mog']})",
            f"- mean retention P_max (none) : {h['mean_p_max_none']:.4f} "
            f"(N={h['n_p_max_none']})",
            f"- Hedges' g (mog vs none) : {h['hedges_g_mog_vs_none']:.4f}",
            f"- Welch t : {h['welch_t']:.4f}",
            (
                f"- Welch p (two-sided, alpha = "
                f"{h['alpha_per_test']:.4f}) : "
                f"{h['welch_p_two_sided']:.4f} -> "
                f"fail_to_reject_h0 = {h['fail_to_reject_h0']}"
            ),
            "",
            (
                f"**H4-C verdict** : RECOMBINE empty confirmed "
                f"= {h['h4c_recombine_empty_confirmed']} "
                "(positive empirical claim mog ≈ none if True)."
            ),
            "",
            (
                "*Honest reading* : Welch fail-to-reject = no "
                "evidence at this N for a difference between mog "
                "and none ; this is **the predicted outcome under "
                "H4-C**, framed as absence of evidence supporting "
                "RECOMBINE adding any signal beyond the "
                "REPLAY+DOWNSCALE coupling already provided by "
                "P_min."
            ),
        ]
    lines += [
        "",
        "### Secondary observation — AE strategy vs none placebo",
        "",
    ]
    if ae_obs.get("insufficient_samples"):
        lines.append("INSUFFICIENT SAMPLES")
    else:
        lines += [
            f"- mean retention P_max (ae) : {ae_obs['mean_p_max_ae']:.4f} "
            f"(N={ae_obs['n_p_max_ae']})",
            f"- mean retention P_max (none) : {ae_obs['mean_p_max_none']:.4f} "
            f"(N={ae_obs['n_p_max_none']})",
            f"- Welch t : {ae_obs['welch_t']:.4f}",
            f"- Welch p (two-sided) : {ae_obs['welch_p_two_sided']:.4f}",
        ]
    lines += [
        "",
        "## Provenance",
        "",
        (
            "- Pre-registration : "
            "[docs/osf-prereg-g4-quater-pilot.md]"
            "(../osf-prereg-g4-quater-pilot.md)"
        ),
        (
            "- Driver : `experiments/g4_quater_test/"
            "run_step3_recombine_strategies.py`"
        ),
        (
            "- Substrate : `experiments.g4_ter_hp_sweep."
            "dream_wrap_hier.G4HierarchicalClassifier`"
        ),
        (
            "- Strategies : `experiments.g4_quater_test."
            "recombine_strategies.sample_synthetic_latents`"
        ),
        (
            "- Run registry : `harness/storage/run_registry.RunRegistry`"
            " (db `.run_registry.sqlite`)"
        ),
        "",
    ]
    return "\n".join(lines)


def run_pilot(
    *,
    data_dir: Path,
    seeds: tuple[int, ...],
    strategies: tuple[RecombineStrategy, ...],
    out_json: Path,
    out_md: Path,
    registry_db: Path,
    epochs: int,
    batch_size: int,
    lr: float,
    smoke: bool = False,
) -> dict[str, Any]:
    tasks = load_split_fmnist_5tasks(data_dir)
    if len(tasks) != 5:
        raise RuntimeError(
            f"Split-FMNIST loader returned {len(tasks)} tasks (expected 5)"
        )

    registry = RunRegistry(registry_db)
    commit_sha = _resolve_commit_sha()
    c5 = representative_combo()
    cells: list[dict[str, Any]] = []
    sweep_start = time.time()

    for strategy in strategies:
        for arm in ARMS:
            for seed in seeds:
                cell = _run_cell(
                    arm,
                    seed,
                    c5,
                    strategy,
                    tasks,
                    epochs=epochs,
                    batch_size=batch_size,
                    lr=lr,
                )
                profile_key = (
                    f"g4-quater/step3/{arm}/{c5.combo_id}/{strategy}"
                )
                run_id = registry.register(
                    c_version=C_VERSION,
                    profile=profile_key,
                    seed=seed,
                    commit_sha=commit_sha,
                )
                cell["run_id"] = run_id
                cells.append(cell)

    wall = time.time() - sweep_start
    verdict = {
        "h4c_recombine_strategy": _h4c_verdict(cells),
        "ae_observation": _h4c_ae_observation(cells),
    }
    payload = {
        "date": "2026-05-03",
        "c_version": C_VERSION,
        "commit_sha": commit_sha,
        "n_seeds": len(seeds),
        "strategies": list(strategies),
        "arms": list(ARMS),
        "data_dir": str(data_dir),
        "wall_time_s": wall,
        "smoke": smoke,
        "cells": cells,
        "verdict": verdict,
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True))
    out_md.write_text(_render_md_report(payload))
    return payload


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="G4-quater Step 3 driver")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--registry-db", type=Path, default=DEFAULT_REGISTRY_DB
    )
    parser.add_argument("--n-seeds", type=int, default=DEFAULT_N_SEEDS)
    parser.add_argument("--seeds", type=int, nargs="+", default=None)
    parser.add_argument(
        "--strategies", type=str, nargs="+",
        default=list(STRATEGIES),
    )
    args = parser.parse_args(argv)

    if args.smoke:
        seeds: tuple[int, ...] = (0,)
        strategies: tuple[RecombineStrategy, ...] = ("mog",)
    else:
        seeds = tuple(args.seeds) if args.seeds else tuple(range(args.n_seeds))
        strategies = tuple(args.strategies)  # type: ignore[assignment]

    payload = run_pilot(
        data_dir=args.data_dir,
        seeds=seeds,
        strategies=strategies,
        out_json=args.out_json,
        out_md=args.out_md,
        registry_db=args.registry_db,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        smoke=args.smoke,
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Cells : {len(payload['cells'])}")
    h = payload["verdict"]["h4c_recombine_strategy"]
    if not h.get("insufficient_samples"):
        print(
            f"H4-C : RECOMBINE empty confirmed = "
            f"{h['h4c_recombine_empty_confirmed']} "
            f"(p={h['welch_p_two_sided']:.4f})"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
