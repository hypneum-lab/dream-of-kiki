"""NH-03 driver : TTT continuous consolidation vs discrete episodes.

Benchmark = G4-ter richer substrate (Split-FMNIST 5 binary tasks,
hierarchical MLP 784-32-16-2, 3 epochs, batch 64, lr 0.01, beta
FIFO 256 filled with 32 exemplars per task, retention = acc_final /
acc_initial on task 0 ; G4-ter C5 consolidation HPs). Every
consolidation arm spends the SAME write budget : 6 writes per
inter-task interval x 4 intervals = 24 writes per cell, metered by
the substrate ledger (cap = 24, hard).

Arms (pre-reg §3) :
- ``baseline``          : no consolidation write (reference only).
- ``A_discrete``        : (a) one DE per interval before task k,
                          ops (REPLAY, DOWNSCALE) = 5 slow replay
                          steps + 1 slow downscale (G4-ter P_min C5).
- ``B_ttt_continuous``  : (b) the 6 writes go to the TTT fast
                          weights (5 TTT-descent steps + 1 forgetting
                          gate), spread evenly over task k's wake
                          steps, each as a single-op micro-DE.
- ``C_sgd_matched``     : (c) 6 slow SGD writes on a wake-stream
                          mini-batch of task k (no beta access),
                          placed where (a) places its DE.
- ``A_continuous``      : exploratory ; (a)'s writes, (b)'s clock.
- ``B_ttt_chunked``     : exploratory ; (b)'s writes, (a)'s clock.

Pre-reg : docs/milestones/2026-10-25-nh03-ttt-vs-episodes.md
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import mlx.core as mx  # noqa: E402
import numpy as np  # noqa: E402

from experiments.g4_split_fmnist.dataset import (  # noqa: E402
    SplitFMNISTTask,
    load_split_fmnist_5tasks,
)
from experiments.g4_ter_hp_sweep.dream_wrap_hier import (  # noqa: E402
    BetaBufferHierFIFO,
)
from experiments.g4_ter_hp_sweep.hp_grid import representative_combo  # noqa: E402
from experiments.nh03_ttt_vs_episodes.ablation_analysis import (  # noqa: E402
    analyse_ablation,
)
from experiments.nh03_ttt_vs_episodes.analysis import analyse  # noqa: E402
from experiments.nh03_ttt_vs_episodes.ttt_classifier import (  # noqa: E402
    NH03Classifier,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.dream.episode import (  # noqa: E402
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.runtime import DreamRuntime  # noqa: E402
from kiki_oniric.substrates.ttt_fast_weights import WriteLedger  # noqa: E402

C_VERSION = "C-v0.25.0+PARTIAL"
ARMS_PRIMARY = ("baseline", "A_discrete", "B_ttt_continuous", "C_sgd_matched")
ARMS_EXPLORATORY = ("A_continuous", "B_ttt_chunked")
ALL_ARMS = ARMS_PRIMARY + ARMS_EXPLORATORY
# Schedule ablation (pre-reg 2026-10-27) : 2 substrates x 5 schedules.
SCHEDULES = ("early_block", "early_spread", "late_block", "late_spread", "uniform")
ABLATION_ARMS = tuple(f"{fam}_{sch}" for fam in ("slow", "ttt") for sch in SCHEDULES)
ABLATION_SEEDS = tuple(range(100, 130))
ARM_SPEC: dict[str, tuple[str, str]] = {
    "baseline": ("none", "early_block"),
    "A_discrete": ("slow", "early_block"),
    "A_continuous": ("slow", "uniform"),
    "B_ttt_continuous": ("ttt", "uniform"),
    "B_ttt_chunked": ("ttt", "early_block"),
    "C_sgd_matched": ("sgd", "early_block"),
    **{f"{fam}_{sch}": (fam, sch) for fam in ("slow", "ttt") for sch in SCHEDULES},
}
CONFIRMATORY_SEEDS = tuple(range(30))
PILOT_SEEDS = (1000, 1001, 1002, 1003, 1004)
TTT_LR_GRID = (0.01, 0.03, 0.1, 0.3, 1.0)

HIDDEN_1 = 32
HIDDEN_2 = 16
N_CLASSES = 2
EPOCHS = 3
BATCH_SIZE = 64
WAKE_LR = 0.01
BETA_CAPACITY = 256
BETA_FILL_PER_TASK = 32
RETENTION_EPS = 1e-6
WRITES_PER_INTERVAL = 6  # 5 replay-type + 1 rescale (G4-ter C5)
N_INTERVALS = 4
WRITE_BUDGET = WRITES_PER_INTERVAL * N_INTERVALS  # 24
CONTROL_BATCH = 32

DEFAULT_DATA_DIR = REPO_ROOT / "experiments" / "g4_split_fmnist" / "data"


def _commit_sha() -> str:
    env = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=5,
        )
        return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        return "unknown"


def _flops_per_write(clf: NH03Classifier, op: str, n: int, in_dim: int) -> float:
    """Rough FLOP estimate of one write (secondary meter only)."""
    slow_macs = in_dim * HIDDEN_1 + HIDDEN_1 * HIDDEN_2 + HIDDEN_2 * N_CLASSES
    if op in ("replay_slow", "sgd_control"):
        return 6.0 * slow_macs * n
    if op == "replay_ttt":
        enc = 2.0 * in_dim * HIDDEN_1 * n
        tail = HIDDEN_1 * HIDDEN_1 + HIDDEN_1 * HIDDEN_2 + HIDDEN_2 * N_CLASSES
        return enc + 6.0 * tail * n
    if op == "downscale_slow":
        return float(clf._n_slow)
    if op == "decay_ttt":
        return float(HIDDEN_1 * HIDDEN_1)
    raise ValueError(op)


def _positions(n_steps: int, continuous: bool) -> list[int]:
    """Wake-step index before which write j (0..5) is applied."""
    return schedule_positions(n_steps, "uniform" if continuous else "early_block")


def schedule_positions(n_steps: int, schedule: str) -> list[int]:
    """Wake-step index before which write j (0..5) is applied.

    ``n_steps`` means "after the last wake step of the task" (end of
    the interval, i.e. just before the next task / final evaluation).
    Spread windows are the first / last third of the task.
    """
    w = WRITES_PER_INTERVAL
    third = n_steps // 3
    if schedule == "early_block":
        return [0] * w
    if schedule == "late_block":
        return [n_steps] * w
    if schedule == "uniform":
        return [(j * n_steps) // w for j in range(w)]
    if schedule == "early_spread":
        return [(j * third) // w for j in range(w)]
    if schedule == "late_spread":
        return [n_steps - third + ((j + 1) * third) // w for j in range(w)]
    raise ValueError(f"unknown schedule {schedule!r}")


def run_cell(
    arm: str,
    seed: int,
    tasks: list[SplitFMNISTTask],
    *,
    ttt_lr: float,
) -> dict[str, Any]:
    if arm not in ARM_SPEC:
        raise ValueError(f"unknown arm {arm!r}")
    family, schedule = ARM_SPEC[arm]
    t0 = time.time()
    c5 = representative_combo()
    in_dim = int(tasks[0]["x_train"].shape[1])
    ledger = WriteLedger(cap=WRITE_BUDGET)
    clf = NH03Classifier(
        in_dim=in_dim, hidden_1=HIDDEN_1, hidden_2=HIDDEN_2,
        n_classes=N_CLASSES, seed=seed, ledger=ledger,
    )
    buffer = BetaBufferHierFIFO(capacity=BETA_CAPACITY)
    fill_rng = np.random.default_rng(seed + 5_000)
    runtime = DreamRuntime()
    flops = 0.0

    def push(task: SplitFMNISTTask) -> None:
        n = task["x_train"].shape[0]
        idx = fill_rng.choice(n, size=min(BETA_FILL_PER_TASK, n), replace=False)
        for i in idx.tolist():
            buffer.push(x=task["x_train"][i], y=int(task["y_train"][i]), latent=None)

    def train_plain(task: SplitFMNISTTask) -> None:
        for idx in clf.wake_batches(task, epochs=EPOCHS, batch_size=BATCH_SIZE):
            clf.wake_step(task, idx, lr=WAKE_LR)

    train_plain(tasks[0])
    acc_initial = clf.eval_accuracy(tasks[0]["x_test"], tasks[0]["y_test"])
    push(tasks[0])

    slow_family = family == "slow"
    ttt_family = family == "ttt"
    block = schedule in ("early_block", "late_block")

    for k in range(1, len(tasks)):
        task = tasks[k]
        batches = clf.wake_batches(task, epochs=EPOCHS, batch_size=BATCH_SIZE)
        writes: list[Callable[[str], None]] = []
        ops: list[Operation] = []
        if slow_family or ttt_family:
            records = buffer.sample(n=c5.replay_batch, seed=seed + k)
            n_rec = len(records)
            for _ in range(c5.replay_n_steps):
                if slow_family:
                    writes.append(
                        lambda eid, r=records: clf.write_slow_replay(
                            r, lr=c5.replay_lr, episode_id=eid))
                    flops += _flops_per_write(clf, "replay_slow", n_rec, in_dim)
                else:
                    writes.append(
                        lambda eid, r=records: clf.write_ttt_replay(
                            r, lr=ttt_lr, episode_id=eid))
                    flops += _flops_per_write(clf, "replay_ttt", n_rec, in_dim)
                ops.append(Operation.REPLAY)
            if slow_family:
                writes.append(lambda eid: clf.write_slow_downscale(
                    factor=c5.downscale_factor, episode_id=eid))
                flops += _flops_per_write(clf, "downscale_slow", 0, in_dim)
            else:
                writes.append(lambda eid: clf.write_ttt_decay(
                    factor=c5.downscale_factor, episode_id=eid))
                flops += _flops_per_write(clf, "decay_ttt", 0, in_dim)
            ops.append(Operation.DOWNSCALE)
        elif family == "sgd":
            crng = np.random.default_rng(seed + k + 7_000)
            cidx = crng.choice(task["x_train"].shape[0], size=CONTROL_BATCH, replace=False)
            xb, yb = task["x_train"][cidx], task["y_train"][cidx]
            for _ in range(WRITES_PER_INTERVAL):
                writes.append(lambda eid, a=xb, b=yb: clf.write_sgd_control(
                    a, b, lr=c5.replay_lr, episode_id=eid))
                flops += _flops_per_write(clf, "sgd_control", CONTROL_BATCH, in_dim)

        n_steps = len(batches)
        pos = schedule_positions(n_steps, schedule) if writes else []
        is_de_arm = slow_family or ttt_family

        def run_block(queue: list[Callable[[str], None]]) -> None:
            # One bounded DE holding all 6 writes (G4 order R x5, D).
            eid = f"nh03-{arm}-s{seed}-k{k}"

            def h_replay(ep: DreamEpisode) -> None:
                for _ in range(c5.replay_n_steps):
                    queue.pop(0)(ep.episode_id)

            def h_down(ep: DreamEpisode) -> None:
                queue.pop(0)(ep.episode_id)

            runtime.register_handler(Operation.REPLAY, h_replay)
            runtime.register_handler(Operation.DOWNSCALE, h_down)
            runtime.execute(_episode(eid, (Operation.REPLAY, Operation.DOWNSCALE)))

        def fire_due(step: int) -> None:
            while pos and pos[0] == step:
                pos.pop(0)
                w = writes.pop(0)
                if is_de_arm:
                    op = ops.pop(0)
                    eid = f"nh03-{arm}-s{seed}-k{k}-w{WRITES_PER_INTERVAL - len(writes) - 1}"
                    runtime.register_handler(op, lambda ep, fn=w: fn(ep.episode_id))
                    runtime.execute(_episode(eid, (op,)))
                else:
                    w(f"ctrl-{arm}-s{seed}-k{k}")

        if is_de_arm and block and schedule == "early_block":
            run_block(writes)
            pos = []
        for step, idx in enumerate(batches):
            fire_due(step)
            clf.wake_step(task, idx, lr=WAKE_LR)
        if is_de_arm and block and schedule == "late_block":
            run_block(writes)
            pos = []
        fire_due(n_steps)
        if writes:
            raise RuntimeError("unscheduled writes left (schedule bug)")
        push(task)

    acc_final = clf.eval_accuracy(tasks[0]["x_test"], tasks[0]["y_test"])
    expected = 0 if arm == "baseline" else WRITE_BUDGET
    if ledger.total != expected:
        raise RuntimeError(
            f"budget mismatch arm={arm} writes={ledger.total} expected={expected}"
        )
    logged = {e.episode_id for e in runtime.log if e.completed}
    traced = all(e in logged for e in ledger.episode_ids() if not e.startswith("ctrl-"))
    return {
        "arm": arm,
        "seed": seed,
        "ttt_lr": ttt_lr,
        "acc_task1_initial": float(acc_initial),
        "acc_task1_final": float(acc_final),
        "retention": float(acc_final / max(acc_initial, RETENTION_EPS)),
        "excluded_underperforming_baseline": bool(acc_initial < 0.5),
        "writes_total": ledger.total,
        "writes_by_op": ledger.by_op(),
        "sample_gradients": ledger.sample_gradients,
        "elements_written": ledger.elements_written,
        "consolidation_flops": flops,
        "n_episodes_logged": len(runtime.log),
        "dr0_all_writes_traced": bool(traced),
        "fast_weight_fro_norm": float(np.linalg.norm(clf.fast.numpy())),
        "wall_time_s": time.time() - t0,
    }


def _episode(eid: str, ops: tuple[Operation, ...]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={},
        operation_set=ops,
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=10.0, energy_j=1.0),
        episode_id=eid,
    )


def run(
    *,
    arms: tuple[str, ...],
    seeds: tuple[int, ...],
    ttt_lr: float,
    data_dir: Path,
    out_json: Path,
    registry_db: Path | None,
    label: str,
) -> dict[str, Any]:
    tasks = load_split_fmnist_5tasks(data_dir)
    commit = _commit_sha()
    registry = RunRegistry(registry_db) if registry_db else None
    cells: list[dict[str, Any]] = []
    t0 = time.time()
    out_json.parent.mkdir(parents=True, exist_ok=True)
    for seed in seeds:
        for arm in arms:
            cell = run_cell(arm, seed, tasks, ttt_lr=ttt_lr)
            if registry is not None:
                cell["run_id"] = registry.register(
                    c_version=C_VERSION, profile=f"nh03/{arm}/ttt_lr={ttt_lr}",
                    seed=seed, commit_sha=commit,
                )
            cells.append(cell)
            print(
                f"[{label}] seed={seed} arm={arm:17s} ret={cell['retention']:.4f} "
                f"writes={cell['writes_total']} t={cell['wall_time_s']:.1f}s",
                flush=True,
            )
        partial = {"label": label, "cells": cells, "partial": True}
        out_json.write_text(json.dumps(partial, indent=1, sort_keys=True))
    payload = {
        "label": label,
        "c_version": C_VERSION,
        "commit_sha": commit,
        "host": platform.node(),
        "mlx_version": mx.__version__,
        "mlx_device": str(mx.default_device()),
        "arms": list(arms),
        "seeds": list(seeds),
        "ttt_lr": ttt_lr,
        "write_budget_per_cell": WRITE_BUDGET,
        "wall_time_s": time.time() - t0,
        "cells": cells,
        "analysis": analyse(cells),
        "ablation_analysis": (
            analyse_ablation(cells)
            if any(c["arm"] in ABLATION_ARMS for c in cells) else None
        ),
        "partial": False,
    }
    out_json.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return payload


def merge(inputs: list[Path], out_json: Path) -> dict[str, Any]:
    """Merge worker JSONs (disjoint seeds) and re-run the analyses."""
    parts = [json.loads(Path(i).read_text()) for i in inputs]
    if any(pp.get("partial", True) for pp in parts):
        raise RuntimeError("refusing to merge a partial worker output")
    cells = [c for pp in parts for c in pp["cells"]]
    keys = [(c["arm"], c["seed"]) for c in cells]
    if len(keys) != len(set(keys)):
        raise RuntimeError("duplicate (arm, seed) cells across workers")
    head = parts[0]
    payload = {
        **{k: head[k] for k in ("c_version", "commit_sha", "host", "mlx_version",
                                "mlx_device", "ttt_lr", "write_budget_per_cell")},
        "label": "merged:" + "+".join(pp["label"] for pp in parts),
        "workers": [{"seeds": pp["seeds"], "wall_time_s": pp["wall_time_s"],
                     "host": pp["host"]} for pp in parts],
        "arms": sorted({c["arm"] for c in cells}),
        "seeds": sorted({c["seed"] for c in cells}),
        "cells": cells,
        "analysis": analyse(cells),
        "ablation_analysis": analyse_ablation(cells),
        "partial": False,
    }
    out_json.write_text(json.dumps(payload, indent=1, sort_keys=True))
    return payload


def calibrate(
    *, data_dir: Path, out_json: Path, grid: tuple[float, ...] = TTT_LR_GRID
) -> dict[str, Any]:
    """Pilot : pick TTT inner lr on PILOT_SEEDS (disjoint from 0..29).

    Rule (pre-reg §4) : argmax of mean B_ttt_continuous retention,
    ties -> smaller lr ; any S2-aborted cell disqualifies that lr.
    """
    tasks = load_split_fmnist_5tasks(data_dir)
    res: dict[str, Any] = {"grid": {}, "seeds": list(PILOT_SEEDS)}
    for lr in grid:
        rets: list[float] = []
        aborted = False
        for s in PILOT_SEEDS:
            try:
                c = run_cell("B_ttt_continuous", s, tasks, ttt_lr=lr)
                rets.append(c["retention"])
            except Exception as exc:  # S2 abort or budget error
                aborted = True
                print(f"[calib] lr={lr} seed={s} aborted: {exc}", flush=True)
                break
        res["grid"][str(lr)] = {
            "mean_retention": float(np.mean(rets)) if rets else None,
            "retentions": rets,
            "aborted": aborted,
        }
        print(f"[calib] lr={lr} mean={res['grid'][str(lr)]['mean_retention']}", flush=True)
    ok = [(float(k), v["mean_retention"]) for k, v in res["grid"].items()
          if not v["aborted"] and v["mean_retention"] is not None]
    best = max(ok, key=lambda kv: (round(kv[1], 6), -kv[0]))
    res["selected_ttt_lr"] = best[0]
    base = [run_cell("A_discrete", s, tasks, ttt_lr=best[0])["retention"]
            for s in PILOT_SEEDS]
    res["pilot_A_discrete_mean_retention"] = float(np.mean(base))
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(res, indent=1, sort_keys=True))
    return res


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="NH-03 driver")
    p.add_argument("--mode", choices=("smoke", "calibrate", "confirmatory", "custom",
                                      "ablation", "merge"),
                   default="smoke")
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--out-json", type=Path, required=True)
    p.add_argument("--registry-db", type=Path, default=None)
    p.add_argument("--ttt-lr", type=float, default=None)
    p.add_argument("--arms", nargs="+", default=list(ALL_ARMS))
    p.add_argument("--seeds", type=int, nargs="+", default=list(CONFIRMATORY_SEEDS))
    p.add_argument("--inputs", type=Path, nargs="+", default=[])
    p.add_argument("--lr-grid", type=float, nargs="+", default=list(TTT_LR_GRID))
    a = p.parse_args(argv)
    if a.mode == "calibrate":
        r = calibrate(data_dir=a.data_dir, out_json=a.out_json, grid=tuple(a.lr_grid))
        print(f"selected_ttt_lr={r['selected_ttt_lr']}")
        return 0
    if a.mode == "merge":
        merge(a.inputs, a.out_json)
        return 0
    if a.ttt_lr is None:
        p.error("--ttt-lr is required outside calibrate mode")
    if a.mode == "smoke":
        seeds, arms = (1000,), tuple(ALL_ARMS)
    elif a.mode == "confirmatory":
        seeds, arms = CONFIRMATORY_SEEDS, tuple(ALL_ARMS)
    elif a.mode == "ablation":
        seeds = tuple(a.seeds) if a.seeds != list(CONFIRMATORY_SEEDS) else ABLATION_SEEDS
        if not set(seeds) <= set(ABLATION_SEEDS):
            p.error("ablation seeds must be within the declared set 100..129")
        arms = ABLATION_ARMS
    else:
        seeds, arms = tuple(a.seeds), tuple(a.arms)
    payload = run(arms=arms, seeds=seeds, ttt_lr=a.ttt_lr, data_dir=a.data_dir,
                  out_json=a.out_json, registry_db=a.registry_db, label=a.mode)
    if payload.get("ablation_analysis"):
        print(json.dumps(payload["ablation_analysis"]["primary"]["verdict"]))
    prim = payload["analysis"]["primary"]
    print(json.dumps(prim, indent=1) if prim else "no primary contrast")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
