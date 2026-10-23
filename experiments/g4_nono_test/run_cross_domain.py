"""G4-nono driver: run ONE (seed, cell) and write its outcome record.

Spec: docs/osf-prereg-g4-nono-cross-domain-2026-10-23.md section 5
(draft PR #41). A cell is (arm, strategy); 8 cells per seed:
``baseline`` and ``P_min`` (strategy ``na``) plus ``P_equ`` / ``P_max``
x {cross, within, none}. This module never loops over seeds or cells;
orchestration (and the aggregator) live elsewhere.

Example::

    uv run python -m experiments.g4_nono_test.run_cross_domain \
        --seed 900 --arm P_max --strategy cross --out /tmp/cell.json
"""
from __future__ import annotations

import argparse
import json
import os
import sqlite3
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from experiments.g4_nono_test.metrics import (  # noqa: E402
    NonFiniteLogitsError,
    compute_cdpa,
    compute_final_accuracy,
    compute_retention_domain_a,
    compute_two_way_accuracy,
    compute_wdpa,
)
from experiments.g4_nono_test.recombine_cross import (  # noqa: E402
    CrossStrategy,
    sample_cross_domain_latents,
    soft_target_fc2_step,
)
from experiments.g4_nono_test.stream import (  # noqa: E402
    TASK_CLASS_PAIRS,
    DomainTask,
    build_domain_stream,
)
from experiments.g4_quinto_test.small_cnn import G4SmallCNN  # noqa: E402
from experiments.g4_sexto_test.cifar100_dataset import (  # noqa: E402
    load_cifar100_coarse_arrays_auto,
)
from experiments.g4_sexto_test.run_step1_cifar100 import (  # noqa: E402
    _BetaBufferCNN,
)
from experiments.g4_split_fmnist.dream_wrap import (  # noqa: E402
    build_profile,
)
from experiments.g4_ter_hp_sweep.hp_grid import (  # noqa: E402
    HPCombo,
    representative_combo,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402

C_VERSION = "C-v0.25.0+PARTIAL"
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
STRATEGIES: tuple[str, ...] = ("cross", "within", "none")
LATENT_DIM = 64
N_CLASSES = 20
EPOCHS = 3
BATCH_SIZE = 64
LR = 0.01
BUFFER_CAPACITY = 320
PUSH_PER_SUPERCLASS = 16
N_SYNTHETIC = 16
RECOMBINE_LR = 0.01
RESTRUCTURE_FACTOR = 0.05
MANIP_THRESHOLD = 0.60
SMOKE_SEEDS: tuple[int, ...] = (900, 901, 902)
DEFAULT_DATA_DIR = REPO_ROOT / "experiments" / "g4_sexto_test" / "data"
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
EVAL_CHUNK = 2000
EXIT_RUN_ID_COLLISION = 3


def cells_for_seed() -> list[tuple[str, str]]:
    """The 8 (arm, strategy) cells of one seed."""
    cells: list[tuple[str, str]] = [("baseline", "na"), ("P_min", "na")]
    for arm in ("P_equ", "P_max"):
        cells += [(arm, s) for s in STRATEGIES]
    return cells


def validate_cell(arm: str, strategy: str) -> None:
    """Raise ``ValueError`` unless (arm, strategy) is one of the 8 cells."""
    if (arm, strategy) not in cells_for_seed():
        raise ValueError(
            f"invalid G4-nono cell (arm={arm!r}, strategy={strategy!r}); "
            f"valid: {cells_for_seed()}"
        )


def new_buffer() -> _BetaBufferCNN:
    """Beta buffer, capacity 320 (never evicts in this stream)."""
    return _BetaBufferCNN(capacity=BUFFER_CAPACITY)


def push_task(
    buffer: _BetaBufferCNN,
    cnn: G4SmallCNN,
    task: DomainTask,
    fill_rng: np.random.Generator,
) -> None:
    """Push 16 records per superclass (32 per task), without
    replacement, from ``fill_rng`` (``default_rng(seed + 5000)``).

    The 64-d post-ReLU ``fc1`` latent is computed by the network at
    push time and never refreshed (ladder policy).
    """
    y = task["y_train"]
    for c in sorted(set(y.tolist())):
        idx_c = np.flatnonzero(y == c)
        pick = fill_rng.choice(
            len(idx_c), size=min(PUSH_PER_SUPERCLASS, len(idx_c)),
            replace=False,
        )
        for i in idx_c[pick].tolist():
            x = task["x_train_nhwc"][i]
            buffer.push(
                x=x, y=int(y[i]), latent=cnn.latent(x[None, ...])[0]
            )


def recombine(
    cnn: G4SmallCNN,
    *,
    strategy: CrossStrategy,
    beta_buffer: _BetaBufferCNN,
    seed: int,
) -> None:
    """RECOMBINE with the section 5 strategies (no-op for ``none``)."""
    populated = beta_buffer.latents()
    if not populated:
        return
    lat = np.asarray([z for z, _ in populated], dtype=np.float32)
    lab = np.asarray([c for _, c in populated], dtype=np.int64)
    batch = sample_cross_domain_latents(
        strategy=strategy, latents=lat, labels=lab,
        n_synthetic=N_SYNTHETIC, seed=seed,
    )
    if batch is None:
        return
    soft_target_fc2_step(cnn, batch["x"], batch["y"], lr=RECOMBINE_LR)


def dream_episode(
    cnn: G4SmallCNN,
    profile: object,
    seed: int,
    *,
    beta_buffer: _BetaBufferCNN,
    combo: HPCombo,
    strategy: CrossStrategy,
) -> None:
    """One DE (ladder schedule). ``seed`` is the DE seed ``seed + k``;
    the recombine seed is ``seed + 30000`` (= ``seed + k + 30000``)."""
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
            Operation.REPLAY, Operation.DOWNSCALE,
        )
        channels: tuple[OutputChannel, ...] = (
            OutputChannel.WEIGHT_DELTA,
        )
    else:
        ops = (
            Operation.REPLAY, Operation.DOWNSCALE,
            Operation.RESTRUCTURE, Operation.RECOMBINE,
        )
        channels = (
            OutputChannel.WEIGHT_DELTA,
            OutputChannel.HIERARCHY_CHG,
            OutputChannel.LATENT_SAMPLE,
        )
    rng = np.random.default_rng(seed + 10_000)
    episode = DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={
            "beta_records": [
                {
                    "x": rng.standard_normal(4).astype(np.float32).tolist(),
                    "y": rng.standard_normal(4).astype(np.float32).tolist(),
                }
                for _ in range(4)
            ],
            "shrink_factor": 0.99,
            "topo_op": "reroute",
            "swap_indices": [0, 1],
            "delta_latents": [
                rng.standard_normal(4).astype(np.float32).tolist()
                for _ in range(2)
            ],
        },
        operation_set=ops,
        output_channels=channels,
        budget=BudgetCap(flops=10_000_000, wall_time_s=10.0, energy_j=1.0),
        episode_id=(
            f"g4-nono-{type(profile).__name__}-{strategy}-seed{seed}"
        ),
    )
    profile.runtime.execute(episode)  # type: ignore[attr-defined]

    sampled = beta_buffer.sample(n=combo.replay_batch, seed=seed)
    cnn.replay_optimizer_step(
        [{"x": r["x"], "y": r["y"]} for r in sampled],
        lr=combo.replay_lr, n_steps=combo.replay_n_steps,
    )
    cnn.downscale_step(factor=combo.downscale_factor)
    if Operation.RESTRUCTURE in ops:
        cnn.restructure_step(factor=RESTRUCTURE_FACTOR, seed=seed + 20_000)
    if Operation.RECOMBINE in ops:
        recombine(
            cnn, strategy=strategy, beta_buffer=beta_buffer,
            seed=seed + 30_000,
        )


def manipulation_accuracy(
    logits: np.ndarray, labels: np.ndarray, task1_classes: tuple[int, int]
) -> float:
    """Section 4: task-1 accuracy on the two task-1 logits only."""
    return compute_two_way_accuracy(logits, labels, task1_classes)


def _logits_chunked(cnn: G4SmallCNN, x: np.ndarray) -> np.ndarray:
    return np.concatenate(
        [
            cnn.predict_logits(x[i : i + EVAL_CHUNK])
            for i in range(0, len(x), EVAL_CHUNK)
        ]
    )


def run_cell(
    *,
    arm: str,
    strategy: str,
    seed: int,
    stream: list[DomainTask],
    epochs: int = EPOCHS,
    batch_size: int = BATCH_SIZE,
    lr: float = LR,
) -> dict[str, Any]:
    """Run one cell and return its outcome record (no I/O)."""
    validate_cell(arm, strategy)
    if len(stream) != len(TASK_CLASS_PAIRS):
        raise ValueError(f"stream has {len(stream)} tasks, expected 10")
    start = time.time()
    combo = representative_combo()
    cnn = G4SmallCNN(latent_dim=LATENT_DIM, n_classes=N_CLASSES, seed=seed)
    buffer = new_buffer()
    fill_rng = np.random.default_rng(seed + 5_000)

    def _train(task: DomainTask) -> None:
        cnn.train_task(
            task["x_train_nhwc"], task["y_train"],
            epochs=epochs, batch_size=batch_size, lr=lr,
        )

    rec: dict[str, Any] = {
        "arm": arm,
        "strategy": strategy,
        "seed": seed,
        "c_version": C_VERSION,
        "n_classes": N_CLASSES,
        "epochs": epochs,
        "batch_size": batch_size,
        "lr": lr,
        "hp_combo_id": combo.combo_id,
        "cdpa": None,
        "wdpa": None,
        "retention_a": None,
        "acc20": None,
        "manip_acc": None,
        "manip_pass": False,
        "finite": True,
        "excluded": False,
        "exclusion_reason": None,
    }
    try:
        _train(stream[0])
        manip_acc = manipulation_accuracy(
            cnn.predict_logits(stream[0]["x_test_nhwc"]),
            stream[0]["y_test"],
            TASK_CLASS_PAIRS[0],
        )
        rec["manip_acc"] = manip_acc
        rec["manip_pass"] = bool(manip_acc >= MANIP_THRESHOLD)
        push_task(buffer, cnn, stream[0], fill_rng)
        profile = None if arm == "baseline" else build_profile(arm, seed=seed)
        for k in range(1, len(stream)):
            if profile is not None:
                dream_episode(
                    cnn, profile, seed + k, beta_buffer=buffer,
                    combo=combo,
                    strategy=strategy if strategy != "na" else "none",  # type: ignore[arg-type]
                )
            _train(stream[k])
            push_task(buffer, cnn, stream[k], fill_rng)

        x_test = np.concatenate([t["x_test_nhwc"] for t in stream])
        y_test = np.concatenate([t["y_test"] for t in stream])
        logits = _logits_chunked(cnn, x_test)
        rec["cdpa"] = compute_cdpa(logits, y_test)
        rec["wdpa"] = compute_wdpa(logits, y_test)
        rec["acc20"] = compute_final_accuracy(logits, y_test)
        rec["retention_a"] = compute_retention_domain_a(logits, y_test)
    except NonFiniteLogitsError:
        for key in ("cdpa", "wdpa", "retention_a", "acc20"):
            rec[key] = None
        rec["finite"] = False
        rec["excluded"] = True
        rec["exclusion_reason"] = "non_finite_logits"
    rec["wall_time_s"] = time.time() - start
    return rec


def load_stream(data_dir: Path) -> list[DomainTask]:
    """Load the pinned CIFAR-100 files and build the two-domain stream."""
    x_tr, c_tr, x_te, c_te = load_cifar100_coarse_arrays_auto(data_dir)
    return build_domain_stream(x_tr, c_tr, x_te, c_te)


def _commit_sha() -> str:
    env = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env:
        return env
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def claim_run_id(
    db_path: Path, *, c_version: str, profile: str, seed: int,
    commit_sha: str,
) -> bool:
    """Atomically claim the cell's run_id; True iff this caller won.

    One plain ``INSERT`` on the ``runs`` PRIMARY KEY inside a
    ``BEGIN IMMEDIATE`` transaction: a concurrent claimant gets
    ``IntegrityError`` (or waits for the write lock, then fails), so
    there is no lookup-then-register window.
    """
    registry = RunRegistry(db_path)  # creates / migrates the schema
    run_id = registry._compute_run_id(c_version, profile, seed, commit_sha)
    conn = sqlite3.connect(db_path, timeout=60, isolation_level=None)
    try:
        conn.execute("BEGIN IMMEDIATE")
        try:
            conn.execute(
                "INSERT INTO runs (run_id, c_version, profile, seed, "
                "commit_sha) VALUES (?, ?, ?, ?, ?)",
                (run_id, c_version, profile, seed, commit_sha),
            )
        except sqlite3.IntegrityError:
            conn.execute("ROLLBACK")
            return False
        conn.execute("COMMIT")
        return True
    finally:
        conn.close()


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="G4-nono: run one cell")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--arm", choices=ARMS, required=True)
    p.add_argument("--strategy", choices=("na",) + STRATEGIES, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    p.add_argument("--registry-db", type=Path, default=DEFAULT_REGISTRY_DB)
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    p.add_argument("--lr", type=float, default=LR)
    args = p.parse_args(argv)
    try:
        validate_cell(args.arm, args.strategy)
    except ValueError as exc:
        p.error(str(exc))
    if args.out.exists():
        p.error(f"refusing to overwrite existing outcome record {args.out}")

    profile_key = f"g4-nono/{args.arm}/{args.strategy}"
    sha = _commit_sha()
    run_id = RunRegistry(args.registry_db)._compute_run_id(
        C_VERSION, profile_key, args.seed, sha
    )
    # Atomic claim before executing; the loser aborts explicitly.
    if not claim_run_id(
        args.registry_db, c_version=C_VERSION, profile=profile_key,
        seed=args.seed, commit_sha=sha,
    ):
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps({
            "status": "aborted",
            "abort_reason": "run_id_collision",
            "arm": args.arm, "strategy": args.strategy, "seed": args.seed,
            "profile_key": profile_key, "commit_sha": sha,
            "run_id": run_id, "c_version": C_VERSION,
        }, indent=2, sort_keys=True))
        print(f"ABORT: run_id {run_id} already registered; wrote {args.out}")
        return EXIT_RUN_ID_COLLISION
    stream = load_stream(args.data_dir)
    rec = run_cell(
        arm=args.arm, strategy=args.strategy, seed=args.seed,
        stream=stream, epochs=args.epochs, batch_size=args.batch_size,
        lr=args.lr,
    )
    rec["status"] = "ok"
    rec["profile_key"] = profile_key
    rec["commit_sha"] = sha
    rec["run_id"] = run_id
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(rec, indent=2, sort_keys=True))
    print(f"Wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
