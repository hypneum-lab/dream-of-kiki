"""Integration smoke test — Wave 3b M4 diffusion ablation driver.

Four assertions on the M4 wiring, per the plan §4 M4 acceptance :

1. ``scripts.ablation_cycle3_diffusion.main`` runs the ``--smoke``
   path end-to-end without exception, in ≤ 30 s wall-clock on M5.
2. The substrate factory resolves ``"mlx_latent_diffusion"`` to a
   :class:`MLXLatentDiffusionSubstrate` instance.
3. The Split-CIFAR-100 smoke loader yields batches with the
   expected shape contract for task 0.
4. The R1 output hash captured by the smoke run is deterministic
   (re-running with the same seed re-emits the same hash) — the
   existing R1 contract surface from
   :mod:`harness.storage.run_registry` is exercised end-to-end.

The MLX-only branches are skipped on Linux CI runners via the
``conftest.py``-level ``mlx`` import-skip pattern already in use
across :mod:`tests.reproducibility`.
"""
from __future__ import annotations

import time
from pathlib import Path

import pytest

mx = pytest.importorskip("mlx.core")

from harness.diffusion_eval.cifar100_split_loader import (  # noqa: E402
    N_CLASSES_PER_TASK,
    SMOKE_FEATURE_DIM,
    SMOKE_N_TRAIN_PER_TASK,
    load_split_cifar100,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.substrates.factory import (  # noqa: E402
    build_substrate_adapter,
)
from kiki_oniric.substrates.mlx_latent_diffusion import (  # noqa: E402
    MLXLatentDiffusionSubstrate,
)
from scripts import ablation_cycle3_diffusion as driver  # noqa: E402


def test_smoke_main_runs_end_to_end(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """plan §4 M4 acceptance — ``--smoke`` exits 0 in ≤ 30 s."""
    # Redirect the run registry to a tmp path so the test does not
    # leak rows into the repo-level ``.run_registry.sqlite`` (per
    # ``tests/CLAUDE.md`` anti-pattern : "don't rely on filesystem
    # state between tests").
    monkeypatch.setenv(
        "DREAMOFKIKI_RUN_REGISTRY", str(tmp_path / "r.sqlite")
    )
    t0 = time.perf_counter()
    rc = driver.main(["--smoke"])
    wall = time.perf_counter() - t0
    assert rc == 0
    # Generous bound : 30 s wall on M5. The smoke path is sub-second
    # in practice (synthetic latents + 4-batch trainer).
    assert wall <= 30.0, (
        f"smoke wall_time {wall:.2f}s exceeds 30 s bound"
    )


def test_factory_resolves_mlx_latent_diffusion() -> None:
    """plan §3.3 — factory dispatches the new substrate name."""
    adapter = build_substrate_adapter("mlx_latent_diffusion")
    assert isinstance(adapter, MLXLatentDiffusionSubstrate)


def test_cifar100_smoke_loader_shapes() -> None:
    """plan §2.1 D1 — Split-CIFAR-100 task 0 yields right shapes."""
    batches = list(
        load_split_cifar100(
            task_idx=0,
            batch_size=32,
            seed=0,
            smoke=True,
            split="train",
        )
    )
    assert batches, "smoke loader yielded zero batches"
    total = sum(b.features.shape[0] for b in batches)
    assert total == SMOKE_N_TRAIN_PER_TASK
    first = batches[0]
    # batch_size 32 divides SMOKE_N_TRAIN_PER_TASK 256, so the
    # first (and all) batches have shape (32, SMOKE_FEATURE_DIM).
    assert first.features.shape == (32, SMOKE_FEATURE_DIM)
    assert first.labels.shape == (32,)
    # Labels are task-local : in 0..19.
    assert int(mx.max(first.labels).item()) < N_CLASSES_PER_TASK
    assert int(mx.min(first.labels).item()) >= 0
    assert first.task_idx == 0


def test_smoke_r1_hash_deterministic(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """plan §2.3 D3 — same seed + same loader → same output hash."""
    cfg = driver.AblationConfig(
        task_idx=0,
        profile="p_min",
        substrate="mlx_latent_diffusion",
        seed=0,
    )
    # Use a stable commit_sha so the run_id is reproducible across
    # the two invocations even if the working tree is dirty.
    monkeypatch.setenv("DREAMOFKIKI_COMMIT_SHA", "0" * 40)
    metrics_a = driver.run_one_cell(cfg, smoke=True)
    metrics_b = driver.run_one_cell(cfg, smoke=True)
    hash_a = driver._r1_hash_metrics(metrics_a)
    hash_b = driver._r1_hash_metrics(metrics_b)
    assert hash_a == hash_b, (
        "R1 contract broken : same seed yielded different output hash "
        f"({hash_a} != {hash_b})"
    )

    # Round-trip the hash through the registry to exercise the
    # R1 surface end-to-end.
    registry = RunRegistry(tmp_path / "r.sqlite")
    run_id = registry.register(
        c_version=driver.HARNESS_VERSION,
        profile=driver._registry_profile_tag(cfg),
        seed=cfg.seed,
        commit_sha="0" * 40,
    )
    registry.register_output_hash(run_id, hash_a)
    # Re-registering the same hash is idempotent.
    registry.register_output_hash(run_id, hash_a)


def test_enumerate_configs_full_grid() -> None:
    """plan §4 M4 — full grid = 5 tasks × 3 profiles × 1 sub × 60 seeds."""
    configs = list(driver.enumerate_configs())
    assert len(configs) == 5 * 3 * 1 * 60
    # Outer-to-inner order : task → profile → substrate → seed.
    assert configs[0].task_idx == 0 and configs[0].seed == 0
    assert configs[-1].task_idx == 4 and configs[-1].seed == 59


def test_compute_run_id_deterministic() -> None:
    """plan §2.3 D3 — run_id is bit-stable on (cfg, commit_sha)."""
    cfg = driver.AblationConfig(
        task_idx=2,
        profile="p_equ",
        substrate="mlx_latent_diffusion",
        seed=7,
    )
    a = driver.compute_run_id(cfg, commit_sha="abc")
    b = driver.compute_run_id(cfg, commit_sha="abc")
    c = driver.compute_run_id(cfg, commit_sha="def")
    assert a == b
    assert a != c
    # 32 hex chars per R1 contract (run_registry slice width).
    assert len(a) == 32


def test_parse_cli_flags() -> None:
    """All four CLI flags parse correctly."""
    opts = driver._parse_cli(["--smoke", "--dry-run"])
    assert opts["smoke"] is True
    assert opts["dry_run"] is True
    opts = driver._parse_cli(["--resume", "--max-runs", "5"])
    assert opts["resume"] is True
    assert opts["max_runs"] == 5


def test_parse_cli_rejects_bad_max_runs() -> None:
    with pytest.raises(SystemExit):
        driver._parse_cli(["--max-runs", "abc"])
    with pytest.raises(SystemExit):
        driver._parse_cli(["--max-runs", "0"])
    with pytest.raises(SystemExit):
        driver._parse_cli(["--max-runs"])


def test_main_dry_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """``--dry-run`` returns 0 without touching the registry."""
    monkeypatch.setenv(
        "DREAMOFKIKI_RUN_REGISTRY", str(tmp_path / "dry.sqlite")
    )
    assert driver.main(["--dry-run", "--max-runs", "3"]) == 0


def test_main_envelope_mode_registers_cells(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Prod path (no --smoke) registers cells then returns 0."""
    monkeypatch.setenv(
        "DREAMOFKIKI_RUN_REGISTRY", str(tmp_path / "env.sqlite")
    )
    monkeypatch.setenv("DREAMOFKIKI_COMMIT_SHA", "0" * 40)
    assert driver.main(["--max-runs", "5"]) == 0
    # The registry should now have 5 rows.
    import sqlite3
    conn = sqlite3.connect(tmp_path / "env.sqlite")
    n = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    conn.close()
    assert n == 5


def test_loader_prod_mode_not_implemented() -> None:
    """plan §4 M4 acceptance — prod loader is M5 scope, raises."""
    from harness.diffusion_eval.cifar100_split_loader import (
        load_split_cifar100,
    )
    with pytest.raises(FileNotFoundError, match="M4"):
        list(load_split_cifar100(task_idx=0, smoke=False))


def test_loader_rejects_bad_args() -> None:
    from harness.diffusion_eval.cifar100_split_loader import (
        load_split_cifar100,
        task_classes,
    )
    with pytest.raises(ValueError, match="batch_size"):
        list(load_split_cifar100(task_idx=0, batch_size=0, smoke=True))
    with pytest.raises(ValueError, match="split"):
        list(
            load_split_cifar100(
                task_idx=0, smoke=True, split="weird"  # type: ignore[arg-type]
            )
        )
    with pytest.raises(ValueError, match="task_idx"):
        task_classes(99)


def test_loader_val_split_distinct_from_train() -> None:
    """smoke train vs val use distinct seed derivations."""
    from harness.diffusion_eval.cifar100_split_loader import (
        load_split_cifar100,
    )
    train = list(
        load_split_cifar100(
            task_idx=0, batch_size=32, seed=0, smoke=True, split="train"
        )
    )
    val = list(
        load_split_cifar100(
            task_idx=0, batch_size=32, seed=0, smoke=True, split="val"
        )
    )
    assert len(train) > 0 and len(val) > 0
    assert not bool(
        mx.all(train[0].features == val[0].features).item()
    )


def test_metrics_average_accuracy() -> None:
    from harness.diffusion_eval.diffusion_metrics import (
        average_accuracy,
        forgetting,
        accuracy_per_task,
    )
    assert average_accuracy([0.5, 0.5, 0.5]) == 0.5
    assert forgetting([0.9, 0.8], [0.6, 0.5]) == pytest.approx(0.3)
    assert accuracy_per_task([0.1, 0.9]) == [0.1, 0.9]


def test_metrics_reject_bad_input() -> None:
    from harness.diffusion_eval.diffusion_metrics import (
        accuracy_per_task,
        average_accuracy,
        forgetting,
    )
    with pytest.raises(ValueError, match="NaN"):
        accuracy_per_task([float("nan")])
    with pytest.raises(ValueError, match="outside"):
        accuracy_per_task([1.5])
    with pytest.raises(ValueError, match="empty"):
        average_accuracy([])
    with pytest.raises(ValueError, match="len mismatch"):
        forgetting([0.5], [0.5, 0.5])
    with pytest.raises(ValueError, match="empty"):
        forgetting([], [])
