"""Cycle-3 diffusion ablation runner — Wave 3b M4.

Fork of :mod:`scripts.ablation_cycle3` that wires the new
``mlx_latent_diffusion`` substrate on the **Split-CIFAR-100 5-task**
loader (D1 decision per the plan §2.1).

Scope split :

- **--smoke** mode runs 1 task × 1 profile × 1 seed = 1 cell, in
  ≤ ~10 min wall-clock on M5. Validates pipeline only.
- Production mode (no flag) lays out the full Cartesian product
  ``5 tasks × 3 profiles × N seeds`` and **delegates execution to
  M5** — at M4 the prod path enumerates + registers cells in the
  :class:`~harness.storage.run_registry.RunRegistry` but stops
  short of the long-running compute (matching the ``ablation_cycle3.py``
  precedent which also halts at the "plan-only" boundary line 369).

Key contracts preserved from ``ablation_cycle3.py`` :

- DualVer HARNESS_VERSION tag baked into every registered ``run_id``.
- ``AblationConfig`` frozen-dataclass enumeration.
- Resume semantics via ``RunRegistry`` ``run_id`` diff.

What changes vs the sibling ``ablation_cycle3.py`` :

- ``SUBSTRATES`` is fixed to ``("mlx_latent_diffusion",)`` — this
  driver is dedicated to the new substrate ; the cycle-3 grid file
  keeps the multi-substrate matrix.
- The **scale** axis is replaced by the Split-CIFAR-100 **task**
  axis (one of 5 disjoint 20-class windows).
- The ``HARNESS_VERSION`` matches ``MLX_LATENT_DIFFUSION_SUBSTRATE_VERSION``
  (``C-v0.14.0+PARTIAL``) so the rows are unambiguously Wave-3b.

References:
- ``docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md``
  §3.1 (module layout) + §3.3 (factory integration) + §4 M4
- ``scripts/ablation_cycle3.py`` (sibling pattern, do not diverge
  on the enumeration / resume contract)
- ``kiki_oniric/substrates/mlx_latent_diffusion.py`` (M3 substrate
  surface — ``execute_profile``)
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.diffusion_eval.cifar100_split_loader import (  # noqa: E402
    N_TASKS,
    SplitCifar100Batch,
    load_split_cifar100,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.substrates.mlx_latent_diffusion import (  # noqa: E402
    MLX_LATENT_DIFFUSION_SUBSTRATE_NAME,
    MLX_LATENT_DIFFUSION_SUBSTRATE_VERSION,
    MLXLatentDiffusionSubstrate,
)

# Bake the substrate-version tag into the run_id so Wave-3b rows
# are unambiguously attributable on the empirical axis.
HARNESS_VERSION = MLX_LATENT_DIFFUSION_SUBSTRATE_VERSION

# Axis values — Split-CIFAR-100 over the new substrate.
TASKS: tuple[int, ...] = tuple(range(N_TASKS))
PROFILES: tuple[str, ...] = ("p_min", "p_equ", "p_max")
SUBSTRATES: tuple[str, ...] = (MLX_LATENT_DIFFUSION_SUBSTRATE_NAME,)
# Default seed grid mirrors ``ablation_cycle3.py`` (0..59) ; the
# smoke run pins seed=0 only.
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(60))


def _resolve_commit_sha() -> str:
    """Best-effort git HEAD lookup — mirrors ``ablation_cycle3``."""
    env_sha = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
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


@dataclass(frozen=True)
class AblationConfig:
    """One cell of the Wave-3b diffusion ablation grid.

    Attributes
    ----------
    task_idx:
        Split-CIFAR-100 task index in ``0..N_TASKS - 1``.
    profile:
        Profile name ``p_min`` / ``p_equ`` / ``p_max``.
    substrate:
        Always ``"mlx_latent_diffusion"`` in this driver.
    seed:
        Seed used for the run_id tuple (R1 contract) and for the
        loader's batch RNG.
    """

    task_idx: int
    profile: str
    substrate: str
    seed: int


def enumerate_configs(
    tasks: Iterable[int] | None = None,
    profiles: Iterable[str] | None = None,
    substrates: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> Iterator[AblationConfig]:
    """Yield the cartesian product in (task → profile → substrate →
    seed) order — outer-to-inner mirrors ``ablation_cycle3``."""
    task_vals = tuple(tasks) if tasks is not None else TASKS
    profile_vals = (
        tuple(profiles) if profiles is not None else PROFILES
    )
    substrate_vals = (
        tuple(substrates) if substrates is not None else SUBSTRATES
    )
    seed_vals = tuple(seeds) if seeds is not None else DEFAULT_SEEDS
    for task_idx in task_vals:
        for profile in profile_vals:
            for substrate in substrate_vals:
                for seed in seed_vals:
                    yield AblationConfig(
                        task_idx=task_idx,
                        profile=profile,
                        substrate=substrate,
                        seed=seed,
                    )


def _registry_profile_tag(cfg: AblationConfig) -> str:
    """Composite profile tag baked into the R1 ``run_id``.

    Pattern : ``wave3b/task{N}/{profile}/{substrate}``. Mirrors
    ``ablation_cycle3._registry_profile_tag`` shape so the SQLite
    schema does not require a migration.
    """
    return (
        f"wave3b/task{cfg.task_idx}/{cfg.profile}/{cfg.substrate}"
    )


def compute_run_id(cfg: AblationConfig, commit_sha: str) -> str:
    """Deterministic ``run_id`` for one config — R1-compliant."""
    profile_tag = _registry_profile_tag(cfg)
    key = (
        f"{HARNESS_VERSION}|{profile_tag}|{cfg.seed}|{commit_sha}"
    ).encode()
    return hashlib.sha256(key).hexdigest()[:32]


@dataclass
class _CellRequest:
    """Minimal stand-in for ``factory.CellRequest`` so the substrate
    can pull ``seed`` / ``profile`` via ``getattr``. Avoids importing
    the heavy ``factory`` module (numpy / Norse / MLX adapters)
    when only the smoke path is exercised.
    """

    seed: int
    profile: str
    task_idx: int


def run_one_cell(
    cfg: AblationConfig,
    *,
    smoke: bool,
) -> dict[str, object]:
    """Execute one ablation cell end-to-end.

    M4 wiring :

    1. Materialise the Split-CIFAR-100 batches for ``cfg.task_idx``
       (smoke=True for the M4 path — see loader docstring).
    2. Hand the substrate a stand-in ``CellRequest`` (the substrate's
       ``execute_profile`` consumes ``seed`` + ``profile`` only at
       M3 ; the M5 wiring will pass batches through to the
       trainer).
    3. Return the metrics dict the substrate emits, augmented with
       ``task_idx``, ``cell_smoke``, and a ``loader_n_batches``
       trace (the latter proves the loader was actually consumed).

    The substrate's own ``execute_profile`` currently drives a tiny
    synthetic-latent training pass (M3 path). M5 will swap the
    substrate's internal driver to consume ``loader_batches``
    directly. M4 therefore exercises both surfaces — loader +
    substrate — without coupling them yet, which is the documented
    M4 acceptance from the plan (§4 M4 : "smoke run, 1 substrate ×
    1 task × 1 profile × 1 seed").
    """
    # Materialise the loader so we capture the batch count even
    # if the substrate ignores it. Forces FileNotFoundError early
    # if a caller invokes prod-mode at M4 (see loader docstring).
    loader_batches: list[SplitCifar100Batch] = list(
        load_split_cifar100(
            task_idx=cfg.task_idx,
            batch_size=32,
            seed=cfg.seed,
            smoke=smoke,
            split="train",
        )
    )

    # Seed the MLX global RNG **before** instantiating the
    # substrate so the denoiser / encoder ``nn.Linear`` weight
    # inits are deterministic — matches the
    # ``tests.reproducibility.test_r1_diffusion._make_models``
    # pattern. Without this, two cells with the same ``cfg.seed``
    # would diverge on the weight init alone (the R1 contract
    # tightens to "same cfg + same chip → same output hash").
    import mlx.core as mx

    mx.random.seed(cfg.seed)
    substrate = MLXLatentDiffusionSubstrate()
    request = _CellRequest(
        seed=cfg.seed, profile=cfg.profile, task_idx=cfg.task_idx
    )
    metrics = substrate.execute_profile(request)
    substrate.teardown()

    out = dict(metrics)
    out["task_idx"] = cfg.task_idx
    out["cell_smoke"] = bool(smoke)
    out["loader_n_batches"] = len(loader_batches)
    out["loader_first_batch_shape"] = (
        list(loader_batches[0].features.shape) if loader_batches else []
    )
    return out


def _r1_hash_metrics(metrics: dict[str, object]) -> str:
    """SHA-256 of the JSON-serialisable metric dict.

    Used by the smoke path to populate the R1 output-hash on the
    registry. Floats are formatted via ``repr`` for bit-stable
    serialisation — same convention as ``tests.reproducibility.
    _r1_helpers.canonical_hash`` for tensor data.
    """
    # Exclude wall_time_s : non-deterministic across machines.
    cleaned = {
        k: v
        for k, v in metrics.items()
        if k != "wall_time_s"
    }
    payload = json.dumps(
        cleaned, sort_keys=True, default=repr
    ).encode()
    return hashlib.sha256(payload).hexdigest()


def _parse_cli(argv: list[str]) -> dict[str, object]:
    """Minimal argv parser ; mirrors ``ablation_cycle3._parse_cli``."""
    opts: dict[str, object] = {
        "smoke": False,
        "resume": False,
        "dry_run": False,
        "max_runs": None,
    }

    def _next_value(flag: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise SystemExit(f"{flag} requires a value") from None

    it = iter(argv)
    for token in it:
        if token == "--smoke":
            opts["smoke"] = True
        elif token == "--resume":
            opts["resume"] = True
        elif token == "--dry-run":
            opts["dry_run"] = True
        elif token == "--max-runs":
            raw = _next_value("--max-runs")
            try:
                value = int(raw)
            except ValueError:
                raise SystemExit(
                    f"--max-runs expects an integer, got {raw!r}"
                ) from None
            if value <= 0:
                raise SystemExit(
                    f"--max-runs must be > 0, got {value}"
                )
            opts["max_runs"] = value
    return opts


def main(argv: list[str] | None = None) -> int:
    """CLI entrypoint — enumerate or smoke-run the matrix.

    Returns the process exit code (0 = success).
    """
    opts = _parse_cli(list(argv) if argv is not None else sys.argv[1:])
    commit_sha = _resolve_commit_sha()

    if opts["smoke"]:
        # 1 task × 1 profile × 1 seed = 1 cell, ≤ 10 min wall.
        configs = [
            AblationConfig(
                task_idx=0,
                profile="p_min",
                substrate=MLX_LATENT_DIFFUSION_SUBSTRATE_NAME,
                seed=0,
            )
        ]
    else:
        configs = list(enumerate_configs())

    max_runs = opts["max_runs"]
    if isinstance(max_runs, int):
        configs = configs[:max_runs]

    registry_path = Path(
        os.environ.get(
            "DREAMOFKIKI_RUN_REGISTRY",
            REPO_ROOT / ".run_registry.sqlite",
        )
    )
    registry = RunRegistry(registry_path)

    print("=" * 64)
    mode = "SMOKE" if opts["smoke"] else "ENUMERATE (M5 = exec)"
    print(f"WAVE 3b DIFFUSION ABLATION RUNNER ({mode})")
    print("=" * 64)
    print(f"harness_version : {HARNESS_VERSION}")
    print(f"commit_sha      : {commit_sha}")
    print(f"tasks           : {TASKS}")
    print(f"profiles        : {PROFILES}")
    print(f"substrates      : {SUBSTRATES}")
    print(f"seeds           : {len(DEFAULT_SEEDS)} default")
    print(f"pending configs : {len(configs)}")
    print("-" * 64)

    if opts["dry_run"]:
        print("[dry-run] no execution — exiting after enumeration.")
        return 0

    if not opts["smoke"]:
        # Prod-grid execution is M5 scope per plan §4. Register the
        # envelope only (mirrors ``ablation_cycle3.main`` line 369-372).
        for cfg in configs:
            registry.register(
                c_version=HARNESS_VERSION,
                profile=_registry_profile_tag(cfg),
                seed=cfg.seed,
                commit_sha=commit_sha,
            )
        print(
            "[plan-only] per-cell execution deferred to M5 ; "
            f"registered {len(configs)} cells in the envelope."
        )
        return 0

    # Smoke execution path.
    t0 = time.perf_counter()
    for cfg in configs:
        run_id = registry.register(
            c_version=HARNESS_VERSION,
            profile=_registry_profile_tag(cfg),
            seed=cfg.seed,
            commit_sha=commit_sha,
        )
        metrics = run_one_cell(cfg, smoke=True)
        output_hash = _r1_hash_metrics(metrics)
        registry.register_output_hash(run_id, output_hash)
        print(
            f"[smoke] cfg={asdict(cfg)} "
            f"run_id={run_id} hash={output_hash[:16]}…"
        )
    wall = time.perf_counter() - t0
    print(f"[smoke] wall_time_s={wall:.2f}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
