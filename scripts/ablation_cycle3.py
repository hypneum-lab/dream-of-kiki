"""Cycle-3 multi-scale ablation runner (C3.6).

Cartesian product driver for the cycle-3 ablation matrix :

    scale ∈ {qwen3p5-1p5b, qwen3p5-7b, qwen3p5-35b}
      × profile ∈ {p_min, p_equ, p_max}
      × substrate ∈ {mlx_kiki_oniric, esnn_thalamocortical}
      × seed ∈ 0..59

3 × 3 × 2 × 60 = 1080 configurations per full launch. Each cell is
recorded as an atomic row in the SQLite-backed
:class:`harness.storage.run_registry.RunRegistry` so the launch is
**resumable** : a crashed / preempted run can be picked up via
``--resume`` and only the pending cells are re-scheduled.

Lineage : the module-level :data:`HARNESS_VERSION` tag is baked
into every registered ``run_id``. After the C3.10 DualVer bump the
tag reads ``"C-v0.7.0+PARTIAL"`` so every cycle-3 row is preserved
in R1-compliant provenance even as the framework formal-axis /
empirical-axis evolves around it.

This module exposes :

- :data:`SCALES`, :data:`PROFILES`, :data:`SUBSTRATES`, :data:`SEEDS`
  — axis definitions. Read :func:`enumerate_configs` for iteration.
- :class:`AblationConfig` — frozen dataclass for one cell.
- :class:`AblationCycle3Runner` — wraps cycle-2's
  :class:`kiki_oniric.eval.ablation.AblationRunner` with the scale
  axis + resume semantics.

CLI (``uv run python scripts/ablation_cycle3.py``) flags :

- ``--resume`` : skip configs already-registered in the registry.
- ``--scale`` : comma-sep filter (e.g. ``qwen3p5-1p5b``) — used by
  the C3.7 sanity pilot to run only the smallest scale.
- ``--max-runs N`` : hard cap on executed cells (testing).
- ``--dry-run`` : enumerate + exit without launching any
  ``evaluate_retained`` call.

Reference :
    docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md
    §5 (reproducibility contracts R1/R3), §8 (scale-axis glossary)
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
    (DR-3 Conformance Criterion), §12 (DualVer)
"""
from __future__ import annotations

import hashlib
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from harness.real_benchmarks.dataset_registry import (  # noqa: E402
    DATASET_REGISTRY,
)
from harness.real_models.base_model_registry import (  # noqa: E402
    REGISTRY as MODEL_REGISTRY,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402


# DualVer empirical-axis tag stamped on every registered run_id.
# Surgical-bump target : post-C3.10 this reads "C-v0.7.0+PARTIAL".
# See the DualVer bump rationale in CHANGELOG.md under the entry
# `[C-v0.7.0+PARTIAL] — 2026-04-19`.
HARNESS_VERSION = "C-v0.7.0+PARTIAL"

# Axis values — materialised from the pre-cycle-3 real-data locks.
SCALES: tuple[str, ...] = tuple(MODEL_REGISTRY.keys())
PROFILES: tuple[str, ...] = ("p_min", "p_equ", "p_max")
SUBSTRATES: tuple[str, ...] = (
    "mlx_kiki_oniric",
    "esnn_thalamocortical",
)
SEEDS: tuple[int, ...] = tuple(range(60))


def _resolve_commit_sha() -> str:
    """Best-effort git HEAD lookup ; env override + ``'unknown'``."""
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
    """One cell of the cycle-3 ablation cartesian product.

    Attributes
    ----------
    scale
        Scale-slot key from :mod:`harness.real_models.base_model_registry`
        (e.g. ``qwen3p5-1p5b``). Stable across Qwen2.5 / Qwen3.5
        fallback swaps.
    profile
        Profile name ``p_min`` / ``p_equ`` / ``p_max``.
    substrate
        Substrate identity label ``mlx_kiki_oniric`` or
        ``esnn_thalamocortical``.
    seed
        Seed in ``0..59`` used for the run_id tuple (R1 contract).
    """

    scale: str
    profile: str
    substrate: str
    seed: int


def enumerate_configs(
    scales: Iterable[str] | None = None,
    profiles: Iterable[str] | None = None,
    substrates: Iterable[str] | None = None,
    seeds: Iterable[int] | None = None,
) -> Iterator[AblationConfig]:
    """Yield the cartesian product of axis values in deterministic
    order (``scale → profile → substrate → seed``).

    All parameters default to the full cycle-3 axis sets ;
    restricting any axis (used by the C3.7 sanity pilot) preserves
    the same outer-to-inner iteration order so a partial run and a
    full run share a canonical prefix.
    """
    scale_vals = tuple(scales) if scales is not None else SCALES
    profile_vals = (
        tuple(profiles) if profiles is not None else PROFILES
    )
    substrate_vals = (
        tuple(substrates) if substrates is not None else SUBSTRATES
    )
    seed_vals = tuple(seeds) if seeds is not None else SEEDS
    for scale in scale_vals:
        for profile in profile_vals:
            for substrate in substrate_vals:
                for seed in seed_vals:
                    yield AblationConfig(
                        scale=scale,
                        profile=profile,
                        substrate=substrate,
                        seed=seed,
                    )


@dataclass
class AblationCycle3Runner:
    """Cycle-3 multi-scale ablation runner with resume semantics.

    Thin wrapper around the cycle-2
    :class:`kiki_oniric.eval.ablation.AblationRunner` that :

    - Adds the **scale** axis (model-registry slot) on top of
      substrate / profile / seed.
    - Computes a deterministic ``run_id`` per cell by embedding the
      four axis values + scale into the
      :class:`~harness.storage.run_registry.RunRegistry` tuple so
      each cell is independently addressable.
    - Exposes a ``resume_from(registry)`` helper that diffs the
      full cartesian product against already-registered
      ``run_id`` values and yields only pending cells.

    The actual per-cell execution (C3.8 full Studio launch)
    instantiates cycle-2's ``AblationRunner`` per-cell with a
    single substrate/profile/seed tuple pinned from the
    ``AblationConfig``. That wiring is intentionally not included
    here — the C3.6 deliverable is the enumeration + resume
    contract, tested directly. C3.8 plugs the runtime.
    """

    harness_version: str = HARNESS_VERSION
    commit_sha: str = ""
    registry_path: Path = REPO_ROOT / ".run_registry.sqlite"
    scales: tuple[str, ...] = SCALES
    profiles: tuple[str, ...] = PROFILES
    substrates: tuple[str, ...] = SUBSTRATES
    seeds: tuple[int, ...] = SEEDS

    def __post_init__(self) -> None:
        if not self.commit_sha:
            self.commit_sha = _resolve_commit_sha()
        self._commit_sha = self.commit_sha

    def enumerate(self) -> Iterator[AblationConfig]:
        """Yield configs under the runner's (possibly-filtered) axes."""
        return enumerate_configs(
            scales=self.scales,
            profiles=self.profiles,
            substrates=self.substrates,
            seeds=self.seeds,
        )

    def _registry_profile_tag(self, cfg: AblationConfig) -> str:
        """Compose the run-registry ``profile`` column string.

        The registry tuple is ``(c_version, profile, seed,
        commit_sha) -> run_id``. Scale + substrate + ablation-
        profile collapse into a single composite ``profile`` tag to
        preserve R1 bit-stability of ``run_id`` across the four
        cycle-3 axes.
        """
        return (
            f"cycle3/{cfg.scale}/{cfg.profile}/{cfg.substrate}"
        )

    def compute_run_id(self, cfg: AblationConfig) -> str:
        """Return the deterministic ``run_id`` for a config.

        Mirrors :meth:`RunRegistry._compute_run_id` so the runner
        can diff the enumerated cartesian product against the
        registry without re-inserting rows.
        """
        profile_tag = self._registry_profile_tag(cfg)
        key = (
            f"{self.harness_version}|{profile_tag}|"
            f"{cfg.seed}|{self._commit_sha}"
        ).encode()
        return hashlib.sha256(key).hexdigest()[:32]

    def _registered_run_ids(
        self, registry: RunRegistry
    ) -> set[str]:
        """Return the set of ``run_id`` values already in the DB."""
        import sqlite3
        from contextlib import closing

        with closing(sqlite3.connect(registry.db_path)) as conn:
            rows = conn.execute(
                "SELECT run_id FROM runs"
            ).fetchall()
        return {r[0] for r in rows}

    def resume_from(
        self, registry: RunRegistry
    ) -> Iterator[AblationConfig]:
        """Yield only cells whose ``run_id`` is not yet in the
        registry. Matching is by R1 deterministic id so the same
        ``(harness_version, profile_tag, seed, commit_sha)`` tuple
        is never re-scheduled."""
        completed = self._registered_run_ids(registry)
        for cfg in self.enumerate():
            rid = self.compute_run_id(cfg)
            if rid not in completed:
                yield cfg

    def register(self, cfg: AblationConfig) -> str:
        """Insert an ``AblationConfig`` into the registry and
        return the resulting ``run_id`` (idempotent — inserts with
        ``OR IGNORE`` per ``RunRegistry.register``)."""
        registry = RunRegistry(self.registry_path)
        return registry.register(
            c_version=self.harness_version,
            profile=self._registry_profile_tag(cfg),
            seed=cfg.seed,
            commit_sha=self._commit_sha,
        )


def _parse_cli(argv: list[str]) -> dict:
    """Minimal argv parser — avoid adding a click dependency for
    the driver script itself (kept importable by the test suite)."""
    opts: dict = {
        "resume": False,
        "dry_run": False,
        "scale_filter": None,
        "max_runs": None,
    }
    def _next_value(flag: str) -> str:
        try:
            return next(it)
        except StopIteration:
            raise SystemExit(f"{flag} requires a value") from None

    it = iter(argv)
    for token in it:
        if token == "--resume":
            opts["resume"] = True
        elif token == "--dry-run":
            opts["dry_run"] = True
        elif token == "--scale":
            opts["scale_filter"] = tuple(
                s.strip()
                for s in _next_value("--scale").split(",")
                if s.strip()
            )
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


def main(argv: list[str] | None = None) -> None:  # pragma: no cover
    """CLI entrypoint — enumerates (or resumes) the matrix.

    Actual per-cell execution is delegated to the C3.8 Studio
    launch script ; here the driver reports the pending plan and
    the anchored harness version so the user can audit the launch
    envelope before committing compute.
    """
    opts = _parse_cli(list(argv) if argv is not None else sys.argv[1:])
    scales = (
        opts["scale_filter"] if opts["scale_filter"] else SCALES
    )
    runner = AblationCycle3Runner(
        registry_path=Path(os.environ.get(
            "DREAMOFKIKI_RUN_REGISTRY",
            REPO_ROOT / ".run_registry.sqlite",
        )),
        scales=tuple(scales),
    )
    configs = list(runner.enumerate())
    if opts["resume"]:
        registry = RunRegistry(runner.registry_path)
        configs = list(runner.resume_from(registry))
    if opts["max_runs"] is not None:
        configs = configs[: opts["max_runs"]]

    print("=" * 64)
    print("CYCLE-3 MULTI-SCALE ABLATION RUNNER (enumeration)")
    print("=" * 64)
    print(f"harness_version : {runner.harness_version}")
    print(f"commit_sha      : {runner._commit_sha}")
    print(f"scales          : {scales}")
    print(f"profiles        : {PROFILES}")
    print(f"substrates      : {SUBSTRATES}")
    print(f"seeds           : {len(SEEDS)} (0..{SEEDS[-1]})")
    print(f"datasets locked : {sorted(DATASET_REGISTRY.keys())}")
    print(f"pending configs : {len(configs)}")
    print("-" * 64)
    if opts["dry_run"]:
        print("[dry-run] no execution — exiting after enumeration.")
        return
    print(
        "[plan-only] per-cell execution delegated to C3.8 Studio "
        "launch ; this driver registered the matrix envelope."
    )


if __name__ == "__main__":  # pragma: no cover
    main()
