"""Unit tests for cycle-3 multi-scale ablation runner (C3.6).

Validates :

1. ``enumerate_configs()`` yields the full cartesian product
   ``scale × profile × substrate × seed`` in deterministic order
   with exactly ``3 × 3 × 3 × 60 = 1620`` tuples (Qwen2.5-Q4
   scale axis × 3 substrates post-micro_kiki wiring).
2. ``resume_from(registry)`` skips already-completed configs by
   matching ``run_id`` in a provided ``RunRegistry``.
3. Every config's ``run_id`` lineage tracks the current
   ``HARNESS_VERSION`` string so post-C3.10 bump runs log
   ``C-v0.7.0+PARTIAL`` from the start (R1 provenance).

Reference :
    docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md
    §5 (R1 reproducibility) + §8 (scale-axis glossary)
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §12
    (DualVer)
"""
from __future__ import annotations

from pathlib import Path


from harness.storage.run_registry import RunRegistry
from scripts import ablation_cycle3 as runner_mod
from scripts.ablation_cycle3 import (
    AblationCycle3Runner,
    AblationConfig,
    enumerate_configs,
)


def test_enumerate_configs_yields_1620_tuples() -> None:
    """Full cartesian product : 3 scales × 3 profiles × 3 substrates
    × 60 seeds = 1620 unique ``AblationConfig`` entries (post
    micro_kiki wiring), in deterministic iteration order.

    Restricted to the Qwen2.5-Q4 scale axis — fp16 + local
    exploratory slots are excluded from the full matrix contract
    (they live in the registry but are outside the 3-scale lock
    per ``SCALES`` filter in the runner module)."""
    configs = list(
        enumerate_configs(
            scales=("qwen3p5-1p5b", "qwen3p5-7b", "qwen3p5-35b"),
        )
    )
    assert len(configs) == 1620
    # Uniqueness on the four-axis tuple
    unique_keys = {
        (c.scale, c.profile, c.substrate, c.seed) for c in configs
    }
    assert len(unique_keys) == 1620
    # Axis coverage
    assert {c.scale for c in configs} == {
        "qwen3p5-1p5b", "qwen3p5-7b", "qwen3p5-35b",
    }
    assert {c.profile for c in configs} == {"p_min", "p_equ", "p_max"}
    assert {c.substrate for c in configs} == {
        "mlx_kiki_oniric", "esnn_thalamocortical", "micro_kiki",
    }
    assert {c.seed for c in configs} == set(range(60))
    # Determinism : two back-to-back enumerations match exactly
    second = list(
        enumerate_configs(
            scales=("qwen3p5-1p5b", "qwen3p5-7b", "qwen3p5-35b"),
        )
    )
    assert [
        (c.scale, c.profile, c.substrate, c.seed) for c in configs
    ] == [
        (c.scale, c.profile, c.substrate, c.seed) for c in second
    ]


def test_resume_from_registry_skips_completed_configs(
    tmp_path: Path,
) -> None:
    """Pre-register run_ids for a subset of configs, then confirm
    the runner's resume helper returns only the pending remainder
    (identity by ``run_id`` per R1)."""
    registry_path = tmp_path / "reg.sqlite"
    registry = RunRegistry(registry_path)
    arunner = AblationCycle3Runner(registry_path=registry_path)
    all_configs = list(arunner.enumerate())
    # Mark the first 3 configs as completed by pre-registering them.
    for cfg in all_configs[:3]:
        registry.register(
            c_version=arunner.harness_version,
            profile=arunner._registry_profile_tag(cfg),
            seed=cfg.seed,
            commit_sha=arunner._commit_sha,
        )
    pending = list(arunner.resume_from(registry))
    # 1080 - 3 = 1077 configs remain
    assert len(pending) == len(all_configs) - 3
    completed_keys = {
        (c.scale, c.profile, c.substrate, c.seed)
        for c in all_configs[:3]
    }
    pending_keys = {
        (c.scale, c.profile, c.substrate, c.seed) for c in pending
    }
    assert completed_keys.isdisjoint(pending_keys)


def test_run_id_lineage_tracks_harness_version(
    tmp_path: Path,
) -> None:
    """Every config's ``run_id`` is derived from
    ``(harness_version, registry_profile_tag, seed, commit_sha)``.
    Changing ``harness_version`` changes the run_id — so post-C3.10
    runs logged under ``C-v0.7.0+PARTIAL`` are never confused with
    pre-bump runs under ``C-v0.6.0+STABLE``."""
    cfg = AblationConfig(
        scale="qwen3p5-1p5b",
        profile="p_min",
        substrate="mlx_kiki_oniric",
        seed=0,
    )
    runner_a = AblationCycle3Runner(
        harness_version="C-v0.7.0+PARTIAL",
        commit_sha="deadbeef",
        registry_path=tmp_path / "reg_a.sqlite",
    )
    runner_b = AblationCycle3Runner(
        harness_version="C-v0.6.0+STABLE",
        commit_sha="deadbeef",
        registry_path=tmp_path / "reg_b.sqlite",
    )
    run_id_a = runner_a.compute_run_id(cfg)
    run_id_b = runner_b.compute_run_id(cfg)
    # Both must be 32-hex (R1 128-bit slice) and must differ across
    # harness_version values : provenance is preserved.
    assert len(run_id_a) == 32
    assert len(run_id_b) == 32
    assert run_id_a != run_id_b
    # Current module-level constant post-bump reads the target
    # harness version at import time.
    assert runner_mod.HARNESS_VERSION.startswith("C-v")
