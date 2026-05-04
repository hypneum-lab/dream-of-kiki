"""Unit tests for the live-tensor dream episode runner.

The four MicroKikiSubstrate handlers MUST mutate the live LoRA
delta dict (passed by reference). This is the load-bearing fix vs
the G6 Path B spectator pattern documented at
``experiments/g6_mmlu_stream/run_g6.py:539-578``.

Tests run against the substrate's pure-numpy stub mode (no MLX
required) so they exercise on Linux CI.

Reference :
- Plan : ``docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md`` Task 4 step 1
"""
from __future__ import annotations

import numpy as np

from experiments.g6_studio_path_a.dream_episode_real import (
    DEFAULT_PRIMARY_KEY,
    PROFILE_OPS_REAL,
    PROFILE_RECOMBINE_ALPHA,
    dream_episode_real,
)
from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate


def _delta(out_dim: int, rank: int, seed: int) -> dict[str, np.ndarray]:
    """Build a deterministic LoRA delta dict for tests."""
    rng = np.random.default_rng(seed)
    return {
        DEFAULT_PRIMARY_KEY: rng.standard_normal(
            (out_dim, rank),
        ).astype(np.float32),
    }


def test_baseline_no_mutation() -> None:
    """TDD-4.1 — baseline arm leaves live_delta unchanged."""
    sub = MicroKikiSubstrate(num_layers=4, rank=8, seed=0)
    delta = _delta(8, 8, seed=0)
    before = delta[DEFAULT_PRIMARY_KEY].copy()

    out = dream_episode_real(
        substrate=sub,
        profile="baseline",
        live_delta=delta,
        seed=0,
        subdomain="anatomy",
        prior_deltas=[],
        sibling_deltas=[],
    )
    assert np.array_equal(out[DEFAULT_PRIMARY_KEY], before)
    assert out is delta  # same dict, in-place semantics preserved


def test_p_min_downscales_live_delta() -> None:
    """TDD-4.2 — P_min applies shrink_factor < 1 in place."""
    sub = MicroKikiSubstrate(num_layers=4, rank=8, seed=0)
    delta = _delta(8, 8, seed=0)
    before_norm = float(np.linalg.norm(delta[DEFAULT_PRIMARY_KEY]))

    dream_episode_real(
        substrate=sub,
        profile="P_min",
        live_delta=delta,
        seed=0,
        subdomain="anatomy",
        prior_deltas=[],
        sibling_deltas=[],
    )
    after_norm = float(np.linalg.norm(delta[DEFAULT_PRIMARY_KEY]))
    assert 0 < after_norm < before_norm


def test_p_equ_mutates_via_all_four_handlers() -> None:
    """TDD-4.3 — P_equ runs all 4 ops ; delta mutated AND DR-0 stamps land."""
    sub = MicroKikiSubstrate(num_layers=4, rank=8, seed=0)
    delta = _delta(8, 8, seed=0)
    before = delta[DEFAULT_PRIMARY_KEY].copy()
    rng = np.random.default_rng(42)
    priors = [
        rng.standard_normal((8, 4)).astype(np.float32),
    ]
    siblings = [
        rng.standard_normal((8, 8)).astype(np.float32) for _ in range(2)
    ]

    dream_episode_real(
        substrate=sub,
        profile="P_equ",
        live_delta=delta,
        seed=0,
        subdomain="anatomy",
        prior_deltas=priors,
        sibling_deltas=siblings,
    )
    assert not np.array_equal(delta[DEFAULT_PRIMARY_KEY], before)
    assert sub.restructure_state.total_episodes_handled == 1
    assert sub.recombine_state.total_episodes_handled == 1
    assert (
        sub.restructure_state.last_episode_id
        == "g6s-P_equ-anatomy-seed0"
    )
    assert (
        sub.recombine_state.last_episode_id
        == "g6s-P_equ-anatomy-seed0"
    )


def test_p_max_recombine_alpha_amplified() -> None:
    """TDD-4.4 — P_max uses alpha=2.0 (vs P_equ's 1.0)."""
    assert PROFILE_RECOMBINE_ALPHA["P_max"] == 2.0
    assert PROFILE_RECOMBINE_ALPHA["P_equ"] == 1.0


def test_profile_ops_table_exhaustive() -> None:
    """TDD-4.5 — PROFILE_OPS_REAL covers all 4 arms."""
    assert set(PROFILE_OPS_REAL) == {
        "baseline", "P_min", "P_equ", "P_max",
    }
    assert PROFILE_OPS_REAL["baseline"] == ()
    assert "replay" in PROFILE_OPS_REAL["P_min"]
    assert "downscale" in PROFILE_OPS_REAL["P_min"]
    assert "restructure" in PROFILE_OPS_REAL["P_equ"]
    assert "recombine" in PROFILE_OPS_REAL["P_equ"]
    assert PROFILE_OPS_REAL["P_max"] == PROFILE_OPS_REAL["P_equ"]


def test_unknown_profile_raises() -> None:
    """TDD-4.6 — unknown profile raises ValueError."""
    import pytest

    sub = MicroKikiSubstrate(num_layers=4, rank=8, seed=0)
    delta = _delta(8, 8, seed=0)

    with pytest.raises(ValueError, match="unknown profile"):
        dream_episode_real(
            substrate=sub,
            profile="P_bogus",
            live_delta=delta,
            seed=0,
            subdomain="anatomy",
            prior_deltas=[],
            sibling_deltas=[],
        )


def test_missing_primary_key_raises_when_ops_active() -> None:
    """TDD-4.7 — non-empty op list + missing primary_key raises KeyError."""
    import pytest

    sub = MicroKikiSubstrate(num_layers=4, rank=8, seed=0)
    delta: dict[str, np.ndarray] = {}

    with pytest.raises(KeyError, match="primary_key"):
        dream_episode_real(
            substrate=sub,
            profile="P_min",
            live_delta=delta,
            seed=0,
            subdomain="anatomy",
            prior_deltas=[],
            sibling_deltas=[],
        )


def test_baseline_does_not_require_primary_key() -> None:
    """TDD-4.8 — baseline arm with empty live_delta returns it unchanged."""
    sub = MicroKikiSubstrate(num_layers=4, rank=8, seed=0)
    delta: dict[str, np.ndarray] = {}

    out = dream_episode_real(
        substrate=sub,
        profile="baseline",
        live_delta=delta,
        seed=0,
        subdomain="anatomy",
        prior_deltas=[],
        sibling_deltas=[],
    )
    assert out == {}
