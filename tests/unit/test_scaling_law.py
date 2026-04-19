"""Unit tests for H5 multi-scale scaling-law module (cycle-3 C3.4).

H5 is the cycle-3 multi-scale hypothesis, split into three pre-registered
variants per the OSF pre-registration lock of 2026-04-19 :

- H5-I  invariance  : one-way ANOVA F-test on per-scale effect-size
                       means (rejects when at least one scale mean
                       differs, i.e. between-scale variance > 0).
- H5-II monotonic   : Spearman ρ on (N_params, effect) correlation,
                       two-sided (no post-hoc direction claim).
- H5-III power-law  : bootstrap CI on α in d = c * N^α.

Tests consume synthetic Cohen's d effect-size arrays produced by the
multi-scale ablation runs (C3.3 adapter). Family-wise α_per_test is
0.00625 = 0.05 / 8 for the cycle-3 combined 8-test family.
"""
from __future__ import annotations

import numpy as np

from kiki_oniric.eval.scaling_law import (
    H5Results,
    HypothesisResult,
    PowerLawResult,
    compute_h5,
    h5_invariance,
    h5_monotonic,
    h5_power_law,
)


def _make_effects(
    means: list[float],
    std: float = 0.05,
    n_per_scale: int = 60,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Deterministic Gaussian effect arrays keyed by scale label."""
    rng = np.random.default_rng(seed)
    labels = ["1.5B", "7B", "35B"]
    return {
        label: rng.normal(loc=mu, scale=std, size=n_per_scale)
        for label, mu in zip(labels, means, strict=True)
    }


def test_h5_invariance_returns_hypothesis_result() -> None:
    """H5-I — ANOVA returns a typed HypothesisResult with a p-value."""
    effects = _make_effects([0.3, 0.3, 0.3], std=0.05, seed=1)
    result = h5_invariance(effects, alpha=0.00625)
    assert isinstance(result, HypothesisResult)
    assert 0.0 <= result.p_value <= 1.0
    assert isinstance(result.reject_null, bool)
    assert np.isfinite(result.statistic)


def test_h5_invariance_rejects_null_with_variance() -> None:
    """H5-I — effects with strongly different means → reject invariance."""
    effects = _make_effects([0.1, 0.5, 1.0], std=0.05, seed=2)
    result = h5_invariance(effects, alpha=0.00625)
    assert result.reject_null is True
    assert result.p_value < 0.00625


def test_h5_monotonic_returns_spearman_result() -> None:
    """H5-II — returns (ρ, p-value) two-sided in a HypothesisResult."""
    scales = [1.5e9, 7e9, 35e9]
    effects = _make_effects([0.3, 0.3, 0.3], std=0.05, seed=3)
    result = h5_monotonic(scales, effects, alpha=0.00625)
    assert isinstance(result, HypothesisResult)
    assert -1.0 <= result.statistic <= 1.0
    assert 0.0 <= result.p_value <= 1.0


def test_h5_monotonic_detects_trend() -> None:
    """H5-II — strictly increasing means by scale → ρ ≈ 1 and reject null."""
    scales = [1.5e9, 7e9, 35e9]
    effects = _make_effects([0.1, 0.5, 1.0], std=0.05, seed=4)
    result = h5_monotonic(scales, effects, alpha=0.00625)
    assert result.reject_null is True
    assert result.p_value < 0.00625
    # Two-sided ρ on an increasing pattern of medians is strongly positive.
    assert result.statistic > 0.3


def test_h5_power_law_returns_power_law_result() -> None:
    """H5-III — exposes α, CI95 bounds, c coefficient, reject_null."""
    scales = [1.5e9, 7e9, 35e9]
    effects = _make_effects([0.1, 0.2, 0.3], std=0.02, seed=5)
    result = h5_power_law(
        scales, effects, n_bootstrap=200, seed=5, alpha=0.00625
    )
    assert isinstance(result, PowerLawResult)
    assert np.isfinite(result.alpha)
    assert result.ci95_low <= result.alpha <= result.ci95_high
    assert np.isfinite(result.c_coef)
    assert isinstance(result.reject_null, bool)


def test_h5_power_law_recovers_alpha() -> None:
    """H5-III — injected d = 0.1 * N^0.3 → recovered α within CI95."""
    scales = [1.5e9, 7e9, 35e9]
    true_c = 0.1
    true_alpha = 0.3
    rng = np.random.default_rng(42)
    effects: dict[str, np.ndarray] = {}
    labels = ["1.5B", "7B", "35B"]
    for label, N in zip(labels, scales, strict=True):
        center = true_c * (N ** true_alpha)
        # Tight noise relative to the signal so the bootstrap is stable.
        effects[label] = rng.normal(
            loc=center, scale=max(center * 0.02, 1e-9), size=120
        )
    result = h5_power_law(
        scales, effects, n_bootstrap=500, seed=42, alpha=0.00625
    )
    assert result.ci95_low <= true_alpha <= result.ci95_high
    assert abs(result.alpha - true_alpha) < 0.05
    # A non-zero α with a narrow CI should also reject the 0 ∉ CI null.
    assert result.reject_null is True


def test_h5_power_law_bootstrap_is_reproducible() -> None:
    """H5-III — identical seed and inputs → identical α and CI bounds."""
    scales = [1.5e9, 7e9, 35e9]
    effects = _make_effects([0.12, 0.18, 0.25], std=0.02, seed=7)
    a = h5_power_law(scales, effects, n_bootstrap=300, seed=11)
    b = h5_power_law(scales, effects, n_bootstrap=300, seed=11)
    assert a.alpha == b.alpha
    assert a.ci95_low == b.ci95_low
    assert a.ci95_high == b.ci95_high


def test_compute_h5_consolidates_three_variants() -> None:
    """compute_h5 — wraps all three variants + Bonferroni family α=0.00625."""
    scales = [1.5e9, 7e9, 35e9]
    effects = _make_effects([0.1, 0.5, 1.0], std=0.05, seed=9)
    results = compute_h5(
        scales, effects, alpha_family=0.00625, n_bootstrap=200, seed=9
    )
    assert isinstance(results, H5Results)
    assert isinstance(results.invariance, HypothesisResult)
    assert isinstance(results.monotonic, HypothesisResult)
    assert isinstance(results.power_law, PowerLawResult)
    assert results.family_alpha == 0.00625
    assert isinstance(results.any_significant, bool)
    # Strong trivariant signal → at least one variant must reject null.
    assert results.any_significant is True
