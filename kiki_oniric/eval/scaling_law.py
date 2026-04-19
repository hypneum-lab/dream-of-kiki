"""H5 multi-scale scaling-law module — trivariant hypothesis test.

Per the OSF pre-registration lock of 2026-04-19 the cycle-3 H5
hypothesis is decomposed into three pre-specified variants :

- H5-I  invariance  : one-way ANOVA on across-scale effect means.
                       H0 : all per-scale effect means are equal
                       (effect is scale-invariant — a non-zero
                       between-scale variance component implies
                       at least one scale mean differs).
- H5-II monotonic   : Spearman ρ on (N_params, d) correlation.
                       Two-sided per pre-reg — no post-hoc direction claim.
- H5-III power-law  : bootstrap 95 % CI on α in the fit d = c * N^α.

`compute_h5` consolidates the three variants under a Bonferroni
family-wise α_per_test = 0.05 / 8 = 0.00625 for the cycle-3
combined 8-test family {H1, H2, H3, H4, H5-I, H5-II, H5-III, H6}.

Input shape : `effects_by_scale` is a mapping from scale label
(e.g. ``"1.5B"``) to a 1-D numpy array of per-item Cohen's d
effect sizes at that scale. Keys iterate in the same order as
`scales_params` (e.g. ``[1.5e9, 7e9, 35e9]``).

Reference : docs/specs/2026-04-17-dreamofkiki-master-design.md §5.4
and cycle-3 atomic plan §C3.4.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats
from scipy.optimize import curve_fit


@dataclass(frozen=True)
class HypothesisResult:
    """Uniform result for the H5-I / H5-II scalar hypothesis tests."""

    statistic: float
    p_value: float
    reject_null: bool
    details: str = ""


@dataclass(frozen=True)
class PowerLawResult:
    """Result of the H5-III bootstrap fit d = c * N^α."""

    alpha: float
    ci95_low: float
    ci95_high: float
    reject_null: bool  # True iff 0 ∉ CI95 on α
    c_coef: float


@dataclass(frozen=True)
class H5Results:
    """Consolidated H5 trivariant result under Bonferroni family α."""

    invariance: HypothesisResult
    monotonic: HypothesisResult
    power_law: PowerLawResult
    any_significant: bool
    family_alpha: float


def _ordered_effects(
    effects_by_scale: dict[str, np.ndarray],
) -> list[np.ndarray]:
    """Preserve insertion order and cast to float arrays."""
    return [
        np.asarray(effects_by_scale[k], dtype=float)
        for k in effects_by_scale
    ]


def h5_invariance(
    effects_by_scale: dict[str, np.ndarray],
    alpha: float = 0.00625,
) -> HypothesisResult:
    """H5-I — variance test on across-scale effect means.

    H0 : σ²(d_scales) = 0 (the effect is scale-invariant, i.e. the
    per-scale mean Cohen's d vector has zero variance).
    Rejecting H0 means the between-scale variance component is
    non-zero — the effect is NOT scale-invariant.

    Implementation : one-way ANOVA F-test on the effect-size arrays
    grouped by scale. The F-statistic is exactly the ratio of
    between-scale variance to within-scale variance, so the test's
    null and σ²(d_scales) = 0 coincide.
    """
    arrays = _ordered_effects(effects_by_scale)
    if len(arrays) < 2:
        raise ValueError(
            "ANOVA requires at least 2 scale groups, "
            f"got {len(arrays)}"
        )
    if any(arr.size == 0 for arr in arrays):
        raise ValueError(
            "All scale groups must contain at least one observation"
        )
    statistic, p_value = stats.f_oneway(*arrays)
    return HypothesisResult(
        statistic=float(statistic),
        p_value=float(p_value),
        reject_null=bool(p_value < alpha),
        details=f"one-way ANOVA on {len(arrays)} scale groups",
    )


def h5_monotonic(
    scales_params: list[float],
    effects_by_scale: dict[str, np.ndarray],
    alpha: float = 0.00625,
) -> HypothesisResult:
    """H5-II — Spearman ρ on the (N_params, d) correlation.

    Two-sided per the OSF pre-registration locked 2026-04-19 — we
    make no post-hoc directional claim about the sign of ρ.
    """
    arrays = _ordered_effects(effects_by_scale)
    if len(arrays) != len(scales_params):
        raise ValueError(
            "scales_params length must match number of effect arrays"
        )
    # Expand (N, d_i) pairs : every observation d_i inherits its
    # scale's parameter count so Spearman's rank correlation is
    # evaluated on the full observation-level sample, not just the
    # three scale means.
    xs: list[float] = []
    ys: list[float] = []
    for N, arr in zip(scales_params, arrays, strict=True):
        xs.extend([float(N)] * len(arr))
        ys.extend(arr.tolist())
    if not ys:
        # All scale groups empty — Spearman is undefined. Return a
        # NaN-filled HypothesisResult so callers can see the failure
        # without the underlying SciPy warning/exception leaking out.
        import math

        return HypothesisResult(
            statistic=math.nan,
            p_value=math.nan,
            reject_null=False,
            details="Spearman ρ two-sided — empty input",
        )
    result = stats.spearmanr(xs, ys, alternative="two-sided")
    rho = float(result.statistic)
    p_value = float(result.pvalue)
    return HypothesisResult(
        statistic=rho,
        p_value=p_value,
        reject_null=bool(p_value < alpha),
        details="Spearman ρ two-sided (N_params, d)",
    )


def _power_law_model(n: np.ndarray, c: float, alpha: float) -> np.ndarray:
    """d = c * N^α — the cycle-3 scaling-law ansatz."""
    return c * np.power(n, alpha)


def _fit_power_law(
    scales: np.ndarray, means: np.ndarray
) -> tuple[float, float]:
    """Fit d = c * N^α on (scale, mean-effect) pairs.

    Falls back to a log-linear regression when curve_fit cannot
    converge (e.g. degenerate flat effects) — this keeps the
    bootstrap stable for near-zero signal.
    """
    # Guard against empty / all-NaN means — curve_fit and polyfit
    # would otherwise raise an opaque IndexError / LinAlgError.
    if means.size == 0 or not np.any(np.isfinite(means)):
        raise ValueError(
            "_fit_power_law requires at least one finite effect mean"
        )
    try:
        p0 = (float(means[0]), 0.1)
        popt, _ = curve_fit(_power_law_model, scales, means, p0=p0, maxfev=5000)
        c_hat, alpha_hat = float(popt[0]), float(popt[1])
        if np.isfinite(c_hat) and np.isfinite(alpha_hat):
            return c_hat, alpha_hat
    except (RuntimeError, ValueError):
        pass
    # Log-linear fallback : log d = log c + α log N.
    log_n = np.log(scales)
    # Guard against non-positive means — Levene/Spearman still make
    # sense but a power-law fit requires positive effect medians.
    positive_means = np.where(means > 0, means, 1e-12)
    log_d = np.log(positive_means)
    alpha_hat, log_c_hat = np.polyfit(log_n, log_d, 1)
    return float(np.exp(log_c_hat)), float(alpha_hat)


def h5_power_law(
    scales_params: list[float],
    effects_by_scale: dict[str, np.ndarray],
    n_bootstrap: int = 1000,
    seed: int = 0,
    alpha: float = 0.00625,
) -> PowerLawResult:
    """H5-III — bootstrap 95 % CI on α in d = c * N^α.

    With only three scales the asymptotic power is limited ; the
    honest read is the bootstrap CI, not the scalar p-value.
    `reject_null=True` iff 0 ∉ CI95 on α. The ``alpha`` argument is
    kept for family-wise API symmetry but is unused for CI width
    (which is fixed at 95 %).
    """
    del alpha  # CI width is fixed by the pre-registration at 95 %.
    arrays = _ordered_effects(effects_by_scale)
    if len(arrays) != len(scales_params):
        raise ValueError(
            "scales_params length must match number of effect arrays"
        )
    scales = np.asarray(scales_params, dtype=float)

    means = np.array([arr.mean() for arr in arrays], dtype=float)
    c_hat, alpha_hat = _fit_power_law(scales, means)

    rng = np.random.default_rng(seed)
    boot_alphas = np.empty(n_bootstrap, dtype=float)
    for i in range(n_bootstrap):
        resampled_means = np.empty_like(means)
        for k, arr in enumerate(arrays):
            idx = rng.integers(0, len(arr), size=len(arr))
            resampled_means[k] = float(arr[idx].mean())
        _, a = _fit_power_law(scales, resampled_means)
        boot_alphas[i] = a
    ci_low = float(np.quantile(boot_alphas, 0.025))
    ci_high = float(np.quantile(boot_alphas, 0.975))
    return PowerLawResult(
        alpha=float(alpha_hat),
        ci95_low=ci_low,
        ci95_high=ci_high,
        reject_null=bool(ci_low > 0.0 or ci_high < 0.0),
        c_coef=float(c_hat),
    )


def compute_h5(
    scales_params: list[float],
    effects_by_scale: dict[str, np.ndarray],
    alpha_family: float = 0.00625,
    n_bootstrap: int = 1000,
    seed: int = 0,
) -> H5Results:
    """Consolidate H5-I / H5-II / H5-III under Bonferroni family α.

    `any_significant` is True iff at least one of the three variants
    rejects its null at the family-wise α_per_test.
    """
    invariance = h5_invariance(effects_by_scale, alpha=alpha_family)
    monotonic = h5_monotonic(
        scales_params, effects_by_scale, alpha=alpha_family
    )
    power_law = h5_power_law(
        scales_params,
        effects_by_scale,
        n_bootstrap=n_bootstrap,
        seed=seed,
        alpha=alpha_family,
    )
    any_sig = (
        invariance.reject_null
        or monotonic.reject_null
        or power_law.reject_null
    )
    return H5Results(
        invariance=invariance,
        monotonic=monotonic,
        power_law=power_law,
        any_significant=bool(any_sig),
        family_alpha=float(alpha_family),
    )
