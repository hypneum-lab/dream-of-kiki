"""Statistical eval module — wraps scipy.stats for H1-H6 hypotheses.

Per OSF pre-registration (docs/osf-preregistration-draft.md):
- H1 Welch's t-test (one-sided): treatment improvement vs control
- H2 TOST equivalence (bidirectional): treatment within ±epsilon
- H3 Jonckheere-Terpstra: monotonic trend across ordered groups
- H4 one-sample t-test (upper bound): sample mean below threshold
- H5-I/II/III scaling law trivariant (see kiki_oniric.eval.scaling_law)
- H6 cross-substrate effect retention (cycle-3 extension)

All H1-H4 tests return a StatTestResult with .reject_h0, .p_value,
and .test_name for uniform downstream handling. The BonferroniFamily
helper at the bottom of this module exposes family-wise α correction
for both the cycle-1 baseline (family_size=4, α=0.0125) and the
cycle-3 combined 8-test family (family_size=8, α=0.00625).

Reference: docs/specs/2026-04-17-dreamofkiki-master-design.md §5.4
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats


@dataclass(frozen=True)
class StatTestResult:
    """Uniform result type for all H1-H4 hypothesis tests."""

    test_name: str
    p_value: float
    reject_h0: bool
    statistic: float | None = None


def welch_one_sided(
    treatment: list[float],
    control: list[float],
    alpha: float = 0.05,
) -> StatTestResult:
    """H1: Welch's t-test, one-sided (treatment < control).

    H0: mean(treatment) >= mean(control)
    H1: mean(treatment) < mean(control) — i.e. treatment improves
    (lower is better, e.g. forgetting rate).
    """
    t_arr = np.asarray(treatment, dtype=float)
    c_arr = np.asarray(control, dtype=float)
    res = stats.ttest_ind(t_arr, c_arr, equal_var=False)
    # Convert two-sided p to one-sided (treatment < control)
    if res.statistic < 0:
        p_one_sided = res.pvalue / 2
    else:
        p_one_sided = 1.0 - res.pvalue / 2
    return StatTestResult(
        test_name="Welch's t-test (one-sided)",
        p_value=float(p_one_sided),
        reject_h0=bool(p_one_sided < alpha),
        statistic=float(res.statistic),
    )


def tost_equivalence(
    treatment: list[float],
    control: list[float],
    epsilon: float,
    alpha: float = 0.05,
) -> StatTestResult:
    """H2: Two One-Sided Tests (TOST) for equivalence.

    H0: |mean(treatment) - mean(control)| >= epsilon (not equivalent)
    H1: |mean(treatment) - mean(control)| < epsilon (equivalent)

    Returns reject_h0=True when both one-sided tests pass at alpha.
    """
    t_arr = np.asarray(treatment, dtype=float)
    c_arr = np.asarray(control, dtype=float)
    diff_mean = float(t_arr.mean() - c_arr.mean())
    pooled_se = float(
        np.sqrt(t_arr.var(ddof=1) / len(t_arr)
                + c_arr.var(ddof=1) / len(c_arr))
    )
    df = len(t_arr) + len(c_arr) - 2  # rough Welch-Satterthwaite floor
    # Zero-variance guard : when both samples are constant
    # (pooled_se == 0), the t-statistic is undefined. Treat
    # equivalence as accepted iff the observed mean diff is
    # strictly inside the equivalence interval, otherwise reject.
    if np.isclose(pooled_se, 0.0):
        is_equivalent = abs(diff_mean) < epsilon
        return StatTestResult(
            test_name="TOST equivalence",
            p_value=0.0 if is_equivalent else 1.0,
            reject_h0=bool(is_equivalent),
            statistic=diff_mean,
        )
    # Lower bound test: H0_lower: diff <= -epsilon
    t_lower = (diff_mean - (-epsilon)) / pooled_se
    p_lower = 1.0 - stats.t.cdf(t_lower, df)
    # Upper bound test: H0_upper: diff >= epsilon
    t_upper = (diff_mean - epsilon) / pooled_se
    p_upper = stats.t.cdf(t_upper, df)
    p_tost = max(p_lower, p_upper)  # TOST: max p-value
    return StatTestResult(
        test_name="TOST equivalence",
        p_value=float(p_tost),
        reject_h0=bool(p_tost < alpha),
        statistic=diff_mean,
    )


def jonckheere_trend(
    groups: list[list[float]],
    alpha: float = 0.05,
) -> StatTestResult:
    """H3: Jonckheere-Terpstra monotonic trend test.

    H0: no ordered trend across groups
    H1: groups in increasing order (group_i < group_{i+1})

    Implementation: sum of Mann-Whitney U over ordered pairs,
    z-approx for p-value. Standard non-parametric trend test.
    """
    arrs = [np.asarray(g, dtype=float) for g in groups]
    n = sum(len(a) for a in arrs)
    j_stat = 0.0
    for i in range(len(arrs)):
        for j in range(i + 1, len(arrs)):
            # Count pairs (x in arrs[i], y in arrs[j]) with x < y
            count = sum(
                1 for x in arrs[i] for y in arrs[j] if x < y
            )
            ties = sum(
                0.5 for x in arrs[i] for y in arrs[j] if x == y
            )
            j_stat += count + ties
    # Mean and variance of J under H0
    sizes = [len(a) for a in arrs]
    mean_j = (
        n ** 2 - sum(ni ** 2 for ni in sizes)
    ) / 4.0
    var_j = (
        n ** 2 * (2 * n + 3)
        - sum(ni ** 2 * (2 * ni + 3) for ni in sizes)
    ) / 72.0
    z = (j_stat - mean_j) / np.sqrt(var_j) if var_j > 0 else 0.0
    p_one_sided = 1.0 - stats.norm.cdf(z)
    return StatTestResult(
        test_name="Jonckheere-Terpstra trend",
        p_value=float(p_one_sided),
        reject_h0=bool(p_one_sided < alpha),
        statistic=float(j_stat),
    )


def one_sample_threshold(
    sample: list[float],
    threshold: float,
    alpha: float = 0.05,
) -> StatTestResult:
    """H4: one-sample t-test against upper bound threshold.

    H0: mean(sample) >= threshold (violates budget)
    H1: mean(sample) < threshold (within budget)

    Returns reject_h0=True when sample mean is significantly below
    threshold.
    """
    arr = np.asarray(sample, dtype=float)
    res = stats.ttest_1samp(arr, popmean=threshold)
    # Two-sided p; convert to one-sided (mean < threshold)
    if res.statistic < 0:
        p_one_sided = res.pvalue / 2
    else:
        p_one_sided = 1.0 - res.pvalue / 2
    return StatTestResult(
        test_name="one-sample t-test (upper bound)",
        p_value=float(p_one_sided),
        reject_h0=bool(p_one_sided < alpha),
        statistic=float(res.statistic),
    )


@dataclass(frozen=True)
class BonferroniFamily:
    """Bonferroni family-wise α correction for a fixed hypothesis set.

    `family_size` is the number of pre-registered tests in the family
    (cycle-1 = 4 for {H1, H2, H3, H4} ; cycle-3 = 8 for the combined
    {H1, H2, H3, H4, H5-I, H5-II, H5-III, H6} family). α_per_test is
    the pointwise threshold any individual p-value must beat, computed
    as α_global / family_size.
    """

    family_size: int
    alpha_global: float = 0.05

    @property
    def alpha_per_test(self) -> float:
        """Per-test Bonferroni-corrected threshold (α_global / family_size)."""
        return self.alpha_global / self.family_size


def apply_bonferroni_family(
    p_values: list[float],
    family: BonferroniFamily,
) -> list[bool]:
    """Return per-test reject-null booleans at `family.alpha_per_test`.

    Convention : strictly-less-than, matching the `< alpha` comparisons
    already used by welch_one_sided / tost_equivalence / jonckheere_trend
    / one_sample_threshold above. The returned list preserves input order.
    """
    threshold = family.alpha_per_test
    return [p < threshold for p in p_values]


# Pre-instantiated constants for the two cycles that reference this
# module. Importing these avoids repeating the family_size literal at
# every reporter site and makes the cycle-1 → cycle-3 transition an
# explicit, grep-able change.
CYCLE1_FAMILY = BonferroniFamily(family_size=4)   # α_per_test = 0.0125
CYCLE3_FAMILY = BonferroniFamily(family_size=8)   # α_per_test = 0.00625


def compute_hedges_g(
    treatment: list[float],
    control: list[float],
) -> float:
    """Bias-corrected Cohen's d (Hedges & Olkin 1985) for two samples.

    Standardised mean difference :

        d = (mean(treatment) - mean(control)) / pooled_sd
        pooled_sd = sqrt(((n1-1) * var(t) + (n2-1) * var(c)) / (n1+n2-2))
        J = 1 - 3 / (4 * df - 1),  df = n1 + n2 - 2
        g = J * d

    The small-sample correction factor ``J`` is the closed-form
    Hedges-Olkin approximation, exact to four decimal places for
    df >= 5 and within 1 % for df = 2-4. For our G4 pilot (N=5 seeds
    per arm, df=8) the approximation error is < 0.001.

    Returns 0.0 when both samples are constant **and equal** (no
    effect, no spread). Raises ValueError when both samples are
    constant but have different means (undefined Cohen's d).

    Parameters :
        treatment : observed values for the treatment / dream-active
                    arm. Must contain >=2 finite floats.
        control   : observed values for the no-dream baseline arm.
                    Must contain >=2 finite floats.

    Returns :
        Hedges' g — positive when treatment mean exceeds control mean,
        negative otherwise.

    Raises :
        ValueError : empty input, singleton input, or zero pooled SD
                     with non-equal means.
    """
    if not treatment or not control:
        raise ValueError(
            "compute_hedges_g requires non-empty treatment and control"
        )
    if len(treatment) < 2 or len(control) < 2:
        raise ValueError(
            "compute_hedges_g requires at least 2 observations per arm "
            f"(got n_t={len(treatment)}, n_c={len(control)})"
        )
    t_arr = np.asarray(treatment, dtype=float)
    c_arr = np.asarray(control, dtype=float)
    n1 = t_arr.size
    n2 = c_arr.size
    df = n1 + n2 - 2
    var_t = float(t_arr.var(ddof=1))
    var_c = float(c_arr.var(ddof=1))
    pooled_var = ((n1 - 1) * var_t + (n2 - 1) * var_c) / df
    pooled_sd = pooled_var ** 0.5
    diff = float(t_arr.mean() - c_arr.mean())
    if pooled_sd == 0.0:
        if diff == 0.0:
            return 0.0
        raise ValueError(
            "compute_hedges_g: zero pooled SD with non-equal means — "
            "Cohen's d undefined"
        )
    cohens_d = diff / pooled_sd
    correction_j = 1.0 - 3.0 / (4.0 * df - 1.0)
    return float(correction_j * cohens_d)
