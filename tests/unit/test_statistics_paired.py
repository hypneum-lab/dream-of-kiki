"""Unit tests for paired TOST / non-inferiority helpers.

Spec: E1 (H14) pre-registration, section 2 "Paired procedures
(exact)": D = x - y, se = sd(D, ddof=1)/sqrt(n), df = n - 1,
q = t_{1-alpha, n-1}, L = mean - q*se, U = mean + q*se.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from kiki_oniric.eval.statistics import (
    PairedNIResult,
    PairedTOSTResult,
    paired_noninferiority,
    paired_tost,
)

X = [0.52, 0.55, 0.50, 0.58, 0.54, 0.51]
Y = [0.50, 0.54, 0.51, 0.55, 0.53, 0.50]
# D = [0.02, 0.01, -0.01, 0.03, 0.01, 0.01]
# mean = 0.07/6, sd (ddof=1) = sqrt(0.0010833.../5)


def test_hand_computed_summary() -> None:
    res = paired_tost(X, Y, delta=0.01, alpha=0.05)
    assert isinstance(res, PairedTOSTResult)
    d = np.array([0.02, 0.01, -0.01, 0.03, 0.01, 0.01])
    mean = d.sum() / 6  # 0.011666...
    ss = ((d - mean) ** 2).sum()
    sd = math.sqrt(ss / 5)
    se = sd / math.sqrt(6)
    assert res.n == 6
    assert res.df == 5
    assert res.mean_diff == pytest.approx(0.07 / 6, abs=1e-12)
    assert res.sd_diff == pytest.approx(sd, abs=1e-12)
    assert res.se_diff == pytest.approx(se, abs=1e-12)
    # t_{0.95, 5} = 2.015048373...
    q = 2.0150483733
    assert res.ci_lower == pytest.approx(mean - q * se, abs=1e-8)
    assert res.ci_upper == pytest.approx(mean + q * se, abs=1e-8)
    assert res.t_lower == pytest.approx((mean + 0.01) / se, abs=1e-10)
    assert res.t_upper == pytest.approx((mean - 0.01) / se, abs=1e-10)
    assert res.ci_level == pytest.approx(0.90)
    # g_z = mean/sd * (1 - 3/(4(n-1)-1))
    assert res.g_z == pytest.approx(mean / sd * (1 - 3 / 19), abs=1e-12)
    assert res.degenerate is False


def test_matches_scipy_ttest_rel() -> None:
    rng = np.random.default_rng(7)
    x = rng.normal(0.5, 0.05, 40)
    y = x - rng.normal(0.004, 0.03, 40)
    delta, alpha = 0.01, 0.05
    res = paired_tost(x.tolist(), y.tolist(), delta, alpha)
    d = x - y
    # lower test: H1 mean(D) > -delta
    lo = stats.ttest_1samp(d, -delta, alternative="greater")
    up = stats.ttest_1samp(d, delta, alternative="less")
    assert res.t_lower == pytest.approx(lo.statistic, rel=1e-10)
    assert res.p_lower == pytest.approx(lo.pvalue, rel=1e-9)
    assert res.t_upper == pytest.approx(up.statistic, rel=1e-10)
    assert res.p_upper == pytest.approx(up.pvalue, rel=1e-9)
    assert res.p_tost == pytest.approx(max(lo.pvalue, up.pvalue))
    assert res.equivalent == (res.p_tost < alpha)
    # delta = 0 superiority form equals ttest_rel(alternative=greater)
    sup = stats.ttest_rel(x, y, alternative="greater")
    ni0 = paired_noninferiority(x.tolist(), y.tolist(), 1e-300, alpha)
    assert ni0.t_stat == pytest.approx(sup.statistic, rel=1e-9)
    assert ni0.p_value == pytest.approx(sup.pvalue, rel=1e-6)


def test_ni_matches_scipy_both_directions() -> None:
    rng = np.random.default_rng(11)
    x = rng.normal(0.5, 0.05, 30)
    y = x - rng.normal(-0.002, 0.02, 30)
    d = x - y
    g = paired_noninferiority(x.tolist(), y.tolist(), 0.01, 0.05, "greater")
    ref = stats.ttest_1samp(d, -0.01, alternative="greater")
    assert isinstance(g, PairedNIResult)
    assert g.t_stat == pytest.approx(ref.statistic, rel=1e-10)
    assert g.p_value == pytest.approx(ref.pvalue, rel=1e-9)
    assert g.noninferior == (ref.pvalue < 0.05)
    lo = paired_noninferiority(x.tolist(), y.tolist(), 0.01, 0.05, "less")
    ref2 = stats.ttest_1samp(d, 0.01, alternative="less")
    assert lo.t_stat == pytest.approx(ref2.statistic, rel=1e-10)
    assert lo.p_value == pytest.approx(ref2.pvalue, rel=1e-9)


@pytest.mark.parametrize("seed", range(25))
def test_ci_tost_equivalence_property(seed: int) -> None:
    """TOST at alpha  <=>  (1-2alpha) CI inside (-delta, delta);
    NI at alpha  <=>  lower bound > -delta (and mirrored)."""
    rng = np.random.default_rng(seed)
    n = int(rng.integers(3, 60))
    x = rng.normal(0.5, 0.05, n)
    y = x - rng.normal(rng.uniform(-0.02, 0.02), rng.uniform(0.005, 0.05), n)
    delta, alpha = 0.01, 0.05
    xs, ys = x.tolist(), y.tolist()
    t = paired_tost(xs, ys, delta, alpha)
    assert t.equivalent == (t.ci_lower > -delta and t.ci_upper < delta)
    g = paired_noninferiority(xs, ys, delta, alpha, "greater")
    assert g.noninferior == (t.ci_lower > -delta)
    assert g.noninferior == (g.bound > -delta)
    lo = paired_noninferiority(xs, ys, delta, alpha, "less")
    assert lo.noninferior == (t.ci_upper < delta)


def test_equivalent_and_not_equivalent_cases() -> None:
    n = 50
    rng = np.random.default_rng(3)
    base = rng.normal(0, 0.003, n)
    y = rng.normal(0.5, 0.05, n)
    assert paired_tost((y + base).tolist(), y.tolist(), 0.01).equivalent
    assert not paired_tost((y + 0.05 + base).tolist(), y.tolist(), 0.01).equivalent


def test_zero_variance_exact_states() -> None:
    # Exactly constant D (binary-exact values).
    x = [1.0, 2.0, 3.0]
    inside = paired_tost(x, [0.75, 1.75, 2.75], 0.5)  # D = 0.25
    assert inside.degenerate is True
    assert inside.ci_lower == inside.ci_upper == inside.mean_diff == 0.25
    assert inside.t_lower is None and inside.t_upper is None
    assert inside.g_z is None
    assert inside.equivalent is True
    assert inside.p_tost == 0.0
    outside = paired_tost(x, [0.25, 1.25, 2.25], 0.5)  # D = 0.75
    assert outside.equivalent is False
    assert outside.p_tost == 1.0
    on_edge = paired_tost(x, [0.5, 1.5, 2.5], 0.5)  # D = 0.5 == delta
    assert on_edge.equivalent is False  # strict inequalities
    ni = paired_noninferiority(x, [1.0, 2.0, 3.0], 0.5, 0.05)  # D = 0
    assert ni.degenerate and ni.noninferior and ni.t_stat is None
    assert ni.p_value == 0.0
    ni_bad = paired_noninferiority(x, [1.75, 2.75, 3.75], 0.5, 0.05)  # D=-.75
    assert ni_bad.noninferior is False and ni_bad.p_value == 1.0
    ni_edge = paired_noninferiority(x, [1.5, 2.5, 3.5], 0.5, 0.05)  # D=-.5
    assert ni_edge.noninferior is False  # L > -delta is strict
    less = paired_noninferiority(x, [0.75, 1.75, 2.75], 0.5, 0.05, "less")
    assert less.noninferior is True  # U = 0.25 < 0.5


def test_n_less_than_two_raises() -> None:
    with pytest.raises(ValueError, match="at least 2"):
        paired_tost([0.5], [0.4], 0.01)
    with pytest.raises(ValueError, match="at least 2"):
        paired_noninferiority([], [], 0.01)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"delta": 0.0},
        {"delta": -0.01},
        {"delta": float("nan")},
        {"alpha": 0.0},
        {"alpha": 0.5},
        {"alpha": 1.0},
    ],
)
def test_invalid_parameters_raise(kwargs: dict[str, float]) -> None:
    params = {"delta": 0.01, "alpha": 0.05, **kwargs}
    with pytest.raises(ValueError):
        paired_tost(X, Y, **params)
    with pytest.raises(ValueError):
        paired_noninferiority(X, Y, **params)


def test_invalid_inputs_raise() -> None:
    with pytest.raises(ValueError, match="same length"):
        paired_tost([0.1, 0.2, 0.3], [0.1, 0.2], 0.01)
    with pytest.raises(ValueError, match="finite"):
        paired_tost([0.1, float("nan"), 0.3], [0.1, 0.2, 0.3], 0.01)
    with pytest.raises(ValueError, match="direction"):
        paired_noninferiority(X, Y, 0.01, 0.05, "sideways")  # type: ignore[arg-type]


def test_does_not_mutate_inputs_and_existing_tost_unchanged() -> None:
    xs, ys = list(X), list(Y)
    paired_tost(xs, ys, 0.01)
    assert xs == X and ys == Y
    from kiki_oniric.eval.statistics import tost_equivalence

    assert tost_equivalence(X, Y, 0.05).test_name == "TOST equivalence"
