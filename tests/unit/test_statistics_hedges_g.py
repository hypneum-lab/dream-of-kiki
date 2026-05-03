"""Unit tests for compute_hedges_g (standardised mean difference).

Hedges' g is the bias-corrected Cohen's d, used to compare an
observed treatment-vs-control effect to published meta-analytic
anchors (Hu 2020, Javadi 2024 — see harness/benchmarks/
effect_size_targets.py).
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from kiki_oniric.eval.statistics import compute_hedges_g


def test_zero_effect_returns_zero() -> None:
    """Identical samples → g exactly 0.0."""
    g = compute_hedges_g([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
    assert g == 0.0


def test_positive_shift_returns_positive_g() -> None:
    """Treatment mean above control mean → positive g."""
    treatment = [1.0, 1.1, 0.9, 1.05, 0.95]
    control = [0.0, 0.1, -0.1, 0.05, -0.05]
    g = compute_hedges_g(treatment, control)
    assert g > 0.0


def test_negative_shift_returns_negative_g() -> None:
    """Treatment mean below control mean → negative g."""
    treatment = [0.0, 0.1, -0.1]
    control = [1.0, 1.1, 0.9]
    g = compute_hedges_g(treatment, control)
    assert g < 0.0


def test_known_value_matches_textbook() -> None:
    """Manual cross-check on a textbook 2-sample case.

    Treatment = [3, 5, 7] (mean 5, var 4, n=3)
    Control   = [1, 3, 5] (mean 3, var 4, n=3)
    Pooled SD = sqrt(((3-1)*4 + (3-1)*4) / (3+3-2)) = 2.0
    Cohen's d = (5 - 3) / 2.0 = 1.0
    Hedges' g = d * J(df=4) where J = 1 - 3 / (4*4 - 1) = 1 - 3/15 = 0.8
    Expected g = 1.0 * 0.8 = 0.8
    """
    g = compute_hedges_g([3.0, 5.0, 7.0], [1.0, 3.0, 5.0])
    assert math.isclose(g, 0.8, rel_tol=1e-6)


def test_rejects_empty_sample() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        compute_hedges_g([], [1.0, 2.0])
    with pytest.raises(ValueError, match="non-empty"):
        compute_hedges_g([1.0, 2.0], [])


def test_rejects_singleton_sample() -> None:
    """n>=2 each side — variance is undefined for n=1."""
    with pytest.raises(ValueError, match="at least 2"):
        compute_hedges_g([1.0], [1.0, 2.0])
    with pytest.raises(ValueError, match="at least 2"):
        compute_hedges_g([1.0, 2.0], [1.0])


def test_zero_variance_returns_zero_when_means_equal() -> None:
    """Both constant + equal means → g=0 (no effect, no spread)."""
    g = compute_hedges_g([0.5, 0.5], [0.5, 0.5])
    assert g == 0.0


def test_zero_variance_raises_when_means_differ() -> None:
    """Both constant + different means → undefined Cohen's d."""
    with pytest.raises(ValueError, match="zero pooled"):
        compute_hedges_g([1.0, 1.0], [0.0, 0.0])


@given(
    treatment=st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=20,
    ),
    control=st.lists(
        st.floats(min_value=-1e3, max_value=1e3, allow_nan=False, allow_infinity=False),
        min_size=2,
        max_size=20,
    ),
)
@settings(max_examples=100, deadline=None)
def test_hedges_g_finite_when_variance_positive(
    treatment: list[float], control: list[float]
) -> None:
    """Property: g is finite whenever pooled SD > 0."""
    import numpy as np

    if np.var(treatment, ddof=1) + np.var(control, ddof=1) <= 0.0:
        return  # excluded by zero-variance branch
    g = compute_hedges_g(treatment, control)
    assert math.isfinite(g)


def test_hedges_g_correction_factor_smaller_than_one_for_small_n() -> None:
    """For small n, J < 1 so |g| < |Cohen's d|.

    With n1=n2=3 (df=4), J = 1 - 3/15 = 0.8 < 1.
    """
    treatment = [2.0, 4.0, 6.0]
    control = [0.0, 2.0, 4.0]
    g = compute_hedges_g(treatment, control)
    # Cohen's d = (4 - 2) / 2 = 1.0, so g should be ~0.8 (J=0.8).
    assert 0.7 < g < 0.85
