"""Unit tests for statistical eval module (H1-H4 OSF pre-reg)."""
from __future__ import annotations


from kiki_oniric.eval.statistics import (
    StatTestResult,
    jonckheere_trend,
    one_sample_threshold,
    tost_equivalence,
    welch_one_sided,
)


def test_h1_welch_one_sided_detects_improvement() -> None:
    """P_equ (lower forgetting) significantly better than baseline."""
    baseline = [0.40, 0.42, 0.39, 0.41, 0.43]  # higher forgetting
    p_equ = [0.20, 0.18, 0.22, 0.19, 0.21]      # lower forgetting
    result = welch_one_sided(
        treatment=p_equ, control=baseline, alpha=0.05
    )
    assert isinstance(result, StatTestResult)
    assert result.reject_h0 is True
    assert result.p_value < 0.05
    assert result.test_name == "Welch's t-test (one-sided)"


def test_h1_welch_one_sided_no_improvement() -> None:
    """When P_equ ≈ baseline, fail to reject H0."""
    baseline = [0.40, 0.42, 0.39, 0.41, 0.43]
    p_equ = [0.40, 0.41, 0.39, 0.42, 0.40]
    result = welch_one_sided(
        treatment=p_equ, control=baseline, alpha=0.05
    )
    assert result.reject_h0 is False


def test_h2_tost_equivalence_passes_within_epsilon() -> None:
    """P_max within ±5% of P_equ → accept equivalence."""
    p_equ = [0.20, 0.21, 0.19, 0.20, 0.22]
    p_max = [0.20, 0.22, 0.18, 0.21, 0.20]  # very close
    result = tost_equivalence(
        treatment=p_max,
        control=p_equ,
        epsilon=0.05,
        alpha=0.05,
    )
    assert result.reject_h0 is True  # H0 = "not equivalent"
    assert result.test_name == "TOST equivalence"


def test_h2_tost_equivalence_fails_far_values() -> None:
    """When P_max is far from P_equ, fail to accept equivalence."""
    p_equ = [0.20, 0.21, 0.19, 0.20, 0.22]
    p_max = [0.50, 0.52, 0.48, 0.51, 0.49]  # far above
    result = tost_equivalence(
        treatment=p_max,
        control=p_equ,
        epsilon=0.05,
        alpha=0.05,
    )
    assert result.reject_h0 is False


def test_h3_jonckheere_detects_monotonic_trend() -> None:
    """Three groups in increasing order → significant trend."""
    p_min = [0.20, 0.22, 0.21, 0.19]
    p_equ = [0.40, 0.42, 0.41, 0.39]
    p_max = [0.60, 0.62, 0.61, 0.59]
    result = jonckheere_trend(
        groups=[p_min, p_equ, p_max], alpha=0.05
    )
    assert result.reject_h0 is True
    assert result.test_name == "Jonckheere-Terpstra trend"


def test_h4_one_sample_threshold_below_passes() -> None:
    """Energy ratio <= 2.0 (budget threshold) → no violation."""
    energy_ratios = [1.5, 1.6, 1.4, 1.7, 1.5]
    result = one_sample_threshold(
        sample=energy_ratios, threshold=2.0, alpha=0.05
    )
    # H0 = "ratio >= 2.0", reject = ratio is significantly < 2.0
    assert result.reject_h0 is True
    assert result.test_name == "one-sample t-test (upper bound)"
