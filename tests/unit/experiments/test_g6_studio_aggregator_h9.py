"""Unit tests for the H9-A / H9-B / H9-C verdict aggregator.

Decision-rule arithmetic — each test pins one (g, Welch) classification
band per pre-reg §6 + plan Task 6 spec.

Reference :
- Plan : ``docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md`` Task 7 step 1
- Pre-reg : ``docs/osf-prereg-g6-studio-path-a.md`` §6
"""
from __future__ import annotations

from experiments.g6_studio_path_a.aggregator_h9 import (
    H9A_MEDIUM_G_THRESHOLD,
    H9A_STRICT_LARGE_G_THRESHOLD,
    H9_BONFERRONI_ALPHA,
    classify_h9,
)


def test_h9a_strict_large_effect_recovery() -> None:
    """TDD-6.1 — large g (>= 2) + Welch reject → strict H9-A."""
    retention = {
        "baseline": [0.40, 0.42, 0.41, 0.39, 0.43],
        "P_equ": [0.85, 0.87, 0.86, 0.88, 0.84],
        "P_min": [0.55, 0.56, 0.54, 0.57, 0.55],
        "P_max": [0.84, 0.83, 0.82, 0.85, 0.86],
    }
    verdict = classify_h9(retention)
    assert verdict["h9a_classification"] == "H9-A"
    assert verdict["h9a_g"] >= H9A_STRICT_LARGE_G_THRESHOLD
    assert verdict["h9a_welch_reject"] is True
    assert verdict["h9a_positive_sign"] is True
    assert verdict["h9a_strict_large_effect"] is True


def test_h9b_washout_persists() -> None:
    """TDD-6.2 — tiny g + Welch fail → H9-B (washout)."""
    retention = {
        "baseline": [0.50, 0.51, 0.49, 0.50, 0.52],
        "P_equ": [0.51, 0.50, 0.51, 0.50, 0.50],
        "P_min": [0.50, 0.49, 0.51, 0.50, 0.50],
        "P_max": [0.51, 0.50, 0.50, 0.51, 0.50],
    }
    verdict = classify_h9(retention)
    assert verdict["h9a_classification"] == "H9-B"
    assert verdict["h9a_g"] < H9A_MEDIUM_G_THRESHOLD
    assert verdict["h9a_welch_reject"] is False


def test_h9c_dr4_inversion_universal() -> None:
    """TDD-6.3 — Jonckheere fail AND mean(P_min) > mean(P_equ) → H9-C."""
    retention = {
        "baseline": [0.40, 0.41, 0.40, 0.41, 0.40],
        "P_equ": [0.42, 0.41, 0.42, 0.43, 0.41],
        "P_min": [0.65, 0.66, 0.64, 0.67, 0.65],
        "P_max": [0.42, 0.42, 0.41, 0.42, 0.41],
    }
    verdict = classify_h9(retention)
    assert verdict["h9c_classification"] == "H9-C"
    assert verdict["h9c_mean_p_min"] > verdict["h9c_mean_p_equ"]
    assert verdict["h9c_jonckheere_reject"] is False


def test_h9_bonferroni_alpha_constant() -> None:
    """TDD-6.4 — Bonferroni α = 0.05 / 3 (three-hypothesis family)."""
    assert abs(H9_BONFERRONI_ALPHA - 0.05 / 3) < 1e-9


def test_classify_h9_insufficient_samples() -> None:
    """TDD-6.5 — < 2 samples per arm → INSUFFICIENT classification."""
    retention = {
        "baseline": [0.40],
        "P_equ": [0.85],
        "P_min": [],
        "P_max": [0.84],
    }
    verdict = classify_h9(retention)
    assert verdict["h9a_classification"] == "INSUFFICIENT"
    assert verdict["h9c_classification"] == "INSUFFICIENT"


def test_h9a_star_exploratory_medium_band() -> None:
    """TDD-6.6 — medium g (0.5 <= g < 2) lands in {H9-A*, H9-B, INDETERMINATE}."""
    # Engineered for a wide-variance medium-effect band so Hedges' g
    # stays in [0.5, 2). Variance is tuned so the std-deviation of
    # each arm is ~0.10, with the mean separation set to ~0.10
    # (Cohen's d ≈ 1.0 → Hedges' g ≈ 0.91 with N=5).
    retention = {
        "baseline": [0.40, 0.55, 0.45, 0.60, 0.50],
        "P_equ": [0.50, 0.65, 0.55, 0.70, 0.60],
        "P_min": [0.45, 0.55, 0.50, 0.60, 0.55],
        "P_max": [0.50, 0.65, 0.55, 0.70, 0.60],
    }
    verdict = classify_h9(retention)
    # The exact classification depends on Welch's p value at N=5 ;
    # assert g falls in the medium band and the classification is
    # one of the documented outcomes.
    assert (
        H9A_MEDIUM_G_THRESHOLD
        <= verdict["h9a_g"]
        < H9A_STRICT_LARGE_G_THRESHOLD
    )
    assert verdict["h9a_classification"] in {
        "H9-A*", "INDETERMINATE", "H9-B",
    }


def test_h9_verdict_carries_all_keys() -> None:
    """TDD-6.7 — verdict dict carries every documented key."""
    retention = {
        "baseline": [0.40, 0.42, 0.41, 0.39, 0.43],
        "P_equ": [0.85, 0.87, 0.86, 0.88, 0.84],
        "P_min": [0.55, 0.56, 0.54, 0.57, 0.55],
        "P_max": [0.84, 0.83, 0.82, 0.85, 0.86],
    }
    verdict = classify_h9(retention)
    expected_keys = {
        "bonferroni_alpha",
        "h9a_classification",
        "h9a_g",
        "h9a_welch_p",
        "h9a_welch_reject",
        "h9a_positive_sign",
        "h9a_strict_large_effect",
        "h9c_classification",
        "h9c_jonckheere_p",
        "h9c_jonckheere_reject",
        "h9c_mean_p_min",
        "h9c_mean_p_equ",
        "h9c_mean_p_max",
    }
    assert expected_keys.issubset(verdict.keys())
