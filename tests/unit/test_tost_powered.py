"""Unit tests for the powered standardized TOST module (E1).

Validates the Two One-Sided Tests equivalence procedure on synthetic,
in-process data only — no model, no 35B weights, no run registration.
This is FC-axis *outillage*; the tests assert the statistical logic, the
pre-registered bounds, and the power analysis that documents whether the
30-seed pilot is adequately powered.

Cases :
* equivalent — true effect << SESOI g=0.29 → equivalence DETECTED.
* non-equivalent — true effect >> SESOI → equivalence REJECTED.
* bound coherence — symmetric bounds, α convention, monotone power.
"""
from __future__ import annotations

import numpy as np
import pytest

from harness.benchmarks.effect_size_targets import HU_2020_OVERALL
from harness.stats.tost import (
    SESOI_HEDGES_G,
    TOST_ALPHA_BONFERRONI,
    PowerAnalysisResult,
    TostResult,
    paired_hedges_g,
    required_n_for_power,
    tost_equivalence_g,
    tost_power_at_n,
)


# --------------------------------------------------------------------
# Pre-registered constants coherence
# --------------------------------------------------------------------


def test_sesoi_matches_published_floor() -> None:
    """SESOI must equal the Hu-2020 overall pre-registration floor."""
    assert SESOI_HEDGES_G == pytest.approx(0.29)
    assert SESOI_HEDGES_G == pytest.approx(HU_2020_OVERALL.hedges_g)


def test_alpha_is_bonferroni_floor_of_three_family() -> None:
    """α must be the 0.05 / 3-profile Bonferroni per-test floor."""
    assert TOST_ALPHA_BONFERRONI == pytest.approx(0.05 / 3.0)
    assert TOST_ALPHA_BONFERRONI == pytest.approx(0.0166667, abs=1e-6)


# --------------------------------------------------------------------
# paired_hedges_g
# --------------------------------------------------------------------


def test_paired_hedges_g_sign_and_se() -> None:
    """g positive when treatment > control ; SE = 1/sqrt(n)."""
    control = [0.50, 0.51, 0.49, 0.50, 0.52, 0.48, 0.50, 0.51, 0.50]
    treatment = [c + 0.10 for c in control]
    # add jitter so difference SD is non-zero
    rng = np.random.default_rng(7)
    treatment = [t + float(rng.normal(0, 0.02)) for t in treatment]
    g, se_g, n = paired_hedges_g(treatment, control)
    assert g > 0.0
    assert n == len(control)
    assert se_g == pytest.approx(1.0 / np.sqrt(n))


def test_paired_hedges_g_rejects_length_mismatch() -> None:
    with pytest.raises(ValueError, match="equal-length"):
        paired_hedges_g([0.1, 0.2, 0.3], [0.1, 0.2])


def test_paired_hedges_g_rejects_singleton() -> None:
    with pytest.raises(ValueError, match="at least 2 pairs"):
        paired_hedges_g([0.1], [0.2])


def test_paired_hedges_g_zero_diff_is_zero_effect() -> None:
    """Identical arms → g exactly 0 (no effect, no spread)."""
    arm = [0.4, 0.5, 0.6, 0.5, 0.4]
    g, se_g, n = paired_hedges_g(arm, arm)
    assert g == 0.0
    assert se_g == pytest.approx(1.0 / np.sqrt(n))


def test_paired_hedges_g_zero_sd_nonzero_mean_raises() -> None:
    """Constant non-zero difference → undefined standardized effect."""
    control = [0.4, 0.5, 0.6, 0.5, 0.4]
    treatment = [c + 0.1 for c in control]  # constant +0.1 diff
    with pytest.raises(ValueError, match="zero difference SD"):
        paired_hedges_g(treatment, control)


# --------------------------------------------------------------------
# TOST decision — equivalent case
# --------------------------------------------------------------------


def test_tost_detects_equivalence_for_subsesoi_effect() -> None:
    """True effect far below SESOI on a powered n → equivalence."""
    rng = np.random.default_rng(0)
    control = rng.normal(0.5, 0.05, 200).tolist()
    # treatment differs by genuinely tiny, noisy amount (g ~ 0)
    treatment = [c + float(rng.normal(0.0, 0.001)) for c in control]
    res = tost_equivalence_g(treatment, control)
    assert isinstance(res, TostResult)
    assert res.equivalent is True
    assert res.p_tost < TOST_ALPHA_BONFERRONI
    assert abs(res.observed_g) < SESOI_HEDGES_G
    assert res.test_name == "paired TOST (Hedges' g)"


# --------------------------------------------------------------------
# TOST decision — non-equivalent case
# --------------------------------------------------------------------


def test_tost_rejects_equivalence_for_supersesoi_effect() -> None:
    """True effect well beyond SESOI → equivalence NOT declared."""
    rng = np.random.default_rng(1)
    control = rng.normal(0.5, 0.05, 60).tolist()
    # ~ +0.05 mean shift on sd 0.05 → standardized effect ~ 1 >> 0.29
    treatment = [
        c + 0.05 + float(rng.normal(0, 0.05)) for c in control
    ]
    res = tost_equivalence_g(treatment, control)
    assert res.observed_g > SESOI_HEDGES_G
    assert res.equivalent is False
    # at least the upper one-sided test must fail to reject
    assert res.p_upper >= TOST_ALPHA_BONFERRONI


def test_tost_borderline_uses_max_of_one_sided_ps() -> None:
    """p_tost is exactly max(p_lower, p_upper) and drives the decision."""
    rng = np.random.default_rng(2)
    control = rng.normal(0.5, 0.05, 40).tolist()
    treatment = [c + float(rng.normal(0.0, 0.01)) for c in control]
    res = tost_equivalence_g(treatment, control)
    assert res.p_tost == pytest.approx(max(res.p_lower, res.p_upper))
    assert res.equivalent == (res.p_tost < res.alpha)


# --------------------------------------------------------------------
# Bound coherence
# --------------------------------------------------------------------


def test_tost_bounds_are_symmetric_and_reported() -> None:
    """Reported sesoi_g equals the symmetric standardized bound."""
    control = [0.5, 0.55, 0.45, 0.5, 0.52, 0.48]
    rng = np.random.default_rng(3)
    treatment = [c + float(rng.normal(0, 0.01)) for c in control]
    res = tost_equivalence_g(treatment, control, sesoi_g=0.29)
    assert res.sesoi_g == pytest.approx(0.29)


def test_tost_rejects_nonpositive_sesoi() -> None:
    with pytest.raises(ValueError, match="sesoi_g must be > 0"):
        tost_equivalence_g([0.1, 0.2, 0.3], [0.1, 0.2, 0.4], sesoi_g=0.0)


def test_tighter_sesoi_is_harder_to_satisfy() -> None:
    """A tighter equivalence bound yields a larger (worse) p_tost."""
    rng = np.random.default_rng(4)
    control = rng.normal(0.5, 0.05, 80).tolist()
    treatment = [c + float(rng.normal(0, 0.002)) for c in control]
    wide = tost_equivalence_g(treatment, control, sesoi_g=0.50)
    tight = tost_equivalence_g(treatment, control, sesoi_g=0.10)
    assert tight.p_tost >= wide.p_tost


# --------------------------------------------------------------------
# Power analysis
# --------------------------------------------------------------------


def test_power_is_monotone_increasing_in_n() -> None:
    """More pairs never reduce equivalence-detection power."""
    powers = [tost_power_at_n(n) for n in (10, 50, 100, 150, 300)]
    assert powers == sorted(powers)


def test_required_n_paired_reaches_target_power() -> None:
    """The solved n actually achieves >= the target power."""
    res = required_n_for_power(target_power=0.80, design="paired")
    assert isinstance(res, PowerAnalysisResult)
    assert res.achieved_power >= 0.80
    assert res.design == "paired"
    # one fewer pair must fall short (it's the *smallest* such n)
    assert tost_power_at_n(res.required_n - 1) < 0.80


def test_thirty_seeds_are_underpowered_for_sesoi() -> None:
    """KEY FINDING: 30 seeds fall far short of 80% power at g=0.29.

    Documents E1's protocol question — the current 30-seed pilot
    cannot *support* the coupling NULL ; required n is in the hundreds.
    """
    res = required_n_for_power(target_power=0.80, design="paired")
    assert res.required_n > 30
    assert tost_power_at_n(30) < 0.80


def test_two_sample_needs_more_than_paired() -> None:
    """Unpaired design needs more units per arm than paired pairs."""
    paired = required_n_for_power(design="paired")
    two = required_n_for_power(design="two_sample")
    assert two.required_n > paired.required_n


def test_required_n_rejects_bad_target_power() -> None:
    with pytest.raises(ValueError, match="target_power"):
        required_n_for_power(target_power=1.5)


def test_power_at_n_rejects_unknown_design() -> None:
    with pytest.raises(ValueError, match="design must be"):
        tost_power_at_n(50, design="bogus")


def test_power_below_two_pairs_is_zero() -> None:
    assert tost_power_at_n(1) == 0.0


# --------------------------------------------------------------------
# Harness wiring — pilot's _e1_tost reporter (no model, in-process)
# --------------------------------------------------------------------


def test_pilot_e1_tost_wiring_on_synthetic_cells() -> None:
    """The pilot reporter consumes cell dicts and emits the TOST block.

    Pure wiring check : feeds synthetic pre/post cells (no 35B, no
    model load) and asserts the power-analysis + per-profile shape.
    """
    from scripts.pilot_cycle3_real import _e1_tost

    rng = np.random.default_rng(11)
    cells = []
    for profile in ("p_min", "p_equ", "p_max"):
        for seed in range(8):
            pre = float(rng.normal(0.5, 0.05))
            post = pre + float(rng.normal(0.0, 0.001))  # near-zero effect
            cells.append(
                {"profile": profile, "seed": seed, "pre": pre, "post": post}
            )
    out = _e1_tost(cells)
    assert out["sesoi_hedges_g"] == pytest.approx(0.29)
    assert out["alpha_per_test"] == pytest.approx(TOST_ALPHA_BONFERRONI)
    pa = out["power_analysis"]
    assert pa["required_n_pairs"] > 30
    assert pa["adequately_powered"] is False
    assert set(out["per_profile"]) == {"p_min", "p_equ", "p_max"}
    for r in out["per_profile"].values():
        assert r["n"] == 8
        assert "p_tost" in r


def test_pilot_e1_tost_skips_insufficient_cells() -> None:
    """Profiles with < 2 paired cells are flagged, not crashed."""
    from scripts.pilot_cycle3_real import _e1_tost

    cells = [{"profile": "p_min", "seed": 0, "pre": 0.5, "post": 0.5}]
    out = _e1_tost(cells)
    assert out["per_profile"]["p_min"]["equivalent"] is None
    assert out["per_profile"]["p_min"]["note"] == "insufficient samples"
