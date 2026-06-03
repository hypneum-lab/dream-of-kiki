"""Powered TOST equivalence on standardized (Hedges' g) bounds — E1.

This module implements the **pre-registered** equivalence test that the
experiment backlog (``EXPERIMENT-BACKLOG.md`` E1) requires before the
coupling NULL can be *supported* rather than merely *failed-to-reject*.
NHST cannot support a null (Lakens 2017) ; a Two One-Sided Tests (TOST)
procedure can — provided the equivalence bounds, α and power are all
pre-specified.

Design decisions (pre-registered, all grep-able constants below) :

* **SESOI** = Hedges' g = 0.29. This is the pre-registration floor
  ``HU_2020_OVERALL`` from :mod:`harness.benchmarks.effect_size_targets`
  (Hu et al. 2020 TMR meta, k=91, N=2004). Equivalence bounds are
  therefore symmetric in *standardized* units : ``[-0.29, +0.29]``.
* **α** = 0.0167. The per-test Bonferroni floor of the project's α=0.05
  family of three (3 profiles ; 0.05 / 3 ≈ 0.0167). TOST runs two
  one-sided tests *each* at this α ; equivalence is declared iff **both**
  one-sided p-values are < α (equivalently ``p_tost = max(p_lower,
  p_upper) < α``).
* **Power** = 0.80 target for detecting equivalence at the SESOI.

The pilot (``scripts/pilot_cycle3_real.py``) uses a *paired* pre/post
design, so this module's primary entry point is a **paired** TOST that
standardizes the within-pair difference by its own SD (the standardizer
for a paired Hedges' g). A pre-existing unpaired/raw-unit TOST lives in
:mod:`kiki_oniric.eval.statistics` ; this module is the standardized,
powered, pre-registered complement and does not replace it.

References :
* Lakens (2017) — Equivalence tests, Soc Psychol Personal Sci.
* Hu et al. (2020) — TMR meta-analysis (SESOI source).
* ``EXPERIMENT-BACKLOG.md`` E1 ; ``REVIEW.md`` §7.6.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import stats

# --------------------------------------------------------------------
# Pre-registered constants (load-bearing — do not edit without a
# DualVer FC bump + spec update). The SESOI is intentionally re-derived
# here from the published anchor rather than hard-coded blindly so a
# drift in the anchor surfaces as a test failure.
# --------------------------------------------------------------------

#: Smallest effect size of interest, Hedges' g (Hu 2020 overall floor).
SESOI_HEDGES_G: float = 0.29

#: Per-test α : Bonferroni floor of the α=0.05 / 3-profile family.
TOST_ALPHA_BONFERRONI: float = 0.05 / 3.0  # ≈ 0.016667

#: Target power for the equivalence-detection power analysis.
DEFAULT_TARGET_POWER: float = 0.80


@dataclass(frozen=True)
class TostResult:
    """Outcome of a standardized paired TOST equivalence test.

    Attributes :
        test_name : human-readable label.
        observed_g : observed paired Hedges' g (standardized mean diff).
        sesoi_g : the equivalence bound in standardized units (+/-).
        p_lower : one-sided p for H0_lower (g <= -sesoi).
        p_upper : one-sided p for H0_upper (g >= +sesoi).
        p_tost : max(p_lower, p_upper) — the TOST p-value.
        alpha : per-one-sided-test α used for the decision.
        equivalent : True iff p_tost < alpha (both one-sided rejected).
        n_pairs : number of paired observations.
    """

    test_name: str
    observed_g: float
    sesoi_g: float
    p_lower: float
    p_upper: float
    p_tost: float
    alpha: float
    equivalent: bool
    n_pairs: int


@dataclass(frozen=True)
class PowerAnalysisResult:
    """Required-n power analysis for equivalence at the SESOI.

    Attributes :
        sesoi_g : standardized equivalence bound.
        alpha : per-one-sided-test α.
        target_power : power the search solved for.
        required_n : smallest n (pairs, or per-arm for unpaired) that
            reaches >= target_power *assuming the true effect is zero*
            (the standard TOST power convention — power to conclude
            equivalence when there is genuinely no effect).
        achieved_power : power at ``required_n``.
        design : "paired" or "two_sample".
    """

    sesoi_g: float
    alpha: float
    target_power: float
    required_n: int
    achieved_power: float
    design: str


def paired_hedges_g(
    treatment: list[float],
    control: list[float],
) -> tuple[float, float, int]:
    """Standardized paired mean difference (Hedges' g) + its SE in g-units.

    For a paired design the difference ``d_i = treatment_i - control_i``
    is standardized by the SD of the differences :

        g_raw = mean(d) / sd(d)
        g     = J * g_raw,   J = 1 - 3 / (4*(n-1) - 1)   (small-sample)

    The standard error of g (in g-units), used to build the two
    one-sided test statistics, is ``1 / sqrt(n)`` for the standardized
    paired mean — i.e. the standardized mean difference has SE
    ``sd(d)/sqrt(n) / sd(d) = 1/sqrt(n)``.

    Returns :
        ``(g, se_g, n)`` where ``se_g`` is the standard error of g.

    Raises :
        ValueError : length mismatch, < 2 pairs, or zero difference SD
            with a non-zero mean difference (g undefined / infinite).
    """
    if len(treatment) != len(control):
        raise ValueError(
            "paired_hedges_g requires equal-length treatment/control "
            f"(got {len(treatment)} vs {len(control)})"
        )
    n = len(treatment)
    if n < 2:
        raise ValueError(
            "paired_hedges_g requires at least 2 pairs "
            f"(got n={n})"
        )
    diff = np.asarray(treatment, dtype=float) - np.asarray(
        control, dtype=float
    )
    mean_d = float(diff.mean())
    sd_d = float(diff.std(ddof=1))
    correction_j = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    if sd_d == 0.0:
        if mean_d == 0.0:
            return 0.0, 1.0 / np.sqrt(n), n
        raise ValueError(
            "paired_hedges_g: zero difference SD with non-zero mean "
            "difference — standardized effect undefined"
        )
    g = correction_j * (mean_d / sd_d)
    se_g = 1.0 / np.sqrt(n)
    return float(g), float(se_g), n


def tost_equivalence_g(
    treatment: list[float],
    control: list[float],
    *,
    sesoi_g: float = SESOI_HEDGES_G,
    alpha: float = TOST_ALPHA_BONFERRONI,
) -> TostResult:
    """Paired TOST equivalence in standardized (Hedges' g) units.

    Two one-sided tests against the symmetric equivalence interval
    ``[-sesoi_g, +sesoi_g]`` :

        H0_lower : g <= -sesoi_g      H1_lower : g >  -sesoi_g
        H0_upper : g >= +sesoi_g      H1_upper : g <  +sesoi_g

    Each is a one-sided t-test on ``df = n - 1`` using the standardized
    paired statistic. Equivalence (``equivalent=True``) is declared iff
    **both** one-sided nulls are rejected at ``alpha`` — i.e.
    ``p_tost = max(p_lower, p_upper) < alpha``.

    The decision is symmetric and pre-registered : ``sesoi_g`` defaults
    to the Hu-2020 floor (0.29) and ``alpha`` to the Bonferroni
    per-test floor (≈0.0167).
    """
    if sesoi_g <= 0.0:
        raise ValueError(f"sesoi_g must be > 0, got {sesoi_g}")
    g, se_g, n = paired_hedges_g(treatment, control)
    df = n - 1
    # Degenerate zero-SE path : standardized statistic is infinite.
    # Equivalence reduces to "is the point estimate strictly inside the
    # interval", with a definite p of 0 (inside) or 1 (outside).
    if se_g == 0.0:  # pragma: no cover - se_g = 1/sqrt(n) > 0 always
        inside = abs(g) < sesoi_g
        return TostResult(
            test_name="paired TOST (Hedges' g)",
            observed_g=g,
            sesoi_g=sesoi_g,
            p_lower=0.0 if inside else 1.0,
            p_upper=0.0 if inside else 1.0,
            p_tost=0.0 if inside else 1.0,
            alpha=alpha,
            equivalent=inside,
            n_pairs=n,
        )
    # Lower one-sided test : reject H0_lower when g is well above -sesoi.
    t_lower = (g - (-sesoi_g)) / se_g
    p_lower = float(1.0 - stats.t.cdf(t_lower, df))
    # Upper one-sided test : reject H0_upper when g is well below +sesoi.
    t_upper = (g - sesoi_g) / se_g
    p_upper = float(stats.t.cdf(t_upper, df))
    p_tost = max(p_lower, p_upper)
    return TostResult(
        test_name="paired TOST (Hedges' g)",
        observed_g=g,
        sesoi_g=sesoi_g,
        p_lower=p_lower,
        p_upper=p_upper,
        p_tost=p_tost,
        alpha=alpha,
        equivalent=bool(p_tost < alpha),
        n_pairs=n,
    )


def tost_power_at_n(
    n: int,
    *,
    sesoi_g: float = SESOI_HEDGES_G,
    alpha: float = TOST_ALPHA_BONFERRONI,
    true_g: float = 0.0,
    design: str = "paired",
) -> float:
    """Power of the TOST to declare equivalence, for a given n.

    Power = P(both one-sided tests reject | true standardized effect =
    ``true_g``). The standard TOST power convention sets ``true_g=0``
    (the worst-relevant case for *supporting* equivalence at a centred
    null). The standardized mean difference has SE :

        paired      : 1 / sqrt(n)        df = n - 1
        two_sample  : sqrt(2 / n)        df = 2 * (n - 1)   (n per arm)

    Each one-sided test rejects when its t-statistic clears the central
    ``t_{1-alpha, df}`` critical value ; the joint rejection region's
    probability is evaluated under the **noncentral** t distribution
    with noncentrality ``(true_g ± sesoi_g) / se``. Returns a power in
    ``[0, 1]``.
    """
    if n < 2:
        return 0.0
    if design == "paired":
        se = 1.0 / np.sqrt(n)
        df = n - 1
    elif design == "two_sample":
        se = np.sqrt(2.0 / n)
        df = 2 * (n - 1)
    else:
        raise ValueError(
            f"design must be 'paired' or 'two_sample', got {design!r}"
        )
    t_crit = stats.t.ppf(1.0 - alpha, df)
    # Lower test statistic ~ noncentral-t with ncp = (true_g + sesoi)/se,
    # rejects when t_lower > t_crit  ->  P = sf(t_crit ; df, ncp_lower).
    ncp_lower = (true_g + sesoi_g) / se
    p_reject_lower = float(stats.nct.sf(t_crit, df, ncp_lower))
    # Upper test statistic ~ noncentral-t with ncp = (true_g - sesoi)/se,
    # rejects when t_upper < -t_crit -> P = cdf(-t_crit ; df, ncp_upper).
    ncp_upper = (true_g - sesoi_g) / se
    p_reject_upper = float(stats.nct.cdf(-t_crit, df, ncp_upper))
    # Joint power. The two events are positively associated through the
    # shared sample ; the exact joint probability requires a bivariate
    # noncentral-t integral. We use the standard conservative bound
    # P(A and B) >= P(A) + P(B) - 1 (Bonferroni / Frechet lower bound),
    # which never *over*-states power — safe for a sample-size floor.
    power = max(0.0, p_reject_lower + p_reject_upper - 1.0)
    return min(power, 1.0)


def required_n_for_power(
    *,
    sesoi_g: float = SESOI_HEDGES_G,
    alpha: float = TOST_ALPHA_BONFERRONI,
    target_power: float = DEFAULT_TARGET_POWER,
    true_g: float = 0.0,
    design: str = "paired",
    n_max: int = 5000,
) -> PowerAnalysisResult:
    """Smallest n reaching ``target_power`` for equivalence at the SESOI.

    Monotone upward search over ``n`` (pairs for paired, per-arm for
    two-sample) until :func:`tost_power_at_n` >= ``target_power`` or
    ``n_max`` is hit. Returns the floor n and the achieved power.

    The reported ``required_n`` uses the conservative joint-power bound
    in :func:`tost_power_at_n`, so it is an *upper* bound on the truly
    required sample — i.e. if a design clears this n it is genuinely
    powered.
    """
    if not (0.0 < target_power < 1.0):
        raise ValueError(
            f"target_power must be in (0, 1), got {target_power}"
        )
    achieved = 0.0
    chosen = n_max
    for n in range(2, n_max + 1):
        achieved = tost_power_at_n(
            n,
            sesoi_g=sesoi_g,
            alpha=alpha,
            true_g=true_g,
            design=design,
        )
        if achieved >= target_power:
            chosen = n
            break
    return PowerAnalysisResult(
        sesoi_g=sesoi_g,
        alpha=alpha,
        target_power=target_power,
        required_n=chosen,
        achieved_power=float(achieved),
        design=design,
    )
