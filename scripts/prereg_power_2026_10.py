#!/usr/bin/env python3
"""Power numbers for the 2026-10-23 draft pre-registrations.

Reproduces every power / minimum-detectable-effect figure quoted in

- docs/osf-prereg-g4-nono-cross-domain-2026-10-23.md   (H12)
- docs/osf-prereg-dr3-relational-probe-2026-10-23.md   (H13)
- docs/osf-prereg-erratum-g4-power-2026-10-23.md       (erratum)

Two kinds of numbers are produced:

1. *Test-rejection* minimum detectable effects (MDE) at 80 % power,
   computed exactly from the noncentral t distribution.
2. *Complete confirmation rule* powers (test rejection AND the
   pre-registered effect floor), and non-inferiority / TOST powers,
   computed by Monte Carlo simulation. Paired-design simulations draw
   the sufficient statistics (mean, variance) directly when the rule
   only needs the paired differences, and draw full bivariate-normal
   samples when the rule needs the per-arm SDs (g_av floor).
3. *"No practical effect" rules* at the margins fixed by the PI on
   2026-10-23 (G4 eps = 0.01, reverted from 0.005 by a later PI
   decision the same day; DR-3 delta = 0.025):
   power at true difference 0, the largest SD of the paired
   difference at which the registered n still reaches 80 %, and the
   n needed for 80 % at the planning SD. One-sided bound rules
   (U < margin, or L > -margin) at mu = 0 are exact: the event is
   {noncentral t(df = n - 1, ncp = margin sqrt(n) / SD) > t_crit}.
   TOST powers are Monte Carlo (fixed seeds).

numpy + scipy only. Deterministic (fixed seeds). Runs in < 1 min.

    uv run python scripts/prereg_power_2026_10.py
"""
from __future__ import annotations

import numpy as np
from scipy import optimize, stats

POWER = 0.80
N_SIM = 200_000      # sufficient-statistic simulations
N_SIM_FULL = 20_000  # full-sample simulations (g_av floor)


# ---------------------------------------------------------------- exact MDE
def _power_two_sample(d: float, n: int, alpha: float, sided: int) -> float:
    """Equal-n, equal-variance two-sample t power (noncentral t)."""
    df = 2 * n - 2
    ncp = d * np.sqrt(n / 2.0)
    if sided == 1:
        crit = stats.t.ppf(1 - alpha, df)
        return float(stats.nct.sf(crit, df, ncp))
    crit = stats.t.ppf(1 - alpha / 2, df)
    # Lower-tail term is negligible for d > 0 and scipy's nct.cdf can
    # return NaN far in the tail, so guard it.
    lower = float(np.nan_to_num(stats.nct.cdf(-crit, df, ncp)))
    return float(stats.nct.sf(crit, df, ncp)) + lower


def _power_paired(dz: float, n: int, alpha: float) -> float:
    """One-sided paired t power (noncentral t, df = n - 1)."""
    df = n - 1
    crit = stats.t.ppf(1 - alpha, df)
    return float(stats.nct.sf(crit, df, dz * np.sqrt(n)))


def mde_two_sample(n: int, alpha: float, sided: int) -> float:
    return float(optimize.brentq(
        lambda d: _power_two_sample(d, n, alpha, sided) - POWER, 1e-3, 4))


def mde_paired(n: int, alpha: float) -> float:
    return float(optimize.brentq(
        lambda d: _power_paired(d, n, alpha) - POWER, 1e-3, 4))


def hedges_j(df: int) -> float:
    """Small-sample correction J(df) = 1 - 3 / (4 df - 1)."""
    return 1.0 - 3.0 / (4.0 * df - 1.0)


# ------------------------------------------------- paired-difference rules
def _paired_suff(mu: float, sd: float, n: int, seed: int):
    """Simulated (mean, sd) of n paired differences ~ N(mu, sd^2)."""
    rng = np.random.default_rng(seed)
    mean = rng.normal(mu, sd / np.sqrt(n), N_SIM)
    var = sd ** 2 * rng.chisquare(n - 1, N_SIM) / (n - 1)
    return mean, np.sqrt(var)


def _bounds(mean, s, n: int, alpha: float):
    """One-sided (1 - alpha) lower and upper confidence bounds."""
    q = stats.t.ppf(1 - alpha, n - 1)
    se = s / np.sqrt(n)
    return mean - q * se, mean + q * se


def power_superiority_floor(mu, sd, n, alpha, floor, seed=1) -> float:
    """P(lower bound > 0 AND observed mean >= raw floor)."""
    m, s = _paired_suff(mu, sd, n, seed)
    lo, _ = _bounds(m, s, n, alpha)
    return float(np.mean((lo > 0) & (m >= floor)))


def power_upper_bound_below(mu, sd, n, alpha, bound, seed=2) -> float:
    """P(upper (1 - alpha) bound < bound): 'effect bounded below bound'."""
    m, s = _paired_suff(mu, sd, n, seed)
    _, hi = _bounds(m, s, n, alpha)
    return float(np.mean(hi < bound))


def power_noninferiority(mu, sd, n, alpha, margin, seed=3) -> float:
    m, s = _paired_suff(mu, sd, n, seed)
    lo, _ = _bounds(m, s, n, alpha)
    return float(np.mean(lo > -margin))


def power_bng_exact(mu, sd, n, alpha, margin, seed=6) -> float:
    """P(L <= 0 and U < margin): the BOUNDED-NO-GAIN state under the
    first-match order (POSITIVE-NEGLIGIBLE takes L > 0 and U < margin)."""
    m, s = _paired_suff(mu, sd, n, seed)
    lo, hi = _bounds(m, s, n, alpha)
    return float(np.mean((lo <= 0) & (hi < margin)))


def power_tost(mu, sd, n, alpha, margin, seed=4) -> float:
    m, s = _paired_suff(mu, sd, n, seed)
    lo, hi = _bounds(m, s, n, alpha)
    return float(np.mean((lo > -margin) & (hi < margin)))


# ------------------------------------------- full-sample g_av floor (G4)
def power_paired_gav_floor(d_av, r, n, alpha, floor, seed=5,
                           sigma=1.0, raw_floor=None) -> float:
    """P(paired one-sided t rejects AND Hedges g_av >= floor [AND
    mean(D) >= raw_floor]).

    Bivariate normal arms with SD sigma and correlation r; true mean
    difference d_av * sigma (so the population d_av equals the
    argument). g_av = mean(D) / sqrt((s1^2 + s2^2) / 2) * J(n - 1).
    """
    rng = np.random.default_rng(seed)
    cov = np.array([[1.0, r], [r, 1.0]])
    chol = np.linalg.cholesky(cov)
    z = sigma * (rng.standard_normal((N_SIM_FULL, n, 2)) @ chol.T)
    x = z[..., 0] + d_av * sigma
    y = z[..., 1]
    diff = x - y
    m = diff.mean(axis=1)
    sd = diff.std(axis=1, ddof=1)
    t = m / (sd / np.sqrt(n))
    crit = stats.t.ppf(1 - alpha, n - 1)
    s_av = np.sqrt((x.var(axis=1, ddof=1) + y.var(axis=1, ddof=1)) / 2)
    g_av = m / s_av * hedges_j(n - 1)
    ok = (t > crit) & (g_av >= floor)
    if raw_floor is not None:
        ok &= m >= raw_floor
    return float(np.mean(ok))


# --------------------------------------- "no practical effect" helpers
def power_bound_exact(margin_over_sd: float, n: int, alpha: float) -> float:
    """Exact P(U < margin) at mu = 0 (= P(L > -margin) at mu = 0)."""
    df = n - 1
    crit = stats.t.ppf(1 - alpha, df)
    return float(stats.nct.sf(crit, df, margin_over_sd * np.sqrt(n)))


def max_sd_bound(margin: float, n: int, alpha: float) -> float:
    """Largest SD_D at which the one-sided bound rule has 80 % power."""
    return float(optimize.brentq(
        lambda sd: power_bound_exact(margin / sd, n, alpha) - POWER,
        margin / 100, margin * 100))


def n_needed_bound(margin: float, sd: float, alpha: float) -> int:
    n = 3
    while power_bound_exact(margin / sd, n, alpha) < POWER:
        n += 1
    return n


def _tost_std(ratio: float, n: int, alpha: float, seed: int = 11) -> float:
    """TOST power at mu = 0 with margin / SD = ratio (Monte Carlo)."""
    return power_tost(0.0, 1.0, n, alpha, ratio, seed=seed)


def max_sd_tost(margin: float, n: int, alpha: float) -> float:
    return float(margin / optimize.brentq(
        lambda q: _tost_std(q, n, alpha) - POWER, 1e-3, 10))


def n_needed_tost(margin: float, sd: float, alpha: float) -> int:
    lo, hi = 3, 3
    while _tost_std(margin / sd, hi, alpha) < POWER:
        lo, hi = hi, hi * 2
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if _tost_std(margin / sd, mid, alpha) >= POWER:
            hi = mid
        else:
            lo = mid
    return hi


def _fmt_row(cells) -> str:
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    a_g4 = 0.05 / 3      # 0.0167
    a_dr3 = 0.05 / 4     # 0.0125

    print("## 1. Test-rejection MDE at 80 % power (exact noncentral t)\n")
    for label, n, a, sided in [
        ("DR3 independent, n=20/group, one-sided a=.0125", 20, a_dr3, 1),
        ("G4 independent, n=95/group, one-sided a=1/60", 95, a_g4, 1),
        ("G4 independent, n=108/group, one-sided a=1/60", 108, a_g4, 1),
    ]:
        d = mde_two_sample(n, a, sided)
        print(f"- {label}: d = {d:.3f}, Hedges g = "
              f"{d * hedges_j(2 * n - 2):.3f}")
    for label, n, a in [
        ("DR3 paired, n=20, one-sided a=.0125", 20, a_dr3),
        ("G4 paired, n=95, one-sided a=1/60", 95, a_g4),
        ("G4 paired, n=108, one-sided a=1/60", 108, a_g4),
    ]:
        dz = mde_paired(n, a)
        print(f"- {label}: d_z = {dz:.3f}, g_z = {dz * hedges_j(n - 1):.3f}")

    print("\n## 2. Erratum: locked G4 / G5-bis power sentences\n")
    for label, n, a, sided in [
        ("N=30/group, two-sided a=0.05", 30, 0.05, 2),
        ("N=30/group, two-sided a=0.0167", 30, a_g4, 2),
        ("N=30/group, one-sided a=0.0167", 30, a_g4, 1),
        ("N=10/group, two-sided a=0.0125", 10, a_dr3, 2),
        ("N=10/group, one-sided a=0.0125", 10, a_dr3, 1),
        ("N=30/group, two-sided a=0.0125", 30, a_dr3, 2),
        ("N=30/group, one-sided a=0.0125", 30, a_dr3, 1),
    ]:
        d = mde_two_sample(n, a, sided)
        print(f"- {label}: d = {d:.3f}, Hedges g = "
              f"{d * hedges_j(2 * n - 2):.3f}")

    print("\n## 3. G4-nono paired MDE in d_av units, by within-seed r\n")
    rs = (0.0, 0.3, 0.5, 0.7, 0.9)
    print(_fmt_row(["n", *[f"r={r}" for r in rs]]))
    print(_fmt_row(["---"] * (len(rs) + 1)))
    for n in (95, 108):
        dz = mde_paired(n, a_g4)
        print(_fmt_row([str(n), *[f"{dz * np.sqrt(2 * (1 - r)):.3f}"
                                  for r in rs]]))

    print("\n## 4. G4-nono complete rule (paired t AND g_av >= 0.5)\n")
    d_grid = (0.5, 0.6, 0.8, 1.0)
    for n in (95, 108):
        print(f"\nn = {n} (rows: r; columns: true d_av)\n")
        print(_fmt_row(["r", *[f"d_av={d}" for d in d_grid]]))
        print(_fmt_row(["---"] * (len(d_grid) + 1)))
        for r in rs:
            cells = [f"{power_paired_gav_floor(d, r, n, a_g4, 0.5):.2f}"
                     for d in d_grid]
            print(_fmt_row([str(r), *cells]))

    eps = 0.01
    print("\n### 4b. Same rule plus raw floor mean(D) >= eps = 0.01"
          " (n = 95, r = 0.5)\n")
    sigmas = (0.01, 0.02, 0.04)
    print(_fmt_row(["arm SD", *[f"d_av={d}" for d in d_grid]]))
    print(_fmt_row(["---"] * (len(d_grid) + 1)))
    for sg in sigmas:
        cells = [f"{power_paired_gav_floor(d, 0.5, 95, a_g4, 0.5, sigma=sg, raw_floor=eps):.2f}"
                 for d in d_grid]
        print(_fmt_row([str(sg), *cells]))

    print("\n## 5. G4-nono 'no practical effect' rules, eps = 0.01,"
          " true diff 0\n")
    sds = (0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04)
    g4_ns = (95, 108)
    print("All powers assume the full planned N is retained. H12-C TOST"
          " (EQUIVALENT); 'U<eps' = P(U < eps), which under first-match"
          " ordering is BOUNDED-NO-GAIN or POSITIVE-NEGLIGIBLE (exact"
          " noncentral t); 'BNG' = the exact BOUNDED-NO-GAIN event"
          " L <= 0 and U < eps (Monte Carlo). By SD of the paired"
          " difference:\n")
    print(_fmt_row(["rule, n", *[f"SD_D={x}" for x in sds]]))
    print(_fmt_row(["---"] * (len(sds) + 1)))
    for n in g4_ns:
        print(_fmt_row([f"TOST, n={n}",
                        *[f"{power_tost(0, x, n, a_g4, eps):.2f}" for x in sds]]))
    for n in g4_ns:
        print(_fmt_row([f"U<eps, n={n}",
                        *[f"{power_bound_exact(eps / x, n, a_g4):.2f}"
                          for x in sds]]))
    for n in g4_ns:
        print(_fmt_row([f"BNG, n={n}",
                        *[f"{power_bng_exact(0, x, n, a_g4, eps):.2f}"
                          for x in sds]]))
    print()
    for n in g4_ns:
        print(f"- n = {n}: 80 % needs SD_D <= {max_sd_tost(eps, n, a_g4):.4f}"
              f" (TOST), <= {max_sd_bound(eps, n, a_g4):.4f} (U < eps)")
    for x in (0.015, 0.02, 0.025, 0.03, 0.04):
        print(f"- SD_D = {x}: n for 80 % = {n_needed_tost(eps, x, a_g4)}"
              f" (TOST), {n_needed_bound(eps, x, a_g4)} (U < eps)")

    print("\n### 5b. N-selection rule, unconditional power"
          " (PI decision 2026-10-24)\n")
    print("N = 108 if the smoke SD^_D (3 seeds, ddof 1) >= 0.015, else"
          " N = 95. sigma_sel is the SD of WDPA(within) - WDPA(none),"
          " which drives the selection; with normal D,"
          " 2 SD^_D^2 / sigma_sel^2 ~ chi2(2), so P(N = 108) ="
          " exp(-(0.015 / sigma_sel)^2). Smoke seeds are disjoint from"
          " the confirmatory seeds, so power = P(N = 95) pow(95) +"
          " P(N = 108) pow(108). Full planned N retained, true diff 0.\n")
    sel_sds = (0.0075, 0.01, 0.015, 0.02, 0.025, 0.03, 0.04)

    def _p108(sig: float) -> float:
        return float(np.exp(-(0.015 / sig) ** 2))

    def _mix_at(pw, sig_sel: float, sig_con: float, n_lo: int = 95) -> float:
        q = _p108(sig_sel)
        return (1 - q) * pw(sig_con, n_lo) + q * pw(sig_con, 108)

    def _tost_at(x: float, n: int) -> float:
        return power_tost(0, x, n, a_g4, eps)

    def _bound_at(x: float, n: int) -> float:
        return power_bound_exact(eps / x, n, a_g4)

    def _bng_at(x: float, n: int) -> float:
        return power_bng_exact(0, x, n, a_g4, eps)

    print("H12-C (its contrast is the selector difference, so"
          " sigma_contrast = sigma_sel = SD_D):\n")
    print(_fmt_row(["", *[f"SD_D={x}" for x in sel_sds]]))
    print(_fmt_row(["---"] * (len(sel_sds) + 1)))
    print(_fmt_row(["P(N = 108)", *[f"{_p108(x):.2f}" for x in sel_sds]]))
    print(_fmt_row(["EQUIVALENT",
                    *[f"{_mix_at(_tost_at, x, x):.3f}" for x in sel_sds]]))
    print("\nFor comparison, the removed rule (30 / 95 / 108) when its r^"
          " line does not trigger (N = 30 unless SD^_D >= 0.015):\n")
    print(_fmt_row(["EQUIVALENT, old",
                    *[f"{_mix_at(_tost_at, x, x, 30):.3f}" for x in sel_sds]]))

    con_sds = (0.01, 0.015, 0.02, 0.025, 0.03, 0.04)
    sel_grid = (0.0075, 0.015, 0.03)
    for label, pw in (("U<eps (BNG or PN)", _bound_at), ("BNG exact", _bng_at)):
        print(f"\nH12-A/B {label}: rows sigma_sel (WDPA), columns"
              " sigma_contrast (CDPA SD of D):\n")
        print(_fmt_row(["sigma_sel", *[f"s_con={x}" for x in con_sds]]))
        print(_fmt_row(["---"] * (len(con_sds) + 1)))
        for sg in sel_grid:
            print(_fmt_row([str(sg), *[f"{_mix_at(pw, sg, x):.3f}"
                                       for x in con_sds]]))

    print("\n## 6. DR3 paired complete rules (units of SD of D), n = 20\n")
    print("SURVIVES = lower bound > 0 AND mean(D) >= delta;"
          " FAILS = upper bound < delta. a = .0125.\n")
    n_dr3 = 20
    ratios = (0.25, 0.5, 1.0, 1.5)
    mus = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    print(_fmt_row(["delta/SD", *[f"SURV mu={m}" for m in mus[1:]],
                    "FAILS mu=0"]))
    print(_fmt_row(["---"] * (len(mus) + 1)))
    for q in ratios:
        cells = [f"{power_superiority_floor(m, 1.0, n_dr3, a_dr3, q):.2f}"
                 for m in mus[1:]]
        cells.append(f"{power_bound_exact(q, n_dr3, a_dr3):.2f}")
        print(_fmt_row([str(q), *cells]))
    d3 = 0.025
    print(f"\nRaw delta = {d3}. FAILS (U < delta) power at mu = 0, n = 20:")
    for x in (0.0125, 0.018, 0.025, 0.035, 0.05):
        print(f"- SD(D) = {x}: {power_bound_exact(d3 / x, n_dr3, a_dr3):.2f};"
              f" n for 80 % = {n_needed_bound(d3, x, a_dr3)}")
    print(f"- n = 20 reaches 80 % iff SD(D) <= "
          f"{max_sd_bound(d3, n_dr3, a_dr3):.4f}")


if __name__ == "__main__":
    main()
