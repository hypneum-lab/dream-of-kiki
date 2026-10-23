#!/usr/bin/env python3
"""Power numbers for the 2026-10-23 draft pre-registrations.

Reproduces every power / minimum-detectable-effect figure quoted in

- docs/osf-prereg-g4-nono-cross-domain-2026-10-23.md   (H12)
- docs/osf-prereg-dr3-relational-probe-2026-10-23.md   (H13)
- docs/osf-prereg-e1-replay-controls-2026-10-23.md     (H14)
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


def _fmt_row(cells) -> str:
    return "| " + " | ".join(cells) + " |"


def main() -> None:
    a_g4 = 0.05 / 3      # 0.0167
    a_dr3 = 0.05 / 4     # 0.0125
    a_e1 = 0.05 / 3      # 0.0167

    print("## 1. Test-rejection MDE at 80 % power (exact noncentral t)\n")
    for label, n, a, sided in [
        ("DR3 independent, n=10/group, one-sided a=.0125", 10, a_dr3, 1),
        ("G4 independent, n=30/group, one-sided a=1/60", 30, a_g4, 1),
        ("G4 independent, n=95/group, one-sided a=1/60", 95, a_g4, 1),
    ]:
        d = mde_two_sample(n, a, sided)
        print(f"- {label}: d = {d:.3f}, Hedges g = "
              f"{d * hedges_j(2 * n - 2):.3f}")
    for label, n, a in [
        ("DR3 paired, n=10, one-sided a=.0125", 10, a_dr3),
        ("G4 paired, n=30, one-sided a=1/60", 30, a_g4),
        ("G4 paired, n=95, one-sided a=1/60", 95, a_g4),
        ("E1 paired, n=141, one-sided a=1/60", 141, a_e1),
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
    for n in (30, 95):
        dz = mde_paired(n, a_g4)
        print(_fmt_row([str(n), *[f"{dz * np.sqrt(2 * (1 - r)):.3f}"
                                  for r in rs]]))

    print("\n## 4. G4-nono complete rule (paired t AND g_av >= 0.5)\n")
    d_grid = (0.5, 0.6, 0.8, 1.0)
    for n in (30, 95):
        print(f"\nn = {n} (rows: r; columns: true d_av)\n")
        print(_fmt_row(["r", *[f"d_av={d}" for d in d_grid]]))
        print(_fmt_row(["---"] * (len(d_grid) + 1)))
        for r in rs:
            cells = [f"{power_paired_gav_floor(d, r, n, a_g4, 0.5):.2f}"
                     for d in d_grid]
            print(_fmt_row([str(r), *cells]))

    print("\n### 4b. Same rule plus raw floor mean(D) >= 0.01 (n = 30, r = 0.5)\n")
    sigmas = (0.01, 0.02, 0.04)
    print(_fmt_row(["arm SD", *[f"d_av={d}" for d in d_grid]]))
    print(_fmt_row(["---"] * (len(d_grid) + 1)))
    for sg in sigmas:
        cells = [f"{power_paired_gav_floor(d, 0.5, 30, a_g4, 0.5, sigma=sg, raw_floor=0.01):.2f}"
                 for d in d_grid]
        print(_fmt_row([str(sg), *cells]))

    print("\n## 5. G4-nono H12-C paired TOST, raw margin 0.01, true diff 0\n")
    sds = (0.005, 0.0075, 0.01, 0.015, 0.02)
    print(_fmt_row(["n", *[f"SD_D={s}" for s in sds]]))
    print(_fmt_row(["---"] * (len(sds) + 1)))
    for n in (30, 95):
        print(_fmt_row([str(n), *[f"{power_tost(0, s, n, a_g4, 0.01):.2f}"
                                  for s in sds]]))

    print("\n## 6. DR3 paired complete rules (units of SD of D)\n")
    print("SURVIVES = lower bound > 0 AND mean(D) >= delta;"
          " FAILS = upper bound < delta. n = 10, a = .0125.\n")
    ratios = (0.25, 0.5, 1.0)
    mus = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0)
    print(_fmt_row(["delta/SD", *[f"SURV mu={m}" for m in mus[1:]],
                    "FAILS mu=0"]))
    print(_fmt_row(["---"] * (len(mus) + 1)))
    for q in ratios:
        cells = [f"{power_superiority_floor(m, 1.0, 10, a_dr3, q):.2f}"
                 for m in mus[1:]]
        cells.append(f"{power_upper_bound_below(0.0, 1.0, 10, a_dr3, q):.2f}")
        print(_fmt_row([str(q), *cells]))

    print("\n## 7. E1 (n = 141, a = 1/60)\n")
    print("Standardised check, margin = 0.29 SD, true diff 0:"
          f" NI = {power_noninferiority(0, 1, 141, a_e1, 0.29):.3f},"
          f" TOST = {power_tost(0, 1, 141, a_e1, 0.29):.3f}\n")
    delta = 0.02
    sds = (0.03, 0.05, 0.069, 0.10)
    print(f"Raw SESOI delta = {delta} composite accuracy.\n")
    print(_fmt_row(["SD_D", "delta/SD_D", "SUP mu=0.02", "SUP mu=0.03",
                    "SUP mu=0.04", "BOUNDED<delta mu=0", "NI mu=0",
                    "TOST mu=0"]))
    print(_fmt_row(["---"] * 8))
    for s in sds:
        print(_fmt_row([
            str(s), f"{delta / s:.2f}",
            f"{power_superiority_floor(0.02, s, 141, a_e1, delta):.2f}",
            f"{power_superiority_floor(0.03, s, 141, a_e1, delta):.2f}",
            f"{power_superiority_floor(0.04, s, 141, a_e1, delta):.2f}",
            f"{power_upper_bound_below(0.0, s, 141, a_e1, delta):.2f}",
            f"{power_noninferiority(0.0, s, 141, a_e1, delta):.2f}",
            f"{power_tost(0.0, s, 141, a_e1, delta):.2f}",
        ]))


if __name__ == "__main__":
    main()
