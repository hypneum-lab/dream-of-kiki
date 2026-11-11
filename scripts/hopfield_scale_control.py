"""Hopfield control arm of the DOWNSCALE dose-response (S5).

Pre-registration draft (L0), section 6 bis:
docs/milestones/2026-11-11-downscale-dose-response.md.

In a Hopfield network the stationary law of Glauber dynamics depends on the
couplings J and the temperature T only through J / T: multiplying J by
(1 - eps) is the same model as dividing T by (1 - eps), and at T = 0 the
update ``sign(J s)`` does not see the scale. This script runs the arms of the
dose-response on such a network and applies the decision rule of section 4 to
the final overlaps, to check that the rule returns the verdict known in
advance. Pure numpy, CPU.
"""

from __future__ import annotations

import argparse
import json
from typing import Any

import numpy as np

from scripts.e1_dose_response import (
    BROKEN_SHARE,
    DOSES,
    N_BOOT,
    RT_SEEDS,
    _boot_idx,
    equivalent,
    paired_interval,
    tolerance_rank,
)

N_UNITS = 1000
LOADS: tuple[float, ...] = (0.02, 0.05, 0.10)
TEMPERATURES: tuple[float, ...] = (0.0, 0.2, 0.4)
N_CUES = 200
SWEEPS = 50
CUE_OVERLAP = 0.8
PATTERN_SEED = 20261111
DYNAMICS_SEED = 1
INDEPENDENT_SEED = 2
IDENTITY_TOL = 1e-9
WIDE_NOISE_DOSES: tuple[float, ...] = (1.0, 2.0, 4.0)  # noise arm only, descriptive


def patterns(n: int, load: float, seed: int = PATTERN_SEED) -> np.ndarray:
    p = max(1, round(load * n))
    xi: np.ndarray = np.random.default_rng(seed).choice(np.array([-1.0, 1.0]), size=(p, n))
    return xi


def hebbian(xi: np.ndarray) -> np.ndarray:
    j: np.ndarray = xi.T @ xi / xi.shape[1]
    np.fill_diagonal(j, 0.0)
    return j


def noisy(j: np.ndarray, eps: float, seed: int) -> np.ndarray:
    """J plus a symmetric Gaussian matrix of Frobenius norm eps * ||J||, zero diagonal."""
    g = np.random.default_rng(1000 + seed).normal(size=j.shape)
    g = (g + g.T) / 2
    np.fill_diagonal(g, 0.0)
    out: np.ndarray = j + g * (eps * np.linalg.norm(j) / np.linalg.norm(g))
    return out


def cues(xi: np.ndarray, n_cues: int, overlap: float = CUE_OVERLAP) -> tuple[np.ndarray, np.ndarray]:
    """Cue c is pattern c mod P with a fraction (1 - overlap) / 2 of its units flipped."""
    p, n = xi.shape
    rng = np.random.default_rng(PATTERN_SEED + 1)
    which = np.arange(n_cues) % p
    s = xi[which].copy()
    k = round(n * (1 - overlap) / 2)
    for c in range(n_cues):
        s[c, rng.choice(n, size=k, replace=False)] *= -1
    return s, which


def run(j: np.ndarray, temp: float, s0: np.ndarray, target: np.ndarray, sweeps: int,
        seed: int) -> np.ndarray:
    """Final overlap of each cue with its pattern after asynchronous Glauber sweeps.

    All cues share the update order and each has its own uniform draw per
    update; both come from ``seed`` only, so two calls with the same seed use
    common random numbers.
    """
    rng = np.random.default_rng(seed)
    s = s0.copy()
    n = s.shape[1]
    h = s @ j  # j is symmetric: h[c, i] = sum_k j[i, k] s[c, k]
    for _ in range(sweeps):
        order = rng.permutation(n)
        u = rng.random((n, s.shape[0]))
        for step, i in enumerate(order):
            hi = h[:, i]
            if temp > 0:
                new = np.where(u[step] < 1.0 / (1.0 + np.exp(-2.0 * hi / temp)), 1.0, -1.0)
            else:
                new = np.where(hi > 0, 1.0, np.where(hi < 0, -1.0, s[:, i]))
            delta = new - s[:, i]
            moved = np.nonzero(delta)[0]
            if moved.size:
                h[moved] += delta[moved, None] * j[i][None, :]
                s[moved, i] = new[moved]
    out: np.ndarray = (s * target).mean(1)
    return out


def broken(arm: np.ndarray, base: np.ndarray, idx: np.ndarray) -> bool:
    """Rule of section 4 with overlap in place of accuracy: chance level is 0."""
    margin = float(base.mean())
    if margin <= 0:
        return False
    drop = base - arm
    lo, _ = paired_interval(drop, idx)
    return float(drop.mean()) >= BROKEN_SHARE * margin and lo > 0


def c1(eq_d: list[bool], eq_rt: list[bool], br_rt: list[bool]) -> tuple[str, int, int]:
    rank_d, rank_rt = tolerance_rank(eq_d), tolerance_rank(eq_rt)
    if not (eq_rt[0] and any(br_rt)):
        return "non_test", rank_d, rank_rt
    if rank_d - rank_rt >= 2:
        return "supported", rank_d, rank_rt
    if rank_d <= rank_rt:
        return "refuted", rank_d, rank_rt
    return "inconclusive", rank_d, rank_rt


def cell(load: float, temp: float, n: int = N_UNITS, n_cues: int = N_CUES,
         sweeps: int = SWEEPS, n_boot: int = N_BOOT) -> dict[str, Any]:
    xi = patterns(n, load)
    j = hebbian(xi)
    s0, which = cues(xi, n_cues)
    target = xi[which]
    idx = _boot_idx(n_cues, n_boot)
    base = run(j, temp, s0, target, sweeps, DYNAMICS_SEED)
    eq_d, eq_rt, br_rt, br_d, eq_tq, table = [], [], [], [], [], []
    rt0: list[np.ndarray] = []
    identity = 0.0
    for eps in DOSES:
        d = run(j * (1 - eps), temp, s0, target, sweeps, DYNAMICS_SEED)
        tq = run(j, temp / (1 - eps), s0, target, sweeps, DYNAMICS_SEED)
        identity = max(identity, float(np.abs(d - tq).max()))
        seeds = [run(noisy(j, eps, k), temp, s0, target, sweeps, DYNAMICS_SEED) for k in RT_SEEDS]
        rt = np.mean(seeds, axis=0)
        rt0.append(seeds[0])
        eq_d.append(equivalent(d, base, idx))
        eq_rt.append(equivalent(rt, base, idx))
        br_rt.append(broken(rt, base, idx))
        br_d.append(broken(d, base, idx))
        row = {"eps": eps, "m_D": float(d.mean()), "m_Rt": float(rt.mean()),
               "equiv_D": eq_d[-1], "equiv_Rt": eq_rt[-1], "broken_D": br_d[-1],
               "broken_Rt": br_rt[-1]}
        if temp > 0:  # C2 analogue: D against the base network at T / (1 - eps), other stream
            ind = run(j, temp / (1 - eps), s0, target, sweeps, INDEPENDENT_SEED)
            eq_tq.append(equivalent(d, ind, idx))
            row |= {"m_Tq_independent": float(ind.mean()), "equiv_D_Tq": eq_tq[-1]}
        table.append(row)
    wide = []
    for eps in WIDE_NOISE_DOSES:  # beyond the grid: D is undefined at eps >= 1
        rt = np.mean([run(noisy(j, eps, k), temp, s0, target, sweeps, DYNAMICS_SEED)
                      for k in RT_SEEDS], axis=0)
        wide.append({"eps": eps, "m_Rt": float(rt.mean()), "equiv_Rt": equivalent(rt, base, idx),
                     "broken_Rt": broken(rt, base, idx)})
    verdict, rank_d, rank_rt = c1(eq_d, eq_rt, br_rt)
    eq_fake = [equivalent(x, base, idx) for x in rt0]  # Rt seed 0 relabelled as D
    negative, _, _ = c1(eq_fake, eq_rt, br_rt)
    return {"load": load, "T": temp, "n_units": n, "n_patterns": int(xi.shape[0]),
            "m_base": float(base.mean()), "identity_max_abs_diff": identity,
            "identity_ok": identity <= IDENTITY_TOL, "rank_D": rank_d, "rank_Rt": rank_rt,
            "C1": verdict, "C1_negative_control": negative,
            "C2_analogue": ("not_applicable" if temp == 0
                            else "supported" if all(eq_tq) else "refuted"),
            "table": table, "wide_noise": wide}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--units", type=int, default=N_UNITS)
    ap.add_argument("--cues", type=int, default=N_CUES)
    ap.add_argument("--sweeps", type=int, default=SWEEPS)
    args = ap.parse_args()
    out = [cell(a, t, args.units, args.cues, args.sweeps) for a in LOADS for t in TEMPERATURES]
    print(json.dumps(out, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
