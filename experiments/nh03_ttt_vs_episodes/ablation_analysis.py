"""NH-03 schedule ablation : recency vs spreading (2 x 2 + uniform).

Per substrate ``fam`` in {ttt (primary), slow (secondary)} and per
seed, with retention points ``EB, ES, LB, LS`` (early/late x
block/spread) :

- recency   R = (LB + LS - EB - ES) / 2   (late minus early)
- spreading S = (ES + LS - EB - LB) / 2   (spread minus block)
- D = R - S = LB - ES                     (late block vs early spread)

Each estimate = mean over seeds, 95 % percentile bootstrap CI over
seeds (10 000 resamples, RNG 20261027). Decision (pre-reg §5) :

- ``recency``      : R > 3, CI_R low > 0 and CI_D low > 0
- ``spreading``    : S > 3, CI_S low > 0 and CI_D high < 0
- ``both``         : R > 3 and S > 3, both CI lows > 0, CI_D spans 0
- ``inconclusive`` : otherwise

Reference : docs/milestones/2026-10-27-nh03-schedule-ablation.md
"""
from __future__ import annotations

from typing import Any

import numpy as np

from experiments.nh03_ttt_vs_episodes.analysis import bootstrap_ci

ABLATION_BOOTSTRAP_SEED = 20261027
MARGIN = 3.0
FAMILIES = ("ttt", "slow")
CELLS = ("early_block", "early_spread", "late_block", "late_spread", "uniform")


def _per_seed(cells: list[dict[str, Any]], fam: str) -> dict[str, np.ndarray]:
    by: dict[tuple[str, int], dict[str, Any]] = {
        (c["arm"], int(c["seed"])): c for c in cells
    }
    seeds = sorted(
        s for s in {int(c["seed"]) for c in cells}
        if all((f"{fam}_{k}", s) in by for k in CELLS)
        and not any(by[(f"{fam}_{k}", s)]["excluded_underperforming_baseline"]
                    for k in CELLS)
    )
    out = {k: np.asarray([100.0 * by[(f"{fam}_{k}", s)]["retention"] for s in seeds])
           for k in CELLS}
    out["_seeds"] = np.asarray(seeds)
    return out


def _est(x: np.ndarray) -> dict[str, Any]:
    lo, hi = bootstrap_ci(x, seed=ABLATION_BOOTSTRAP_SEED)
    sd = float(x.std(ddof=1)) if x.size > 1 else float("nan")
    return {"mean": float(x.mean()), "sd": sd, "ci95": [lo, hi]}


def decide(r: dict[str, Any], s: dict[str, Any], d: dict[str, Any]) -> str:
    r_ok = r["mean"] > MARGIN and r["ci95"][0] > 0.0
    s_ok = s["mean"] > MARGIN and s["ci95"][0] > 0.0
    if r_ok and d["ci95"][0] > 0.0:
        return "recency"
    if s_ok and d["ci95"][1] < 0.0:
        return "spreading"
    if r_ok and s_ok and d["ci95"][0] <= 0.0 <= d["ci95"][1]:
        return "both"
    return "inconclusive"


def analyse_family(cells: list[dict[str, Any]], fam: str) -> dict[str, Any]:
    v = _per_seed(cells, fam)
    n = int(v["_seeds"].size)
    if n < 2:
        return {"family": fam, "n": n, "insufficient": True}
    eb, es, lb, ls, u = (v[k] for k in CELLS)
    r = _est((lb + ls - eb - es) / 2.0)
    s = _est((es + ls - eb - lb) / 2.0)
    d = _est(lb - es)
    inter = _est((lb - eb) - (ls - es))
    return {
        "family": fam,
        "n": n,
        "seeds": v["_seeds"].tolist(),
        "arm_means": {k: float(v[k].mean()) for k in CELLS},
        "recency_R": r,
        "spreading_S": s,
        "late_block_minus_early_spread_D": d,
        "interaction_recency_x_block": inter,
        "uniform_minus": {k: _est(u - v[k]) for k in CELLS if k != "uniform"},
        "verdict": decide(r, s, d),
    }


def analyse_ablation(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "primary": analyse_family(cells, "ttt"),
        "secondary": analyse_family(cells, "slow"),
    }
