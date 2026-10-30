"""NH-03 deferred evaluation : does recency survive a write-free delay ?

Same 2 x 2 schedule factorial as the schedule ablation (early/late x
block/spread, + uniform) on two substrates (``ttt`` primary, ``slow``
secondary), but task-0 retention is read at three delay levels after
the last write : ``d0`` (no delay, = schedule ablation), ``d1`` (one
epoch of write-free wake training on the delay task) and ``d3`` (three
epochs = one full task, primary level).

Per substrate, delay level L and seed, with retention points
``EB, ES, LB, LS`` at L :

- recency   R_L = (LB + LS - EB - ES) / 2
- spreading S_L = (ES + LS - EB - LB) / 2
- D_L = LB - ES
- attenuation dR_L = R_L - R_d0 (paired by seed)

Each estimate = mean over seeds, 95 % percentile bootstrap CI over
seeds (10 000 resamples, RNG 20261030). Decision (pre-reg §6, TTT
substrate, L = d3) :

- control gate : R_d0 > 3 and CI low > 0, otherwise ``invalid_control``
- ``recency_survives`` : R_d3 > 3 and CI low > 0
- ``recency_below_margin`` : CI high of R_d3 < 3
- ``inconclusive`` : otherwise

Secondary labels (simple sign tests on the CI, no margin) :
``attenuates`` (CI high of dR_d3 < 0) and ``spreading_positive``
(CI low of S_d3 > 0).

Reference : docs/milestones/2026-10-30-nh03-deferred-eval.md
"""
from __future__ import annotations

from typing import Any

import numpy as np

from experiments.nh03_ttt_vs_episodes.analysis import bootstrap_ci

DEFERRED_BOOTSTRAP_SEED = 20261030
MARGIN = 3.0
FAMILIES = ("ttt", "slow")
CELLS = ("early_block", "early_spread", "late_block", "late_spread", "uniform")
LEVELS = ("d0", "d1", "d3")
PRIMARY_LEVEL = "d3"


def _per_seed(
    cells: list[dict[str, Any]], fam: str
) -> tuple[list[int], dict[str, dict[str, np.ndarray]]]:
    """Seeds complete for ``fam`` (5 schedules, 3 levels, not excluded)
    and per-level, per-schedule retention points."""
    by: dict[tuple[str, int], dict[str, Any]] = {
        (c["arm"], int(c["seed"])): c for c in cells
    }

    def ok(s: int) -> bool:
        for k in CELLS:
            c = by.get((f"{fam}_{k}", s))
            if c is None or c["excluded_underperforming_baseline"]:
                return False
            rbd = c.get("retention_by_delay")
            if not rbd or any(lv not in rbd for lv in LEVELS):
                return False
        return True

    seeds = sorted(s for s in {int(c["seed"]) for c in cells} if ok(s))
    out = {
        lv: {k: np.asarray(
            [100.0 * by[(f"{fam}_{k}", s)]["retention_by_delay"][lv] for s in seeds])
            for k in CELLS}
        for lv in LEVELS
    }
    return seeds, out


def _est(x: np.ndarray) -> dict[str, Any]:
    lo, hi = bootstrap_ci(x, seed=DEFERRED_BOOTSTRAP_SEED)
    sd = float(x.std(ddof=1)) if x.size > 1 else float("nan")
    return {"mean": float(x.mean()), "sd": sd, "ci95": [lo, hi]}


def _effects(v: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    eb, es, lb, ls = (v[k] for k in ("early_block", "early_spread",
                                     "late_block", "late_spread"))
    return {
        "R": (lb + ls - eb - es) / 2.0,
        "S": (es + ls - eb - lb) / 2.0,
        "D": lb - es,
    }


def control_ok(r_d0: dict[str, Any]) -> bool:
    return r_d0["mean"] > MARGIN and r_d0["ci95"][0] > 0.0


def decide(r_d0: dict[str, Any], r_l: dict[str, Any]) -> str:
    if not control_ok(r_d0):
        return "invalid_control"
    if r_l["mean"] > MARGIN and r_l["ci95"][0] > 0.0:
        return "recency_survives"
    if r_l["ci95"][1] < MARGIN:
        return "recency_below_margin"
    return "inconclusive"


def secondary_labels(dr_l: dict[str, Any], s_l: dict[str, Any]) -> dict[str, bool]:
    return {
        "attenuates": bool(dr_l["ci95"][1] < 0.0),
        "spreading_positive": bool(s_l["ci95"][0] > 0.0),
    }


def analyse_family(cells: list[dict[str, Any]], fam: str) -> dict[str, Any]:
    seeds, v = _per_seed(cells, fam)
    n = len(seeds)
    if n < 2:
        return {"family": fam, "n": n, "insufficient": True}
    eff = {lv: _effects(v[lv]) for lv in LEVELS}
    levels: dict[str, Any] = {}
    for lv in LEVELS:
        levels[lv] = {
            "arm_means": {k: float(v[lv][k].mean()) for k in CELLS},
            "recency_R": _est(eff[lv]["R"]),
            "spreading_S": _est(eff[lv]["S"]),
            "late_block_minus_early_spread_D": _est(eff[lv]["D"]),
            "uniform_minus_early_block": _est(v[lv]["uniform"] - v[lv]["early_block"]),
        }
        if lv != "d0":
            levels[lv]["attenuation_dR"] = _est(eff[lv]["R"] - eff["d0"]["R"])
            levels[lv]["spreading_change_dS"] = _est(eff[lv]["S"] - eff["d0"]["S"])
    r_d0 = levels["d0"]["recency_R"]
    r_l = levels[PRIMARY_LEVEL]["recency_R"]
    return {
        "family": fam,
        "n": n,
        "seeds": seeds,
        "levels": levels,
        "control_d0": control_ok(r_d0),
        "verdict": decide(r_d0, r_l),
        "secondary": secondary_labels(
            levels[PRIMARY_LEVEL]["attenuation_dR"],
            levels[PRIMARY_LEVEL]["spreading_S"],
        ),
    }


def cross_substrate(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Descriptive : TTT minus slow per schedule and level, and the NH-03
    primary contrast (ttt_uniform - slow_early_block) per level."""
    st, vt = _per_seed(cells, "ttt")
    ss, vs = _per_seed(cells, "slow")
    if st != ss or len(st) < 2:
        return {"insufficient": True}
    return {
        lv: {
            **{f"ttt_minus_slow_{k}": _est(vt[lv][k] - vs[lv][k]) for k in CELLS},
            "nh03_primary_ttt_uniform_minus_slow_early_block":
                _est(vt[lv]["uniform"] - vs[lv]["early_block"]),
        }
        for lv in LEVELS
    }


def analyse_deferred(cells: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "primary": analyse_family(cells, "ttt"),
        "secondary": analyse_family(cells, "slow"),
        "cross_substrate": cross_substrate(cells),
    }
