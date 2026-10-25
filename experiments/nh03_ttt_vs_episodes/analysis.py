"""NH-03 paired analysis : retention differences with bootstrap CI.

Unit = seed. For each contrast ``X - Y`` the statistic is the mean
over seeds of ``100 * (retention_X - retention_Y)`` (retention points),
with a percentile bootstrap 95 % CI over seeds (10 000 resamples,
fixed RNG). Pre-registered decision rule (primary contrast
``B_ttt_continuous - A_discrete``) :

- ``continuous_gt_bounded``     : mean > 3 and CI low > 0
- ``equivalent_at_this_scale``  : |mean| < 1 and CI within [-3, 3]
- ``inconclusive``              : otherwise

Reference : docs/milestones/2026-10-25-nh03-ttt-vs-episodes.md §6
"""
from __future__ import annotations

import math
from typing import Any

import numpy as np

BOOTSTRAP_RESAMPLES = 10_000
BOOTSTRAP_SEED = 20261025
SUPERIORITY_MARGIN = 3.0
EQUIV_POINT = 1.0
EQUIV_BOUND = 3.0

PRIMARY = ("B_ttt_continuous", "A_discrete")
SECONDARY: tuple[tuple[str, str], ...] = (
    ("A_discrete", "C_sgd_matched"),
    ("B_ttt_continuous", "C_sgd_matched"),
    ("A_discrete", "baseline"),
    ("B_ttt_continuous", "baseline"),
    ("C_sgd_matched", "baseline"),
)
EXPLORATORY: tuple[tuple[str, str], ...] = (
    ("A_continuous", "A_discrete"),
    ("B_ttt_continuous", "B_ttt_chunked"),
    ("B_ttt_chunked", "A_discrete"),
)


def paired_points(
    cells: list[dict[str, Any]], arm_x: str, arm_y: str
) -> tuple[list[int], np.ndarray]:
    """Per-seed ``100 * (ret_X - ret_Y)`` over seeds present in both
    arms and not excluded (acc_initial < 0.5, G4 rule)."""
    by: dict[tuple[str, int], dict[str, Any]] = {
        (c["arm"], int(c["seed"])): c for c in cells
    }
    seeds = sorted(
        {s for (a, s) in by if a == arm_x} & {s for (a, s) in by if a == arm_y}
    )
    kept: list[int] = []
    diffs: list[float] = []
    for s in seeds:
        cx, cy = by[(arm_x, s)], by[(arm_y, s)]
        if cx["excluded_underperforming_baseline"] or cy[
            "excluded_underperforming_baseline"
        ]:
            continue
        kept.append(s)
        diffs.append(100.0 * (cx["retention"] - cy["retention"]))
    return kept, np.asarray(diffs, dtype=np.float64)


def bootstrap_ci(
    diffs: np.ndarray,
    *,
    n_resamples: int = BOOTSTRAP_RESAMPLES,
    seed: int = BOOTSTRAP_SEED,
    level: float = 0.95,
) -> tuple[float, float]:
    if diffs.size == 0:
        return (math.nan, math.nan)
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, diffs.size, size=(n_resamples, diffs.size))
    means = diffs[idx].mean(axis=1)
    lo = float(np.quantile(means, (1.0 - level) / 2.0))
    hi = float(np.quantile(means, 1.0 - (1.0 - level) / 2.0))
    return lo, hi


def classify(mean: float, ci: tuple[float, float]) -> dict[str, str]:
    lo, hi = ci
    if mean > SUPERIORITY_MARGIN and lo > 0.0:
        verdict = "continuous_gt_bounded"
    elif abs(mean) < EQUIV_POINT and lo >= -EQUIV_BOUND and hi <= EQUIV_BOUND:
        verdict = "equivalent_at_this_scale"
    else:
        verdict = "inconclusive"
    if hi < 0.0:
        direction = "bounded_gt_continuous_descriptive"
    elif lo > 0.0:
        direction = "continuous_gt_bounded_descriptive"
    else:
        direction = "ci_includes_zero"
    return {"verdict": verdict, "direction": direction}


def contrast(cells: list[dict[str, Any]], arm_x: str, arm_y: str) -> dict[str, Any]:
    seeds, d = paired_points(cells, arm_x, arm_y)
    if d.size < 2:
        return {"x": arm_x, "y": arm_y, "n": int(d.size), "insufficient": True}
    mean = float(d.mean())
    sd = float(d.std(ddof=1))
    ci = bootstrap_ci(d)
    dz = mean / sd if sd > 0 else math.inf
    # Hedges small-sample correction for paired d_z.
    n = d.size
    j = 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)
    return {
        "x": arm_x,
        "y": arm_y,
        "n": int(n),
        "seeds": seeds,
        "mean_points": mean,
        "sd_points": sd,
        "ci95_points": list(ci),
        "hedges_gz": float(dz * j) if math.isfinite(dz) else dz,
        "n_seeds_diff_gt_3": int((d > SUPERIORITY_MARGIN).sum()),
        "n_seeds_diff_lt_minus_3": int((d < -SUPERIORITY_MARGIN).sum()),
        **classify(mean, ci),
    }


def arm_summary(cells: list[dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for arm in sorted({c["arm"] for c in cells}):
        rs = [
            c for c in cells
            if c["arm"] == arm and not c["excluded_underperforming_baseline"]
        ]
        ret = np.asarray([c["retention"] for c in rs], dtype=np.float64)
        out[arm] = {
            "n": int(ret.size),
            "mean_retention_points": float(100 * ret.mean()) if ret.size else math.nan,
            "sd_retention_points": float(100 * ret.std(ddof=1)) if ret.size > 1 else math.nan,
            "writes_per_cell": sorted({int(c["writes_total"]) for c in rs}),
            "mean_consolidation_flops": float(
                np.mean([c["consolidation_flops"] for c in rs])
            ) if rs else math.nan,
        }
    return out


def analyse(cells: list[dict[str, Any]]) -> dict[str, Any]:
    present = {c["arm"] for c in cells}

    def ok(pair: tuple[str, str]) -> bool:
        return pair[0] in present and pair[1] in present

    return {
        "primary": contrast(cells, *PRIMARY) if ok(PRIMARY) else None,
        "secondary": [contrast(cells, *p) for p in SECONDARY if ok(p)],
        "exploratory": [contrast(cells, *p) for p in EXPLORATORY if ok(p)],
        "arms": arm_summary(cells),
    }
