"""DOWNSCALE dose-response on the base model: plan of the passes and verdict.

Pre-registration draft (L0): docs/milestones/2026-11-11-downscale-dose-response.md.
The harness mode ``dose-response`` writes one JSON line per scoring pass; this
module reads them and prints the verdict of section 4 (C1, quasi-symmetry;
C2, temperature). Pure numpy: no model, no GPU.
"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np

DOSES: tuple[float, ...] = (0.01, 0.02, 0.05, 0.1, 0.2, 0.5)  # eps = 1 - f, relative norm
RT_SEEDS: tuple[int, ...] = (0, 1, 2)
RG_DOSES: tuple[float, ...] = (0.01, 0.05, 0.2)
PRIMARY_SOURCES: tuple[str, ...] = ("mmlu_cf", "realtimeqa")
CHANCE = 0.25
EQUIV_MARGIN = 0.03
BROKEN_SHARE = 0.5
LEVEL = 0.99
N_BOOT = 10_000
BOOT_SEED = 20261111
T_GRID = np.exp(np.linspace(np.log(0.25), np.log(4.0), 61))
KL_FLOOR = 0.005
RESTORE_CHECK_ITEMS = 50


def plan() -> list[tuple[str, float | None, int | None]]:
    """The 28 scoring passes, in run order: (arm, eps, seed)."""
    out: list[tuple[str, float | None, int | None]] = [("base", None, None)]
    for eps in DOSES:
        out.append(("D", eps, None))
        out.extend(("Rt", eps, s) for s in RT_SEEDS)
        if eps in RG_DOSES:
            out.append(("Rg", eps, 0))
    return out


def _softmax(x: np.ndarray) -> np.ndarray:
    z = x - x.max(-1, keepdims=True)
    e = np.exp(z)
    out: np.ndarray = e / e.sum(-1, keepdims=True)
    return out


def _boot_idx(n: int, n_boot: int) -> np.ndarray:
    idx: np.ndarray = np.random.default_rng(BOOT_SEED).integers(0, n, size=(n_boot, n))
    return idx


def paired_interval(diff: np.ndarray, idx: np.ndarray) -> tuple[float, float]:
    """Percentile interval at LEVEL of the mean of a per-item difference."""
    means = diff[idx].mean(-1)
    lo, hi = np.percentile(means, [100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2])
    return float(lo), float(hi)


def equivalent(arm: np.ndarray, base: np.ndarray, idx: np.ndarray) -> bool:
    lo, hi = paired_interval(arm - base, idx)
    return -EQUIV_MARGIN <= lo and hi <= EQUIV_MARGIN


def broken(arm: np.ndarray, base: np.ndarray, idx: np.ndarray) -> bool:
    margin = float(base.mean()) - CHANCE
    if margin <= 0:
        return False
    drop = base - arm
    lo, _ = paired_interval(drop, idx)
    return float(drop.mean()) >= BROKEN_SHARE * margin and lo > 0


def tolerance_rank(equiv: Sequence[bool]) -> int:
    """Number of leading doses at which the arm is equivalent (0 = none)."""
    rank = 0
    for ok in equiv:
        if not ok:
            break
        rank += 1
    return rank


def temperature_fit(s_base: np.ndarray, s_arm: np.ndarray, idx: np.ndarray) -> dict[str, Any]:
    """One temperature on the base scores against the arm's option distribution."""
    p = _softmax(s_arm)
    logp = np.log(np.clip(p, 1e-300, None))
    kl = np.empty((s_base.shape[0], T_GRID.size))
    for j, t in enumerate(T_GRID):
        q = _softmax(s_base / t)
        kl[:, j] = (p * (logp - np.log(np.clip(q, 1e-300, None)))).sum(-1)
    j1 = int(np.argmin(np.abs(T_GRID - 1.0)))
    mean = kl.mean(0)
    jb = int(np.argmin(mean))
    kl1 = float(mean[j1])
    ts = np.empty(idx.shape[0])
    shares = np.empty(idx.shape[0])
    for lo in range(0, idx.shape[0], 500):
        m = kl[idx[lo:lo + 500]].mean(1)  # (b, T)
        k = m.argmin(1)
        ts[lo:lo + 500] = T_GRID[k]
        d = m[:, j1]
        shares[lo:lo + 500] = np.where(d > 0, 1 - m[np.arange(k.size), k] / np.where(d > 0, d, 1), 0)
    q_lo, q_hi = 100 * (1 - LEVEL) / 2, 100 * (1 + LEVEL) / 2
    return {
        "T": float(T_GRID[jb]), "T_lo": float(np.percentile(ts, q_lo)),
        "T_hi": float(np.percentile(ts, q_hi)), "kl_at_1": kl1,
        "share_removed": float(1 - mean[jb] / kl1) if kl1 > 0 else 0.0,
        "share_lo": float(np.percentile(shares, q_lo)),
    }


def c2_verdict(rank_d: int, fit: dict[str, Any] | None) -> str:
    if rank_d == 0 or fit is None or fit["kl_at_1"] < KL_FLOOR:
        return "not_applicable"
    if fit["T_lo"] > 1 and fit["share_removed"] >= 0.8 and fit["share_lo"] >= 0.6:
        return "supported"
    if fit["T_lo"] <= 1 <= fit["T_hi"] or fit["share_removed"] < 0.5:
        return "refuted"
    return "inconclusive"


def analyse(passes: Sequence[dict[str, Any]], answers: np.ndarray, keep: np.ndarray,
            n_boot: int = N_BOOT) -> dict[str, Any]:
    """Verdict from the passes. ``passes``: dicts with arm, eps, seed, scores (n, K)."""
    ans = answers[keep]
    idx = _boot_idx(int(keep.sum()), n_boot)

    def scores(arm: str, eps: float | None, seed: int | None) -> np.ndarray:
        hit = [p for p in passes if p["arm"] == arm and p["eps"] == eps and p["seed"] == seed]
        if len(hit) != 1:
            raise SystemExit(f"expected one pass for {arm} eps={eps} seed={seed}, got {len(hit)}")
        arr: np.ndarray = np.asarray(hit[0]["scores"], dtype=np.float64)[keep]
        return arr

    def correct(s: np.ndarray) -> np.ndarray:
        ok: np.ndarray = (s.argmax(-1) == ans).astype(np.float64)  # ties -> lowest index
        return ok

    s0 = scores("base", None, None)
    base = correct(s0)
    table: list[dict[str, Any]] = []
    eq_d, eq_rt, br_rt = [], [], []
    for eps in DOSES:
        sd = scores("D", eps, None)
        d = correct(sd)
        rt = np.mean([correct(scores("Rt", eps, s)) for s in RT_SEEDS], axis=0)
        eq_d.append(equivalent(d, base, idx))
        eq_rt.append(equivalent(rt, base, idx))
        br_rt.append(broken(rt, base, idx))
        table.append({"eps": eps, "acc_D": float(d.mean()), "acc_Rt": float(rt.mean()),
                      "equiv_D": eq_d[-1], "equiv_Rt": eq_rt[-1], "broken_Rt": br_rt[-1],
                      "conf_D": float(_softmax(sd).max(-1).mean())})
    rank_d, rank_rt = tolerance_rank(eq_d), tolerance_rank(eq_rt)
    if not (eq_rt[0] and any(br_rt)):
        c1 = "non_test"
    elif rank_d - rank_rt >= 2:
        c1 = "supported"
    elif rank_d <= rank_rt:
        c1 = "refuted"
    else:
        c1 = "inconclusive"
    fit = temperature_fit(s0, scores("D", DOSES[rank_d - 1], None), idx) if rank_d else None
    return {"n_items": int(keep.sum()), "acc_base": float(base.mean()),
            "conf_base": float(_softmax(s0).max(-1).mean()),
            "rank_D": rank_d, "rank_Rt": rank_rt, "C1": c1,
            "C2": "not_applicable" if c1 == "non_test" else c2_verdict(rank_d, fit),
            "temperature": fit, "table": table}


def load(path: Path) -> tuple[list[dict[str, Any]], np.ndarray, list[str]]:
    lines = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    header = [x for x in lines if x.get("kind") == "header"]
    if len(header) != 1:
        raise SystemExit("expected exactly one header line")
    keys = sorted(header[0]["items"])
    passes = []
    for x in lines:
        if x.get("kind") != "pass":
            continue
        if set(x["scores"]) != set(keys):
            raise SystemExit(f"pass {x['arm']} eps={x['eps']} does not cover the items")
        passes.append({**x, "scores": [x["scores"][k] for k in keys]})
    want = plan()
    got = [(p["arm"], p["eps"], p["seed"]) for p in passes]
    if sorted(map(str, got)) != sorted(map(str, want)):
        raise SystemExit(f"incomplete run: {len(got)} passes, {len(want)} planned; nothing is read")
    answers = np.array([header[0]["items"][k]["answer"] for k in keys])
    return passes, answers, [header[0]["items"][k]["source"] for k in keys]


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("passes", type=Path)
    args = ap.parse_args()
    passes, answers, sources = load(args.passes)
    src = np.array(sources)
    out = {"primary": analyse(passes, answers, np.isin(src, PRIMARY_SOURCES)),
           "all_items": analyse(passes, answers, np.ones(src.size, dtype=bool))}
    print(json.dumps(out, indent=1, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
