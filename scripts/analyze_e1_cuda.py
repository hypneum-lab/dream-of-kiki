#!/usr/bin/env python3
"""Analyse an E1 CUDA run: is the dream effect defensible against the controls?

Consumes the resumable cell cache (``pilot-cycle3-cuda-<slug>.cells.jsonl``) and
reports, per profile, the within-profile paired effect (mean delta, Hedges g) and
— the part that matters — the BETWEEN-condition comparison: does the dream beat
the controls?

The defensible signature (what turns "a delta" into "consolidation"):
    p_min/p_equ/p_max  >  p_ctrl_sft  >  p_ctrl_shuffle  ≈  p_noise
i.e. dream > matched-compute neutral SGD > content-destroyed > no-op. If the
dream sits in the control band, that is an honest null, not a finding.

  python analyze_e1_cuda.py <cells.jsonl | run_dir>
"""
from __future__ import annotations

import json
import math
import sys
from pathlib import Path

DREAM = ("p_min", "p_equ", "p_max")
CONTROLS = ("p_ctrl_sft", "p_ctrl_shuffle", "p_noise")


def _load(path: Path) -> dict:
    if path.is_dir():
        hits = list(path.glob("*.cells.jsonl"))
        if not hits:
            raise SystemExit(f"no *.cells.jsonl in {path}")
        path = hits[0]
    by = {}
    for line in path.read_text().splitlines():
        if line.strip():
            r = json.loads(line)
            by.setdefault(r["profile"], []).append(float(r["delta"]))
    return by


def _mean(xs):
    return sum(xs) / len(xs)


def _sd(xs):
    if len(xs) < 2:
        return 0.0
    m = _mean(xs)
    return math.sqrt(sum((x - m) ** 2 for x in xs) / (len(xs) - 1))


def _hedges_g_paired(xs):
    s = _sd(xs)
    if s == 0:
        return float("inf") if _mean(xs) else 0.0
    g = _mean(xs) / s
    return g * (1 - 3 / (4 * len(xs) - 1))   # small-sample correction


def _between(a, b):
    """Standardised mean difference (Cohen's d, pooled) of a vs b + p if scipy."""
    if len(a) < 2 or len(b) < 2:
        return None
    na, nb = len(a), len(b)
    sp = math.sqrt(((na - 1) * _sd(a) ** 2 + (nb - 1) * _sd(b) ** 2) / (na + nb - 2))
    d = (_mean(a) - _mean(b)) / sp if sp else float("inf")
    p = None
    try:
        from scipy import stats
        p = float(stats.ttest_ind(a, b, equal_var=False).pvalue)
    except Exception:  # noqa: BLE001
        pass
    return {"d": d, "p": p, "mean_diff": _mean(a) - _mean(b)}


def main() -> int:
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    by = _load(Path(sys.argv[1]))

    print("=== per profile (within: paired pre->post) ===")
    print(f"{'profile':16}{'n':>4}{'mean_d':>9}{'sd':>8}{'g':>8}{'%pos':>6}")
    for p in DREAM + CONTROLS:
        xs = by.get(p)
        if not xs:
            continue
        pos = 100 * sum(1 for x in xs if x > 0) / len(xs)
        print(f"{p:16}{len(xs):>4}{_mean(xs):>+9.4f}{_sd(xs):>8.4f}"
              f"{_hedges_g_paired(xs):>8.2f}{pos:>5.0f}%")

    noise = by.get("p_noise", [])
    print("\n=== between condition (the defensibility test) ===")
    verdict_ok = True
    for d in DREAM:
        if d not in by:
            continue
        print(f"\n[{d}] vs controls:")
        for c in CONTROLS:
            if c not in by:
                continue
            r = _between(by[d], by[c])
            if r is None:
                continue
            ps = f"p={r['p']:.3g}" if r["p"] is not None else "p=n/a(scipy)"
            verdict = "DREAM>" if r["mean_diff"] > 0 else "<=ctrl"
            print(f"   vs {c:15} Δmean={r['mean_diff']:+.4f}  d={r['d']:+.2f}  {ps}  {verdict}")
        # defensible if this dream profile beats the noise floor at all
        if noise and _mean(by[d]) <= _mean(noise):
            verdict_ok = False

    print("\n=== verdict ===")
    if not noise:
        print("no p_noise control -> cannot bound the effect; run the full suite.")
    elif verdict_ok and any(d in by and _mean(by[d]) > _mean(noise) for d in DREAM):
        print("Dream profiles exceed the noise floor. Check the d / p above vs "
              "p_ctrl_sft and p_ctrl_shuffle: clearing BOTH = defensible consolidation; "
              "only beating p_noise = beware it may be generic fine-tuning.")
    else:
        print("Dream does NOT beat the noise floor -> honest null at this scale "
              "(effect, if any, needs more scale/seeds/replay).")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
