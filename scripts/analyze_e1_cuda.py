#!/usr/bin/env python3
"""E1 decision code: section 2 states, section 4 / 6 gates, section 7 tables.

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` sections
2, 4, 6, 7, 8 (task 6 of section 9).
**Mode** : consumes the confirmatory cell cache written by
``pilot_cycle3_cuda.py --mode confirmatory`` and applies the registered
paired procedures. Nothing is tuned here; every constant is the
registered one.

Contrasts (unit = seed, ``D_s = delta_s(X) - delta_s(Y)``, ``delta`` =
post - pre ``acc``):

- H14-A ``p_min`` vs ``p_ctrl_template`` (superiority family)
- H14-B ``p_equ`` vs ``p_ctrl_vanilla_replay_tuned`` (superiority family)
- H14-C ``p_equ`` vs ``p_ctrl_onpolicy_replay`` (non-inferiority at delta)

Bonferroni alpha = 1/60; delta = 0.01. ``L`` / ``U`` are the one-sided
(1 - alpha) bounds (``kiki_oniric.eval.statistics.paired_tost``).
States (first match): UNINTERPRETABLE (a gate fails), then section 2.

Gates (section 4, 6, 8):

- cell-level exclusion: BLOCKING invariant violated, non-finite pre /
  post, missing record, or (for contrasts involving OPR) a
  ``scoring-overbudget`` OPR cell; pairwise deletion per contrast;
- differential exclusion: more than 7 of 141 seeds lost to cell-level
  exclusions -> UNINTERPRETABLE;
- OPR feasibility: seeds whose OPR corpus is empty are dropped from
  H14-C only; more than 7 -> H14-C UNINTERPRETABLE;
- collapse: collapse rate of either profile above 20 % -> UNINTERPRETABLE;
- incomplete run (fewer than 141 complete seeds): every hypothesis is
  reported as exploratory.

    python scripts/analyze_e1_cuda.py <cells.jsonl> --out decision.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from kiki_oniric.eval.statistics import paired_tost
from scripts.e1_diagnostics import COLLAPSE_GATE, collapse_rate, mean_ci95
from scripts.e1_profiles import EXPLORATORY, REGISTERED

N_PLANNED = 141
SEEDS = tuple(range(N_PLANNED))
DELTA = 0.01
ALPHA = 1.0 / 60.0
ALPHA_SECONDARY = 0.05
MAX_LOST = 7  # differential-exclusion gate and OPR feasibility (> 5 %)
OPR = "p_ctrl_onpolicy_replay"

FAMILY: dict[str, tuple[str, str, str]] = {
    "H14-A": ("p_min", "p_ctrl_template", "superiority"),
    "H14-B": ("p_equ", "p_ctrl_vanilla_replay_tuned", "superiority"),
    "H14-C": ("p_equ", OPR, "noninferiority"),
}
DREAM = ("p_min", "p_equ", "p_max")
CONTROLS = ("p_ctrl_sft", "p_ctrl_shuffle", "p_noise", "p_ctrl_template",
            "p_ctrl_vanilla_replay_tuned", OPR)
DIAG_FIELDS = ("drift", "kl_ref", "lost_correct", "router_drift", "usage_l1",
               "mean_replay_loss")

Cell = Mapping[str, Any]


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def cell_exclusion(cell: Cell | None, matched_compute: bool) -> str | None:
    """Reason a cell is excluded from a contrast, or None if retained."""
    if cell is None:
        return "missing_record"
    if cell.get("invariant_violation"):
        return "blocking_invariant"
    if not (_finite(cell.get("pre")) and _finite(cell.get("post"))):
        return "non_finite"
    if matched_compute and cell.get("budget", {}).get("status") == "scoring-overbudget":
        return "scoring_overbudget"
    return None


def state_superiority(L: float, U: float, mean: float, delta: float) -> str:
    if L > 0 and mean >= delta:
        return "SUPERIOR"
    if L > 0:
        return "POSITIVE-BELOW-SESOI"
    if U < delta:
        return "BOUNDED-NO-GAIN"
    return "INCONCLUSIVE"


def state_noninferiority(L: float, U: float, delta: float) -> str:
    if L > -delta:
        return "NON-INFERIOR"
    if U < -delta:
        return "INFERIOR"
    return "INCONCLUSIVE"


def _index(cells: Iterable[Cell]) -> dict[tuple[str, int], Cell]:
    out: dict[tuple[str, int], Cell] = {}
    for c in cells:
        key = (str(c["profile"]), int(c["seed"]))
        if key in out:
            raise ValueError(f"duplicate cell {key}")
        out[key] = c
    return out


def _delta(c: Cell) -> float:
    return float(c["post"]) - float(c["pre"])


def profile_collapse(idx: Mapping[tuple[str, int], Cell], profile: str) -> float:
    """Section 6: collapse rate of a profile over its seeds (cells with finite pre / post)."""
    flags = [bool(c.get("collapse")) for (p, _), c in idx.items()
             if p == profile and _finite(c.get("pre")) and _finite(c.get("post"))]
    return collapse_rate(flags)


def contrast(idx: Mapping[tuple[str, int], Cell], x: str, y: str, kind: str,
             alpha: float, seeds: Sequence[int] = SEEDS) -> dict[str, Any]:
    """Paired contrast with gates; ``kind`` superiority or noninferiority."""
    matched = OPR in (x, y)
    lost: dict[str, int] = {}
    opr_dropped = 0
    dx, dy, kept = [], [], []
    for s in seeds:
        cx, cy = idx.get((x, s)), idx.get((y, s))
        if OPR in (x, y):
            opr_cell = cx if x == OPR else cy
            if opr_cell is not None and int(opr_cell.get("opr_corpus_size", 1)) == 0:
                opr_dropped += 1  # feasibility, decided before any treatment
                continue
        why = cell_exclusion(cx, matched) or cell_exclusion(cy, matched)
        if why:
            lost[why] = lost.get(why, 0) + 1
            continue
        assert cx is not None and cy is not None
        dx.append(_delta(cx))
        dy.append(_delta(cy))
        kept.append(s)
    n_lost = sum(lost.values())
    out: dict[str, Any] = {"x": x, "y": y, "kind": kind, "alpha": alpha, "delta": DELTA,
                           "n": len(kept), "n_lost_cell_level": n_lost, "lost_by_reason": lost,
                           "opr_feasibility_dropped": opr_dropped}
    rates = {x: profile_collapse(idx, x), y: profile_collapse(idx, y)}
    out["collapse_rate"] = rates
    out["kept_seeds"] = kept
    gates = []
    if n_lost > MAX_LOST:
        gates.append("differential_exclusion")
    if opr_dropped > MAX_LOST:
        gates.append("opr_feasibility")
    if max(rates.values()) > COLLAPSE_GATE:
        gates.append("collapse")
    if len(kept) < 2:
        gates.append("fewer_than_2_pairs")
    out["gates_failed"] = gates
    if len(kept) >= 2:
        r = paired_tost(dx, dy, delta=DELTA, alpha=alpha)
        L, U, mean = r.ci_lower, r.ci_upper, r.mean_diff
        out.update({"mean": mean, "sd": r.sd_diff, "L": L, "U": U, "g_z": r.g_z,
                    "degenerate": r.degenerate, "tost_equivalent": r.equivalent,
                    "U_below_0": U < 0})
        state = (state_superiority(L, U, mean, DELTA) if kind == "superiority"
                 else state_noninferiority(L, U, DELTA))
    else:
        state = "INCONCLUSIVE"
    out["state"] = "UNINTERPRETABLE" if gates else state
    out["state_ungated"] = state
    return out


def joint_headline(states: Mapping[str, str]) -> str:
    a, b, c = states["H14-A"], states["H14-B"], states["H14-C"]
    if a == "SUPERIOR" and b == "SUPERIOR" and c == "NON-INFERIOR":
        return "J1"
    if a == "SUPERIOR" and b == "BOUNDED-NO-GAIN":
        return "J2"
    if a == "BOUNDED-NO-GAIN":
        return "J3"
    return "J4"


def _source_delta(c: Cell, src: str) -> float | None:
    mask = [e == src for e in c.get("eval_sources") or []]
    pre, post = c.get("pre_correct") or [], c.get("post_correct") or []
    n = sum(mask)
    if not n or len(pre) != len(mask) or len(post) != len(mask):
        return None
    return (sum(b for b, m in zip(post, mask, strict=True) if m)
            - sum(a for a, m in zip(pre, mask, strict=True) if m)) / n


def per_source(idx: Mapping[tuple[str, int], Cell], res: Mapping[str, Any]) -> dict[str, Any]:
    """Section 5.1a rule 10: every section 2 statistic per source, descriptive, no state.

    Uses the seeds the pooled contrast retained (same exclusions). For
    ``p_noise`` the post items differ from the pre items, so no per-source
    difference is defined.
    """
    x, y = str(res["x"]), str(res["y"])
    if "p_noise" in (x, y):
        return {}
    kept = list(res.get("kept_seeds") or [])
    sources = sorted({e for s in kept for e in idx[(x, s)].get("eval_sources") or []})
    out: dict[str, Any] = {}
    for src in sources:
        dx, dy = [], []
        for s in kept:
            a, b = _source_delta(idx[(x, s)], src), _source_delta(idx[(y, s)], src)
            if a is not None and b is not None:
                dx.append(a)
                dy.append(b)
        row: dict[str, Any] = {"n": len(dx)}
        if len(dx) >= 2:
            r = paired_tost(dx, dy, delta=DELTA, alpha=float(res["alpha"]))
            row.update({"mean": r.mean_diff, "sd": r.sd_diff, "se": r.se_diff,
                        "L": r.ci_lower, "U": r.ci_upper, "g_z": r.g_z,
                        "degenerate": r.degenerate})
        out[src] = row
    return out


def decide(cells: Iterable[Cell]) -> dict[str, Any]:
    cells = list(cells)
    bad_mode = {str(c.get("mode")) for c in cells} - {"confirmatory"}
    if bad_mode:
        raise ValueError(f"non-confirmatory cells in the input: {sorted(bad_mode)}")
    for field in ("corpus", "model", "revision", "amendment_sha256", "run_identity"):
        vals = {json.dumps(c.get(field), sort_keys=True) for c in cells}
        if len(vals) > 1:
            raise ValueError(f"cells disagree on {field}: more than one corpus / run identity")
    idx = _index(cells)
    outside = sorted({s for (_, s) in idx} - set(SEEDS))
    if outside:
        raise ValueError(f"seeds outside 0..140: {outside[:5]}")
    complete = [s for s in SEEDS if all((p, s) in idx for p in REGISTERED)]
    run_complete = len(complete) == N_PLANNED
    hyps = {h: contrast(idx, x, y, k, ALPHA) for h, (x, y, k) in FAMILY.items()}
    states = {h: r["state"] for h, r in hyps.items()}
    secondary = {f"{d}-{c}": contrast(idx, d, c, "superiority", ALPHA_SECONDARY)
                 for d in DREAM for c in CONTROLS}
    explo = {"p_equ-p_ctrl_lr_matched": contrast(idx, "p_equ", EXPLORATORY[0], "superiority",
                                                 ALPHA_SECONDARY)}
    diag: dict[str, Any] = {}
    for p in REGISTERED + EXPLORATORY:
        cs = [idx[(p, s)] for s in SEEDS if (p, s) in idx]
        diag[p] = {"n": len(cs), "collapse_rate": collapse_rate([bool(c.get("collapse"))
                                                                for c in cs])}
        for f in DIAG_FIELDS:
            diag[p][f] = mean_ci95([c.get(f) for c in cs])
        diag[p]["n_updates"] = mean_ci95([c.get("n_updates") for c in cs])
        diag[p]["achieved_loss_tokens"] = mean_ci95([c.get("achieved_loss_tokens") for c in cs])
    return {
        "prereg": "docs/osf-prereg-e1-replay-controls-2026-10-23.md",
        "n_planned": N_PLANNED, "n_complete_seeds": len(complete),
        "run_complete": run_complete,
        "confirmatory": run_complete,
        "hypotheses": hyps, "states": states,
        "joint_headline": joint_headline(states),
        "secondary": secondary, "exploratory_arms": explo,
        "per_source": {h: per_source(idx, r) for h, r in hyps.items()},
        "opr_corpus_size": _size_distribution(idx),
        "diagnostics": diag,
    }


def _size_distribution(idx: Mapping[tuple[str, int], Cell]) -> dict[str, int]:
    sizes = [int(c["opr_corpus_size"]) for (p, _), c in idx.items()
             if p == OPR and c.get("opr_corpus_size") is not None]
    hist: dict[str, int] = {}
    for v in sizes:
        hist[str(v)] = hist.get(str(v), 0) + 1
    return dict(sorted(hist.items(), key=lambda kv: int(kv[0])))


def load_cells(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()
            if line.strip()]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="E1 decision code (sections 2, 4, 6, 7)")
    ap.add_argument("cells", type=Path, help="confirmatory *.cells.jsonl")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    res = decide(load_cells(args.cells))
    args.out.write_text(json.dumps(res, indent=1, sort_keys=True) + "\n", encoding="utf-8")
    tag = "" if res["confirmatory"] else " (EXPLORATORY: run incomplete)"
    for h, r in res["hypotheses"].items():
        print(f"{h}: {r['state']} n={r['n']}{tag}")
    print(f"joint headline: {res['joint_headline']}{tag}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
