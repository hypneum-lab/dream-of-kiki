"""G4-nono aggregator: section 2 states, section 7 row, section 3 N rule.

Spec: ``docs/osf-prereg-g4-nono-cross-domain-2026-10-23.md`` (draft
PR #41). Consumes the per-cell JSON records written by
:mod:`experiments.g4_nono_test.run_cross_domain` and implements:

* section 4: seed-level manipulation-check exclusion, pairwise deletion
  of cell-level exclusions, the differential-exclusion guard and the
  abort conditions (more than 50 % of seeds fail the manipulation
  check, or any ``run_id_collision`` record);
* section 2: paired bounds ``L`` / ``U`` at one-sided alpha = 1/60,
  ``g_av`` (average-SD Hedges form) and ``g_z``; the H12-A / H12-B
  states (CONFIRMED, POSITIVE-NEGLIGIBLE, DETECTED-BELOW-FLOOR,
  BOUNDED-NO-GAIN, INCONCLUSIVE) and the H12-C states (EQUIVALENT,
  MEANINGFULLY-DIFFERENT, INCONCLUSIVE), first match;
* section 7: the first matching DualVer row (0-8);
* section 2 secondary contrasts and the section 6 domain-only probe
  (DOA) contrasts, same state rules at nominal alpha = 0.05;
* section 3: the smoke N-selection rule (``--select-n``).

Bounds come from :func:`kiki_oniric.eval.statistics.paired_tost`
(paired SE, df = n - 1, exact identical-difference handling), so the
aggregator and the E1 analysis share one paired implementation.

Usage::

    uv run python -m experiments.g4_nono_test.aggregator cells/*.json \
        --n-planned 30 --out agg.json
    uv run python -m experiments.g4_nono_test.aggregator smoke/*.json \
        --select-n --out n-selection.json
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from kiki_oniric.eval.statistics import paired_tost  # noqa: E402

EPS = 0.01
ALPHA_FAMILY = 0.05 / 3
ALPHA_SECONDARY = 0.05
G_FLOOR = 0.5
# Differential-exclusion guard (section 4): more than this many seeds
# lost to cell-level exclusions makes the hypothesis exploratory.
GUARD = {30: 3, 95: 9, 108: 10}
SMOKE_SEEDS = (900, 901, 902)
SD_THRESHOLD = 0.015
R_THRESHOLD = 0.5
METRICS = ("cdpa", "wdpa", "retention_a", "acc20", "doa")

Cell = tuple[str, str]
# H12 family: (label, metric, cell 1, cell 2); D = cell 1 - cell 2.
FAMILY: tuple[tuple[str, str, Cell, Cell], ...] = (
    ("H12-A", "cdpa", ("P_max", "cross"), ("P_max", "none")),
    ("H12-B", "cdpa", ("P_max", "cross"), ("P_max", "within")),
    ("H12-C", "wdpa", ("P_max", "within"), ("P_max", "none")),
)
SECONDARY: tuple[tuple[str, Cell, Cell], ...] = (
    ("cdpa", ("P_max", "cross"), ("P_equ", "cross")),
    ("cdpa", ("P_max", "cross"), ("P_min", "na")),
    ("cdpa", ("P_max", "within"), ("P_max", "none")),
    ("retention_a", ("P_max", "cross"), ("P_max", "none")),
    ("retention_a", ("P_max", "cross"), ("P_max", "within")),
    ("acc20", ("P_max", "cross"), ("P_max", "none")),
    ("acc20", ("P_max", "cross"), ("P_max", "within")),
    ("doa", ("P_max", "cross"), ("P_max", "none")),
    ("doa", ("P_max", "cross"), ("P_max", "within")),
)


def _hedges_j(n: int) -> float:
    return 1.0 - 3.0 / (4.0 * (n - 1) - 1.0)


def paired_summary(
    x: Iterable[float], y: Iterable[float], alpha: float = ALPHA_FAMILY
) -> dict[str, Any]:
    """Paired bounds and effect sizes for ``D = x - y`` (section 2).

    ``L`` / ``U`` are the one-sided (1 - alpha) bounds (the limits of
    the two-sided 1 - 2 alpha interval). ``g_av`` and ``g_z`` are None
    when their denominator is zero (an undefined effect size never
    meets the ``g_av >= 0.5`` floor).
    """
    xa = np.asarray(list(x), dtype=float)
    ya = np.asarray(list(y), dtype=float)
    # delta only parametrises paired_tost's equivalence flag, which is
    # not used here; the bounds do not depend on it.
    r = paired_tost(xa, ya, delta=1.0, alpha=alpha)
    n = r.n
    j = _hedges_j(n)
    s1 = float(np.std(xa, ddof=1))
    s2 = float(np.std(ya, ddof=1))
    s_av = math.sqrt((s1 * s1 + s2 * s2) / 2.0)
    return {
        "n": n,
        "mean": r.mean_diff,
        "sd": r.sd_diff,
        "L": r.ci_lower,
        "U": r.ci_upper,
        "alpha": alpha,
        "degenerate": r.degenerate,
        "g_z": r.g_z,
        "g_av": (r.mean_diff / s_av * j) if s_av > 0.0 else None,
    }


def contrast_state_ac(s: Mapping[str, Any], eps: float = EPS) -> str:
    """H12-A / H12-B state, first match (section 2)."""
    lo, up, mean, g_av = s["L"], s["U"], s["mean"], s["g_av"]
    if lo > 0 and g_av is not None and g_av >= G_FLOOR and mean >= eps:
        return "CONFIRMED"
    if lo > 0 and up < eps:
        return "POSITIVE-NEGLIGIBLE"
    if lo > 0:
        return "DETECTED-BELOW-FLOOR"
    if up < eps:
        return "BOUNDED-NO-GAIN"
    return "INCONCLUSIVE"


def contrast_state_eq(s: Mapping[str, Any], eps: float = EPS) -> str:
    """H12-C state, first match (section 2)."""
    lo, up = s["L"], s["U"]
    if lo > -eps and up < eps:
        return "EQUIVALENT"
    if lo > eps or up < -eps:
        return "MEANINGFULLY-DIFFERENT"
    return "INCONCLUSIVE"


def outcome_row(a: str, b: str, c: str, *, abort: bool) -> int:
    """First matching section 7 row (0-8); the table is exhaustive."""
    if abort:
        return 0
    if c == "MEANINGFULLY-DIFFERENT":
        return 1
    if a == "CONFIRMED" and b == "CONFIRMED":
        return 2 if c == "EQUIVALENT" else 3
    if a == "CONFIRMED" and b in ("BOUNDED-NO-GAIN", "POSITIVE-NEGLIGIBLE"):
        return 4
    if a == "CONFIRMED":
        return 5
    if a in ("BOUNDED-NO-GAIN", "POSITIVE-NEGLIGIBLE"):
        return 6
    if a == "DETECTED-BELOW-FLOOR":
        return 7
    return 8


def _index(records: Iterable[Mapping[str, Any]]) -> tuple[
    dict[tuple[int, str, str], Mapping[str, Any]], list[Mapping[str, Any]]
]:
    cells: dict[tuple[int, str, str], Mapping[str, Any]] = {}
    aborted: list[Mapping[str, Any]] = []
    for rec in records:
        if rec.get("status") == "aborted":
            aborted.append(rec)
            continue
        key = (int(rec["seed"]), str(rec["arm"]), str(rec["strategy"]))
        if key in cells:
            raise ValueError(f"duplicate cell record {key}")
        cells[key] = rec
    return cells, aborted


def _usable(rec: Mapping[str, Any] | None, metric: str) -> bool:
    if rec is None or rec.get("excluded") or rec.get("status", "ok") != "ok":
        return False
    v = rec.get(metric)
    return v is not None and math.isfinite(float(v))


def _contrast(
    cells: Mapping[tuple[int, str, str], Mapping[str, Any]],
    seeds: list[int],
    metric: str,
    c1: Cell,
    c2: Cell,
    alpha: float,
) -> dict[str, Any]:
    kept, lost = [], []
    reasons: dict[str, str] = {}
    for s in seeds:
        r1, r2 = cells.get((s, *c1)), cells.get((s, *c2))
        if _usable(r1, metric) and _usable(r2, metric):
            kept.append(s)
            continue
        lost.append(s)
        why = []
        for c, r in ((c1, r1), (c2, r2)):
            if r is None:
                why.append(f"missing_record:{c[0]}/{c[1]}")
            elif not _usable(r, metric):
                tag = r.get("exclusion_reason") or "unusable"
                why.append(f"{tag}:{c[0]}/{c[1]}")
        reasons[str(s)] = ";".join(why)
    out: dict[str, Any] = {
        "metric": metric,
        "cells": [f"{c1[0]}/{c1[1]}", f"{c2[0]}/{c2[1]}"],
        "seeds_lost_cell_level": lost,
        "n_lost_cell_level": len(lost),
        "loss_reasons": reasons,
    }
    if len(kept) < 2:
        out.update(n=len(kept), state="INCONCLUSIVE", note="fewer than 2 complete pairs")
        return out
    x = [float(cells[(s, *c1)][metric]) for s in kept]
    y = [float(cells[(s, *c2)][metric]) for s in kept]
    out.update(paired_summary(x, y, alpha=alpha))
    return out


def aggregate(
    records: Iterable[Mapping[str, Any]],
    n_planned: int,
    eps: float = EPS,
) -> dict[str, Any]:
    """Apply sections 2, 4 and 7 to the cell records of one run."""
    if n_planned not in GUARD:
        raise ValueError(f"n_planned must be one of {sorted(GUARD)} (got {n_planned})")
    cells, aborted = _index(records)
    seeds = list(range(n_planned))
    seen = {k[0] for k in cells} | {int(r["seed"]) for r in aborted}
    outside = sorted(seen - set(seeds))
    if outside:
        raise ValueError(f"seed(s) {outside} outside the planned list 0..{n_planned - 1}")
    incomplete = len(cells) < 8 * n_planned
    abort_reasons: list[str] = []
    if aborted:
        abort_reasons.append(f"{len(aborted)} run_id_collision / aborted cell record(s)")

    seed_excl: list[dict[str, Any]] = []
    for s in seeds:
        recs = [r for k, r in cells.items() if k[0] == s]
        if any(r.get("manip_pass") is False for r in recs):
            accs = [r.get("manip_acc") for r in recs if r.get("manip_acc") is not None]
            seed_excl.append({
                "seed": s, "reason": "manipulation_check",
                "manip_acc": accs[0] if accs else None,
            })
    excluded_seeds = {e["seed"] for e in seed_excl}
    if seeds and len(excluded_seeds) > 0.5 * len(seeds):
        abort_reasons.append(
            f"manipulation check failed on {len(excluded_seeds)} of {len(seeds)} seeds (> 50 %)"
        )
    retained = [s for s in seeds if s not in excluded_seeds]
    abort = bool(abort_reasons)

    hyps: dict[str, dict[str, Any]] = {}
    for label, metric, c1, c2 in FAMILY:
        h = _contrast(cells, retained, metric, c1, c2, ALPHA_FAMILY)
        if "L" in h:
            h["state"] = (contrast_state_eq if label == "H12-C" else contrast_state_ac)(h, eps)
        h["exploratory_guard"] = h["n_lost_cell_level"] > GUARD[n_planned]
        hyps[label] = h

    sec = {}
    for metric, c1, c2 in SECONDARY:
        key = f"{metric}:{c1[0]}/{c1[1]}-{c2[0]}/{c2[1]}"
        h = _contrast(cells, retained, metric, c1, c2, ALPHA_SECONDARY)
        if "L" in h:
            h["state"] = contrast_state_ac(h, eps)
        sec[key] = h

    row = outcome_row(
        hyps["H12-A"]["state"], hyps["H12-B"]["state"], hyps["H12-C"]["state"], abort=abort
    )
    return {
        "prereg": "docs/osf-prereg-g4-nono-cross-domain-2026-10-23.md",
        "n_planned": n_planned,
        "n_seeds_seen": len(seen),
        "incomplete_run": incomplete,
        "n_seeds_retained": len(retained),
        "seed_exclusions": seed_excl,
        "eps": eps,
        "alpha_family": ALPHA_FAMILY,
        "abort": abort,
        "abort_reasons": abort_reasons,
        "hypotheses": hyps,
        "row": row,
        "secondary": {"alpha": ALPHA_SECONDARY, "exploratory": True, "contrasts": sec},
    }


def _pearson(x: list[float], y: list[float]) -> float | None:
    xa, ya = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if np.std(xa) == 0.0 or np.std(ya) == 0.0:
        return None
    return float(np.corrcoef(xa, ya)[0, 1])


def select_n(records: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """Section 3 N-selection rule from the smoke seeds 900-902.

    Line 1: SD^_D (ddof 1) of WDPA(within) - WDPA(none) >= 0.015 -> 108;
    line 2: min r^ < 0.5 (undefined r^ counts as below) -> 95;
    line 3: otherwise 30. Contrasts are those of the H12 family
    (``P_max`` arm). Every smoke cell must be present and usable.
    """
    cells, aborted = _index(records)
    if aborted:
        raise ValueError("smoke contains aborted cell records")
    vals: dict[tuple[str, Cell], list[float]] = {}
    for _label, metric, c1, c2 in FAMILY:
        for c in (c1, c2):
            col = []
            for s in SMOKE_SEEDS:
                rec = cells.get((s, *c))
                if not _usable(rec, metric):
                    raise ValueError(f"smoke incomplete: seed {s} cell {c} metric {metric}")
                col.append(float(rec[metric]))  # type: ignore[index]
            vals[(metric, c)] = col
    r_hat = {}
    for label, metric, c1, c2 in FAMILY:
        r_hat[label] = _pearson(vals[(metric, c1)], vals[(metric, c2)])
    d = np.asarray(vals[("wdpa", ("P_max", "within"))]) - np.asarray(
        vals[("wdpa", ("P_max", "none"))]
    )
    sd_hat = float(np.std(d, ddof=1))
    min_r = min((r if r is not None else -math.inf) for r in r_hat.values())
    if sd_hat >= SD_THRESHOLD:
        n, line = 108, 1
    elif min_r < R_THRESHOLD:
        n, line = 95, 2
    else:
        n, line = 30, 3
    return {
        "smoke_seeds": list(SMOKE_SEEDS),
        "r_hat": r_hat,
        "min_r_hat": None if math.isinf(min_r) else min_r,
        "sd_hat_d": sd_hat,
        "n": n,
        "line": line,
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="G4-nono aggregator")
    p.add_argument("inputs", nargs="+", type=Path, help="cell JSON records")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--n-planned", type=int, choices=sorted(GUARD))
    p.add_argument("--select-n", action="store_true", help="apply the section 3 smoke rule")
    args = p.parse_args(argv)
    records = [json.loads(f.read_text()) for f in sorted(args.inputs)]
    if args.select_n:
        doc = select_n(records)
    else:
        if args.n_planned is None:
            p.error("--n-planned is required unless --select-n")
        doc = aggregate(records, n_planned=args.n_planned)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
