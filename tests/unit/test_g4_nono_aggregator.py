"""G4-nono aggregator: section 2 states, section 7 rows, section 3 N rule.

Synthetic paired data built so that every section 2 state and every
section 7 row is reached end to end (filing gate (c) of the draft
``docs/osf-prereg-g4-nono-cross-domain-2026-10-23.md``, section 9),
plus a hand-computed paired example (fidelity checklist item 8).
NumPy / SciPy only (no MLX), so it runs on Linux CI.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from scipy import stats

from experiments.g4_nono_test.aggregator import (
    ALPHA_FAMILY,
    EPS,
    aggregate,
    contrast_state_ac,
    contrast_state_eq,
    main,
    outcome_row,
    paired_summary,
    select_n,
)

N = 30


def _z(n: int, seed: int) -> np.ndarray:
    """Fixed sequence with mean exactly 0 and sd (ddof=1) exactly 1."""
    z = np.random.default_rng(seed).normal(size=n)
    z = z - z.mean()
    out: np.ndarray = z / z.std(ddof=1)
    return out


def _vec(mean: float, spread: float, n: int = N, seed: int = 1) -> np.ndarray:
    return mean + spread * _z(n, seed)


def _records(
    d_a: np.ndarray,
    d_b: np.ndarray,
    d_c: np.ndarray,
    base_sd: float = 0.1,
    manip_fail: set[int] | None = None,
    excluded: dict[tuple[int, str, str], str] | None = None,
) -> list[dict[str, Any]]:
    """8 cells per seed. CDPA: none = base, cross = base + d_a,
    within = base + d_a - d_b. WDPA: none = w, within = w + d_c."""
    n = len(d_a)
    base = 0.5 + base_sd * _z(n, 11)
    wbase = 0.6 + base_sd * _z(n, 12)
    manip_fail = manip_fail or set()
    excluded = excluded or {}
    recs: list[dict[str, Any]] = []
    for s in range(n):
        cd = {"none": base[s], "cross": base[s] + d_a[s], "within": base[s] + d_a[s] - d_b[s]}
        wd = {"none": wbase[s], "cross": wbase[s], "within": wbase[s] + d_c[s]}
        cells = [("baseline", "na"), ("P_min", "na")] + [
            (arm, st) for arm in ("P_equ", "P_max") for st in ("cross", "within", "none")
        ]
        for arm, st in cells:
            key = st if st != "na" else "none"
            rec = {
                "status": "ok",
                "seed": s,
                "arm": arm,
                "strategy": st,
                "cdpa": float(cd[key]),
                "wdpa": float(wd[key]),
                "retention_a": 0.4,
                "acc20": 0.3,
                "doa": 0.7,
                "manip_acc": 0.5 if s in manip_fail else 0.9,
                "manip_pass": s not in manip_fail,
                "excluded": False,
                "exclusion_reason": None,
                "finite": True,
            }
            reason = excluded.get((s, arm, st))
            if reason is not None:
                rec.update(excluded=True, exclusion_reason=reason, cdpa=None, wdpa=None)
            recs.append(rec)
    return recs


# Per-contrast D recipes (n = 30, alpha = 1/60, eps = 0.01, arm SD 0.1).
AB = {
    "CONFIRMED": (0.10, 0.01),
    "POSITIVE-NEGLIGIBLE": (0.004, 0.002),
    "DETECTED-BELOW-FLOOR": (0.02, 0.01),
    "BOUNDED-NO-GAIN": (0.0, 0.01),
    "INCONCLUSIVE": (0.0, 0.05),
}
C = {
    "EQUIVALENT": (0.0, 0.01),
    "MEANINGFULLY-DIFFERENT": (0.03, 0.01),
    "INCONCLUSIVE": (0.0, 0.05),
}


def _scenario(a: str, b: str, c: str, **kw: Any) -> list[dict[str, Any]]:
    return _records(_vec(*AB[a], seed=2), _vec(*AB[b], seed=3), _vec(*C[c], seed=4), **kw)


# --- hand-computed example (checklist item 8) -------------------------------


def test_paired_summary_hand_example() -> None:
    x = np.array([0.62, 0.58, 0.66, 0.61, 0.64])
    y = np.array([0.60, 0.57, 0.62, 0.60, 0.61])
    d = x - y  # 0.02 0.01 0.04 0.01 0.03
    n = 5
    mean = 0.022
    sd = math.sqrt(sum((v - mean) ** 2 for v in [0.02, 0.01, 0.04, 0.01, 0.03]) / 4)
    q = stats.t.ppf(1 - 1 / 60, 4)
    j = 1 - 3 / (4 * 4 - 1)
    s1, s2 = np.std(x, ddof=1), np.std(y, ddof=1)
    out = paired_summary(x, y, alpha=1 / 60)
    assert out["n"] == n
    assert out["mean"] == pytest.approx(mean)
    assert out["sd"] == pytest.approx(sd)
    assert out["L"] == pytest.approx(mean - q * sd / math.sqrt(n))
    assert out["U"] == pytest.approx(mean + q * sd / math.sqrt(n))
    assert out["g_z"] == pytest.approx(mean / sd * j)
    assert out["g_av"] == pytest.approx(mean / math.sqrt((s1**2 + s2**2) / 2) * j)
    assert d.mean() == pytest.approx(mean)
    # same decision as scipy's paired one-sided test
    p = stats.ttest_rel(x, y, alternative="greater").pvalue
    assert (out["L"] > 0) == (p < 1 / 60)


def test_paired_summary_degenerate() -> None:
    x = np.array([0.5, 0.25, 0.75])  # binary-exact: identical D
    out = paired_summary(x + 0.125, x, alpha=1 / 60)
    assert out["degenerate"] is True
    assert out["L"] == out["U"] == 0.125
    assert out["g_z"] is None


def test_degenerate_g_av_undefined_misses_floor() -> None:
    x = np.full(5, 0.5)
    out = paired_summary(x + 0.05, x, alpha=1 / 60)
    assert out["g_av"] is None
    assert contrast_state_ac(out, EPS) == "DETECTED-BELOW-FLOOR"


# --- section 2 states -------------------------------------------------------


@pytest.mark.parametrize("state", list(AB))
def test_every_h12_ab_state_reachable(state: str) -> None:
    out = aggregate(_scenario(state, state, "EQUIVALENT"), n_planned=N)
    assert out["hypotheses"]["H12-A"]["state"] == state
    assert out["hypotheses"]["H12-B"]["state"] == state


@pytest.mark.parametrize("state", list(C))
def test_every_h12_c_state_reachable(state: str) -> None:
    out = aggregate(_scenario("BOUNDED-NO-GAIN", "BOUNDED-NO-GAIN", state), n_planned=N)
    assert out["hypotheses"]["H12-C"]["state"] == state


def test_state_order_first_match() -> None:
    # L > 0 and U < eps but g_av >= 0.5 and mean < eps: POSITIVE-NEGLIGIBLE
    s = {"L": 0.001, "U": 0.009, "mean": 0.005, "g_av": 2.0}
    assert contrast_state_ac(s, EPS) == "POSITIVE-NEGLIGIBLE"
    # negative interval: BOUNDED-NO-GAIN
    assert contrast_state_ac({"L": -0.03, "U": -0.01, "mean": -0.02, "g_av": -1}, EPS) == (
        "BOUNDED-NO-GAIN"
    )
    assert contrast_state_eq({"L": -0.03, "U": -0.011}, EPS) == "MEANINGFULLY-DIFFERENT"
    # strict inequalities at the margin
    assert contrast_state_eq({"L": -0.01, "U": 0.0}, EPS) == "INCONCLUSIVE"


# --- section 7 rows ---------------------------------------------------------

ROWS = [
    (1, ("CONFIRMED", "CONFIRMED", "MEANINGFULLY-DIFFERENT")),
    (2, ("CONFIRMED", "CONFIRMED", "EQUIVALENT")),
    (3, ("CONFIRMED", "CONFIRMED", "INCONCLUSIVE")),
    (4, ("CONFIRMED", "BOUNDED-NO-GAIN", "EQUIVALENT")),
    (4, ("CONFIRMED", "POSITIVE-NEGLIGIBLE", "EQUIVALENT")),
    (5, ("CONFIRMED", "DETECTED-BELOW-FLOOR", "EQUIVALENT")),
    (5, ("CONFIRMED", "INCONCLUSIVE", "EQUIVALENT")),
    (6, ("BOUNDED-NO-GAIN", "INCONCLUSIVE", "EQUIVALENT")),
    (6, ("POSITIVE-NEGLIGIBLE", "CONFIRMED", "INCONCLUSIVE")),
    (7, ("DETECTED-BELOW-FLOOR", "BOUNDED-NO-GAIN", "EQUIVALENT")),
    (8, ("INCONCLUSIVE", "INCONCLUSIVE", "INCONCLUSIVE")),
]


@pytest.mark.parametrize(("row", "states"), ROWS)
def test_every_section7_row_end_to_end(row: int, states: tuple[str, str, str]) -> None:
    out = aggregate(_scenario(*states), n_planned=N)
    got = tuple(out["hypotheses"][h]["state"] for h in ("H12-A", "H12-B", "H12-C"))
    assert got == states
    assert out["row"] == row


def test_row_table_exhaustive() -> None:
    for a in AB:
        for b in AB:
            for c in C:
                assert outcome_row(a, b, c, abort=False) in range(1, 9)
    assert outcome_row("CONFIRMED", "CONFIRMED", "EQUIVALENT", abort=True) == 0


# --- section 4 exclusions, guard and abort ----------------------------------


def test_row0_abort_on_manipulation_failure() -> None:
    recs = _scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT", manip_fail=set(range(16)))
    out = aggregate(recs, n_planned=N)
    assert out["abort"] is True
    assert out["row"] == 0
    assert "manipulation" in out["abort_reasons"][0]


def test_abort_on_run_id_collision_record() -> None:
    recs = _scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT")
    recs.append({"status": "aborted", "abort_reason": "run_id_collision", "seed": 3,
                 "arm": "P_max", "strategy": "cross"})
    out = aggregate(recs, n_planned=N)
    assert out["abort"] is True and out["row"] == 0


def test_manipulation_excluded_seed_dropped_everywhere() -> None:
    recs = _scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT", manip_fail={0, 1})
    out = aggregate(recs, n_planned=N)
    assert out["abort"] is False
    for h in ("H12-A", "H12-B", "H12-C"):
        assert out["hypotheses"][h]["n"] == N - 2
        assert out["hypotheses"][h]["n_lost_cell_level"] == 0
    assert sorted(e["seed"] for e in out["seed_exclusions"]) == [0, 1]


def test_pairwise_deletion_and_differential_guard() -> None:
    excl = {(s, "P_max", "cross"): "non_finite_logits" for s in range(4)}
    out = aggregate(_scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT", excluded=excl), n_planned=N)
    a, c = out["hypotheses"]["H12-A"], out["hypotheses"]["H12-C"]
    assert a["n"] == N - 4 and a["n_lost_cell_level"] == 4
    assert a["exploratory_guard"] is True  # > 3 of 30 lost
    assert c["n"] == N and c["exploratory_guard"] is False


def test_guard_not_triggered_at_three_losses() -> None:
    excl = {(s, "P_max", "cross"): "non_finite_logits" for s in range(3)}
    out = aggregate(_scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT", excluded=excl), n_planned=N)
    assert out["hypotheses"]["H12-A"]["exploratory_guard"] is False


def test_rejects_unregistered_n() -> None:
    with pytest.raises(ValueError, match="n_planned"):
        aggregate(_scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT"), n_planned=31)


def test_rejects_duplicate_cells() -> None:
    recs = _scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT")
    recs.append(dict(recs[0]))
    with pytest.raises(ValueError, match="duplicate"):
        aggregate(recs, n_planned=N)


def test_secondary_contrasts_reported_at_nominal_alpha() -> None:
    out = aggregate(_scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT"), n_planned=N)
    sec = out["secondary"]
    assert sec["alpha"] == 0.05
    for key in (
        "cdpa:P_max/cross-P_equ/cross",
        "cdpa:P_max/cross-P_min/na",
        "cdpa:P_max/within-P_max/none",
        "doa:P_max/cross-P_max/none",
        "doa:P_max/cross-P_max/within",
        "retention_a:P_max/cross-P_max/none",
        "acc20:P_max/cross-P_max/none",
    ):
        assert key in sec["contrasts"], key
    assert out["alpha_family"] == pytest.approx(ALPHA_FAMILY)


# --- section 3 N-selection rule ---------------------------------------------


def _smoke(sd_wdpa: float, rho_low: bool) -> list[dict[str, Any]]:
    """Three smoke seeds; d_c SD = sd_wdpa; cross/none correlation."""
    d_c = _vec(0.0, sd_wdpa, n=3, seed=5)
    recs = _records(_vec(0.02, 0.001, 3, 6), _vec(0.01, 0.001, 3, 7), d_c, base_sd=0.05)
    if rho_low:
        # perfectly negative cross/none correlation
        none = {r["seed"]: r["cdpa"] for r in recs if (r["arm"], r["strategy"]) == ("P_max", "none")}
        for r in recs:
            if (r["arm"], r["strategy"]) == ("P_max", "cross"):
                r["cdpa"] = 1.0 - none[r["seed"]]
    for r in recs:
        r["seed"] += 900
    return recs


def test_select_n_sd_line_first() -> None:
    out = select_n(_smoke(0.02, rho_low=False))
    assert out["sd_hat_d"] == pytest.approx(0.02)
    assert out["n"] == 108 and out["line"] == 1


def test_select_n_r_line() -> None:
    out = select_n(_smoke(0.005, rho_low=True))
    assert out["min_r_hat"] < 0.5
    assert out["n"] == 95 and out["line"] == 2


def test_select_n_default_line() -> None:
    out = select_n(_smoke(0.005, rho_low=False))
    assert out["min_r_hat"] >= 0.5
    assert out["n"] == 30 and out["line"] == 3


def test_select_n_zero_variance_counts_below_half() -> None:
    recs = _smoke(0.005, rho_low=False)
    for r in recs:
        if r["arm"] == "P_max" and r["strategy"] == "none":
            r["cdpa"] = 0.5
    out = select_n(recs)
    assert out["n"] == 95


def test_select_n_requires_complete_smoke() -> None:
    recs = [r for r in _smoke(0.005, rho_low=False) if r["seed"] != 902]
    with pytest.raises(ValueError, match="smoke"):
        select_n(recs)


# --- CLI ----------------------------------------------------------------------


def test_cli_writes_json(tmp_path: Path) -> None:
    files = []
    for i, rec in enumerate(_scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT")):
        p = tmp_path / f"cell{i:04d}.json"
        p.write_text(json.dumps(rec))
        files.append(str(p))
    out = tmp_path / "agg.json"
    assert main([*files, "--n-planned", "30", "--out", str(out)]) == 0
    doc = json.loads(out.read_text())
    assert doc["row"] == 2


def test_cli_select_n(tmp_path: Path) -> None:
    files = []
    for i, rec in enumerate(_smoke(0.02, rho_low=False)):
        p = tmp_path / f"smoke{i:02d}.json"
        p.write_text(json.dumps(rec))
        files.append(str(p))
    out = tmp_path / "n.json"
    assert main([*files, "--select-n", "--out", str(out)]) == 0
    assert json.loads(out.read_text())["n"] == 108
