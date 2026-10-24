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
    n_from_sd,
    outcome_row,
    paired_summary,
    select_n,
)

N = 95


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


# Per-contrast D recipes (n = 95, alpha = 1/60, eps = 0.01, arm SD 0.1).
AB = {
    "CONFIRMED": (0.10, 0.01),
    "POSITIVE-NEGLIGIBLE": (0.004, 0.002),
    "DETECTED-BELOW-FLOOR": (0.02, 0.01),
    "BOUNDED-NO-GAIN": (0.0, 0.01),
    "INCONCLUSIVE": (0.0, 0.08),
}
C = {
    "EQUIVALENT": (0.0, 0.01),
    "MEANINGFULLY-DIFFERENT": (0.03, 0.01),
    "INCONCLUSIVE": (0.0, 0.08),
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
    recs = _scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT", manip_fail=set(range(48)))
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
    excl = {(s, "P_max", "cross"): "non_finite_logits" for s in range(10)}
    out = aggregate(_scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT", excluded=excl), n_planned=N)
    a, c = out["hypotheses"]["H12-A"], out["hypotheses"]["H12-C"]
    assert a["n"] == N - 10 and a["n_lost_cell_level"] == 10
    assert a["exploratory_guard"] is True  # > 9 of 95 lost
    assert c["n"] == N and c["exploratory_guard"] is False


def test_guard_not_triggered_at_nine_losses() -> None:
    excl = {(s, "P_max", "cross"): "non_finite_logits" for s in range(9)}
    out = aggregate(_scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT", excluded=excl), n_planned=N)
    assert out["hypotheses"]["H12-A"]["exploratory_guard"] is False


def test_rejects_unregistered_n() -> None:
    with pytest.raises(ValueError, match="n_planned"):
        aggregate(_scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT"), n_planned=31)


def test_rejects_removed_n30() -> None:
    # PI decision 2026-10-24: N = 30 is no longer a registered N
    recs = _records(_vec(0.1, 0.01, 30, 2), _vec(0.1, 0.01, 30, 3), _vec(0.0, 0.01, 30, 4))
    with pytest.raises(ValueError, match="n_planned"):
        aggregate(recs, n_planned=30)


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


def test_select_n_floor_line_with_low_r() -> None:
    out = select_n(_smoke(0.005, rho_low=True))
    assert out["min_r_hat"] < 0.5  # r^ is reported, it no longer selects N
    assert out["n"] == 95 and out["line"] == 2


def test_select_n_floor_line_with_high_r() -> None:
    # PI decision 2026-10-24: the N = 30 line is removed; 95 is the floor
    out = select_n(_smoke(0.005, rho_low=False))
    assert out["min_r_hat"] >= 0.5
    assert out["n"] == 95 and out["line"] == 2


def test_select_n_just_below_threshold_stays_at_floor() -> None:
    out = select_n(_smoke(0.0149, rho_low=False))
    assert out["sd_hat_d"] == pytest.approx(0.0149)
    assert out["n"] == 95 and out["line"] == 2


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
    assert main([*files, "--n-planned", "95", "--out", str(out)]) == 0
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


def test_missing_cell_record_counts_as_cell_level_loss() -> None:
    # a cell that crashed (e.g. a BLOCKING invariant raised in the DE
    # runtime) writes no record: it is a cell-level loss with a reason
    recs = [
        r for r in _scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT")
        if not (r["seed"] == 5 and r["arm"] == "P_max" and r["strategy"] == "none")
    ]
    out = aggregate(recs, n_planned=N)
    a = out["hypotheses"]["H12-A"]
    assert a["n"] == N - 1 and a["n_lost_cell_level"] == 1
    assert a["loss_reasons"] == {"5": "missing_record:P_max/none"}
    assert out["incomplete_run"] is True


def test_missing_seed_is_lost_and_run_incomplete() -> None:
    recs = [r for r in _scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT") if r["seed"] != 94]
    out = aggregate(recs, n_planned=N)
    assert out["hypotheses"]["H12-A"]["n"] == N - 1
    assert out["incomplete_run"] is True


def test_rejects_seed_outside_planned_list() -> None:
    recs = _scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT")
    recs[0] = dict(recs[0], seed=900)
    with pytest.raises(ValueError, match="seed"):
        aggregate(recs, n_planned=N)


# --- differential-exclusion guard qualifies the row (review fix 2) ---------


def _lose(cell: tuple[str, str], k: int) -> dict[tuple[int, str, str], str]:
    return {(s, *cell): "non_finite_logits" for s in range(k)}


def test_guard_marks_dependent_row_exploratory() -> None:
    # 10 of 95 seeds lose P_max/cross: H12-A and H12-B are guarded
    recs = _scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT",
                     excluded=_lose(("P_max", "cross"), 10))
    out = aggregate(recs, n_planned=N)
    assert out["row"] == 2  # raw row is kept
    assert out["row_confirmatory"] is False
    assert out["row_guarded_by"] == ["H12-A", "H12-B"]
    assert out["hypotheses"]["H12-A"]["confirmatory"] is False
    assert out["hypotheses"]["H12-C"]["confirmatory"] is True


def test_unguarded_row_is_confirmatory() -> None:
    out = aggregate(_scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT"), n_planned=N)
    assert out["row_confirmatory"] is True and out["row_guarded_by"] == []


def test_row1_reads_only_h12c() -> None:
    recs = _scenario("CONFIRMED", "CONFIRMED", "MEANINGFULLY-DIFFERENT",
                     excluded=_lose(("P_max", "cross"), 10))
    out = aggregate(recs, n_planned=N)
    assert out["row"] == 1 and out["row_confirmatory"] is True


def test_guarded_data_never_yields_unqualified_row() -> None:
    # P_max/none feeds H12-A and H12-C, so every row depends on a guarded
    # hypothesis: no combination may produce a confirmatory reading
    for a in AB:
        for b in AB:
            for c in C:
                recs = _scenario(a, b, c, excluded=_lose(("P_max", "none"), 10))
                out = aggregate(recs, n_planned=N)
                assert out["row_confirmatory"] is False, (a, b, c, out["row"])


def test_abort_row_is_never_confirmatory() -> None:
    recs = _scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT", manip_fail=set(range(48)))
    out = aggregate(recs, n_planned=N)
    assert out["row"] == 0 and out["row_confirmatory"] is False


# --- worst-case imputation sensitivity (PI decision 2026-10-24) -----------


def test_worst_case_not_needed_without_cell_losses() -> None:
    out = aggregate(_scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT"), n_planned=N)
    wc = out["worst_case_sensitivity"]
    assert wc["reported_only"] is True
    for h in ("H12-A", "H12-B", "H12-C"):
        assert wc["hypotheses"][h]["n_imputed"] == 0
        assert wc["hypotheses"][h]["state"] == out["hypotheses"][h]["state"]
    assert wc["row"] == out["row"]


def test_worst_case_imputes_against_the_treatment() -> None:
    excl = _lose(("P_max", "cross"), 10)
    recs = _scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT", excluded=excl)
    out = aggregate(recs, n_planned=N)
    wc = out["worst_case_sensitivity"]["hypotheses"]["H12-A"]
    assert wc["n_imputed"] == 10 and wc["n"] == N
    cross = [r["cdpa"] for r in recs if (r["arm"], r["strategy"]) == ("P_max", "cross")
             and r["cdpa"] is not None]
    none = [r["cdpa"] for r in recs if (r["arm"], r["strategy"]) == ("P_max", "none")]
    assert wc["imputed_values"]["P_max/cross"] == pytest.approx(min(cross))
    assert wc["imputed_values"]["P_max/none"] == pytest.approx(max(none))
    assert wc["mean"] < out["hypotheses"]["H12-A"]["mean"]
    # the decided state is never changed by the sensitivity analysis
    plain = aggregate(recs, n_planned=N)["hypotheses"]["H12-A"]["state"]
    assert out["hypotheses"]["H12-A"]["state"] == plain


def test_worst_case_imputes_missing_records_but_not_seed_exclusions() -> None:
    recs = [
        r for r in _scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT", manip_fail={0})
        if not (r["seed"] == 5 and r["arm"] == "P_max" and r["strategy"] == "none")
    ]
    out = aggregate(recs, n_planned=N)
    wc = out["worst_case_sensitivity"]["hypotheses"]["H12-A"]
    assert wc["n_imputed"] == 1 and wc["n"] == N - 1  # seed 0 stays excluded


def test_worst_case_can_change_the_sensitivity_state_only() -> None:
    # heavy one-sided losses flip the worst case away from CONFIRMED
    excl = _lose(("P_max", "cross"), 40)
    out = aggregate(_scenario("CONFIRMED", "CONFIRMED", "EQUIVALENT", excluded=excl), n_planned=N)
    assert out["hypotheses"]["H12-A"]["state"] == "CONFIRMED"
    assert out["worst_case_sensitivity"]["hypotheses"]["H12-A"]["state"] != "CONFIRMED"


# --- smoke validation for the N rule (review fix 3) -----------------------


def test_select_n_requires_every_smoke_cell() -> None:
    recs = [r for r in _smoke(0.005, rho_low=False)
            if not (r["seed"] == 901 and r["arm"] == "baseline")]
    with pytest.raises(ValueError, match="smoke"):
        select_n(recs)


def test_select_n_refused_on_majority_manipulation_failure() -> None:
    recs = _smoke(0.02, rho_low=False)
    for r in recs:
        if r["seed"] in (900, 901):
            r["manip_pass"] = False
    out = select_n(recs)
    assert out["n"] is None and out["abort"] is True
    assert out["n_manip_fail"] == 2


def test_select_n_single_manipulation_failure_uses_two_seeds() -> None:
    recs = _smoke(0.02, rho_low=False)
    for r in recs:
        if r["seed"] == 902:
            r["manip_pass"] = False
    out = select_n(recs)
    assert out["abort"] is False and out["n_manip_fail"] == 1
    assert out["seeds_used"] == [900, 901]
    assert out["n"] in (95, 108)


def test_select_n_refused_on_run_id_collision() -> None:
    recs = _smoke(0.005, rho_low=False)
    recs.append({"status": "aborted", "abort_reason": "run_id_collision", "seed": 900,
                 "arm": "P_max", "strategy": "cross"})
    out = select_n(recs)
    assert out["n"] is None and out["abort"] is True


def test_n_rule_boundary_is_inclusive() -> None:
    assert n_from_sd(0.015) == (108, 1)
    assert n_from_sd(float(np.nextafter(0.015, 0.0))) == (95, 2)


def test_cli_select_n_refusal_exit_code(tmp_path: Path) -> None:
    files = []
    for i, rec in enumerate(_smoke(0.02, rho_low=False)):
        if rec["seed"] in (900, 901):
            rec["manip_pass"] = False
        p = tmp_path / f"smoke{i:02d}.json"
        p.write_text(json.dumps(rec))
        files.append(str(p))
    out = tmp_path / "n.json"
    assert main([*files, "--select-n", "--out", str(out)]) == 1
    assert json.loads(out.read_text())["n"] is None


def test_select_n_validates_the_dropped_seed_too() -> None:
    # review round 2: a single manipulation-failing seed with a NaN cell
    # must refuse the selection, not be silently dropped
    recs = _smoke(0.005, rho_low=False)
    for r in recs:
        if r["seed"] == 902:
            r["manip_pass"] = False
            if (r["arm"], r["strategy"]) == ("P_equ", "cross"):
                r["cdpa"] = float("nan")
    with pytest.raises(ValueError, match="smoke"):
        select_n(recs)
