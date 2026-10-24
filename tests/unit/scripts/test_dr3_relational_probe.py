"""DR-3 relational probe script: .npz input contract and JSON output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from scripts.dr3_relational_probe import (
    analyse_seed,
    calibrate,
    decide,
    eligibility_problems,
    permutation_null,
    load_seed_npz,
    main,
    outcome_row,
    statistic_state,
)


def _write(path: Path, seed: int = 100, n: int = 40, **over: object) -> Path:
    rng = np.random.default_rng(seed)
    a = rng.normal(size=(n, 6)).astype(np.float32)
    arrays: dict[str, object] = {
        "seed": np.int64(seed),
        "a_trained": a,
        "b_trained": (a + 0.1 * rng.normal(size=a.shape)).astype(np.float32),
        "a_null": rng.normal(size=(n, 6)).astype(np.float32),
        "b_null": rng.normal(size=(n, 6)).astype(np.float32),
    }
    arrays.update(over)
    np.savez(path, **arrays)  # type: ignore[arg-type]
    return path


def test_load_valid_npz(tmp_path: Path) -> None:
    rec = load_seed_npz(_write(tmp_path / "s100.npz"))
    assert rec["seed"] == 100
    assert rec["a_trained"].shape == (40, 6)
    assert rec["sha256"] is None  # no sidecar


def test_load_rejects_missing_key(tmp_path: Path) -> None:
    p = tmp_path / "s.npz"
    np.savez(p, seed=np.int64(1), a_trained=np.ones((5, 2)))
    with pytest.raises(ValueError, match="missing key"):
        load_seed_npz(p)


def test_load_rejects_row_mismatch(tmp_path: Path) -> None:
    p = _write(tmp_path / "s.npz", b_null=np.ones((30, 6), dtype=np.float32))
    with pytest.raises(ValueError, match="same number of rows"):
        load_seed_npz(p)


def test_load_rejects_non_2d(tmp_path: Path) -> None:
    p = _write(tmp_path / "s.npz", a_null=np.ones(40, dtype=np.float32))
    with pytest.raises(ValueError, match="2-D"):
        load_seed_npz(p)


def test_load_rejects_non_numeric(tmp_path: Path) -> None:
    p = _write(tmp_path / "s.npz", a_null=np.array([["x"] * 6] * 40))
    with pytest.raises(ValueError, match="numeric"):
        load_seed_npz(p)


def test_sha256_sidecar_verified(tmp_path: Path) -> None:
    p = _write(tmp_path / "s.npz")
    digest = hashlib.sha256(p.read_bytes()).hexdigest()
    (tmp_path / "s.npz.sha256").write_text(f"{digest}  s.npz\n")
    assert load_seed_npz(p)["sha256"] == digest
    (tmp_path / "s.npz.sha256").write_text("0" * 64 + "  s.npz\n")
    with pytest.raises(ValueError, match="sha256 mismatch"):
        load_seed_npz(p)


def test_require_hash_flag(tmp_path: Path) -> None:
    p = _write(tmp_path / "s.npz")
    with pytest.raises(ValueError, match="sha256"):
        load_seed_npz(p, require_hash=True)


def test_analyse_seed_ok_has_differences(tmp_path: Path) -> None:
    rec = analyse_seed(load_seed_npz(_write(tmp_path / "s.npz")), k=5, tau=0.1)
    assert rec["status"] == "ok"
    for key in ("S_mKNN", "S_H0", "S_mKNN_tau", "S_H0_tau"):
        assert rec["D"][key] == pytest.approx(rec["trained"][key] - rec["null"][key])
    # trained pair is a noisy copy, null pair is independent
    assert rec["D"]["S_mKNN"] > 0.3


def test_analyse_seed_excludes_coincident_rows(tmp_path: Path) -> None:
    p = _write(tmp_path / "s.npz", a_null=np.ones((40, 6), dtype=np.float32))
    rec = analyse_seed(load_seed_npz(p), k=5, tau=0.1)
    assert rec["status"] == "excluded"
    assert any("a_null" in r and "coincide" in r for r in rec["reasons"])
    assert "D" not in rec


def _gate(acc_a: float, acc_b: float) -> dict[str, Any]:
    return {
        "acc_a_trained": np.float64(acc_a),
        "acc_b_trained": np.float64(acc_b),
        "acc_a_null": np.float64(0.08),
        "acc_b_null": np.float64(0.09),
        "gate_pass": np.bool_(acc_a >= 0.25 and acc_b >= 0.25),
    }


def test_learning_gate_fields_from_export_helper(tmp_path: Path) -> None:
    rec = load_seed_npz(_write(tmp_path / "s.npz", **_gate(0.6, 0.4)))
    assert rec["gate"] == {
        "gate_pass": True,
        "acc_a_trained": pytest.approx(0.6),
        "acc_b_trained": pytest.approx(0.4),
        "acc_a_null": pytest.approx(0.08),
        "acc_b_null": pytest.approx(0.09),
    }
    out = analyse_seed(rec, k=5, tau=0.1)
    assert out["status"] == "ok"
    assert out["gate"]["gate_pass"] is True


def test_learning_gate_failure_excludes_seed(tmp_path: Path) -> None:
    rec = load_seed_npz(_write(tmp_path / "s.npz", **_gate(0.6, 0.2)))
    out = analyse_seed(rec, k=5, tau=0.1)
    assert out["status"] == "excluded"
    assert any("learning gate" in r and "b_trained" in r for r in out["reasons"])
    assert "D" not in out


def test_learning_gate_absent_means_not_scored(tmp_path: Path) -> None:
    rec = load_seed_npz(_write(tmp_path / "s.npz"))
    assert rec["gate"] is None
    assert analyse_seed(rec, k=5, tau=0.1)["gate"] is None


def test_learning_gate_failure_counts_towards_abort(tmp_path: Path) -> None:
    ok = _write(tmp_path / "s100.npz", seed=100, **_gate(0.6, 0.4))
    bad = _write(tmp_path / "s101.npz", seed=101, **_gate(0.1, 0.4))
    out = tmp_path / "out.json"
    main([str(ok), str(bad), "--k", "5", "--output", str(out), "--abort-threshold", "0"])
    doc = json.loads(out.read_text())
    assert doc["n_excluded"] == 1 and doc["abort"] is True


def test_analyse_seed_sensitivity_blocks(tmp_path: Path) -> None:
    rec = analyse_seed(load_seed_npz(_write(tmp_path / "s.npz")), k=5, tau=0.1, sensitivity=True)
    sens = rec["sensitivity"]
    assert set(sens["pipelines"]) == {"l2", "center_l2", "raw"}
    assert set(sens["tau_grid"]) == {"1", "0.1", "0.01", "0.001", "0.0001"}
    assert set(sens["k"]) == {"5", "20"}


def test_main_writes_json_and_flags_abort(tmp_path: Path) -> None:
    files = [str(_write(tmp_path / f"s{s}.npz", seed=s)) for s in (100, 101)]
    bad = _write(tmp_path / "s102.npz", seed=102, a_null=np.ones((40, 6), dtype=np.float32))
    out = tmp_path / "out.json"
    rc = main([*files, str(bad), "--k", "5", "--output", str(out), "--abort-threshold", "0"])
    doc = json.loads(out.read_text())
    assert rc == 0
    assert [s["seed"] for s in doc["seeds"]] == [100, 101, 102]
    assert doc["n_retained"] == 2 and doc["n_excluded"] == 1
    assert doc["abort"] is True
    assert doc["protocol"]["preprocess"] == "l2" and doc["protocol"]["tau"] == 0.1


# --- 1-NN comparator block ------------------------------------------------


def test_analyse_seed_has_comparator(tmp_path: Path) -> None:
    rec = analyse_seed(load_seed_npz(_write(tmp_path / "s.npz")), k=5, tau=0.1)
    comp = rec["comparator"]
    for key in ("S_mKNN_k1", "S_H0_res"):
        assert comp["D"][key] == pytest.approx(comp["trained"][key] - comp["null"][key])


# --- section 2 states and section 7 rows (decision code) ------------------

N20 = 20
DELTA = 0.025


def _z(n: int, seed: int) -> np.ndarray:
    z = np.random.default_rng(seed).normal(size=n)
    z = z - z.mean()
    out: np.ndarray = z / z.std(ddof=1)
    return out


def _d(mean: float, spread: float, seed: int = 1) -> list[float]:
    return list(mean + spread * _z(N20, seed))


# n = 20, alpha = 0.0125: q / sqrt(n) ~ 0.54
STATE_D = {
    "SURVIVES": (0.05, 0.01),
    "FAILS": (0.0, 0.01),
    "INDETERMINATE": (0.0, 0.08),
}


@pytest.mark.parametrize("state", list(STATE_D))
def test_every_statistic_state(state: str) -> None:
    out = statistic_state(_d(*STATE_D[state]), delta=DELTA, alpha=0.0125)
    assert out["state"] == state


def test_state_survives_needs_point_estimate_at_delta() -> None:
    # L > 0 but mean < delta and U >= delta: INDETERMINATE, not SURVIVES
    out = statistic_state(_d(0.02, 0.01), delta=DELTA, alpha=0.0125)
    assert out["L"] > 0 and out["mean"] < DELTA and out["U"] >= DELTA
    assert out["state"] == "INDETERMINATE"


def test_state_degenerate_identical_differences() -> None:
    out = statistic_state([0.125] * N20, delta=DELTA, alpha=0.0125)
    assert out["degenerate"] is True and out["L"] == out["U"] == 0.125
    assert out["state"] == "SURVIVES"


ROWS = [
    (1, ("SURVIVES", "SURVIVES", "FAILS", "FAILS")),
    (2, ("SURVIVES", "SURVIVES", "SURVIVES", "FAILS")),
    (2, ("SURVIVES", "SURVIVES", "INDETERMINATE", "SURVIVES")),
    (3, ("SURVIVES", "FAILS", "FAILS", "FAILS")),
    (4, ("FAILS", "SURVIVES", "INDETERMINATE", "FAILS")),
    (5, ("SURVIVES", "SURVIVES", "INDETERMINATE", "FAILS")),
    (5, ("SURVIVES", "INDETERMINATE", "FAILS", "FAILS")),
    (5, ("INDETERMINATE", "SURVIVES", "FAILS", "FAILS")),
]


@pytest.mark.parametrize(("row", "states"), ROWS)
def test_every_section7_row(row: int, states: tuple[str, str, str, str]) -> None:
    assert outcome_row(states, abort=False) == row


def test_row_table_exhaustive_and_abort() -> None:
    import itertools

    for combo in itertools.product(STATE_D, repeat=4):
        assert outcome_row(combo, abort=False) in range(1, 6)
    assert outcome_row(("SURVIVES",) * 4, abort=True) == 0


def _seed_rec(seed: int, d: dict[str, float], res: float) -> dict[str, object]:
    return {
        "seed": seed, "status": "ok", "sha256": "ab" * 32, "pinned": True,
        "dims": {k: [1024, 16] for k in ("a_trained", "b_trained", "a_null", "b_null")},
        "gate": {"gate_pass": True, "acc_a_trained": 0.5, "acc_b_trained": 0.5},
        "D": d,
        "comparator": {"D": {"S_mKNN_k1": res, "S_H0_res": res}},
    }


PIN = "95ac6b78477c3ce999b31bca775769041e082b46"


def _eligible(seeds: list[Any], abort: bool = False) -> dict[str, Any]:
    return {
        "seeds": seeds, "abort": abort,
        "protocol": {"k": 10, "tau": 0.1, "preprocess": "l2"},
        "abort_threshold": 6,
        "provenance": {"manifest_hash_ok": True, "confirmatory_eligible": True,
                       "seed_hash_mismatch": [], "seeds_not_in_manifest": [],
                       "nerve_wml": {"commit": PIN, "head": PIN}},
    }


def _seeds_for(states: tuple[str, ...], res_mean: float = 0.05) -> list[Any]:
    keys = ("S_mKNN", "S_H0", "S_mKNN_tau", "S_H0_tau")
    cols = {k: _d(*STATE_D[s], seed=i + 2) for i, (k, s) in enumerate(zip(keys, states))}
    res = _d(res_mean, 0.01, seed=9)
    return [_seed_rec(100 + i, {k: cols[k][i] for k in keys}, res[i]) for i in range(N20)]


@pytest.mark.parametrize(("row", "states"), ROWS)
def test_decide_end_to_end_rows(row: int, states: tuple[str, ...]) -> None:
    keys = ("S_mKNN", "S_H0", "S_mKNN_tau", "S_H0_tau")
    cols = {k: _d(*STATE_D[s], seed=i + 2) for i, (k, s) in enumerate(zip(keys, states))}
    res = _d(0.05, 0.01, seed=9)
    seeds = [
        _seed_rec(100 + i, {k: cols[k][i] for k in keys}, res[i]) for i in range(N20)
    ]
    out = decide(_eligible(seeds), delta=DELTA)
    assert out["eligible"] is True, out["eligibility_problems"]
    assert tuple(out["states"][k]["state"] for k in keys) == states
    assert out["row"] == row
    assert out["confirmatory_seed_set"] is True


def test_decide_global_wording_rule() -> None:
    keys = ("S_mKNN", "S_H0", "S_mKNN_tau", "S_H0_tau")
    base = {k: _d(*STATE_D[s], seed=i + 2)
            for i, (k, s) in enumerate(zip(keys, ("SURVIVES", "SURVIVES", "FAILS", "FAILS")))}
    for res_mean, allowed in ((0.05, True), (0.01, False), (0.0, False)):
        res = _d(res_mean, 0.01, seed=9)
        seeds = [_seed_rec(100 + i, {k: base[k][i] for k in keys}, res[i]) for i in range(N20)]
        out = decide(_eligible(seeds), delta=DELTA)
        assert out["row"] == 1
        assert out["global_wording_allowed"] is allowed
        assert out["comparator"]["S_H0_res"]["ci95"][0] <= out["comparator"]["S_H0_res"]["mean"]


def test_decide_abort_and_excluded_seeds() -> None:
    keys = ("S_mKNN", "S_H0", "S_mKNN_tau", "S_H0_tau")
    seeds = [_seed_rec(100 + i, {k: 0.05 + 0.001 * i for k in keys}, 0.0) for i in range(N20)]
    seeds[0] = dict(seeds[0], status="excluded", reasons=["x"])
    out = decide(_eligible(seeds), delta=DELTA)
    assert out["n_retained"] == N20 - 1
    out = decide(_eligible(seeds, abort=True), delta=DELTA)
    assert out["row"] == 0 and out["global_wording_allowed"] is False


def test_decide_flags_nonconfirmatory_seed_set() -> None:
    keys = ("S_mKNN", "S_H0", "S_mKNN_tau", "S_H0_tau")
    seeds = [_seed_rec(900 + i, {k: 0.05 + 0.001 * i for k in keys}, 0.0) for i in range(3)]
    out = decide(_eligible(seeds), delta=DELTA)
    assert out["confirmatory_seed_set"] is False
    assert out["eligible"] is False and out["row"] is None and out["states"] is None
    assert out["exploratory"]["row"] in range(0, 6)


def test_main_decide_flag(tmp_path: Path) -> None:
    files = [str(_write(tmp_path / f"s{s}.npz", seed=s)) for s in (100, 101, 102)]
    out = tmp_path / "out.json"
    assert main([*files, "--output", str(out), "--decide"]) == 0
    doc = json.loads(out.read_text())
    assert doc["decision"]["eligible"] is False and doc["decision"]["row"] is None
    assert doc["decision"]["exploratory"]["row"] in range(0, 6)
    assert doc["decision"]["confirmatory_seed_set"] is False


@pytest.mark.parametrize("flags", [["--k", "5"], ["--tau", "0.01"]])
def test_decide_rejects_non_primary_protocol(tmp_path: Path, flags: list[str]) -> None:
    f = str(_write(tmp_path / "s100.npz"))
    with pytest.raises(SystemExit):
        main([f, *flags, "--output", str(tmp_path / "o.json"), "--decide"])


# --- confirmatory eligibility (review fix 1) --------------------------------


def _ok_seeds() -> list[Any]:
    return _seeds_for(("SURVIVES", "SURVIVES", "FAILS", "FAILS"))


def test_eligibility_clean_doc_has_no_problems() -> None:
    assert eligibility_problems(_eligible(_ok_seeds())) == []


@pytest.mark.parametrize(
    ("mutate", "needle"),
    [
        (lambda d: d.pop("provenance"), "manifest"),
        (lambda d: d["provenance"].update(confirmatory_eligible=False), "unpinned"),
        (lambda d: d["provenance"].update(manifest_hash_ok=False), "hash not verified"),
        (lambda d: d["provenance"].update(seed_hash_mismatch=[101]), "do not match"),
        (lambda d: d["protocol"].update(tau=0.01), "non-primary"),
        (lambda d: d["seeds"].pop(), "100..119"),
        (lambda d: d["seeds"].append(dict(d["seeds"][0])), "duplicate"),
        (lambda d: d["seeds"][3].update(pinned=False), "unpinned"),
        (lambda d: d["seeds"][3].update(sha256=None), "hash not verified"),
        (lambda d: d["seeds"][3]["dims"].update(a_null=[1024, 8]), "1024"),
        (lambda d: d["seeds"][3].pop("gate"), "gate not scored"),
        (lambda d: d["seeds"][3]["gate"].update(acc_b_trained=float("nan")), "gate not scored"),
    ],
)
def test_eligibility_problems_block_the_decision(mutate: Any, needle: str) -> None:
    doc = _eligible(_ok_seeds())
    mutate(doc)
    probs = eligibility_problems(doc)
    assert any(needle in p for p in probs), probs
    out = decide(doc, delta=DELTA)
    assert out["eligible"] is False
    assert out["row"] is None and out["states"] is None
    assert out["global_wording_allowed"] is False
    assert "exploratory" in out


# --- "global" wording rule needs eligibility, no abort, row 1-2 (fix 6) -----


@pytest.mark.parametrize(("row", "states"), ROWS)
def test_global_wording_only_on_rows_1_and_2(row: int, states: tuple[str, ...]) -> None:
    out = decide(_eligible(_seeds_for(states, res_mean=0.05)), delta=DELTA)
    assert out["row"] == row
    assert out["global_wording_allowed"] is (row in (1, 2))


def test_global_wording_never_on_abort_or_ineligible() -> None:
    seeds = _seeds_for(("SURVIVES", "SURVIVES", "FAILS", "FAILS"), res_mean=0.05)
    assert decide(_eligible(seeds), delta=DELTA)["global_wording_allowed"] is True
    assert decide(_eligible(seeds, abort=True), delta=DELTA)["global_wording_allowed"] is False
    doc = _eligible(seeds)
    doc["provenance"]["confirmatory_eligible"] = False
    assert decide(doc, delta=DELTA)["global_wording_allowed"] is False


# --- gate recomputed, supplied gate_pass not trusted (fix 1) ---------------


def test_gate_pass_true_but_low_accuracy_is_excluded(tmp_path: Path) -> None:
    g = _gate(0.6, 0.2)
    g["gate_pass"] = np.bool_(True)  # lying flag
    out = analyse_seed(load_seed_npz(_write(tmp_path / "s.npz", **g)), k=5, tau=0.1)
    assert out["status"] == "excluded" and out["gate_pass_mismatch"] is True


def test_gate_non_finite_accuracy_is_excluded(tmp_path: Path) -> None:
    g = _gate(float("nan"), 0.6)
    out = analyse_seed(load_seed_npz(_write(tmp_path / "s.npz", **g)), k=5, tau=0.1)
    assert out["status"] == "excluded"
    assert any("non-finite" in r for r in out["reasons"])


# --- permutation-null secondary (PI decision 2026-10-24) -------------------


def test_calibrate_order_statistic_and_rescaling() -> None:
    perm = np.arange(19) / 100.0  # 0.00 .. 0.18
    out = calibrate(0.5, perm, alpha=0.05)
    # pool of 20 values; rank ceil(0.95 * 20) = 19 -> 0.18
    assert out["c_alpha"] == pytest.approx(0.18)
    assert out["T_cal"] == pytest.approx((0.5 - 0.18) / (1 - 0.18))
    assert out["K"] == 19 and out["p_add_one"] == pytest.approx(1 / 20)


def test_calibrate_clips_to_zero_and_handles_c_one() -> None:
    assert calibrate(0.1, np.full(19, 0.3))["T_cal"] == 0.0
    assert calibrate(1.0, np.ones(19))["T_cal"] == 0.0


def test_calibrate_95th_percentile_at_k500() -> None:
    perm = np.linspace(0.0, 1.0, 500, endpoint=False)
    out = calibrate(0.2, perm)
    pool = np.sort(np.concatenate([[0.2], perm]))
    assert out["c_alpha"] == pool[int(np.ceil(0.95 * 501)) - 1]


def test_permutation_null_identical_pair_calibrates_high() -> None:
    x = np.random.default_rng(0).normal(size=(40, 6))
    out = permutation_null(x, x.copy(), k=5, tau=0.1, n_perm=19, seed=3)
    assert set(out) == {"S_mKNN", "S_H0", "S_mKNN_tau", "S_H0_tau"}
    assert out["S_mKNN"]["T_obs"] == pytest.approx(1.0)
    assert out["S_mKNN"]["T_cal"] > 0.5
    again = permutation_null(x, x.copy(), k=5, tau=0.1, n_perm=19, seed=3)
    assert again == out  # fixed stream


def test_permutation_null_reported_in_seed_record(tmp_path: Path) -> None:
    out = tmp_path / "o.json"
    f = str(_write(tmp_path / "s100.npz"))
    assert main([f, "--k", "5", "--output", str(out), "--permutation-null", "--n-perm", "9"]) == 0
    rec = json.loads(out.read_text())["seeds"][0]
    pn = rec["permutation_null"]
    assert pn["reported_only"] is True and pn["trained"]["S_H0"]["K"] == 9
    assert "D" in rec  # primary statistics unchanged alongside


# --- review round 2 ------------------------------------------------------------


def test_missing_pinned_flag_is_not_eligible() -> None:
    doc = _eligible(_ok_seeds())
    doc["seeds"][4].pop("pinned")
    assert any("unpinned" in p for p in eligibility_problems(doc))
    assert decide(doc, delta=DELTA)["eligible"] is False


def test_manifest_pin_must_be_the_registered_commit() -> None:
    doc = _eligible(_ok_seeds())
    doc["provenance"]["nerve_wml"] = {"commit": "0" * 40, "head": "0" * 40}
    assert any("pin" in p for p in eligibility_problems(doc))


def test_registered_abort_recomputed_despite_threshold_override() -> None:
    seeds = _ok_seeds()
    for s in seeds[:7]:
        s["status"] = "excluded"
        s["reasons"] = ["x"]
    doc = _eligible(seeds)
    doc["abort"] = False  # computed with --abort-threshold 20
    doc["abort_threshold"] = 20
    out = decide(doc, delta=DELTA)
    assert out["abort"] is True
    assert out["eligible"] is False
    assert any("abort threshold" in p for p in out["eligibility_problems"])
    assert out["global_wording_allowed"] is False
    doc["abort_threshold"] = 6
    out = decide(doc, delta=DELTA)
    assert out["abort"] is True and out["row"] == 0
    assert out["global_wording_allowed"] is False


def test_decide_rejects_abort_threshold_override(tmp_path: Path) -> None:
    f = str(_write(tmp_path / "s100.npz"))
    with pytest.raises(SystemExit):
        main([f, "--abort-threshold", "20", "--output", str(tmp_path / "o.json"), "--decide"])
