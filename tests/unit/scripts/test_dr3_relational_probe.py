"""DR-3 relational probe script: .npz input contract and JSON output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.dr3_relational_probe import (
    analyse_seed,
    decide,
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
        "seed": seed, "status": "ok", "sha256": "ab" * 32,
        "D": d,
        "comparator": {"D": {"S_mKNN_k1": res, "S_H0_res": res}},
    }


@pytest.mark.parametrize(("row", "states"), ROWS)
def test_decide_end_to_end_rows(row: int, states: tuple[str, ...]) -> None:
    keys = ("S_mKNN", "S_H0", "S_mKNN_tau", "S_H0_tau")
    cols = {k: _d(*STATE_D[s], seed=i + 2) for i, (k, s) in enumerate(zip(keys, states))}
    res = _d(0.05, 0.01, seed=9)
    seeds = [
        _seed_rec(100 + i, {k: cols[k][i] for k in keys}, res[i]) for i in range(N20)
    ]
    out = decide({"seeds": seeds, "abort": False}, delta=DELTA)
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
        out = decide({"seeds": seeds, "abort": False}, delta=DELTA)
        assert out["row"] == 1
        assert out["global_wording_allowed"] is allowed
        assert out["comparator"]["S_H0_res"]["ci95"][0] <= out["comparator"]["S_H0_res"]["mean"]


def test_decide_abort_and_excluded_seeds() -> None:
    keys = ("S_mKNN", "S_H0", "S_mKNN_tau", "S_H0_tau")
    seeds = [_seed_rec(100 + i, {k: 0.05 + 0.001 * i for k in keys}, 0.0) for i in range(N20)]
    seeds[0] = {"seed": 100, "status": "excluded", "reasons": ["x"]}
    out = decide({"seeds": seeds, "abort": False}, delta=DELTA)
    assert out["n_retained"] == N20 - 1
    out = decide({"seeds": seeds, "abort": True}, delta=DELTA)
    assert out["row"] == 0


def test_decide_flags_nonconfirmatory_seed_set() -> None:
    keys = ("S_mKNN", "S_H0", "S_mKNN_tau", "S_H0_tau")
    seeds = [_seed_rec(900 + i, {k: 0.05 + 0.001 * i for k in keys}, 0.0) for i in range(3)]
    out = decide({"seeds": seeds, "abort": False}, delta=DELTA)
    assert out["confirmatory_seed_set"] is False


def test_main_decide_flag(tmp_path: Path) -> None:
    files = [str(_write(tmp_path / f"s{s}.npz", seed=s)) for s in (100, 101, 102)]
    out = tmp_path / "out.json"
    assert main([*files, "--k", "5", "--output", str(out), "--decide"]) == 0
    doc = json.loads(out.read_text())
    assert doc["decision"]["row"] in range(0, 6)
    assert doc["decision"]["confirmatory_seed_set"] is False
