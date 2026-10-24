"""DR-3 relational probe script: .npz input contract and JSON output."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.dr3_relational_probe import (
    analyse_seed,
    load_seed_npz,
    main,
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
