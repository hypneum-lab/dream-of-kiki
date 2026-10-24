"""G4-nono driver: synthetic stream, no CIFAR download."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.g4_nono_test import run_cross_domain as drv
from experiments.g4_nono_test.stream import (
    DOMAIN_A,
    TASK_CLASS_PAIRS,
    build_domain_stream,
)


def _synthetic_stream(n_train: int = 20, n_test: int = 6, seed: int = 0):
    rng = np.random.default_rng(seed)

    def mk(n):
        y = np.repeat(np.arange(20, dtype=np.uint8), n)
        x = rng.integers(0, 256, (len(y), 32, 32, 3), dtype=np.uint8)
        # class-dependent brightness so the net can learn something
        x = np.clip(x // 4 + (y[:, None, None, None] * 9), 0, 255).astype(np.uint8)
        return x, y

    xtr, ytr = mk(n_train)
    xte, yte = mk(n_test)
    return build_domain_stream(xtr, ytr, xte, yte)


@pytest.fixture(scope="module")
def stream():
    return _synthetic_stream()


def test_cell_grid_is_8_per_seed() -> None:
    cells = drv.cells_for_seed()
    assert len(cells) == 8
    assert ("baseline", "na") in cells and ("P_min", "na") in cells
    for arm in ("P_equ", "P_max"):
        for s in ("cross", "within", "none"):
            assert (arm, s) in cells


def test_invalid_arm_strategy_combo_rejected(stream) -> None:
    with pytest.raises(ValueError):
        drv.validate_cell("baseline", "cross")
    with pytest.raises(ValueError):
        drv.validate_cell("P_min", "within")
    with pytest.raises(ValueError):
        drv.validate_cell("P_max", "na")
    with pytest.raises(ValueError):
        drv.validate_cell("P_zzz", "none")


def test_constants_match_spec() -> None:
    assert drv.BUFFER_CAPACITY == 320
    assert drv.PUSH_PER_SUPERCLASS == 16
    assert drv.N_SYNTHETIC == 16
    assert drv.LATENT_DIM == 64 and drv.N_CLASSES == 20
    assert drv.EPOCHS == 3 and drv.BATCH_SIZE == 64 and drv.LR == 0.01
    assert drv.MANIP_THRESHOLD == 0.60
    assert drv.RESTRUCTURE_FACTOR == 0.05
    assert drv.RECOMBINE_LR == 0.01
    assert drv.SMOKE_SEEDS == (900, 901, 902)


def test_buffer_push_16_plus_16_latents_frozen(stream) -> None:
    from experiments.g4_quinto_test.small_cnn import G4SmallCNN

    cnn = G4SmallCNN(latent_dim=64, n_classes=20, seed=0)
    buf = drv.new_buffer()
    fill_rng = np.random.default_rng(5000)
    drv.push_task(buf, cnn, stream[0], fill_rng)
    assert len(buf) == 32
    ys = [r["y"] for r in buf._records]
    assert ys.count(0) == 16 and ys.count(1) == 16
    before = [list(r["latent"]) for r in buf._records]
    cnn.train_task(
        stream[0]["x_train_nhwc"], stream[0]["y_train"],
        epochs=1, batch_size=64, lr=0.05,
    )
    after = [list(r["latent"]) for r in buf._records]
    assert before == after  # frozen at push time
    # but a fresh latent from the trained net differs
    x = np.asarray(buf._records[0]["x"], np.float32)[None]
    assert not np.allclose(cnn.latent(x)[0], before[0])


def test_buffer_fill_without_replacement(stream) -> None:
    from experiments.g4_quinto_test.small_cnn import G4SmallCNN

    cnn = G4SmallCNN(latent_dim=64, n_classes=20, seed=0)
    buf = drv.new_buffer()
    drv.push_task(buf, cnn, stream[0], np.random.default_rng(1))
    xs = {np.asarray(r["x"], np.float32).tobytes() for r in buf._records}
    assert len(xs) == 32


def _weights(cnn):
    return [np.asarray(cnn._fc2.weight).copy(), np.asarray(cnn._fc2.bias).copy()]


def _state(stream, upto_task: int, seed: int = 3):
    """Network + buffer after training/pushing tasks 0..upto_task-1."""
    from experiments.g4_quinto_test.small_cnn import G4SmallCNN

    cnn = G4SmallCNN(latent_dim=64, n_classes=20, seed=seed)
    buf = drv.new_buffer()
    rng = np.random.default_rng(seed + 5000)
    for k in range(upto_task):
        cnn.train_task(
            stream[k]["x_train_nhwc"], stream[k]["y_train"],
            epochs=1, batch_size=64, lr=0.01,
        )
        drv.push_task(buf, cnn, stream[k], rng)
    return cnn, buf


def _de(cnn, buf, strategy, k, seed=3):
    from experiments.g4_split_fmnist.dream_wrap import build_profile
    from experiments.g4_ter_hp_sweep.hp_grid import representative_combo

    drv.dream_episode(
        cnn, build_profile("P_max", seed=seed), seed + k,
        beta_buffer=buf, combo=representative_combo(), strategy=strategy,
    )


def test_cross_and_within_noop_before_task6(stream) -> None:
    """Checklist 4: weights after the DE before task 6 (k=5) are
    bit-identical across cross / within / none."""
    outs = {}
    for strat in ("cross", "within", "none"):
        cnn, buf = _state(stream, 5)
        _de(cnn, buf, strat, k=5)
        outs[strat] = _weights(cnn)
    for strat in ("cross", "within"):
        for a, b in zip(outs[strat], outs["none"]):
            assert np.array_equal(a, b)


def test_cross_differs_from_none_once_both_domains_present(stream) -> None:
    res = {}
    for strat in ("cross", "within", "none"):
        cnn, buf = _state(stream, 6)
        _de(cnn, buf, strat, k=6)
        res[strat] = _weights(cnn)
    assert not np.array_equal(res["cross"][0], res["none"][0])
    assert not np.array_equal(res["within"][0], res["none"][0])
    assert not np.array_equal(res["cross"][0], res["within"][0])


def test_soft_target_step_updates_only_fc2(stream) -> None:
    cnn, buf = _state(stream, 6)
    conv_before = np.asarray(cnn._conv1.weight).copy()
    fc2_before = np.asarray(cnn._fc2.weight).copy()
    rng = np.random.default_rng(0)
    x = rng.standard_normal((16, 64)).astype(np.float32)
    t = np.zeros((16, 20), np.float32)
    t[:, 0] = 0.5
    t[:, 2] = 0.5
    drv.soft_target_fc2_step(cnn, x, t, lr=0.01)
    assert np.array_equal(conv_before, np.asarray(cnn._conv1.weight))
    assert not np.array_equal(fc2_before, np.asarray(cnn._fc2.weight))


def test_run_cell_record_and_pairing(stream, tmp_path: Path) -> None:
    rec = drv.run_cell(
        arm="P_max", strategy="cross", seed=1, stream=stream,
        epochs=1, batch_size=64, lr=0.01,
    )
    for key in (
        "arm", "strategy", "seed", "cdpa", "wdpa", "retention_a",
        "acc20", "doa", "manip_acc", "manip_pass", "finite", "wall_time_s",
        "c_version", "n_classes", "epochs",
    ):
        assert key in rec
    assert 0.0 <= rec["cdpa"] <= 1.0 and 0.0 <= rec["wdpa"] <= 1.0
    assert 0.0 <= rec["doa"] <= 1.0
    assert rec["finite"] is True
    assert rec["manip_pass"] == (rec["manip_acc"] >= 0.60)
    # matched-block pairing: same seed, same task-1 manip accuracy
    other = drv.run_cell(
        arm="P_max", strategy="none", seed=1, stream=stream,
        epochs=1, batch_size=64, lr=0.01,
    )
    assert other["manip_acc"] == rec["manip_acc"]


def test_baseline_cell_has_na_strategy(stream) -> None:
    rec = drv.run_cell(
        arm="baseline", strategy="na", seed=2, stream=stream,
        epochs=1, batch_size=64, lr=0.01,
    )
    assert rec["strategy"] == "na"


def test_none_matches_independently_disabled_recombine(
    stream, monkeypatch
) -> None:
    """P_max/none equals P_max with the RECOMBINE step patched out."""
    a = drv.run_cell(arm="P_max", strategy="none", seed=4, stream=stream,
                     epochs=1, batch_size=64, lr=0.01)
    monkeypatch.setattr(drv, "recombine", lambda *a, **k: None)
    b = drv.run_cell(arm="P_max", strategy="cross", seed=4, stream=stream,
                     epochs=1, batch_size=64, lr=0.01)
    assert a["cdpa"] == b["cdpa"] and a["wdpa"] == b["wdpa"]
    assert a["acc20"] == b["acc20"]


def test_run_cell_non_finite_logits_marks_exclusion(
    stream, monkeypatch
) -> None:
    def nan_logits(cnn, x):
        return np.full((len(x), 20), np.nan, np.float32)

    monkeypatch.setattr(drv, "_logits_chunked", nan_logits)
    rec = drv.run_cell(arm="baseline", strategy="na", seed=6, stream=stream,
                       epochs=1, batch_size=64, lr=0.01)
    assert rec["finite"] is False
    assert rec["excluded"] is True
    assert rec["exclusion_reason"] == "non_finite_logits"
    assert rec["cdpa"] is None and rec["wdpa"] is None
    assert rec["doa"] is None


def test_run_cell_non_finite_manip_marks_exclusion(
    stream, monkeypatch
) -> None:
    real = drv.G4SmallCNN.predict_logits

    def bad(self, x):
        out = real(self, x).copy()
        out[0, 0] = np.nan
        return out

    monkeypatch.setattr(drv.G4SmallCNN, "predict_logits", bad)
    rec = drv.run_cell(arm="baseline", strategy="na", seed=6, stream=stream,
                       epochs=1, batch_size=64, lr=0.01)
    assert rec["finite"] is False
    assert rec["exclusion_reason"] == "non_finite_logits"


def test_clean_cell_not_excluded(stream) -> None:
    rec = drv.run_cell(arm="baseline", strategy="na", seed=6, stream=stream,
                       epochs=1, batch_size=64, lr=0.01)
    assert rec["excluded"] is False and rec["exclusion_reason"] is None


def test_duplicate_cell_different_out_aborts(
    stream, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(drv, "load_stream", lambda data_dir: stream)
    db = tmp_path / "r.sqlite"
    base = ["--seed", "7", "--arm", "P_min", "--strategy", "na",
            "--registry-db", str(db), "--epochs", "1"]
    out1 = tmp_path / "a.json"
    assert drv.main(base + ["--out", str(out1)]) == 0
    first = json.loads(out1.read_text())
    out2 = tmp_path / "b.json"
    rc = drv.main(base + ["--out", str(out2), "--lr", "0.02"])
    assert rc == drv.EXIT_RUN_ID_COLLISION
    rec = json.loads(out2.read_text())
    assert rec["status"] == "aborted"
    assert rec["abort_reason"] == "run_id_collision"
    assert rec["run_id"] == first["run_id"]
    assert "cdpa" not in rec
    assert json.loads(out1.read_text()) == first


def test_main_writes_outcome_record(stream, tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(drv, "load_stream", lambda data_dir: stream)
    out = tmp_path / "cell.json"
    rc = drv.main([
        "--seed", "5", "--arm", "P_min", "--strategy", "na",
        "--out", str(out), "--registry-db", str(tmp_path / "r.sqlite"),
        "--epochs", "1",
    ])
    assert rc == 0
    rec = json.loads(out.read_text())
    assert rec["seed"] == 5 and rec["arm"] == "P_min"
    assert rec["profile_key"] == "g4-nono/P_min/na"
    assert len(rec["run_id"]) > 8


def test_main_refuses_bad_combo(tmp_path: Path) -> None:
    with pytest.raises(SystemExit):
        drv.main(["--seed", "0", "--arm", "baseline", "--strategy", "cross",
                  "--out", str(tmp_path / "x.json")])


def test_manip_accuracy_is_two_way_restricted() -> None:
    z = np.zeros((4, 20), np.float32)
    z[:, 7] = 50.0
    z[:, 0] = [1, 1, 0, 0]
    z[:, 1] = [0, 0, 1, 1]
    y = np.array([0, 0, 1, 1])
    assert drv.manipulation_accuracy(z, y, TASK_CLASS_PAIRS[0]) == 1.0
    assert set(DOMAIN_A) >= set(TASK_CLASS_PAIRS[0])


def test_main_refuses_to_overwrite_existing_record(
    stream, tmp_path: Path, monkeypatch
) -> None:
    monkeypatch.setattr(drv, "load_stream", lambda data_dir: stream)
    out = tmp_path / "cell.json"
    out.write_text("{}")
    with pytest.raises(SystemExit):
        drv.main(["--seed", "0", "--arm", "baseline", "--strategy", "na",
                  "--out", str(out), "--epochs", "1"])
    assert out.read_text() == "{}"


def test_concurrent_claim_has_exactly_one_winner(tmp_path: Path) -> None:
    """Racing claims on one sqlite registry: a single UNIQUE-checked
    INSERT makes the claim atomic (no lookup-then-register window)."""
    import threading

    db = tmp_path / "race.sqlite"
    n = 8
    barrier = threading.Barrier(n)
    results: list[bool] = []
    lock = threading.Lock()

    def worker() -> None:
        barrier.wait()
        won = drv.claim_run_id(
            db, c_version=drv.C_VERSION, profile="g4-nono/P_min/na",
            seed=1, commit_sha="abc",
        )
        with lock:
            results.append(won)

    threads = [threading.Thread(target=worker) for _ in range(n)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert len(results) == n and results.count(True) == 1


def test_claim_then_second_claim_loses(tmp_path: Path) -> None:
    db = tmp_path / "c.sqlite"
    kw = dict(c_version=drv.C_VERSION, profile="p", seed=0, commit_sha="x")
    assert drv.claim_run_id(db, **kw) is True
    assert drv.claim_run_id(db, **kw) is False
