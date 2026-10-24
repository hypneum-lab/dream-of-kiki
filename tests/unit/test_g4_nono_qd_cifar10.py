"""G4-nono domain corpus: CIFAR-10 (A, grayscale) + Quick, Draw! (B).

Pure numpy (no MLX, no download): preprocessing, label map, seeded
subsampling, corpus assembly and the pinned-file loader. Checklist
item 5 on the real files is the last test (skipped where absent).
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from experiments.g4_nono_test import qd_cifar10 as qd
from experiments.g4_nono_test.stream import DOMAIN_A, DOMAIN_B, TASK_CLASS_PAIRS


def test_gray3_uses_bt601_luma_and_equal_channels() -> None:
    x = np.array([[[[255, 0, 0], [0, 255, 0]], [[0, 0, 255], [10, 20, 30]]]], np.uint8)
    g = qd.to_gray3(x)
    assert g.dtype == np.uint8 and g.shape == x.shape
    assert (g[..., 0] == g[..., 1]).all() and (g[..., 1] == g[..., 2]).all()
    want = np.rint(0.299 * x[..., 0] + 0.587 * x[..., 1] + 0.114 * x[..., 2])
    assert np.array_equal(g[..., 0], want.astype(np.uint8))


def test_colour_shortcut_neutralised() -> None:
    # after preprocessing, no image of either domain carries colour
    rng = np.random.default_rng(0)
    cif = rng.integers(0, 256, (5, 32, 32, 3), dtype=np.uint8)
    dra = rng.integers(0, 256, (5, 784), dtype=np.uint8)
    for arr in (qd.to_gray3(cif), qd.pad_drawings(dra)):
        assert (arr[..., 0] == arr[..., 1]).all() and (arr[..., 0] == arr[..., 2]).all()


def test_pad_drawings_28_to_32_centred_with_zero_border() -> None:
    d = np.arange(784, dtype=np.uint32).astype(np.uint8)[None]
    out = qd.pad_drawings(d)
    assert out.shape == (1, 32, 32, 3) and out.dtype == np.uint8
    assert np.array_equal(out[0, 2:30, 2:30, 0], d.reshape(28, 28))
    border = np.ones((32, 32), bool)
    border[2:30, 2:30] = False
    assert (out[0][border] == 0).all()


def test_label_map_puts_cifar_in_domain_a_and_drawings_in_domain_b() -> None:
    assert len(qd.CIFAR10_CLASSES) == 10 and len(qd.QD_CLASSES) == 10
    assert [qd.LABEL_OF[("cifar10", k)] for k in range(10)] == list(DOMAIN_A)
    assert [qd.LABEL_OF[("quickdraw", k)] for k in range(10)] == list(DOMAIN_B)
    # every registered task pair stays within one corpus
    for a, b in TASK_CLASS_PAIRS:
        assert (a in DOMAIN_A) == (b in DOMAIN_A)


def test_no_semantic_overlap_between_domains() -> None:
    cifar = set(qd.CIFAR10_CLASSES)
    assert not cifar & set(qd.QD_CLASSES)
    assert not {"car", "truck", "cat", "dog", "bird", "frog", "horse", "airplane"} & set(qd.QD_CLASSES)


def test_subsample_is_seeded_disjoint_and_sized() -> None:
    a = qd.subsample(10_000, n_train=2500, n_test=500, key=("quickdraw", 3))
    b = qd.subsample(10_000, n_train=2500, n_test=500, key=("quickdraw", 3))
    c = qd.subsample(10_000, n_train=2500, n_test=500, key=("quickdraw", 4))
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])
    assert not np.array_equal(a[0], c[0])
    assert len(a[0]) == 2500 and len(a[1]) == 500
    assert not set(a[0].tolist()) & set(a[1].tolist())
    with pytest.raises(ValueError):
        qd.subsample(100, n_train=2500, n_test=500, key=("quickdraw", 0))


def _fake_sources(
    n_cifar_tr: int = 30, n_cifar_te: int = 12, n_qd: int = 40
) -> tuple[Any, Any, Any]:
    rng = np.random.default_rng(1)
    ytr = np.repeat(np.arange(10, dtype=np.uint8), n_cifar_tr)
    yte = np.repeat(np.arange(10, dtype=np.uint8), n_cifar_te)
    xtr = rng.integers(0, 256, (len(ytr), 32, 32, 3), dtype=np.uint8)
    xte = rng.integers(0, 256, (len(yte), 32, 32, 3), dtype=np.uint8)
    draws = {c: rng.integers(0, 256, (n_qd, 784), dtype=np.uint8) for c in qd.QD_CLASSES}
    return (xtr, ytr), (xte, yte), draws


def test_build_corpus_counts_labels_and_indices() -> None:
    cif_tr, cif_te, draws = _fake_sources()
    out = qd.build_corpus(cif_tr, cif_te, draws, n_train=20, n_test=8)
    assert out["x_train"].shape == (20 * 20, 32, 32, 3)
    assert out["x_test"].shape == (20 * 8, 32, 32, 3)
    assert sorted(set(out["y_train"].tolist())) == list(range(20))
    assert np.bincount(out["y_train"], minlength=20).tolist() == [20] * 20
    assert np.bincount(out["y_test"], minlength=20).tolist() == [8] * 20
    idx = out["indices"]
    assert set(idx) == {f"cifar10/{c}" for c in qd.CIFAR10_CLASSES} | {
        f"quickdraw/{c}" for c in qd.QD_CLASSES
    }
    # CIFAR test images come from the CIFAR test split, train from train
    k = qd.CIFAR10_CLASSES.index("cat")
    rows = np.flatnonzero(cif_te[1] == k)
    assert set(idx["cifar10/cat"]["test"]) <= set(rows.tolist())
    # the stored images are the preprocessed source rows
    first = idx["quickdraw/apple"]["train"][0]
    lab = qd.LABEL_OF[("quickdraw", qd.QD_CLASSES.index("apple"))]
    pos = int(np.flatnonzero(out["y_train"] == lab)[0])
    assert np.array_equal(out["x_train"][pos], qd.pad_drawings(draws["apple"][first][None])[0])


def test_build_corpus_is_deterministic() -> None:
    cif_tr, cif_te, draws = _fake_sources()
    a = qd.build_corpus(cif_tr, cif_te, draws, n_train=20, n_test=8)
    b = qd.build_corpus(cif_tr, cif_te, draws, n_train=20, n_test=8)
    for k in ("x_train", "y_train", "x_test", "y_test"):
        assert np.array_equal(a[k], b[k])
    assert json.dumps(a["indices"], sort_keys=True) == json.dumps(b["indices"], sort_keys=True)


def test_loader_verifies_pinned_hash(tmp_path: Path) -> None:
    cif_tr, cif_te, draws = _fake_sources()
    out = qd.build_corpus(cif_tr, cif_te, draws, n_train=20, n_test=8)
    path = qd.save_corpus(out, tmp_path)
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    x_tr, y_tr, x_te, y_te = qd.load_qd_cifar10(tmp_path, expected_sha256=digest)
    assert x_tr.shape[0] == 400 and y_te.shape[0] == 160
    with pytest.raises(ValueError, match="SHA-256"):
        qd.load_qd_cifar10(tmp_path, expected_sha256="0" * 64)


def test_raw_pins_are_recorded() -> None:
    for c in qd.QD_CLASSES:
        pin = qd.QD_PINS[c]
        assert set(pin) == {"sha256", "gcs_md5_b64", "gcs_crc32c_b64", "bytes"}
        assert len(pin["sha256"]) == 64
    assert qd.CIFAR10_TRAIN_SHA256.startswith("8428b53a")
    assert qd.CIFAR10_TEST_SHA256.startswith("841389e6")


_DATA = Path(__file__).resolve().parents[2] / "experiments" / "g4_nono_test" / "data"


@pytest.mark.skipif(
    not (_DATA / qd.CORPUS_FILE).exists(),
    reason="pinned QD-CIFAR10 corpus not built on this host",
)
def test_checklist_item5_real_corpus() -> None:
    """Checklist item 5 (new corpus): pinned file, counts, labels, no colour."""
    x_tr, y_tr, x_te, y_te = qd.load_qd_cifar10(_DATA)
    assert x_tr.shape == (50_000, 32, 32, 3) and x_te.shape == (10_000, 32, 32, 3)
    assert np.bincount(y_tr, minlength=20).tolist() == [2500] * 20
    assert np.bincount(y_te, minlength=20).tolist() == [500] * 20
    for x in (x_tr, x_te):
        assert (x[..., 0] == x[..., 1]).all() and (x[..., 0] == x[..., 2]).all()
    idx = json.loads(qd.INDEX_FILE.read_text())
    assert hashlib.sha256(qd.INDEX_FILE.read_bytes()).hexdigest() == qd.INDEX_SHA256
    assert len(idx) == 20


def test_driver_defaults_to_the_new_corpus(monkeypatch: pytest.MonkeyPatch) -> None:
    try:  # the driver imports MLX, absent on Linux CI (see conftest.py)
        import mlx.core  # noqa: F401
    except ImportError:
        pytest.skip("MLX unavailable on this platform")
    from experiments.g4_nono_test import run_cross_domain as drv

    cif_tr, cif_te, draws = _fake_sources()
    out = qd.build_corpus(cif_tr, cif_te, draws, n_train=20, n_test=8)
    seen: dict[str, Path] = {}

    def fake(data_dir: Path) -> tuple[Any, ...]:
        seen["dir"] = data_dir
        return out["x_train"], out["y_train"], out["x_test"], out["y_test"]

    monkeypatch.setattr(qd, "load_qd_cifar10", fake)
    assert drv.DEFAULT_CORPUS == "qd-cifar10"
    stream = drv.load_stream(Path("/nowhere"))
    assert len(stream) == 10 and seen["dir"] == Path("/nowhere")
    # task 1 is the first CIFAR-10 pair, task 6 the first Quick, Draw! pair
    assert set(stream[0]["y_train"].tolist()) == {DOMAIN_A[0], DOMAIN_A[1]}
    assert set(stream[5]["y_train"].tolist()) == {DOMAIN_B[0], DOMAIN_B[1]}
