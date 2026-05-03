"""2-seed integration smoke test for the G5 driver.

Uses a synthetic 16x16 IDX fixture (mirrors the G4-bis integration
test) to keep wall time under 60 s per arm.
"""
from __future__ import annotations

import gzip
import json
import struct
from pathlib import Path

import numpy as np
import pytest

from experiments.g5_cross_substrate.run_g5 import run_pilot


def _write_idx_image(path: Path, images: np.ndarray) -> None:
    n, h, w = images.shape
    with gzip.open(path, "wb") as fh:
        fh.write(struct.pack(">IIII", 2051, n, h, w))
        fh.write(images.astype(np.uint8).tobytes())


def _write_idx_label(path: Path, labels: np.ndarray) -> None:
    with gzip.open(path, "wb") as fh:
        fh.write(struct.pack(">II", 2049, labels.size))
        fh.write(labels.astype(np.uint8).tobytes())


def _make_synthetic_fmnist(data_dir: Path, n_per_class: int = 8) -> None:
    """Produce a 10-class 16x16 FMNIST mock under `data_dir`."""
    rng = np.random.default_rng(0)
    n_classes = 10
    h, w = 16, 16
    train_imgs = rng.integers(
        0, 256, size=(n_classes * n_per_class, h, w)
    ).astype(np.uint8)
    train_lbls = np.repeat(np.arange(n_classes), n_per_class).astype(
        np.uint8
    )
    test_imgs = rng.integers(
        0, 256, size=(n_classes * 4, h, w)
    ).astype(np.uint8)
    test_lbls = np.repeat(np.arange(n_classes), 4).astype(np.uint8)
    data_dir.mkdir(parents=True, exist_ok=True)
    _write_idx_image(
        data_dir / "train-images-idx3-ubyte.gz", train_imgs
    )
    _write_idx_label(
        data_dir / "train-labels-idx1-ubyte.gz", train_lbls
    )
    _write_idx_image(
        data_dir / "t10k-images-idx3-ubyte.gz", test_imgs
    )
    _write_idx_label(
        data_dir / "t10k-labels-idx1-ubyte.gz", test_lbls
    )


@pytest.mark.slow
def test_run_g5_pilot_smoke_2seeds(tmp_path: Path) -> None:
    """Driver runs end-to-end on a 2-seed synthetic FMNIST fixture."""
    data_dir = tmp_path / "data"
    _make_synthetic_fmnist(data_dir)
    out_json = tmp_path / "g5.json"
    out_md = tmp_path / "g5.md"
    db = tmp_path / ".registry.sqlite"
    payload = run_pilot(
        data_dir=data_dir,
        seeds=(0, 1),
        out_json=out_json,
        out_md=out_md,
        registry_db=db,
        epochs=1,
        batch_size=4,
        hidden_dim=8,
        lr=0.1,
        n_steps=5,
    )
    # Shape contract — same keys as G4-bis + `substrate`
    expected_keys = {
        "arms",
        "c_version",
        "cells",
        "commit_sha",
        "data_dir",
        "date",
        "n_seeds",
        "substrate",
        "verdict",
        "wall_time_s",
    }
    assert expected_keys <= set(payload.keys())
    assert payload["substrate"] == "esnn_thalamocortical"
    assert len(payload["cells"]) == 4 * 2  # 4 arms x 2 seeds
    # Milestone dump persisted
    assert out_json.exists()
    assert out_md.exists()
    written = json.loads(out_json.read_text())
    assert written["substrate"] == "esnn_thalamocortical"


@pytest.mark.slow
def test_run_g5_pilot_register_run_ids_are_stable(tmp_path: Path) -> None:
    """Two consecutive runs with same seeds -> same `run_id` per cell."""
    data_dir = tmp_path / "data"
    _make_synthetic_fmnist(data_dir)
    db = tmp_path / ".registry.sqlite"
    payload_a = run_pilot(
        data_dir=data_dir,
        seeds=(0,),
        out_json=tmp_path / "a.json",
        out_md=tmp_path / "a.md",
        registry_db=db,
        epochs=1,
        batch_size=4,
        hidden_dim=8,
        lr=0.1,
        n_steps=5,
    )
    payload_b = run_pilot(
        data_dir=data_dir,
        seeds=(0,),
        out_json=tmp_path / "b.json",
        out_md=tmp_path / "b.md",
        registry_db=db,
        epochs=1,
        batch_size=4,
        hidden_dim=8,
        lr=0.1,
        n_steps=5,
    )
    ids_a = sorted(c["run_id"] for c in payload_a["cells"])
    ids_b = sorted(c["run_id"] for c in payload_b["cells"])
    assert ids_a == ids_b
