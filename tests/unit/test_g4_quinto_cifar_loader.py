"""Unit tests for G4-quinto CIFAR-10 loader (synthetic tmp_path fixture)."""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from experiments.g4_quinto_test.cifar10_dataset import (
    CIFAR10_RECORD_SIZE,
    decode_cifar10_bin,
    load_split_cifar10_5tasks,
)


def _write_batch(
    path: Path, labels: list[int], rng: np.random.Generator
) -> None:
    rows = [
        bytes([lbl])
        + rng.integers(0, 256, size=3072, dtype=np.uint8).tobytes()
        for lbl in labels
    ]
    path.write_bytes(b"".join(rows))


def test_decode_cifar10_bin_shape(tmp_path: Path) -> None:
    f = tmp_path / "batch.bin"
    _write_batch(f, [0, 1, 2, 3], np.random.default_rng(0))
    images, labels = decode_cifar10_bin(f)
    assert images.shape == (4, 32, 32, 3) and images.dtype == np.uint8
    assert labels.tolist() == [0, 1, 2, 3]


def test_decode_cifar10_bin_truncated_raises(tmp_path: Path) -> None:
    f = tmp_path / "bad.bin"
    f.write_bytes(b"\x00" * (CIFAR10_RECORD_SIZE - 1))
    with pytest.raises(ValueError, match="truncated"):
        decode_cifar10_bin(f)


def test_load_split_cifar10_5tasks_split(tmp_path: Path) -> None:
    rng = np.random.default_rng(0)
    bin_dir = tmp_path / "cifar-10-batches-bin"
    bin_dir.mkdir()
    # 2 records per class × 10 classes = 20 records in batch_1.
    _write_batch(
        bin_dir / "data_batch_1.bin",
        [c for c in range(10) for _ in range(2)],
        rng,
    )
    for k in range(2, 6):
        _write_batch(bin_dir / f"data_batch_{k}.bin", [], rng)
    _write_batch(bin_dir / "test_batch.bin", list(range(10)), rng)
    tasks = load_split_cifar10_5tasks(bin_dir)
    assert len(tasks) == 5
    for task in tasks:
        assert task["x_train"].shape[1] == 3072
        assert task["x_train_nhwc"].shape[1:] == (32, 32, 3)
        assert task["x_train"].dtype == np.float32
        assert task["x_train_nhwc"].dtype == np.float32
        assert set(task["y_train"].tolist()) <= {0, 1}
        assert set(task["y_test"].tolist()) <= {0, 1}
