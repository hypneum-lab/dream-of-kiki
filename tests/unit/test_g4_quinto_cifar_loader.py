"""Unit tests for G4-quinto CIFAR-10 loader (synthetic tmp_path fixture).

Covers both the canonical binary path and the HF parquet
fallback path (pre-reg §9.1 deviation) — both produce the
same SplitCIFAR10Task contract.
"""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from experiments.g4_quinto_test.cifar10_dataset import (
    CIFAR10_RECORD_SIZE,
    decode_cifar10_bin,
    load_split_cifar10_5tasks,
    load_split_cifar10_5tasks_hf,
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


def _write_hf_parquet(
    path: Path, labels: list[int], rng: np.random.Generator
) -> None:
    """Write a synthetic HF-compatible parquet shard.

    Schema mirrors uoft-cs/cifar10 plain_text :
        img : struct<bytes: binary, path: string>
        label : int64
    Each ``img.bytes`` is a PNG-encoded 32x32 RGB image.
    """
    import pyarrow as pa  # type: ignore[import-untyped]
    import pyarrow.parquet as pq  # type: ignore[import-untyped]
    from PIL import Image

    img_bytes_list: list[bytes] = []
    for _ in labels:
        arr = rng.integers(0, 256, size=(32, 32, 3), dtype=np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).save(buf, format="PNG")
        img_bytes_list.append(buf.getvalue())
    img_struct = pa.StructArray.from_arrays(
        [
            pa.array(img_bytes_list, type=pa.binary()),
            pa.array([""] * len(labels), type=pa.string()),
        ],
        fields=[
            pa.field("bytes", pa.binary()),
            pa.field("path", pa.string()),
        ],
    )
    table = pa.table(
        {
            "img": img_struct,
            "label": pa.array(labels, type=pa.int64()),
        }
    )
    pq.write_table(table, path)


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


def test_load_split_cifar10_5tasks_hf_split(tmp_path: Path) -> None:
    """HF parquet fallback returns the same SplitCIFAR10Task contract."""
    rng = np.random.default_rng(0)
    train = tmp_path / "train.parquet"
    test = tmp_path / "test.parquet"
    # 2 records per class x 10 classes for both shards.
    labels = [c for c in range(10) for _ in range(2)]
    _write_hf_parquet(train, labels, rng)
    _write_hf_parquet(test, labels, rng)
    tasks = load_split_cifar10_5tasks_hf(train, test)
    assert len(tasks) == 5
    for task in tasks:
        assert task["x_train"].shape[1] == 3072
        assert task["x_train_nhwc"].shape[1:] == (32, 32, 3)
        assert task["x_train"].dtype == np.float32
        assert task["x_train_nhwc"].dtype == np.float32
        assert set(task["y_train"].tolist()) <= {0, 1}
        assert set(task["y_test"].tolist()) <= {0, 1}
