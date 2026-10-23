"""Coarse-label decode path for G4-nono (synthetic fixtures only)."""
from __future__ import annotations

import io
from pathlib import Path

import numpy as np
import pytest

from experiments.g4_sexto_test.cifar100_dataset import (
    decode_cifar100_bin,
    decode_cifar100_bin_with_coarse,
    decode_parquet_shard_with_coarse,
)


def _write_bin(path: Path, coarse: list[int], fine: list[int]) -> None:
    rng = np.random.default_rng(0)
    rows = [
        bytes([c, f]) + rng.integers(0, 256, 3072, dtype=np.uint8).tobytes()
        for c, f in zip(coarse, fine)
    ]
    path.write_bytes(b"".join(rows))


def test_bin_with_coarse_returns_coarse_byte(tmp_path: Path) -> None:
    f = tmp_path / "train.bin"
    _write_bin(f, [0, 19, 7, 3], [4, 99, 35, 12])
    images, fine, coarse = decode_cifar100_bin_with_coarse(f)
    assert images.shape == (4, 32, 32, 3)
    assert coarse.tolist() == [0, 19, 7, 3]
    assert fine.tolist() == [4, 99, 35, 12]


def test_legacy_decoder_unchanged(tmp_path: Path) -> None:
    f = tmp_path / "train.bin"
    _write_bin(f, [0, 19], [4, 99])
    out = decode_cifar100_bin(f)
    assert len(out) == 2
    assert out[1].tolist() == [4, 99]
    imgs, _fine, _coarse = decode_cifar100_bin_with_coarse(f)
    assert np.array_equal(out[0], imgs)


def test_bin_with_coarse_empty_and_truncated(tmp_path: Path) -> None:
    f = tmp_path / "e.bin"
    f.write_bytes(b"")
    imgs, fine, coarse = decode_cifar100_bin_with_coarse(f)
    assert imgs.shape == (0, 32, 32, 3) and coarse.shape == (0,)
    g = tmp_path / "t.bin"
    g.write_bytes(b"\x00" * 10)
    with pytest.raises(ValueError, match="truncated"):
        decode_cifar100_bin_with_coarse(g)


def _png(arr: np.ndarray) -> bytes:
    from PIL import Image

    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return buf.getvalue()


def test_parquet_with_coarse_matches_column(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    rng = np.random.default_rng(1)
    coarse = [2, 18, 0, 11]
    fine = [60, 90, 4, 55]
    imgs = [rng.integers(0, 256, (32, 32, 3), dtype=np.uint8) for _ in coarse]
    df = pd.DataFrame(
        {
            "img": [{"bytes": _png(i)} for i in imgs],
            "fine_label": fine,
            "coarse_label": coarse,
        }
    )
    p = tmp_path / "shard.parquet"
    df.to_parquet(p)
    x, y_fine, y_coarse = decode_parquet_shard_with_coarse(p)
    assert y_coarse.tolist() == coarse
    assert y_fine.tolist() == fine
    assert np.array_equal(x[0], imgs[0])


def test_parquet_without_coarse_column_raises(tmp_path: Path) -> None:
    pd = pytest.importorskip("pandas")
    pytest.importorskip("pyarrow")
    img = np.zeros((32, 32, 3), np.uint8)
    df = pd.DataFrame({"img": [{"bytes": _png(img)}], "fine_label": [1]})
    p = tmp_path / "s.parquet"
    df.to_parquet(p)
    with pytest.raises(ValueError, match="coarse_label"):
        decode_parquet_shard_with_coarse(p)


_REAL = Path("experiments/g4_sexto_test/data")


@pytest.mark.skipif(
    not (
        (_REAL / "cifar-100-binary" / "train.bin").exists()
        and (_REAL / "cifar100-train.parquet").exists()
    ),
    reason="pinned CIFAR-100 files not present locally (no download in tests)",
)
def test_pinned_bin_coarse_matches_parquet_first_1000() -> None:
    _x, _f, c_bin = decode_cifar100_bin_with_coarse(
        _REAL / "cifar-100-binary" / "train.bin"
    )
    _x2, _f2, c_pq = decode_parquet_shard_with_coarse(
        _REAL / "cifar100-train.parquet"
    )
    assert c_bin[:1000].tolist() == c_pq[:1000].tolist()
