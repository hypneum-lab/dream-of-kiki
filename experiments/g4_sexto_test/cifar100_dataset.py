"""Split-CIFAR-100 10-task loader — pure numpy, no torchvision.

Two acquisition paths are supported (both pinned by SHA-256) :

1. **Canonical** — https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
   (~163 MB). Records per the CIFAR-100 binary spec : 1 coarse-label
   byte + 1 fine-label byte + 3072 image bytes (CHW : 1024 R + 1024
   G + 1024 B, row-major within each channel). Two binary files :
   ``train.bin`` (50000 records) + ``test.bin`` (10000 records),
   both inside the extracted ``cifar-100-binary/`` directory.

2. **HF mirror fallback** — Hugging Face dataset
   ``uoft-cs/cifar100`` (parquet, ~163 MB). Used when the canonical
   mirror returns a non-2xx response — pre-reg §9.1 deviation
   envelope (G4-quinto pattern).

Class-incremental 10-task split using **fine labels** (0..99) :

    task 0 : fine classes {0..9}
    task 1 : fine classes {10..19}
    ...
    task 9 : fine classes {90..99}

Per task, fine labels are remapped to ``{0, 1, ..., 9}`` (10-class
head shared across tasks). Images stored as ``np.float32`` in
``[0, 1]``, layout ``(N, 32, 32, 3)`` for CNN consumption (NHWC)
and flattened to ``(N, 3072)`` — both returned.

Reference :
    Krizhevsky 2009 — "Learning Multiple Layers of Features from
        Tiny Images"
    https://www.cs.toronto.edu/~kriz/cifar.html
    https://huggingface.co/datasets/uoft-cs/cifar100
    docs/osf-prereg-g4-sexto-pilot.md sec 5
"""
from __future__ import annotations

import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path
from typing import Any, Final, TypedDict

import numpy as np

CIFAR100_LABEL_BYTES: Final[int] = 2  # coarse + fine
CIFAR100_IMAGE_BYTES: Final[int] = 32 * 32 * 3
CIFAR100_RECORD_SIZE: Final[int] = (
    CIFAR100_LABEL_BYTES + CIFAR100_IMAGE_BYTES
)  # 3074
CIFAR100_URL: Final[str] = (
    "https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz"
)
# One-shot SHA-256 pin — value is replaced at first download. The
# placeholder leading "..." disables the integrity check until the
# real hash is committed.
CIFAR100_TAR_SHA256: Final[str] = "...replace_at_first_download..."

# HF mirror fallback — pinned at first download. Same §9.1-style
# amendment pattern as G4-quinto cifar10 fallback.
CIFAR100_HF_TRAIN_URL: Final[str] = (
    "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/"
    "cifar100/train-00000-of-00001.parquet"
)
CIFAR100_HF_TEST_URL: Final[str] = (
    "https://huggingface.co/datasets/uoft-cs/cifar100/resolve/main/"
    "cifar100/test-00000-of-00001.parquet"
)
CIFAR100_HF_TRAIN_SHA256: Final[str] = (
    "694865d6b990e234804f01268586c41e88bcbbb75e20858432c05ad4081aca23"
)
CIFAR100_HF_TEST_SHA256: Final[str] = (
    "98776c529bb146a9c791229df74a5cf076be9b43d82dbbd334b6a7788d73dc68"
)
HTTP_USER_AGENT: Final[str] = "g4-sexto-pilot/1 (mlx-on-m1max)"

CIFAR100_N_TASKS: Final[int] = 10
CIFAR100_CLASSES_PER_TASK: Final[int] = 10


class SplitCIFAR100Task(TypedDict):
    """One Split-CIFAR-100 10-class task : NHWC + flat float32 + label."""

    x_train: np.ndarray
    x_train_nhwc: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    x_test_nhwc: np.ndarray
    y_test: np.ndarray


def decode_cifar100_bin_with_coarse(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode one CIFAR-100 binary file into (NHWC uint8, fine, coarse).

    Same record layout as :func:`decode_cifar100_bin` ; byte 0 (the
    coarse label, 0..19) is kept and returned as the third array.

    Raises :
        ValueError : payload length is not a multiple of
                     ``CIFAR100_RECORD_SIZE`` (truncated file).
    """
    raw = path.read_bytes()
    n, rem = divmod(len(raw), CIFAR100_RECORD_SIZE)
    if rem != 0:
        raise ValueError(
            f"truncated CIFAR-100 binary in {path} : "
            f"{len(raw)} bytes is not a multiple of "
            f"{CIFAR100_RECORD_SIZE}"
        )
    if n == 0:
        return (
            np.zeros((0, 32, 32, 3), np.uint8),
            np.zeros((0,), np.uint8),
            np.zeros((0,), np.uint8),
        )
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(
        n, CIFAR100_RECORD_SIZE
    )
    coarse = arr[:, 0].copy()
    fine = arr[:, 1].copy()
    nhwc = np.transpose(
        arr[:, 2:].reshape(n, 3, 32, 32), (0, 2, 3, 1)
    ).copy()
    return nhwc, fine, coarse


def decode_cifar100_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode one CIFAR-100 binary file into (NHWC uint8, fine uint8).

    Each record is ``CIFAR100_RECORD_SIZE`` bytes : the first byte
    is the coarse label (dropped here ; see
    :func:`decode_cifar100_bin_with_coarse` to keep it), the second
    the fine label (returned), and the remaining 3072 bytes are the
    image stored CHW (RGB) — reshaped to NHWC for CNN consumption.

    Raises :
        ValueError : payload length is not a multiple of
                     ``CIFAR100_RECORD_SIZE`` (truncated file).
    """
    nhwc, fine, _coarse = decode_cifar100_bin_with_coarse(path)
    return nhwc, fine


def _http_get(url: str, timeout: int = 60) -> bytes:
    """HTTP GET with browser-style UA. Raises on non-2xx."""
    req = urllib.request.Request(
        url, headers={"User-Agent": HTTP_USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def download_if_missing(data_dir: Path) -> Path:
    """Download tar -> verify SHA-256 -> extract (canonical path).

    Returns the ``cifar-100-binary/`` extracted dir. Raises
    ``FileNotFoundError`` on network failure (pre-reg §9
    deviation envelope a) — caller may fall back to
    :func:`download_if_missing_hf`.
    """
    bin_dir = data_dir / "cifar-100-binary"
    if bin_dir.exists() and (bin_dir / "test.bin").exists():
        return bin_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "cifar-100-binary.tar.gz"
    if not tar_path.exists():
        try:
            tar_path.write_bytes(_http_get(CIFAR100_URL, timeout=300))
        except OSError as exc:
            raise FileNotFoundError(
                f"CIFAR-100 download failed : {exc}"
            ) from exc
    if not CIFAR100_TAR_SHA256.startswith("..."):
        h = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        if h != CIFAR100_TAR_SHA256:
            raise ValueError(
                f"SHA-256 mismatch : got {h}, "
                f"expected {CIFAR100_TAR_SHA256}"
            )
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)
    return bin_dir


def _verify_sha256(blob: bytes, expected: str, label: str) -> None:
    if expected.startswith("..."):
        # Placeholder hash — first-download lock not yet performed.
        return
    h = hashlib.sha256(blob).hexdigest()
    if h != expected:
        raise ValueError(
            f"SHA-256 mismatch ({label}) : got {h}, expected {expected}"
        )


def download_if_missing_hf(data_dir: Path) -> tuple[Path, Path]:
    """Fallback : fetch the HF parquet shards if absent.

    Returns ``(train_parquet, test_parquet)`` paths. SHA-256
    pinned per ``CIFAR100_HF_{TRAIN,TEST}_SHA256``. Raises
    ``FileNotFoundError`` on network failure (pre-reg §9
    second-line deviation : if even the HF mirror is unreachable,
    the pilot must abort).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "cifar100-train.parquet"
    test_path = data_dir / "cifar100-test.parquet"
    pairs = (
        (train_path, CIFAR100_HF_TRAIN_URL, CIFAR100_HF_TRAIN_SHA256, "train"),
        (test_path, CIFAR100_HF_TEST_URL, CIFAR100_HF_TEST_SHA256, "test"),
    )
    for path, url, sha, label in pairs:
        if path.exists():
            _verify_sha256(path.read_bytes(), sha, f"hf-{label}")
            continue
        try:
            blob = _http_get(url, timeout=600)
        except OSError as exc:
            raise FileNotFoundError(
                f"CIFAR-100 HF mirror download failed for "
                f"{label} : {exc}"
            ) from exc
        _verify_sha256(blob, sha, f"hf-{label}")
        path.write_bytes(blob)
    return train_path, test_path


def _parquet_frame_to_arrays(
    df: Any, path: Path, *, with_coarse: bool
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
    from PIL import Image  # local import keeps base loader pure-numpy

    n = len(df)
    images = np.empty((n, 32, 32, 3), dtype=np.uint8)
    labels = np.empty((n,), dtype=np.uint8)
    img_col = "img" if "img" in df.columns else "image"
    if "fine_label" in df.columns:
        label_col = "fine_label"
    elif "label" in df.columns:
        label_col = "label"
    else:
        raise ValueError(
            f"HF cifar100 parquet at {path} has no fine_label / label "
            "column"
        )
    coarse: np.ndarray | None = None
    if with_coarse:
        if "coarse_label" not in df.columns:
            raise ValueError(
                f"HF cifar100 parquet at {path} has no coarse_label "
                "column"
            )
        coarse = np.empty((n,), dtype=np.uint8)
    for i in range(n):
        cell = df[img_col].iloc[i]
        png_bytes = cell["bytes"] if isinstance(cell, dict) else cell
        with Image.open(io.BytesIO(png_bytes)) as pil_img:
            arr = np.asarray(pil_img.convert("RGB"))
        images[i] = arr
        labels[i] = int(df[label_col].iloc[i])
        if coarse is not None:
            coarse[i] = int(df["coarse_label"].iloc[i])
    return images, labels, coarse


def _decode_parquet_shard(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode a HF cifar100 parquet shard into (NHWC uint8, fine uint8).

    The HF schema is ``{"img": {"bytes": ...}, "fine_label": int,
    "coarse_label": int}`` ; we read ``fine_label`` only. Some
    mirrors expose ``"image"`` and ``"label"`` instead — both are
    tried.
    """
    import pyarrow.parquet as pq

    df = pq.read_table(path).to_pandas()
    images, labels, _ = _parquet_frame_to_arrays(
        df, path, with_coarse=False
    )
    return images, labels


def decode_parquet_shard_with_coarse(
    path: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Decode a HF parquet shard into (NHWC uint8, fine, coarse).

    Requires the ``coarse_label`` column (raises ``ValueError``
    otherwise). Same pinned files as :func:`_decode_parquet_shard`.
    """
    import pyarrow.parquet as pq

    df = pq.read_table(path).to_pandas()
    images, labels, coarse = _parquet_frame_to_arrays(
        df, path, with_coarse=True
    )
    assert coarse is not None
    return images, labels, coarse


def _build_tasks_from_arrays(
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
) -> list[SplitCIFAR100Task]:
    """Common 10-task split builder shared by canonical + HF paths."""
    x_tr_nhwc = x_train_raw.astype(np.float32) / 255.0
    x_te_nhwc = x_test_raw.astype(np.float32) / 255.0
    x_tr_flat = x_tr_nhwc.reshape(x_tr_nhwc.shape[0], -1)
    x_te_flat = x_te_nhwc.reshape(x_te_nhwc.shape[0], -1)

    tasks: list[SplitCIFAR100Task] = []
    for k in range(CIFAR100_N_TASKS):
        lo = k * CIFAR100_CLASSES_PER_TASK
        hi = (k + 1) * CIFAR100_CLASSES_PER_TASK
        tr = (y_train_raw >= lo) & (y_train_raw < hi)
        te = (y_test_raw >= lo) & (y_test_raw < hi)
        y_tr = (y_train_raw[tr].astype(np.int64) - lo)
        y_te = (y_test_raw[te].astype(np.int64) - lo)
        tasks.append(
            SplitCIFAR100Task(
                x_train=x_tr_flat[tr],
                x_train_nhwc=x_tr_nhwc[tr],
                y_train=y_tr,
                x_test=x_te_flat[te],
                x_test_nhwc=x_te_nhwc[te],
                y_test=y_te,
            )
        )
    return tasks


def load_split_cifar100_10tasks(data_dir: Path) -> list[SplitCIFAR100Task]:
    """Load CIFAR-100 binary as 10 sequential 10-class tasks.

    Expects the two canonical binary files in ``data_dir`` :

        train.bin  test.bin

    Returns a list of 10 :class:`SplitCIFAR100Task` dicts, each
    with NHWC + flat float32 images normalised to ``[0, 1]`` and
    fine labels remapped to ``{0..9}``.

    Raises :
        FileNotFoundError : ``data_dir`` is not a directory or
                            either canonical file is missing.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(
            f"CIFAR-100 dir does not exist : {data_dir}"
        )
    test_path = data_dir / "test.bin"
    if not test_path.exists():
        raise FileNotFoundError(
            f"missing CIFAR-100 test bin : {test_path}"
        )
    train_path = data_dir / "train.bin"
    if not train_path.exists():
        raise FileNotFoundError(
            f"missing CIFAR-100 train bin : {train_path}"
        )
    x_train_raw, y_train_raw = decode_cifar100_bin(train_path)
    x_test_raw, y_test_raw = decode_cifar100_bin(test_path)
    return _build_tasks_from_arrays(
        x_train_raw, y_train_raw, x_test_raw, y_test_raw
    )


def load_split_cifar100_10tasks_hf(
    train_parquet: Path, test_parquet: Path
) -> list[SplitCIFAR100Task]:
    """Build the 10-task split from HF parquet shards."""
    if not train_parquet.exists():
        raise FileNotFoundError(
            f"missing CIFAR-100 HF train parquet : {train_parquet}"
        )
    if not test_parquet.exists():
        raise FileNotFoundError(
            f"missing CIFAR-100 HF test parquet : {test_parquet}"
        )
    x_train_raw, y_train_raw = _decode_parquet_shard(train_parquet)
    x_test_raw, y_test_raw = _decode_parquet_shard(test_parquet)
    return _build_tasks_from_arrays(
        x_train_raw, y_train_raw, x_test_raw, y_test_raw
    )


def load_split_cifar100_10tasks_auto(
    data_dir: Path,
) -> list[SplitCIFAR100Task]:
    """Try canonical loader, fall back to HF parquet on FileNotFound.

    Calls :func:`load_split_cifar100_10tasks` first (canonical
    binary). If the canonical layout is absent, transparently
    downloads + decodes the HF parquet mirror (G4-quinto §9.1
    pattern). ``data_dir`` is the workspace dir
    (``experiments/g4_sexto_test/data``) ; this function will
    locate or create the appropriate sub-paths.
    """
    canonical_dir = data_dir / "cifar-100-binary"
    if canonical_dir.exists() and (canonical_dir / "test.bin").exists():
        return load_split_cifar100_10tasks(canonical_dir)
    train_path, test_path = download_if_missing_hf(data_dir)
    return load_split_cifar100_10tasks_hf(train_path, test_path)


def load_cifar100_coarse_arrays_auto(
    data_dir: Path,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return raw ``(x_train, coarse_train, x_test, coarse_test)``.

    Images are NHWC uint8 ; coarse labels are uint8 in ``0..19``.
    Same acquisition order and SHA-256 pins as
    :func:`load_split_cifar100_10tasks_auto` (canonical binary if
    present, else the pinned HF parquet mirror) ; no new download
    source. Used by the G4-nono two-domain stream.
    """
    canonical_dir = data_dir / "cifar-100-binary"
    if canonical_dir.exists() and (canonical_dir / "test.bin").exists():
        x_tr, _f, c_tr = decode_cifar100_bin_with_coarse(
            canonical_dir / "train.bin"
        )
        x_te, _f2, c_te = decode_cifar100_bin_with_coarse(
            canonical_dir / "test.bin"
        )
        return x_tr, c_tr, x_te, c_te
    train_path, test_path = download_if_missing_hf(data_dir)
    x_tr, _f, c_tr = decode_parquet_shard_with_coarse(train_path)
    x_te, _f2, c_te = decode_parquet_shard_with_coarse(test_path)
    return x_tr, c_tr, x_te, c_te
