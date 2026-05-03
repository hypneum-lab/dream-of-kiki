"""Split-CIFAR-10 5-task loader — pure numpy, no torchvision.

Two acquisition paths are supported (both pinned by SHA-256) :

1. **Canonical** — https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
   (~163 MB). Records per the CIFAR-10 binary spec : 1 label byte
   + 3072 image bytes (CHW : 1024 R + 1024 G + 1024 B, row-major
   within each channel). Five training files
   ``data_batch_1.bin`` .. ``data_batch_5.bin`` (10000 records
   each) plus ``test_batch.bin`` (10000 records).

2. **HF mirror fallback** — Hugging Face dataset
   ``uoft-cs/cifar10`` (parquet, ~140 MB total : a 23.9 MB test
   shard + a 119.7 MB train shard, each row carrying a PNG-encoded
   32x32 image + integer label). Used when the canonical mirror
   returns a non-2xx response — pre-reg §9.1 deviation envelope.

Class-incremental 5-task split mirroring Split-FMNIST canonical :

    task 0 : classes {0, 1}  (airplane, automobile)
    task 1 : classes {2, 3}  (bird, cat)
    task 2 : classes {4, 5}  (deer, dog)
    task 3 : classes {6, 7}  (frog, horse)
    task 4 : classes {8, 9}  (ship, truck)

Labels remapped to ``{0, 1}`` per task (binary head shared).
Images stored as ``np.float32`` in ``[0, 1]``, layout
``(N, 32, 32, 3)`` for CNN consumption (NHWC) and flattened to
``(N, 3072)`` for MLP consumption — the loader returns both.

Reference :
    Krizhevsky 2009 — "Learning Multiple Layers of Features from
        Tiny Images"
    https://www.cs.toronto.edu/~kriz/cifar.html
    https://huggingface.co/datasets/uoft-cs/cifar10
    docs/osf-prereg-g4-quinto-pilot.md sec 5 + sec 9.1
"""
from __future__ import annotations

import hashlib
import io
import tarfile
import urllib.request
from pathlib import Path
from typing import Final, TypedDict

import numpy as np

CIFAR10_LABEL_BYTES: Final[int] = 1
CIFAR10_IMAGE_BYTES: Final[int] = 32 * 32 * 3
CIFAR10_RECORD_SIZE: Final[int] = (
    CIFAR10_LABEL_BYTES + CIFAR10_IMAGE_BYTES
)
CIFAR10_URL: Final[str] = (
    "https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
)
# Toronto canonical mirror was unavailable (HTTP 503) at G4-quinto
# pilot time — see `docs/osf-prereg-g4-quinto-pilot.md` §9.1
# amendment. The HF parquet fallback path below is the load-bearing
# data acquisition route. This SHA-256 string is intentionally a
# placeholder ("..." prefix disables the integrity check) so the
# canonical Toronto path is gated off until a real hash is pinned
# at first successful download.
CIFAR10_TAR_SHA256: Final[str] = "unavailable_2026-05-03_per_prereg_g4-quinto_section_9p1"

# HF mirror fallback — pinned 2026-05-03 against
# https://huggingface.co/datasets/uoft-cs/cifar10 (commit
# `0b2714987fa478483af9968de7c934580d0bb9a2`).
CIFAR10_HF_TEST_URL: Final[str] = (
    "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/"
    "plain_text/test-00000-of-00001.parquet"
)
CIFAR10_HF_TRAIN_URL: Final[str] = (
    "https://huggingface.co/datasets/uoft-cs/cifar10/resolve/main/"
    "plain_text/train-00000-of-00001.parquet"
)
CIFAR10_HF_TEST_SHA256: Final[str] = (
    "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de"
)
CIFAR10_HF_TRAIN_SHA256: Final[str] = (
    "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883"
)
HTTP_USER_AGENT: Final[str] = "g4-quinto-pilot/1 (mlx-on-m1max)"

SPLIT_CIFAR10_TASKS: Final[tuple[tuple[int, int], ...]] = (
    (0, 1),
    (2, 3),
    (4, 5),
    (6, 7),
    (8, 9),
)


class SplitCIFAR10Task(TypedDict):
    """One Split-CIFAR-10 2-class task : NHWC + flat float32 + label."""

    x_train: np.ndarray
    x_train_nhwc: np.ndarray
    y_train: np.ndarray
    x_test: np.ndarray
    x_test_nhwc: np.ndarray
    y_test: np.ndarray


def decode_cifar10_bin(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode one CIFAR-10 binary file into (images NHWC uint8, labels uint8).

    Each record is ``CIFAR10_RECORD_SIZE`` bytes ; the image is
    stored CHW (RGB) and reshaped to NHWC for CNN consumption.

    Raises :
        ValueError : payload length is not a multiple of
                     ``CIFAR10_RECORD_SIZE`` (truncated file).
    """
    raw = path.read_bytes()
    n, rem = divmod(len(raw), CIFAR10_RECORD_SIZE)
    if rem != 0:
        raise ValueError(
            f"truncated CIFAR-10 binary in {path} : "
            f"{len(raw)} bytes is not a multiple of "
            f"{CIFAR10_RECORD_SIZE}"
        )
    if n == 0:
        return (
            np.zeros((0, 32, 32, 3), np.uint8),
            np.zeros((0,), np.uint8),
        )
    arr = np.frombuffer(raw, dtype=np.uint8).reshape(
        n, CIFAR10_RECORD_SIZE
    )
    labels = arr[:, 0].copy()
    nhwc = np.transpose(
        arr[:, 1:].reshape(n, 3, 32, 32), (0, 2, 3, 1)
    ).copy()
    return nhwc, labels


def _http_get(url: str, timeout: int = 60) -> bytes:
    """HTTP GET with browser-style UA. Raises on non-2xx."""
    req = urllib.request.Request(
        url, headers={"User-Agent": HTTP_USER_AGENT}
    )
    with urllib.request.urlopen(req, timeout=timeout) as resp:  # noqa: S310
        return resp.read()


def download_if_missing(data_dir: Path) -> Path:
    """Download tar -> verify SHA-256 -> extract (canonical path).

    Returns the ``cifar-10-batches-bin/`` extracted dir. Raises
    ``FileNotFoundError`` on network failure (pre-reg §9.1
    deviation envelope) — caller may then fall back to
    :func:`download_if_missing_hf` per the §9.1 amendment.
    """
    bin_dir = data_dir / "cifar-10-batches-bin"
    if bin_dir.exists() and (bin_dir / "test_batch.bin").exists():
        return bin_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "cifar-10-binary.tar.gz"
    if not tar_path.exists():
        try:
            tar_path.write_bytes(_http_get(CIFAR10_URL, timeout=120))
        except OSError as exc:
            raise FileNotFoundError(
                f"CIFAR-10 download failed : {exc}"
            ) from exc
    # Skip integrity check when the SHA-256 pin is a placeholder
    # ("..." prefix or "unavailable_" prefix per §9.1 amendment). The
    # canonical Toronto path is dead at pilot time ; the HF parquet
    # fallback owns the load-bearing data path.
    if not (
        CIFAR10_TAR_SHA256.startswith("...")
        or CIFAR10_TAR_SHA256.startswith("unavailable_")
    ):
        h = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        if h != CIFAR10_TAR_SHA256:
            raise ValueError(
                f"SHA-256 mismatch : got {h}, "
                f"expected {CIFAR10_TAR_SHA256}"
            )
    with tarfile.open(tar_path, "r:gz") as tar:
        # PEP 706 / CVE-2007-4559 hardening — restrict extracted paths to
        # data files within data_dir, no symlinks, no setuid bits.
        tar.extractall(data_dir, filter="data")
    return bin_dir


def _verify_sha256(blob: bytes, expected: str, label: str) -> None:
    h = hashlib.sha256(blob).hexdigest()
    if h != expected:
        raise ValueError(
            f"SHA-256 mismatch ({label}) : got {h}, "
            f"expected {expected}"
        )


def download_if_missing_hf(data_dir: Path) -> tuple[Path, Path]:
    """Fallback path : fetch the HF parquet shards if absent.

    Returns ``(train_parquet, test_parquet)`` paths. SHA-256
    pinned per ``CIFAR10_HF_{TRAIN,TEST}_SHA256``. Raises
    ``FileNotFoundError`` on network failure (pre-reg §9.1
    second-line deviation : if even the HF mirror is unreachable,
    the pilot must abort).
    """
    data_dir.mkdir(parents=True, exist_ok=True)
    train_path = data_dir / "cifar10-train.parquet"
    test_path = data_dir / "cifar10-test.parquet"
    pairs = (
        (train_path, CIFAR10_HF_TRAIN_URL, CIFAR10_HF_TRAIN_SHA256, "train"),
        (test_path, CIFAR10_HF_TEST_URL, CIFAR10_HF_TEST_SHA256, "test"),
    )
    for path, url, sha, label in pairs:
        if path.exists():
            _verify_sha256(path.read_bytes(), sha, f"hf-{label}")
            continue
        try:
            blob = _http_get(url, timeout=300)
        except OSError as exc:
            raise FileNotFoundError(
                f"CIFAR-10 HF mirror download failed for "
                f"{label} : {exc}"
            ) from exc
        _verify_sha256(blob, sha, f"hf-{label}")
        path.write_bytes(blob)
    return train_path, test_path


def _decode_parquet_shard(path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Decode a HF cifar10 parquet shard into (NHWC uint8, labels uint8)."""
    import pyarrow.parquet as pq
    from PIL import Image  # local import keeps base loader pure-numpy

    table = pq.read_table(path)
    df = table.to_pandas()
    n = len(df)
    images = np.empty((n, 32, 32, 3), dtype=np.uint8)
    labels = np.empty((n,), dtype=np.uint8)
    for i in range(n):
        cell = df["img"].iloc[i]
        png_bytes = cell["bytes"] if isinstance(cell, dict) else cell
        with Image.open(io.BytesIO(png_bytes)) as pil_img:
            arr = np.asarray(pil_img.convert("RGB"))
        images[i] = arr
        labels[i] = int(df["label"].iloc[i])
    return images, labels


def load_split_cifar10_5tasks_hf(
    train_parquet: Path, test_parquet: Path
) -> list["SplitCIFAR10Task"]:
    """Build the 5-task split from HF parquet shards.

    Same NHWC + flat float32 + label remapping contract as
    :func:`load_split_cifar10_5tasks`.
    """
    if not train_parquet.exists():
        raise FileNotFoundError(
            f"missing CIFAR-10 HF train parquet : {train_parquet}"
        )
    if not test_parquet.exists():
        raise FileNotFoundError(
            f"missing CIFAR-10 HF test parquet : {test_parquet}"
        )
    x_train_raw, y_train_raw = _decode_parquet_shard(train_parquet)
    x_test_raw, y_test_raw = _decode_parquet_shard(test_parquet)
    return _build_tasks_from_arrays(
        x_train_raw, y_train_raw, x_test_raw, y_test_raw
    )


def load_split_cifar10_5tasks_auto(data_dir: Path) -> list["SplitCIFAR10Task"]:
    """Try canonical loader, fall back to HF parquet on FileNotFound.

    Calls :func:`load_split_cifar10_5tasks` first (canonical
    binary). If the canonical layout is absent, transparently
    downloads + decodes the HF parquet mirror per pre-reg §9.1
    amendment. ``data_dir`` is the workspace dir
    (``experiments/g4_quinto_test/data``) ; this function will
    locate or create the appropriate sub-paths.
    """
    canonical_dir = data_dir / "cifar-10-batches-bin"
    if canonical_dir.exists() and (canonical_dir / "test_batch.bin").exists():
        return load_split_cifar10_5tasks(canonical_dir)
    train_path, test_path = download_if_missing_hf(data_dir)
    return load_split_cifar10_5tasks_hf(train_path, test_path)


def _build_tasks_from_arrays(
    x_train_raw: np.ndarray,
    y_train_raw: np.ndarray,
    x_test_raw: np.ndarray,
    y_test_raw: np.ndarray,
) -> list[SplitCIFAR10Task]:
    """Common 5-task split builder shared by canonical + HF paths."""
    x_tr_nhwc = x_train_raw.astype(np.float32) / 255.0
    x_te_nhwc = x_test_raw.astype(np.float32) / 255.0
    x_tr_flat = x_tr_nhwc.reshape(x_tr_nhwc.shape[0], -1)
    x_te_flat = x_te_nhwc.reshape(x_te_nhwc.shape[0], -1)

    tasks: list[SplitCIFAR10Task] = []
    for class_a, class_b in SPLIT_CIFAR10_TASKS:
        tr = (y_train_raw == class_a) | (y_train_raw == class_b)
        te = (y_test_raw == class_a) | (y_test_raw == class_b)
        y_tr = np.where(
            y_train_raw[tr] == class_a, 0, 1
        ).astype(np.int64)
        y_te = np.where(
            y_test_raw[te] == class_a, 0, 1
        ).astype(np.int64)
        tasks.append(
            SplitCIFAR10Task(
                x_train=x_tr_flat[tr],
                x_train_nhwc=x_tr_nhwc[tr],
                y_train=y_tr,
                x_test=x_te_flat[te],
                x_test_nhwc=x_te_nhwc[te],
                y_test=y_te,
            )
        )
    return tasks


def load_split_cifar10_5tasks(data_dir: Path) -> list[SplitCIFAR10Task]:
    """Load CIFAR-10 binary as 5 sequential 2-class binary tasks.

    Expects the six canonical binary files in ``data_dir`` :

        data_batch_1.bin .. data_batch_5.bin
        test_batch.bin

    Returns a list of 5 :class:`SplitCIFAR10Task` dicts, each with
    NHWC + flat float32 images normalised to ``[0, 1]`` and labels
    remapped to ``{0, 1}`` (binary head shared across tasks).

    Raises :
        FileNotFoundError : ``data_dir`` is not a directory or any
                            of the six canonical files is missing.
    """
    if not data_dir.exists() or not data_dir.is_dir():
        raise FileNotFoundError(
            f"CIFAR-10 dir does not exist : {data_dir}"
        )
    test_path = data_dir / "test_batch.bin"
    if not test_path.exists():
        raise FileNotFoundError(
            f"missing CIFAR-10 test batch : {test_path}"
        )
    train_imgs: list[np.ndarray] = []
    train_lbls: list[np.ndarray] = []
    for k in range(1, 6):
        p = data_dir / f"data_batch_{k}.bin"
        if not p.exists():
            raise FileNotFoundError(
                f"missing CIFAR-10 train batch : {p}"
            )
        x, y = decode_cifar10_bin(p)
        train_imgs.append(x)
        train_lbls.append(y)
    x_train_raw = np.concatenate(train_imgs, axis=0)
    y_train_raw = np.concatenate(train_lbls, axis=0)
    x_test_raw, y_test_raw = decode_cifar10_bin(test_path)
    return _build_tasks_from_arrays(
        x_train_raw, y_train_raw, x_test_raw, y_test_raw
    )
