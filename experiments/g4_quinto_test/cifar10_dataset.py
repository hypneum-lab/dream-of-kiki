"""Split-CIFAR-10 5-task loader — pure numpy, no torchvision.

Source : https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
Records (per CIFAR-10 binary spec) : 1 label byte + 3072 image
bytes (CHW : 1024 R + 1024 G + 1024 B, row-major within each
channel). Five training files ``data_batch_1.bin`` ..
``data_batch_5.bin`` (10000 records each) plus ``test_batch.bin``
(10000 records).

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
    docs/osf-prereg-g4-quinto-pilot.md sec 5
"""
from __future__ import annotations

import hashlib
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
# One-shot SHA-256 pin — value is replaced at first download in
# Task 9 §2 of the G4-quinto plan. The placeholder leading "..."
# disables the integrity check until the real hash is committed.
CIFAR10_TAR_SHA256: Final[str] = "...replace_in_task9..."
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


def download_if_missing(data_dir: Path) -> Path:
    """Download tar -> verify SHA-256 -> extract.

    Returns the ``cifar-10-batches-bin/`` extracted dir. Raises
    ``FileNotFoundError`` on network failure (pre-reg §9.1
    deviation envelope).
    """
    bin_dir = data_dir / "cifar-10-batches-bin"
    if bin_dir.exists() and (bin_dir / "test_batch.bin").exists():
        return bin_dir
    data_dir.mkdir(parents=True, exist_ok=True)
    tar_path = data_dir / "cifar-10-binary.tar.gz"
    if not tar_path.exists():
        try:
            urllib.request.urlretrieve(CIFAR10_URL, tar_path)
        except OSError as exc:
            raise FileNotFoundError(
                f"CIFAR-10 download failed : {exc}"
            ) from exc
    if not CIFAR10_TAR_SHA256.startswith("..."):
        h = hashlib.sha256(tar_path.read_bytes()).hexdigest()
        if h != CIFAR10_TAR_SHA256:
            raise ValueError(
                f"SHA-256 mismatch : got {h}, "
                f"expected {CIFAR10_TAR_SHA256}"
            )
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(data_dir)
    return bin_dir


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
