"""G4-nono domain corpus: CIFAR-10 (domain A) + Quick, Draw! (domain B).

PI decision 2026-10-24: a second corpus with a stronger domain shift
replaces the CIFAR-100 animals / non-animals split.

* Domain A: the 10 CIFAR-10 classes, HF ``uoft-cs/cifar10`` at revision
  ``0b2714987fa478483af9968de7c934580d0bb9a2`` (same parquet pins as
  ``experiments.g4_quinto_test.cifar10_dataset``). Converted to
  grayscale (ITU-R BT.601 luma, rounded) and replicated to 3 channels,
  so the domain cannot be read from colour alone.
* Domain B: 10 Quick, Draw! categories (``numpy_bitmap``, 28 x 28
  grayscale, CC BY 4.0, Google Creative Lab), semantically disjoint
  from CIFAR-10, zero-padded 28 -> 32 (2 px each side, background 0)
  and replicated to 3 channels. No official SHA-256 exists; ours are
  pinned below next to the GCS md5 / crc32c and byte size.

Labels reuse the coarse-id space of :mod:`experiments.g4_nono_test.stream`
(CIFAR-10 class k -> ``DOMAIN_A[k]``, Quick, Draw! class k ->
``DOMAIN_B[k]``), so the stream, metrics and RECOMBINE strategies are
unchanged. Each class is subsampled to 2 500 train / 500 test with a
seeded generator (``default_rng([SUBSAMPLE_SEED, corpus, k])``); the
index lists are committed (``qd_cifar10_indices.json``) and hashed.
"""
from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from pathlib import Path
from typing import Any, Final

import numpy as np

from experiments.g4_nono_test.stream import DOMAIN_A, DOMAIN_B

CIFAR10_HF_REVISION: Final[str] = "0b2714987fa478483af9968de7c934580d0bb9a2"
CIFAR10_TRAIN_SHA256: Final[str] = (
    "8428b53a88a11ac374111006708df51469e315a22ac6d66470afd9c78d2ae883"
)
CIFAR10_TEST_SHA256: Final[str] = (
    "841389e6f2d64f28bf17310e430aebac20ec3ba611a3c5e231dc93c645ce84de"
)
CIFAR10_CLASSES: Final[tuple[str, ...]] = (
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck",
)
QD_URL: Final[str] = (
    "https://storage.googleapis.com/quickdraw_dataset/full/numpy_bitmap/{name}.npy"
)
QD_CLASSES: Final[tuple[str, ...]] = (
    "apple", "banana", "clock", "cup", "house",
    "key", "ladder", "moon", "tree", "umbrella",
)
# Downloaded 2026-10-24; md5 / crc32c from the GCS x-goog-hash header.
QD_PINS: Final[dict[str, dict[str, Any]]] = {
    "apple": {"bytes": 113462128, "gcs_md5_b64": "j/EjvWFYD/aDTEpmmYqCsw==",
              "gcs_crc32c_b64": "wcYTXA==",
              "sha256": "a70df0030c146cfe4aa9a17aa2a320d5f64ac166fdc3d2cfb439b8322fd8bb0d"},
    "banana": {"bytes": 241421904, "gcs_md5_b64": "ZI4kmC1it+0U/awgK3qBEA==",
               "gcs_crc32c_b64": "FS3uIg==",
               "sha256": "d67e4bbc7a2e51774a1c1d1960e3406ec2ed2946637db3ef8f9ca5dd0d0ae90d"},
    "clock": {"bytes": 94500304, "gcs_md5_b64": "wcl6q7LkotBiRRBSO8iBtg==",
              "gcs_crc32c_b64": "I338Sw==",
              "sha256": "a109294aab4f0f66be01d753879c3252ac1c0a4955a123b266c36a3c8a3109af"},
    "cup": {"bytes": 102485344, "gcs_md5_b64": "2JB7B0m5OGp+jkr0cl8a6A==",
            "gcs_crc32c_b64": "GWZscg==",
            "sha256": "6f6e119f75d19f6aaf40bce109281e08ce35626a387e789ed832458e8692598e"},
    "house": {"bytes": 106169360, "gcs_md5_b64": "KeJUi1vqPa9x7bzoOhyeVw==",
              "gcs_crc32c_b64": "lgzw+g==",
              "sha256": "92e1de1e129349525cb63f6dba6d1b54a0db0a3fc4f18b77cf9f1be4105b2a19"},
    "key": {"bytes": 126196640, "gcs_md5_b64": "iJxpRMvPke7VGwN2P22xYQ==",
            "gcs_crc32c_b64": "8vLN7Q==",
            "sha256": "06d3d9d976d3382b34e38e331ad93c478528cc5114a6c94697490c4207b52f8c"},
    "ladder": {"bytes": 98305056, "gcs_md5_b64": "5X3eZb+40rOG3PhG8QCIvw==",
               "gcs_crc32c_b64": "rDGo5A==",
               "sha256": "7317bb84f3caef4b458ee311e0831f4f0f11a51feae7a5616ef9d72854393c35"},
    "moon": {"bytes": 95382304, "gcs_md5_b64": "sEYbWOynZRxYGRQJOrEbng==",
             "gcs_crc32c_b64": "8efWrg==",
             "sha256": "6b96ee61ff37e0533ed7774c732ffdba7358037e4daae7485a447904b82247ab"},
    "tree": {"bytes": 113461344, "gcs_md5_b64": "0gFwT95h/42B79Dv8o2JtA==",
             "gcs_crc32c_b64": "ZOJgfQ==",
             "sha256": "174eb2ffd87834defa8a503509e0dee330a3b10c81537d35f2af3736b94e7279"},
    "umbrella": {"bytes": 97281936, "gcs_md5_b64": "BN1DVxS5wgaPf3mREe9ONQ==",
                 "gcs_crc32c_b64": "RDPlEw==",
                 "sha256": "5bf7d6d392246eba64c6b8e2baccf09c34b673d302cb3248636dcbef5a823015"},
}
LABEL_OF: Final[dict[tuple[str, int], int]] = {
    **{("cifar10", k): int(DOMAIN_A[k]) for k in range(10)},
    **{("quickdraw", k): int(DOMAIN_B[k]) for k in range(10)},
}
SUBSAMPLE_SEED: Final[int] = 20261024
N_TRAIN: Final[int] = 2500
N_TEST: Final[int] = 500
CORPUS_FILE: Final[str] = "qd-cifar10-v1.npz"
INDEX_FILE: Final[Path] = Path(__file__).with_name("qd_cifar10_indices.json")
# Filled from the build on 2026-10-24 (scripts/g4_build_qd_cifar10.py).
INDEX_SHA256: Final[str] = (
    "f4092ff2766d1bb77b4aee75e9203c970d1c516eb18a4bc5a9ba7d058eb11ae7"
)
CORPUS_SHA256: Final[str] = (
    "4f7fdf1c42b9df3252e4e755101e26f4f7332dcde643b4e0bb535f7857bf381b"
)
_CORPUS_ID: Final[dict[str, int]] = {"cifar10": 0, "quickdraw": 1}


def to_gray3(x: np.ndarray) -> np.ndarray:
    """RGB uint8 NHWC -> BT.601 luma (rounded), replicated to 3 channels."""
    xf = np.asarray(x, dtype=np.float64)
    y = np.rint(0.299 * xf[..., 0] + 0.587 * xf[..., 1] + 0.114 * xf[..., 2])
    g = np.clip(y, 0, 255).astype(np.uint8)
    return np.repeat(g[..., None], 3, axis=-1)


def pad_drawings(d: np.ndarray) -> np.ndarray:
    """(N, 784) or (N, 28, 28) uint8 -> (N, 32, 32, 3), 2-px zero border."""
    a = np.asarray(d, dtype=np.uint8).reshape(-1, 28, 28)
    out = np.zeros((a.shape[0], 32, 32), dtype=np.uint8)
    out[:, 2:30, 2:30] = a
    return np.repeat(out[..., None], 3, axis=-1)


def subsample(
    n_avail: int, *, n_train: int, n_test: int, key: tuple[str, int]
) -> tuple[np.ndarray, np.ndarray]:
    """Seeded disjoint (train, test) row indices for one class.

    For Quick, Draw! both come from one pool (``n_avail`` drawings);
    for CIFAR-10 call it once per split with ``n_test = 0`` or
    ``n_train = 0``.
    """
    if n_train + n_test > n_avail:
        raise ValueError(f"{key}: need {n_train + n_test} rows, have {n_avail}")
    rng = np.random.default_rng([SUBSAMPLE_SEED, _CORPUS_ID[key[0]], key[1]])
    pick = rng.choice(n_avail, size=n_train + n_test, replace=False)
    return pick[:n_train].astype(np.int64), pick[n_train:].astype(np.int64)


def build_corpus(
    cifar_train: tuple[np.ndarray, np.ndarray],
    cifar_test: tuple[np.ndarray, np.ndarray],
    drawings: Mapping[str, np.ndarray],
    *,
    n_train: int = N_TRAIN,
    n_test: int = N_TEST,
) -> dict[str, Any]:
    """Assemble the 20-class corpus (arrays + per-class index lists)."""
    xs_tr, ys_tr, xs_te, ys_te = [], [], [], []
    indices: dict[str, dict[str, list[int]]] = {}
    for k, name in enumerate(CIFAR10_CLASSES):
        lab = LABEL_OF[("cifar10", k)]
        rows_tr = np.flatnonzero(cifar_train[1] == k)
        rows_te = np.flatnonzero(cifar_test[1] == k)
        tr, _ = subsample(len(rows_tr), n_train=n_train, n_test=0, key=("cifar10", k))
        _, te = subsample(len(rows_te), n_train=0, n_test=n_test, key=("cifar10", 100 + k))
        r_tr, r_te = rows_tr[tr], rows_te[te]
        indices[f"cifar10/{name}"] = {"train": r_tr.tolist(), "test": r_te.tolist()}
        xs_tr.append(to_gray3(cifar_train[0][r_tr]))
        xs_te.append(to_gray3(cifar_test[0][r_te]))
        ys_tr.append(np.full(len(r_tr), lab, np.int64))
        ys_te.append(np.full(len(r_te), lab, np.int64))
    for k, name in enumerate(QD_CLASSES):
        lab = LABEL_OF[("quickdraw", k)]
        pool = drawings[name]
        tr, te = subsample(len(pool), n_train=n_train, n_test=n_test, key=("quickdraw", k))
        indices[f"quickdraw/{name}"] = {"train": tr.tolist(), "test": te.tolist()}
        xs_tr.append(pad_drawings(pool[tr]))
        xs_te.append(pad_drawings(pool[te]))
        ys_tr.append(np.full(len(tr), lab, np.int64))
        ys_te.append(np.full(len(te), lab, np.int64))
    return {
        "x_train": np.concatenate(xs_tr),
        "y_train": np.concatenate(ys_tr),
        "x_test": np.concatenate(xs_te),
        "y_test": np.concatenate(ys_te),
        "indices": indices,
    }


def save_corpus(corpus: Mapping[str, Any], data_dir: Path) -> Path:
    """Write the arrays as ``CORPUS_FILE`` (uncompressed npz)."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / CORPUS_FILE
    np.savez(
        path,
        x_train=corpus["x_train"], y_train=corpus["y_train"],
        x_test=corpus["x_test"], y_test=corpus["y_test"],
    )
    return path


def indices_json(indices: Mapping[str, Any]) -> str:
    """Canonical serialisation of the index lists (hashed as committed)."""
    return json.dumps(indices, sort_keys=True, separators=(",", ":")) + "\n"


def load_qd_cifar10(
    data_dir: Path, expected_sha256: str | None = None
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load the pinned corpus: ``(x_train, y_train, x_test, y_test)``.

    Images are NHWC uint8 (gray replicated to 3 channels), labels int
    in the coarse-id space. The file SHA-256 must equal
    ``expected_sha256`` (default: the pinned ``CORPUS_SHA256``).
    """
    path = data_dir / CORPUS_FILE
    want = expected_sha256 or CORPUS_SHA256
    got = hashlib.sha256(path.read_bytes()).hexdigest()
    if got != want:
        raise ValueError(f"SHA-256 mismatch for {path}: got {got}, expected {want}")
    with np.load(path, allow_pickle=False) as z:
        return (np.asarray(z["x_train"]), np.asarray(z["y_train"]),
                np.asarray(z["x_test"]), np.asarray(z["y_test"]))
