"""Split-CIFAR-100 5-task loader — Wave 3b M4 (D1 decision).

Split-CIFAR-100 splits CIFAR-100's 100 classes into 5 disjoint
**tasks** of 20 classes each. Per the plan §2.2 D2 the M5-local
budget per task is 25 000 training samples (full CIFAR-100 train
split = 50 000, with 10 000 in each 20-class group; M5 plan uses
the full per-task quota = 5 000 train + 1 000 val per class window
when the corpus is materialised end-to-end).

This module ships **two scales** :

- **prod scale** (default) — 5 000 train samples per task. Mirrors
  the M5 budget. Requires a materialised CIFAR-100 cache (either
  a Hugging Face ``datasets`` cache or a pinned ``.npz`` fixture
  under ``tests/fixtures/cifar100/``).
- **smoke scale** (``smoke=True``) — 256 train + 64 val samples per
  task, **deterministic synthetic latents** generated via
  ``mx.random.normal`` with a per-task split-derived sub-key. The
  smoke path is offline-safe and does not touch the network or the
  HF cache. This is the path used by ``--smoke`` in
  ``scripts/ablation_cycle3_diffusion.py``.

The two scales share the same return contract so a smoke run and a
prod run are wired identically downstream :

    iter_batches : Iterable[mx.array]      # shape (batch, d_feat)
    n_classes_per_task : 20
    task_idx : int in 0..4

R1 contract : batch ordering and sample contents are derived from
``seed`` via ``mx.random.key`` + ``mx.random.split`` per the plan
§2.3 D3. Two calls with the same ``(task_idx, seed, batch_size)``
tuple yield bit-identical batches.

References:
- ``docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md``
  §2.1 D1 (Split-CIFAR-100), §2.2 D2 (corpus size),
  §2.3 D3 (R1 RNG contract).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable, Iterator

if TYPE_CHECKING:
    import mlx.core as mx_t

# We use ``pytest.importorskip``-friendly lazy access for mlx so
# Linux CI runners (no MLX) can still import the module for
# enumeration / smoke-shape contract tests.


# Per-channel CIFAR-100 normalisation constants (mean/std over the
# full 50 000-sample training set, channel order R/G/B).
# Source: standard values used across the CL benchmark literature
# (GEM, A-GEM, CLEAR-100 baselines). Keeps prod features roughly
# zero-mean / unit-variance — matching the smoke path (mx.random.normal).
import numpy as np

_CIFAR100_MEAN = np.array([0.5071, 0.4865, 0.4409], dtype=np.float32)
_CIFAR100_STD = np.array([0.2673, 0.2564, 0.2762], dtype=np.float32)

# Public constants — used by smoke test + by the M5 driver.
N_CIFAR100_CLASSES = 100
N_TASKS = 5
N_CLASSES_PER_TASK = N_CIFAR100_CLASSES // N_TASKS  # 20
SMOKE_N_TRAIN_PER_TASK = 256
SMOKE_N_VAL_PER_TASK = 64
PROD_N_TRAIN_PER_TASK = 5_000
PROD_N_VAL_PER_TASK = 1_000
# Flattened CIFAR-100 image size : 32 × 32 × 3 = 3072. The encoder
# downstream reads d_in via the substrate config (default 512), so
# the loader exposes the raw 3072 feature width and the caller
# slices / projects.
RAW_FEATURE_DIM = 32 * 32 * 3
# Smoke-scale feature width — matches MLXLatentDiffusionConfig
# ``d_in`` default = 512 so the smoke loader feeds the substrate
# without an additional projection step.
SMOKE_FEATURE_DIM = 512


@dataclass(frozen=True)
class SplitCifar100Batch:
    """One batch yielded by :func:`load_split_cifar100`.

    Attributes
    ----------
    features:
        ``(batch, feature_dim)`` mx.array of pixels (prod) or
        synthetic latents (smoke). Already normalised to roughly
        zero-mean / unit-variance.
    labels:
        ``(batch,)`` mx.array of int32 class indices **local to the
        task** (i.e. in ``0..N_CLASSES_PER_TASK - 1``). The global
        class index is recoverable as
        ``task_idx * N_CLASSES_PER_TASK + label``.
    task_idx:
        The Split-CIFAR-100 task index this batch belongs to.
    """

    features: "mx_t.array"
    labels: "mx_t.array"
    task_idx: int


def task_classes(task_idx: int) -> tuple[int, ...]:
    """Return the global CIFAR-100 class indices for ``task_idx``.

    Deterministic linear partition : task ``k`` owns classes
    ``[k * 20, k * 20 + 20)``. This is the standard Split-CIFAR-100
    convention used by the CL benchmark literature (GEM, A-GEM,
    CLEAR-100 baselines).
    """
    if not 0 <= task_idx < N_TASKS:
        raise ValueError(
            f"task_idx must be in 0..{N_TASKS - 1}, got {task_idx}"
        )
    start = task_idx * N_CLASSES_PER_TASK
    return tuple(range(start, start + N_CLASSES_PER_TASK))


def _derive_task_keys(
    task_idx: int, seed: int
) -> tuple["mx_t.array", "mx_t.array"]:
    """Split a seed into (feature_key, label_key) for one task.

    Per the plan §2.3 D3 RNG contract : never feed a raw
    ``mx.random.key(seed)`` to a noise consumer ; always pass
    through one or more ``mx.random.split`` calls.
    """
    import mlx.core as mx

    root = mx.random.key(seed)
    task_keys = mx.random.split(root, num=N_TASKS)
    task_root = task_keys[task_idx]
    feature_key, label_key = mx.random.split(task_root, num=2)
    return feature_key, label_key


def _smoke_batches(
    task_idx: int,
    seed: int,
    batch_size: int,
    n_samples: int,
    feature_dim: int,
) -> Iterator[SplitCifar100Batch]:
    """Generate deterministic synthetic batches for the smoke path.

    Samples ``n_samples`` features from a standard normal and
    labels uniformly from ``0..N_CLASSES_PER_TASK - 1``. The same
    ``(task_idx, seed, batch_size, n_samples, feature_dim)`` tuple
    always yields the same bytes — R1-friendly.
    """
    import mlx.core as mx

    feature_key, label_key = _derive_task_keys(task_idx, seed)
    n_batches = (n_samples + batch_size - 1) // batch_size
    feat_keys = mx.random.split(feature_key, num=max(n_batches, 1))
    label_keys = mx.random.split(label_key, num=max(n_batches, 1))
    remaining = n_samples
    for i in range(n_batches):
        this_batch = min(batch_size, remaining)
        if this_batch <= 0:
            break
        features = mx.random.normal(
            shape=(this_batch, feature_dim), key=feat_keys[i]
        )
        labels = mx.random.randint(
            low=0,
            high=N_CLASSES_PER_TASK,
            shape=(this_batch,),
            key=label_keys[i],
        ).astype(mx.int32)
        remaining -= this_batch
        yield SplitCifar100Batch(
            features=features, labels=labels, task_idx=task_idx
        )


def _prod_batches(
    *,
    task_idx: int,
    seed: int,
    batch_size: int,
    split: str,
) -> Iterable["SplitCifar100Batch"]:
    """Real Split-CIFAR-100 batches for one task window.

    Loads ``uoft-cs/cifar100`` from the HuggingFace cache, keeps only
    the 20 fine-label classes of ``task_idx`` (via ``task_classes``),
    remaps labels into the task-local 0..19 range, and yields
    fixed-size batches in a seed-deterministic order.
    """
    import mlx.core as mx

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - env guard
        raise FileNotFoundError(
            "The 'datasets' package is required for the M5 prod "
            "loader. Install with: uv sync --all-extras"
        ) from exc

    hf_split = "train" if split == "train" else "test"
    try:
        dataset = load_dataset("uoft-cs/cifar100", split=hf_split)
    except (FileNotFoundError, ConnectionError, OSError) as exc:
        raise FileNotFoundError(
            "Split-CIFAR-100 prod data not in the HuggingFace cache. "
            "Run: huggingface-cli download uoft-cs/cifar100 "
            "--repo-type dataset"
        ) from exc

    keep_classes = list(task_classes(task_idx))
    class_to_local = {c: i for i, c in enumerate(keep_classes)}

    fine = np.asarray(dataset["fine_label"], dtype=np.int64)
    mask = np.isin(fine, keep_classes)
    rows = np.nonzero(mask)[0]

    cap = PROD_N_TRAIN_PER_TASK if split == "train" else PROD_N_VAL_PER_TASK

    images = dataset.with_format("numpy")["img"]
    raw: np.ndarray[tuple[int, ...], np.dtype[np.uint8]] = np.stack(
        [np.asarray(images[i], dtype=np.uint8) for i in rows]
    )
    # raw shape: (N, 32, 32, 3) uint8 — normalise per-channel before
    # flattening so features are zero-mean / unit-variance, matching
    # the smoke path (mx.random.normal). _CIFAR100_MEAN/STD broadcast
    # over (N, H, W) via the trailing channel axis.
    norm = (raw.astype(np.float32) / 255.0 - _CIFAR100_MEAN) / _CIFAR100_STD
    feats: np.ndarray[tuple[int, ...], np.dtype[np.float32]] = (
        norm.reshape(len(rows), RAW_FEATURE_DIM).astype(np.float32)
    )
    labels = np.array(
        [class_to_local[int(fine[i])] for i in rows], dtype=np.int32
    )

    feature_key, _ = _derive_task_keys(task_idx, seed)
    perm_key = mx.random.split(feature_key, num=2)[1]
    order = np.asarray(mx.random.permutation(len(rows), key=perm_key))
    order = order[:cap]

    feats = feats[order]
    labels = labels[order]

    for start in range(0, len(order), batch_size):
        stop = start + batch_size
        yield SplitCifar100Batch(
            features=mx.array(feats[start:stop]),
            labels=mx.array(labels[start:stop]),
            task_idx=task_idx,
        )


def load_split_cifar100(
    task_idx: int,
    batch_size: int = 32,
    seed: int = 0,
    *,
    smoke: bool = False,
    split: str = "train",
) -> Iterable[SplitCifar100Batch]:
    """Yield Split-CIFAR-100 batches for one task.

    Args:
        task_idx: One of ``0..4``.
        batch_size: Mini-batch size. Per-batch shape is
            ``(min(batch_size, remaining), feature_dim)``.
        seed: R1 seed for batch ordering + (smoke path) for the
            synthetic generator. Same seed → same batches.
        smoke: When True, return deterministic synthetic latents
            sized for smoke validation (~256 train / 64 val per
            task). The synthetic path does not require a CIFAR-100
            cache and is offline-safe. Default False — the prod
            path is selected.
        split: ``"train"`` or ``"val"``. The smoke path supports
            both via the same synthetic generator (with a different
            seed-derived sub-key per split).

    Returns:
        A finite iterable of :class:`SplitCifar100Batch` instances.
        The iterable is **single-use** ; callers materialise it
        with ``list(...)`` if they need re-iteration.

    Raises:
        ValueError: ``task_idx`` out of range or ``batch_size <= 0``
            or ``split not in {"train", "val"}``.
        FileNotFoundError: prod path is selected and the CIFAR-100
            cache is not materialised. M4 deliberately does not
            implement the prod path (M5 milestone) — callers must
            pass ``smoke=True`` until then.

    Memory footprint (M5 prod-mode estimate per plan §2.2 D2) :
        25 000 samples × 3072 floats × 4 bytes ≈ 300 MB per task,
        held as one float32 mx.array in VRAM on M5/Studio. The
        smoke path holds ≤ 256 × 512 × 4 ≈ 0.5 MB per task —
        negligible.
    """
    if batch_size <= 0:
        raise ValueError(
            f"batch_size must be positive, got {batch_size}"
        )
    if split not in ("train", "val"):
        raise ValueError(
            f'split must be "train" or "val", got {split!r}'
        )
    # task_idx range-check happens inside _derive_task_keys via
    # task_classes — call it eagerly to fail fast.
    task_classes(task_idx)

    if smoke:
        # Use distinct seed derivations for train vs val so the two
        # splits cover non-overlapping pseudo-random material (still
        # deterministic given the user-supplied ``seed``).
        split_seed = seed if split == "train" else seed + 10_007
        n_samples = (
            SMOKE_N_TRAIN_PER_TASK
            if split == "train"
            else SMOKE_N_VAL_PER_TASK
        )
        return _smoke_batches(
            task_idx=task_idx,
            seed=split_seed,
            batch_size=batch_size,
            n_samples=n_samples,
            feature_dim=SMOKE_FEATURE_DIM,
        )

    # Prod path (M5) — real CIFAR-100 from the HuggingFace cache.
    split_seed = seed if split == "train" else seed + 10_007
    return _prod_batches(
        task_idx=task_idx,
        seed=split_seed,
        batch_size=batch_size,
        split=split,
    )
