"""TIES-Merge implementation for the micro_kiki substrate.

Numpy port of TIES-Merge (Yadav et al. arXiv 2306.01708) :
trim → elect-sign → disjoint-mean merge on a list of task-
specific delta tensors.

Extracted from ``_legacy.py`` by N5 Task 6 (2026-05-10). Continues
the M2 audit split started by Tasks 4-5 (handlers, oplora). The
recombine handler in :mod:`.handlers` lazy-imports ``_ties_merge``
from this module ; the legacy holding file re-exports it for any
remaining call site under ``._legacy``.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray


# -----------------------------------------------------------------
# TIES-Merge (Yadav et al., arXiv 2306.01708, §3)
# -----------------------------------------------------------------
# Given ``K`` task-specific delta tensors ``τ_i = W_ft_i - W_base``
# (sharing a common shape), TIES-Merge produces a single merged
# delta via a three-step procedure :
#
#   1. **Trim** : per task ``i``, zero the ``(1 - k)%`` smallest-
#      magnitude entries of ``τ_i``. Keeps only the top-``k``
#      fraction by absolute value — reduces "sign-noise"
#      interference from parameters the task barely updated.
#   2. **Elect sign** : per parameter ``p`` compute the sign of
#      the sum of signed magnitudes across tasks,
#      ``γ_p = sign(Σ_i sign(τ_i[p]) · |τ_i[p]|)``. Parameters
#      with no consensus end up at ``γ = 0`` and contribute zero
#      to the merged delta.
#   3. **Disjoint merge** : per parameter, take the mean over only
#      those tasks whose sign agrees with ``γ_p``. Parameters with
#      ``γ = 0`` or no agreeing tasks remain ``0``.
#
# The merged delta is finally scaled by a merge coefficient
# ``alpha`` (default 1.0 — paper §3 default ; larger values
# amplify the merged contribution when downstream eval suggests
# under-shooting).
#
# Numpy-only port so the dream runtime stays torch / mlx free.
# -----------------------------------------------------------------


def _ties_merge(
    deltas: list[NDArray],
    trim_fraction: float = 0.2,
    alpha: float = 1.0,
) -> NDArray:
    """Merge a list of task-specific delta tensors via TIES-Merge.

    Parameters
    ----------
    deltas
        Non-empty list of per-task delta tensors. All must share
        the same shape ; shape-mismatch raises ``ValueError``.
    trim_fraction
        Fraction of entries to **keep** per task (top-magnitude
        quantile). Default ``0.2`` matches the paper's k=20 %.
        Must lie in ``(0, 1]``.
    alpha
        Merge coefficient scaling the final delta. Default ``1.0``
        — the paper's unscaled merge.

    Returns
    -------
    merged : ndarray
        Same shape as each input delta ; dtype of the first input.

    Raises
    ------
    ValueError
        - Empty ``deltas`` list.
        - Shape-mismatch across inputs.
        - ``trim_fraction`` outside ``(0, 1]``.

    Notes
    -----
    Single-element input fast-paths to ``alpha * deltas[0]`` — no
    election / trimming needed when only one task contributes.

    Reference : Yadav et al., *TIES-Merging : Resolving
    Interference When Merging Models*, arXiv 2306.01708, §3
    (procedure) + §4 (empirical defaults).
    """
    if not deltas:
        raise ValueError(
            "TIES-Merge _ties_merge requires at least one delta ; "
            "got empty list"
        )
    if not (0.0 < trim_fraction <= 1.0):
        raise ValueError(
            f"trim_fraction must lie in (0, 1], got {trim_fraction}"
        )

    first = np.asarray(deltas[0])
    target_dtype = first.dtype
    target_shape = first.shape

    if len(deltas) == 1:
        # Single-task fast path : no election / trim needed.
        return (alpha * first.astype(np.float64)).astype(
            target_dtype, copy=False,
        )

    for i, d in enumerate(deltas):
        arr = np.asarray(d)
        if arr.shape != target_shape:
            raise ValueError(
                f"TIES-Merge: all deltas must share shape "
                f"{target_shape}, got {arr.shape} at index {i}"
            )

    # Stack into shape (K, *delta_shape) in float64 for a stable
    # sign-sum reduction.
    stack = np.stack(
        [np.asarray(d, dtype=np.float64) for d in deltas], axis=0,
    )
    K = stack.shape[0]

    # Step 1 — Trim : per task, zero entries below the
    # (1 - trim_fraction) magnitude quantile.
    abs_stack = np.abs(stack)
    # Flatten per-task axis so we can take a per-row quantile.
    flat_abs = abs_stack.reshape(K, -1)
    # Quantile threshold : keep entries >= quantile(|τ_i|, 1-k).
    # When trim_fraction == 1.0 the threshold is the min (keep
    # everything) ; when trim_fraction is tiny the threshold is
    # near the max (drop all but the largest entries).
    q = 1.0 - trim_fraction
    thresholds = np.quantile(flat_abs, q, axis=1)  # shape (K,)
    # Broadcast threshold over the per-task slice.
    keep_mask_shape = (K,) + (1,) * (stack.ndim - 1)
    thresholds_b = thresholds.reshape(keep_mask_shape)
    keep_mask = abs_stack >= thresholds_b
    trimmed = np.where(keep_mask, stack, 0.0)

    # Step 2 — Elect sign : per-parameter sign of the signed-
    # magnitude sum across tasks. ``np.sign`` maps {<0, 0, >0} to
    # {-1, 0, +1}.
    signed_sum = np.sum(trimmed, axis=0)  # drop K axis
    elected = np.sign(signed_sum)  # shape == target_shape

    # Step 3 — Disjoint merge : per parameter, mean over tasks
    # whose sign agrees with the elected sign.
    trimmed_signs = np.sign(trimmed)
    agree_mask = trimmed_signs == elected[None, ...]
    # Exclude elected == 0 entries (no consensus → merged = 0).
    agree_mask &= elected[None, ...] != 0

    contrib_count = np.sum(agree_mask, axis=0)  # shape target
    numerator = np.sum(np.where(agree_mask, trimmed, 0.0), axis=0)
    # Divide-by-zero guard : parameters with zero contributors
    # stay at 0 in the merged delta.
    merged = np.where(
        contrib_count > 0,
        numerator / np.maximum(contrib_count, 1),
        0.0,
    )

    merged *= alpha
    return merged.astype(target_dtype, copy=False)
