"""OPLoRA projection logic for the micro_kiki substrate.

Extracted from :mod:`._legacy` by N5 Task 5 (2026-05-10). Continues
the M2 audit split started by Task 4 (handlers). Tasks 6-7 will
peel TIES-Merge + safetensors loaders out next.

OPLoRA (Orthogonal-Projection LoRA, arXiv 2510.13003, Du et al.) :
given ``k`` prior LoRA deltas ``Δ_i = B_i · A_i`` (each of shape
``(out_dim, in_dim)``), OPLoRA constructs a projector ``P`` onto
the orthogonal complement of the column space spanned by the
priors. When a new adapter B-matrix ``B_new`` is restructured we
replace it with ``P · B_new``, which guarantees that the new
contribution ``P · B_new · A_new · x`` is orthogonal to every
prior range — i.e. the new stack cannot overwrite features
already encoded in the retained subspace.

The micro-kiki *training* pipeline uses torch (see local impl
``src/stacks/oplora.py``) ; the substrate runs on the dream
runtime which is pure numpy (no torch / mlx dep), so this is a
pure-linear-algebra port. The algebra is identical :

  torch                                                 numpy
  -----------------------------------------------      -------------------------------------------------
  Q, _ = torch.linalg.qr(prior.float())                U, S, _Vt = np.linalg.svd(prior, full_matrices=False)
  P    = I - Q @ Q.T                                   U_trim = U[:, S > rank_thresh]
                                                       P      = I - U_trim @ U_trim.T

The numpy port uses SVD rather than QR so we can filter out the
near-zero singular values (rank selection via ``rank_thresh``) :
QR would include numerical-noise columns in ``Q``, over-pruning
the projected subspace. SVD + singular-value filter is the
standard OPLoRA recipe (paper §3.2).
"""
from __future__ import annotations

import logging

import numpy as np
from numpy.typing import NDArray


_LOG = logging.getLogger(__name__)


def _oplora_projector(
    prior_deltas: list[NDArray] | tuple[NDArray, ...],
    rank_thresh: float = 1e-4,
) -> NDArray:
    """Build the OPLoRA orthogonal-complement projector ``P``.

    Parameters
    ----------
    prior_deltas
        Iterable of prior-adapter delta matrices (each ``B_i @ A_i``,
        shape ``(out_dim, in_dim)``). All must share the same
        ``out_dim`` — the first axis is what the projector acts on
        (columns of ``B_new``). An empty iterable returns the
        identity (no prior subspace to exclude).
    rank_thresh
        Singular-value magnitude below which a direction is treated
        as numerical noise and dropped from the prior subspace. The
        paper uses ``1e-4`` (§3.2).

    Returns
    -------
    P : ndarray of shape ``(out_dim, out_dim)``
        Orthogonal-complement projector. Satisfies ``P == P.T``,
        ``P @ P ≈ P``, and ``P @ v ≈ 0`` for any ``v`` in the prior
        column space.

    Notes
    -----
    Guarded against two numerical failure modes :

    1. **Zero-singular-value rank collapse** : if *all* singular
       values of the stacked prior fall below ``rank_thresh``
       (pathological : priors are numerical noise), we fall back
       to ``P = I`` with a warning. The new adapter is not
       projected away.
    2. **Shape mismatch across priors** : explicit ``ValueError``
       raised rather than silently broadcasting — the caller
       must make the priors shape-consistent before handing them
       in. Guards against a whole class of latent bugs where a
       reshape earlier in the pipeline slips through.

    Reference : Du et al., *OPLoRA: Orthogonal Projection for
    LoRA Continual Learning*, arXiv 2510.13003, §3.2 (projector
    construction) + §3.3 (rank-threshold robustness study).
    """
    deltas = list(prior_deltas)
    if not deltas:
        # No priors → nothing to project away. Identity is the
        # correct (and well-defined) fallback.
        raise ValueError(
            "OPLoRA _oplora_projector requires at least one "
            "prior delta ; pass ``np.eye(out_dim)`` as the "
            "pseudo-prior if you want the no-op branch"
        )
    out_dims = {d.shape[0] for d in deltas}
    if len(out_dims) != 1:
        raise ValueError(
            f"OPLoRA: all prior deltas must share out_dim "
            f"(axis 0), got {sorted(out_dims)}"
        )
    (out_dim,) = out_dims

    # Stack priors column-wise : the combined column space is the
    # union of each prior's range, exactly what the projector must
    # annihilate.
    stacked = np.concatenate(
        [np.asarray(d, dtype=np.float64) for d in deltas], axis=1
    )
    # SVD on the stacked prior matrix. ``U`` columns span the
    # range of ``stacked`` ; filter by singular-value magnitude
    # to drop numerical-noise directions.
    try:
        U, S, _Vt = np.linalg.svd(stacked, full_matrices=False)
    except np.linalg.LinAlgError as exc:  # pragma: no cover
        _LOG.warning(
            "OPLoRA: SVD failed on stacked prior (%s) ; falling "
            "back to identity projector",
            exc,
        )
        return np.eye(out_dim, dtype=np.float32)

    keep = S > rank_thresh
    if not keep.any():
        _LOG.warning(
            "OPLoRA: all %d singular values below rank_thresh=%g "
            "; prior subspace is effectively empty, returning "
            "identity projector",
            S.size, rank_thresh,
        )
        return np.eye(out_dim, dtype=np.float32)

    U_trim = U[:, keep]
    identity = np.eye(out_dim, dtype=np.float64)
    P = identity - U_trim @ U_trim.T
    # Full-rank saturation : when accumulated priors span the entire
    # output space (a real, non-pathological case in long sequential
    # multi-expert curricula), ``U_trim @ U_trim.T`` collapses to the
    # identity and ``P`` to the zero matrix. The handler would then
    # silently zero every new adapter. Surface this loudly so callers
    # know to prune priors or widen ``out_dim``.
    if np.linalg.norm(P, ord="fro") < rank_thresh * out_dim:
        _LOG.warning(
            "OPLoRA: projector is effectively zero — priors span "
            "the full output space (rank=%d, out_dim=%d) ; new "
            "adapter will be annihilated. Prune priors or widen "
            "out_dim.",
            U_trim.shape[1], out_dim,
        )
    # Cast back to float32 — the rest of the substrate stores LoRA
    # tensors in float32 (matches mlx adapter dtype).
    return np.asarray(P, dtype=np.float32)
