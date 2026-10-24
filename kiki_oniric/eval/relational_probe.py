"""DR-3 relational probe statistics (OSF pre-registration, section 5).

Pure-numpy/scipy implementation of the 2 x 2 statistics of
``docs/osf-prereg-dr3-relational-probe-2026-10-23.md``:

============  ==================  ====================
              local (kNN)         global (H0 / MST)
============  ==================  ====================
structure     ``S_mKNN``          ``S_H0``
geometry      ``S_mKNN_tau``      ``S_H0_tau``
============  ==================  ====================

Definitions follow You et al. (arXiv:2609.27252, Eqs. 1-5) as pinned
by the pre-registration:

* ``S_mKNN``: mean over inputs of ``|N_k^X(i) & N_k^Y(i)| / k``, self
  excluded, neighbour ties broken by lower input index.
* ``S_H0``: ``|T_X & T_Y| / (n - 1)`` where ``T`` is the Euclidean MST
  edge set (NOT a Jaccard index), computed with Prim's algorithm from
  input 0: a parent is updated only on a strictly smaller weight and,
  among equal minimum keys, the lowest index is added next.
* Geometry weights: ``D~ = D / Q_0.9({D_ij > 0})`` per representation
  (off-diagonal entries), ``w_ij = exp(-|log D~_X - log D~_Y| / tau)``;
  ``w = 1`` if both distances are zero, ``w = 0`` if exactly one is.
* ``S_mKNN_tau`` / ``S_H0_tau`` multiply shared-support membership by
  ``w`` with the same denominators as the structure scores.

Primary preprocessing is ``"l2"`` (row L2 normalisation, no
centering). ``"center_l2"`` and ``"raw"`` are sensitivity pipelines.
Calibration against the untrained null is a *difference*
(:func:`calibrated_difference`), never a ratio.

Representation inputs are ``(n, d)`` float arrays whose rows are the
same ``n`` inputs in the same order.
"""

from __future__ import annotations

from typing import Mapping

import numpy as np
from numpy.typing import NDArray
from scipy.spatial.distance import cdist

Array = NDArray[np.float64]

PREPROCESS_MODES = ("l2", "center_l2", "raw")
STAT_KEYS = ("S_mKNN", "S_H0", "S_mKNN_tau", "S_H0_tau")


# --- validation and preprocessing ----------------------------------------


def _as_matrix(x: object, name: str = "representation") -> Array:
    arr = np.asarray(x)
    if arr.ndim != 2:
        raise ValueError(f"{name} must be 2-D (n, d), got ndim={arr.ndim}")
    if arr.shape[0] < 2 or arr.shape[1] < 1:
        raise ValueError(f"{name} must have n >= 2 rows and d >= 1, got {arr.shape}")
    if not np.issubdtype(arr.dtype, np.number) or np.issubdtype(arr.dtype, np.complexfloating):
        raise ValueError(f"{name} must be a real numeric array, got dtype {arr.dtype}")
    return arr.astype(np.float64)


def preprocess(x: object, mode: str = "l2") -> Array:
    """Apply a fixed preprocessing pipeline (pre-registration, section 5)."""
    if mode not in PREPROCESS_MODES:
        raise ValueError(f"unknown preprocess mode {mode!r}; expected one of {PREPROCESS_MODES}")
    arr = _as_matrix(x)
    if mode == "raw":
        return arr
    if mode == "center_l2":
        arr = arr - arr.mean(axis=0, keepdims=True)
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    if not np.all(norms > 0):
        raise ValueError("a row has zero norm: L2 normalisation is undefined")
    return np.asarray(arr / norms, dtype=np.float64)


def check_representation(x: object, mode: str = "l2") -> str | None:
    """Return an exclusion reason (section 4) or ``None`` if usable.

    Excluded: non-finite entry; zero-norm row (L2 undefined); no
    positive pairwise distance after preprocessing (all rows coincide,
    so the geometry quantile is undefined).
    """
    arr = _as_matrix(x)
    if not np.all(np.isfinite(arr)):
        return "non-finite entry"
    if mode in ("l2", "center_l2"):
        # zero-norm rows break L2 normalisation (both L2 pipelines)
        base = arr - arr.mean(axis=0, keepdims=True) if mode == "center_l2" else arr
        if not np.all(np.linalg.norm(base, axis=1) > 0):
            return "row with zero norm (L2 normalisation undefined)"
    d = pairwise_distances(preprocess(arr, mode))
    if not np.any(d > 0):
        return "all rows coincide (no positive pairwise distance)"
    return None


# --- distances, neighbours, MST ------------------------------------------


def pairwise_distances(x: object) -> Array:
    """Euclidean distance matrix (float64, zero diagonal)."""
    arr = _as_matrix(x)
    d = np.asarray(cdist(arr, arr, metric="euclidean"), dtype=np.float64)
    np.fill_diagonal(d, 0.0)
    return d


def knn_indices(d: Array, k: int) -> NDArray[np.int64]:
    """k nearest neighbours per row, self excluded, ties -> lower index."""
    n = d.shape[0]
    if not 1 <= k <= n - 1:
        raise ValueError(f"k must satisfy 1 <= k <= n - 1 = {n - 1}, got {k}")
    masked = d.copy()
    np.fill_diagonal(masked, np.inf)
    order = np.argsort(masked, axis=1, kind="stable")
    return np.asarray(order[:, :k], dtype=np.int64)


def prim_mst_edges(d: Array) -> set[tuple[int, int]]:
    """MST edge set (unordered pairs, ``i < j``) with the pinned tie rule.

    Prim from vertex 0; a vertex's parent is replaced only on a
    strictly smaller weight; among equal minimum keys the lowest index
    is added next (You et al., Appendix C.3).
    """
    n = d.shape[0]
    in_tree = np.zeros(n, dtype=bool)
    key = np.full(n, np.inf)
    parent = np.full(n, -1, dtype=np.int64)
    in_tree[0] = True
    key[1:] = d[0, 1:]
    parent[1:] = 0
    edges: set[tuple[int, int]] = set()
    for _ in range(n - 1):
        cand = np.where(in_tree, np.inf, key)
        v = int(np.argmin(cand))  # first (lowest) index among ties
        p = int(parent[v])
        edges.add((min(p, v), max(p, v)))
        in_tree[v] = True
        better = (~in_tree) & (d[v] < key)
        key[better] = d[v][better]
        parent[better] = v
    return edges


# --- geometry weights -----------------------------------------------------


def zero_distance_pairs(d: Array) -> int:
    """Number of unordered off-diagonal pairs at distance exactly 0."""
    off = d[np.triu_indices(d.shape[0], k=1)]
    return int(np.sum(off == 0))


def _quantile90_positive(d: Array) -> float:
    off = d[~np.eye(d.shape[0], dtype=bool)]
    pos = off[off > 0]
    if pos.size == 0:
        raise ValueError("all rows coincide: no positive distance for the 0.9 quantile")
    return float(np.quantile(pos, 0.9))


def geometry_weights(dx: Array, dy: Array, tau: float) -> Array:
    """Distance-agreement weights ``w_ij`` (You et al. Eq. 3), full matrix."""
    if not tau > 0:
        raise ValueError(f"tau must be > 0, got {tau}")
    tx = dx / _quantile90_positive(dx)
    ty = dy / _quantile90_positive(dy)
    zx, zy = dx == 0, dy == 0
    safe_x = np.where(zx, 1.0, tx)
    safe_y = np.where(zy, 1.0, ty)
    w = np.exp(-np.abs(np.log(safe_x) - np.log(safe_y)) / tau)
    w = np.where(zx & zy, 1.0, np.where(zx | zy, 0.0, w))
    return np.asarray(w, dtype=np.float64)


# --- statistics on distance matrices -------------------------------------


def _shared_knn_mask(dx: Array, dy: Array, k: int) -> NDArray[np.bool_]:
    n = dx.shape[0]
    rows = np.repeat(np.arange(n), k)
    mx = np.zeros((n, n), dtype=bool)
    my = np.zeros((n, n), dtype=bool)
    mx[rows, knn_indices(dx, k).ravel()] = True
    my[rows, knn_indices(dy, k).ravel()] = True
    return mx & my


def _check_pair(x: Array, y: Array) -> None:
    if x.shape[0] != y.shape[0]:
        raise ValueError(
            f"representations must have the same number of rows, got {x.shape[0]} and {y.shape[0]}"
        )


def _prepare(x: object, y: object, mode: str) -> tuple[Array, Array]:
    ax, ay = _as_matrix(x, "x"), _as_matrix(y, "y")
    _check_pair(ax, ay)
    for name, arr in (("x", ax), ("y", ay)):
        reason = check_representation(arr, mode)
        if reason is not None:
            raise ValueError(f"representation {name} excluded: {reason}")
    return pairwise_distances(preprocess(ax, mode)), pairwise_distances(preprocess(ay, mode))


def _mknn_from_d(dx: Array, dy: Array, k: int, w: Array | None) -> float:
    shared = _shared_knn_mask(dx, dy, k)
    vals = shared.astype(np.float64) if w is None else shared * w
    return float(np.mean(vals.sum(axis=1) / k))


def _h0_from_d(dx: Array, dy: Array, w: Array | None) -> float:
    n = int(dx.shape[0])
    common = prim_mst_edges(dx) & prim_mst_edges(dy)
    if w is None:
        return len(common) / (n - 1)
    total = 0.0
    for i, j in common:
        total += float(w[i, j])
    return total / (n - 1)


def mutual_knn(x: object, y: object, k: int = 10, preprocess: str = "l2") -> float:
    """``S_mKNN`` (Eq. 1)."""
    dx, dy = _prepare(x, y, preprocess)
    return _mknn_from_d(dx, dy, k, None)


def mutual_knn_tau(
    x: object, y: object, k: int = 10, tau: float = 0.1, preprocess: str = "l2"
) -> float:
    """``S_mKNN^tau`` (Eq. 4)."""
    dx, dy = _prepare(x, y, preprocess)
    return _mknn_from_d(dx, dy, k, geometry_weights(dx, dy, tau))


def mst_edge_overlap(x: object, y: object, preprocess: str = "l2") -> float:
    """``S_H0`` (Eq. 2): shared MST edges over ``n - 1``."""
    dx, dy = _prepare(x, y, preprocess)
    return _h0_from_d(dx, dy, None)


def mst_edge_overlap_tau(
    x: object, y: object, tau: float = 0.1, preprocess: str = "l2"
) -> float:
    """``S_H0^tau`` (Eq. 5)."""
    dx, dy = _prepare(x, y, preprocess)
    return _h0_from_d(dx, dy, geometry_weights(dx, dy, tau))


def relational_statistics(
    x: object,
    y: object,
    k: int = 10,
    tau: float = 0.1,
    preprocess: str = "l2",
) -> dict[str, float | int]:
    """All four statistics plus zero-distance pair counts for one pair.

    Raises ``ValueError`` if either representation fails the section 4
    representation check (callers exclude the seed in that case).
    """
    dx, dy = _prepare(x, y, preprocess)
    w = geometry_weights(dx, dy, tau)
    return {
        "S_mKNN": _mknn_from_d(dx, dy, k, None),
        "S_H0": _h0_from_d(dx, dy, None),
        "S_mKNN_tau": _mknn_from_d(dx, dy, k, w),
        "S_H0_tau": _h0_from_d(dx, dy, w),
        "zero_distance_pairs_x": zero_distance_pairs(dx),
        "zero_distance_pairs_y": zero_distance_pairs(dy),
    }


def calibrated_difference(
    trained: Mapping[str, float], null: Mapping[str, float]
) -> dict[str, float]:
    """``D = S(trained) - S(null)`` per shared key (never a ratio)."""
    return {key: float(trained[key]) - float(null[key]) for key in trained if key in null}


# --- 1-NN comparator (exploratory, descriptive) ---------------------------


def nn_edges(d: Array) -> set[tuple[int, int]]:
    """Unordered 1-NN edges ``{i, nn(i)}`` (``i < j``), ties -> lower index."""
    nn = knn_indices(d, 1)[:, 0]
    return {(min(i, int(j)), max(i, int(j))) for i, j in enumerate(nn)}


def nn_comparator(x: object, y: object, preprocess: str = "l2") -> dict[str, float]:
    """1-NN comparator for ``S_H0`` (Shi et al., arXiv:2610.20286).

    * ``S_mKNN_k1``: ``(1/n) sum_i 1[nn_X(i) = nn_Y(i)]`` (mutual-kNN
      at k = 1, same tie rule).
    * ``S_H0_res``: ``|(T_X & T_Y) - (E_NN(X) | E_NN(Y))| / (n - 1)``,
      the shared MST edges that are 1-NN edges in neither
      representation.

    Descriptive only: not part of the confirmatory family.
    """
    dx, dy = _prepare(x, y, preprocess)
    n = int(dx.shape[0])
    shared = prim_mst_edges(dx) & prim_mst_edges(dy)
    residual = shared - (nn_edges(dx) | nn_edges(dy))
    return {
        "S_mKNN_k1": _mknn_from_d(dx, dy, 1, None),
        "S_H0_res": len(residual) / (n - 1),
    }
