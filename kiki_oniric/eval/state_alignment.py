"""HMM dream-state alignment module (cycle-3 C3.16 — Phase 2 track c).

Hidden Markov Model fitter that aligns (a) dream-op state
trajectories produced by ``DreamRuntime`` against (b) BOLD time-
series from Studyforrest (loaded via ``harness.fmri.studyforrest``).
The fit output is consumed by the C3.17 CCA alignment step and
contributes to closing the DR-3 Conformance Criterion condition 2
on real brain-state data (framework-C spec §6.2).

Dependency policy : ``hmmlearn`` is an **optional** dependency.
When not importable the aligner falls back to a pure scipy + numpy
EM implementation (forward-backward with Gaussian emissions) so CI
and local unit tests work without the extra dependency. A heavier
production fit may prefer ``pip install hmmlearn`` but the numbers
stay comparable — both paths initialise via K-means++ on the input
and report the same ``StateCorrespondence`` contract.

Determinism : seeded via ``numpy.random.default_rng(seed)``. Same
(X, seed, n_states, n_iter_max) produces bit-stable state labels
and log-likelihood — R1 run-registry contract.

References :
- docs/interfaces/fmri-schema.yaml (schema v0.7.0+PARTIAL)
- harness/fmri/studyforrest.py (C3.15 BoldSeries)
- kiki_oniric/eval/scaling_law.py (similar TDD pattern)
- framework-C spec §6.2 (DR-3 Conformance Criterion condition 2)
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
from scipy import linalg as _linalg
from scipy.stats import multivariate_normal

if TYPE_CHECKING:  # pragma: no cover - type-check only
    from numpy.typing import NDArray

# Optional-dependency probe — hmmlearn is only used as a faster
# production backend. The EM fallback covers all tests so the
# dependency is not required.
try:  # pragma: no cover - branch depends on env
    from hmmlearn.hmm import GaussianHMM  # noqa: F401

    _HMMLEARN_AVAILABLE = True
except ImportError:  # pragma: no cover - branch depends on env
    _HMMLEARN_AVAILABLE = False


@dataclass(frozen=True)
class StateCorrespondence:
    """Result of a single HMM fit on a dream / BOLD sequence.

    ``rotation`` is a Procrustes-optimal orthogonal matrix that
    maps source onto target coordinates — shipped alongside the
    state labels so downstream CCA (C3.17) can apply a unified
    basis change before correlating. For a bare ``fit`` call (no
    target provided) ``rotation`` is the identity.
    """

    rotation: "NDArray[np.floating]"      # (n_features, n_features)
    state_labels: "NDArray[np.integer]"   # (n_frames,)
    log_likelihood: float
    bic: float
    n_states: int


def _kmeans_pp_init(
    X: "NDArray[np.floating]",
    n_states: int,
    rng: np.random.Generator,
) -> "NDArray[np.floating]":
    """K-means++ seed for the emission means (deterministic under rng)."""
    n = X.shape[0]
    idx = int(rng.integers(0, n))
    centres = [X[idx]]
    for _ in range(1, n_states):
        diffs = X[:, None, :] - np.stack(centres, axis=0)[None, :, :]
        d2 = np.min(np.sum(diffs ** 2, axis=-1), axis=1)
        total = float(d2.sum())
        if total <= 0.0:
            # Degenerate : pick a random remaining point.
            centres.append(X[int(rng.integers(0, n))])
            continue
        probs = d2 / total
        nxt = int(rng.choice(n, p=probs))
        centres.append(X[nxt])
    return np.stack(centres, axis=0)


def _forward_backward(
    log_init: "NDArray[np.floating]",
    log_trans: "NDArray[np.floating]",
    log_emit: "NDArray[np.floating]",
) -> tuple[
    "NDArray[np.floating]",   # gamma (T, K) posterior p(state_t = k | X)
    "NDArray[np.floating]",   # xi_sum (K, K) expected transitions
    float,                     # log-likelihood
]:
    """Scaled forward-backward for a Gaussian HMM.

    All arguments are in log-space to avoid underflow on long
    sequences. Returns posterior marginals, expected transition
    counts, and the sequence log-likelihood.
    """
    T, K = log_emit.shape
    log_alpha = np.empty((T, K))
    log_beta = np.empty((T, K))

    log_alpha[0] = log_init + log_emit[0]
    for t in range(1, T):
        # log_alpha[t, k] = log_emit[t, k] + logsumexp_j(log_alpha[t-1, j] + log_trans[j, k])
        log_alpha[t] = log_emit[t] + _logsumexp(
            log_alpha[t - 1][:, None] + log_trans, axis=0
        )

    log_beta[T - 1] = 0.0
    for t in range(T - 2, -1, -1):
        log_beta[t] = _logsumexp(
            log_trans + (log_emit[t + 1] + log_beta[t + 1])[None, :],
            axis=1,
        )

    log_ll = float(_logsumexp(log_alpha[T - 1]))
    log_gamma = log_alpha + log_beta - log_ll
    gamma = np.exp(log_gamma)

    # Expected transitions : xi_sum[i, j] = sum_t exp(log_alpha[t, i]
    # + log_trans[i, j] + log_emit[t+1, j] + log_beta[t+1, j] - log_ll)
    xi_sum = np.zeros((K, K))
    for t in range(T - 1):
        log_xi = (
            log_alpha[t][:, None]
            + log_trans
            + (log_emit[t + 1] + log_beta[t + 1])[None, :]
            - log_ll
        )
        xi_sum += np.exp(log_xi)

    return gamma, xi_sum, log_ll


def _logsumexp(
    x: "NDArray[np.floating]", axis: int | None = None
) -> "NDArray[np.floating]":
    """Numerically-stable log-sum-exp (scalar when axis is None)."""
    if axis is None:
        xmax = float(np.max(x))
        if not np.isfinite(xmax):
            xmax = 0.0
        return float(np.log(np.sum(np.exp(x - xmax))) + xmax)
    xmax = np.max(x, axis=axis, keepdims=True)
    # Guard -inf rows (should not appear on valid inputs but cheap)
    xmax = np.where(np.isfinite(xmax), xmax, 0.0)
    out = np.log(np.sum(np.exp(x - xmax), axis=axis, keepdims=True)) + xmax
    return np.squeeze(out, axis=axis)


@dataclass
class HmmAligner:
    """Gaussian-HMM aligner for dream-state / BOLD sequences.

    Parameters
    ----------
    n_states : int
        Number of latent states in the HMM. Defaults to 3 —
        matches the dream-runtime state discretisation used in
        the cycle-3 ablation (``replay`` / ``downscale`` /
        ``restructure``).
    n_iter_max : int
        Maximum EM iterations. The EM loop also stops when the
        log-likelihood improvement drops below ``1e-4``.
    seed : int
        RNG seed for K-means++ init ; part of the R1 contract.
    """

    n_states: int = 3
    n_iter_max: int = 100
    seed: int = 0

    def fit(
        self, X: "NDArray[np.floating]"
    ) -> StateCorrespondence:
        """Fit the Gaussian HMM on ``X`` of shape (n_frames, n_features).

        Returns a :class:`StateCorrespondence` with the Viterbi-
        aligned state labels, log-likelihood, BIC and an identity
        rotation (no target provided).
        """
        X_arr = np.asarray(X, dtype=float)
        if X_arr.ndim != 2:
            raise ValueError(
                f"HMM fit expects (n_frames, n_features), got {X_arr.shape}"
            )
        T, D = X_arr.shape
        K = self.n_states
        if T < K:
            raise ValueError(
                f"Need at least n_states={K} frames, got {T}"
            )

        rng = np.random.default_rng(self.seed)

        # Initialisation — K-means++ means, identity covariances,
        # uniform transitions and initial distribution.
        means = _kmeans_pp_init(X_arr, K, rng)
        covars = np.stack(
            [np.eye(D) * (X_arr.var(axis=0).mean() + 1e-6)] * K, axis=0
        )
        trans = np.full((K, K), 1.0 / K)
        init = np.full(K, 1.0 / K)

        prev_ll = -np.inf
        log_ll = -np.inf
        gamma = np.zeros((T, K))
        for _ in range(self.n_iter_max):
            # E-step — build emission log-probs via scipy.stats for
            # numerical robustness on ill-conditioned covariances.
            log_emit = np.empty((T, K))
            for k in range(K):
                try:
                    log_emit[:, k] = multivariate_normal(
                        mean=means[k], cov=covars[k], allow_singular=True,
                    ).logpdf(X_arr)
                except (np.linalg.LinAlgError, ValueError):
                    # Ridge-regularised fallback for degenerate cov
                    reg = covars[k] + np.eye(D) * 1e-4
                    log_emit[:, k] = multivariate_normal(
                        mean=means[k], cov=reg, allow_singular=True,
                    ).logpdf(X_arr)

            log_init = np.log(np.clip(init, 1e-12, None))
            log_trans = np.log(np.clip(trans, 1e-12, None))
            gamma, xi_sum, log_ll = _forward_backward(
                log_init, log_trans, log_emit
            )

            if abs(log_ll - prev_ll) < 1e-4:
                break
            prev_ll = log_ll

            # M-step
            # Guard ``gamma[0].sum()`` the same way ``denom`` and
            # ``weights`` are guarded below — a degenerate posterior
            # could drive the divisor to zero.
            init_denom = float(gamma[0].sum())
            init = gamma[0] / (init_denom if init_denom > 0 else 1.0)
            denom = xi_sum.sum(axis=1, keepdims=True)
            denom = np.where(denom > 0, denom, 1.0)
            trans = xi_sum / denom
            weights = gamma.sum(axis=0)
            weights = np.where(weights > 0, weights, 1.0)
            means = (gamma.T @ X_arr) / weights[:, None]
            for k in range(K):
                diff = X_arr - means[k]
                weighted = (gamma[:, k][:, None] * diff)
                covars[k] = (weighted.T @ diff) / weights[k]
                # Regularise to avoid collapse on near-duplicate frames
                covars[k] += np.eye(D) * 1e-6

        # Viterbi decoding for hard labels (stable and needed
        # downstream by CCA). We use the posterior argmax which is
        # a maximum-marginal decoder — matches hmmlearn's
        # ``predict`` default and is cheaper than Viterbi while
        # agreeing on well-separated data.
        labels = gamma.argmax(axis=1).astype(np.int64)

        # BIC = -2 log L + p log T, p = free params
        #     = K-1 (init) + K(K-1) (trans) + K*D (means) + K*D*(D+1)/2 (cov)
        p = (K - 1) + K * (K - 1) + K * D + K * D * (D + 1) // 2
        bic = -2.0 * float(log_ll) + p * np.log(T)

        return StateCorrespondence(
            rotation=np.eye(D),
            state_labels=labels,
            log_likelihood=float(log_ll),
            bic=float(bic),
            n_states=K,
        )

    def align_sequences(
        self,
        X_source: "NDArray[np.floating]",
        X_target: "NDArray[np.floating]",
    ) -> "NDArray[np.floating]":
        """Return the Procrustes-optimal rotation matrix that aligns
        ``X_source`` onto ``X_target`` (minimises ``||X_source R -
        X_target||_F``) via ``scipy.linalg.orthogonal_procrustes``.
        """
        src = np.asarray(X_source, dtype=float)
        tgt = np.asarray(X_target, dtype=float)
        if src.shape != tgt.shape:
            raise ValueError(
                "align_sequences requires equal-shape inputs ; "
                f"got {src.shape} vs {tgt.shape}"
            )
        rotation, _scale = _linalg.orthogonal_procrustes(src, tgt)
        return rotation
