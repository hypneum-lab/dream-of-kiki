"""Conformance tests for the HMM dream-state alignment module
(cycle-3 C3.16 — Phase 2 track c).

These tests run WITHOUT ``hmmlearn`` installed — the aligner has a
pure scipy + numpy EM fallback so the test path is independent of
the optional dependency. The ``hmmlearn`` branch is exercised only
when the env has it (``pragma: no cover`` on the import-dependent
code).

References :
- docs/interfaces/fmri-schema.yaml (v0.7.0+PARTIAL, schema)
- harness/fmri/studyforrest.py (C3.15 BOLD loader)
- framework-C spec §6.2 (DR-3 condition 2), §3 track (c)
"""
from __future__ import annotations

import numpy as np

from kiki_oniric.eval.state_alignment import (
    HmmAligner,
    StateCorrespondence,
)


def _sample_gaussian_hmm(
    transition: np.ndarray,
    means: np.ndarray,
    cov_scale: float,
    n_frames: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a synthetic Gaussian HMM with planted parameters.

    Returns (observations, true_states).
    """
    rng = np.random.default_rng(seed)
    n_states, n_features = means.shape
    states = np.empty(n_frames, dtype=int)
    obs = np.empty((n_frames, n_features), dtype=float)
    states[0] = 0
    obs[0] = rng.normal(means[0], cov_scale, size=n_features)
    for t in range(1, n_frames):
        states[t] = rng.choice(n_states, p=transition[states[t - 1]])
        obs[t] = rng.normal(means[states[t]], cov_scale, size=n_features)
    return obs, states


def test_hmm_fit_recovers_planted_transition() -> None:
    """TDD-1 — HmmAligner.fit converges on synthetic Gaussian HMM
    data (3 states, well-separated means) and recovers a state
    assignment consistent with the planted transition matrix.
    """
    planted = np.array(
        [
            [0.80, 0.15, 0.05],
            [0.10, 0.80, 0.10],
            [0.10, 0.10, 0.80],
        ]
    )
    means = np.array(
        [
            [-6.0, -6.0],
            [0.0, 0.0],
            [6.0, 6.0],
        ]
    )
    X, true_states = _sample_gaussian_hmm(
        transition=planted,
        means=means,
        cov_scale=0.4,
        n_frames=600,
        seed=0,
    )

    aligner = HmmAligner(n_states=3, n_iter_max=60, seed=0)
    result = aligner.fit(X)

    assert isinstance(result, StateCorrespondence)
    assert result.state_labels.shape == (600,)
    assert result.n_states == 3

    # With well-separated Gaussians the per-frame classification is
    # recoverable up to label permutation. Compute best-permutation
    # agreement — for 3 states we enumerate the 6 permutations.
    from itertools import permutations

    best = 0.0
    for perm in permutations(range(3)):
        remapped = np.array([perm[s] for s in result.state_labels])
        best = max(best, float(np.mean(remapped == true_states)))
    assert best > 0.85, (
        f"best-permutation recovery {best:.3f} below 0.85 "
        "— planted HMM not tracked"
    )


def test_hmm_align_sequences_returns_procrustes_rotation() -> None:
    """TDD-2 — align_sequences on two Procrustes-related sequences
    recovers an orthogonal rotation that aligns source onto target.
    """
    rng = np.random.default_rng(seed=1)
    latent = rng.standard_normal((200, 3))

    # Planted rotation — orthogonal by construction (QR decomp)
    A = rng.standard_normal((3, 3))
    Q, _ = np.linalg.qr(A)
    planted_rotation = Q

    X_source = latent
    X_target = latent @ planted_rotation

    aligner = HmmAligner(n_states=3, seed=0)
    recovered = aligner.align_sequences(X_source, X_target)

    # Orthogonality : R R^T ≈ I
    eye = recovered @ recovered.T
    np.testing.assert_allclose(eye, np.eye(3), atol=1e-6)
    # Rotation aligns X_source to X_target (within tolerance)
    aligned = X_source @ recovered
    err = float(np.mean((aligned - X_target) ** 2))
    assert err < 1e-6, f"Procrustes reconstruction error {err:.3e}"


def test_hmm_fit_is_deterministic_under_seed() -> None:
    """TDD-3 — R1 contract : the same (X, seed, n_states, n_iter_max)
    produces bit-stable state_labels and log_likelihood across
    repeated fits.
    """
    rng = np.random.default_rng(seed=2)
    X = rng.standard_normal((150, 4))

    r_a = HmmAligner(n_states=3, n_iter_max=40, seed=7).fit(X)
    r_b = HmmAligner(n_states=3, n_iter_max=40, seed=7).fit(X)

    np.testing.assert_array_equal(r_a.state_labels, r_b.state_labels)
    assert r_a.log_likelihood == r_b.log_likelihood
    assert r_a.bic == r_b.bic
    # Different seed on random init → different result in general
    r_c = HmmAligner(n_states=3, n_iter_max=40, seed=99).fit(X)
    # At minimum, determinism is per-seed (two fits with seed=7
    # match). We do not require seed=99 to diverge — on some
    # random inputs the same local optimum is reached.
    assert r_c.n_states == 3


def test_state_correspondence_exposes_bic_ll_labels() -> None:
    """TDD-4 — the dataclass exposes the attributes downstream
    harness code (C3.17 CCA, paper reporters) needs.
    """
    rng = np.random.default_rng(seed=3)
    X = rng.standard_normal((80, 2))

    result = HmmAligner(n_states=2, n_iter_max=20, seed=0).fit(X)

    assert hasattr(result, "rotation")
    assert hasattr(result, "state_labels")
    assert hasattr(result, "log_likelihood")
    assert hasattr(result, "bic")
    assert hasattr(result, "n_states")

    assert result.rotation.shape == (2, 2)
    assert result.state_labels.shape == (80,)
    assert isinstance(result.log_likelihood, float)
    assert isinstance(result.bic, float)
    # BIC is finite for a valid fit ; log-likelihood must be <= 0
    # only if the data are on a probability density scale — we
    # just require it finite here.
    assert np.isfinite(result.log_likelihood)
    assert np.isfinite(result.bic)
    # state_labels are integer indices in [0, n_states)
    assert result.state_labels.dtype.kind == "i"
    assert int(result.state_labels.min()) >= 0
    assert int(result.state_labels.max()) < 2
