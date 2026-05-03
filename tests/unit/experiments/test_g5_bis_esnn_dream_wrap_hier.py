"""Unit tests for the G5-bis dream wrapper + buffer (Plan G5-bis Task 3)."""
from __future__ import annotations

import numpy as np

from experiments.g5_bis_richer_esnn.esnn_dream_wrap_hier import (
    EsnnHierBetaBuffer,
    build_esnn_richer_profile,
    dream_episode_hier_esnn,
)
from experiments.g5_bis_richer_esnn.esnn_hier_classifier import (
    EsnnG5BisHierarchicalClassifier,
)


def _build_clf(seed: int = 0) -> EsnnG5BisHierarchicalClassifier:
    return EsnnG5BisHierarchicalClassifier(
        in_dim=4,
        hidden_1=8,
        hidden_2=6,
        n_classes=2,
        seed=seed,
        n_steps=3,
    )


def test_buffer_push_and_snapshot_roundtrip() -> None:
    buf = EsnnHierBetaBuffer(capacity=4)
    x = np.zeros(4, dtype=np.float32)
    latent = np.array([0.5, 0.6], dtype=np.float32)
    buf.push(x=x, y=0, latent=latent)
    snap = buf.snapshot()
    assert len(snap) == 1
    assert snap[0]["y"] == 0
    np.testing.assert_allclose(snap[0]["latent"], [0.5, 0.6], rtol=1e-6)


def test_buffer_capacity_evicts_fifo() -> None:
    buf = EsnnHierBetaBuffer(capacity=2)
    x = np.zeros(4, dtype=np.float32)
    for label in range(3):
        buf.push(x=x, y=label, latent=None)
    snap = buf.snapshot()
    assert [r["y"] for r in snap] == [1, 2]


def test_dream_episode_runs_and_mutates_weights() -> None:
    clf = _build_clf(seed=0)
    buf = EsnnHierBetaBuffer(capacity=16)
    rng = np.random.default_rng(0)
    for label in range(8):
        buf.push(
            x=rng.standard_normal(4).astype(np.float32),
            y=label % 2,
            latent=rng.standard_normal(6).astype(np.float32),
        )
    w_in_before = clf.W_in.copy()
    w_h_before = clf.W_h.copy()
    w_out_before = clf.W_out.copy()
    profile = build_esnn_richer_profile("P_equ", seed=0)
    dream_episode_hier_esnn(
        clf,
        profile,
        seed=0,
        beta_buffer=buf,
        replay_n_records=4,
        replay_n_steps=2,
        replay_lr=0.01,
        downscale_factor=0.95,
        restructure_factor=0.05,
        recombine_n_synthetic=4,
        recombine_lr=0.01,
    )
    assert not np.allclose(clf.W_in, w_in_before)
    assert not np.allclose(clf.W_h, w_h_before)
    assert not np.allclose(clf.W_out, w_out_before)


def test_dream_episode_p_min_skips_restructure_recombine() -> None:
    clf = _build_clf(seed=0)
    buf = EsnnHierBetaBuffer(capacity=8)
    rng = np.random.default_rng(0)
    for label in range(4):
        buf.push(
            x=rng.standard_normal(4).astype(np.float32),
            y=label % 2,
            latent=rng.standard_normal(6).astype(np.float32),
        )
    w_h_before = clf.W_h.copy()
    profile = build_esnn_richer_profile("P_min", seed=0)
    dream_episode_hier_esnn(
        clf,
        profile,
        seed=0,
        beta_buffer=buf,
        replay_n_records=4,
        replay_n_steps=2,
        replay_lr=0.0,  # disables REPLAY mutation
        downscale_factor=0.5,
        restructure_factor=0.05,
        recombine_n_synthetic=4,
        recombine_lr=0.01,
    )
    # P_min with replay_lr=0 -> only DOWNSCALE mutates W_h ; with
    # factor=0.5 the resulting W_h equals 0.5 * W_h_before exactly.
    np.testing.assert_allclose(clf.W_h, 0.5 * w_h_before)
