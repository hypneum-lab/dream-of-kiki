"""Unit tests for the G5-ter dream-episode wrapper."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g5_ter_spiking_cnn.dream_wrap_cnn import (
    EsnnCNNBetaBuffer,
    _downscale_step,
    _recombine_step,
    _replay_step,
    _restructure_step,
    build_esnn_cnn_profile,
    dream_episode_cnn_esnn,
)
from experiments.g5_ter_spiking_cnn.spiking_cnn import (
    EsnnG5TerSpikingCNN,
)


def _make_clf() -> EsnnG5TerSpikingCNN:
    return EsnnG5TerSpikingCNN(n_classes=2, seed=0, n_steps=2)


def test_buffer_capacity_fifo() -> None:
    buf = EsnnCNNBetaBuffer(capacity=2)
    for k in range(3):
        buf.push(
            x=np.full((32, 32, 3), float(k), dtype=np.float32),
            y=k % 2,
            latent=None,
        )
    snap = buf.snapshot()
    assert len(snap) == 2 and snap[0]["y"] == 1


def test_buffer_capacity_zero_raises() -> None:
    with pytest.raises(ValueError):
        EsnnCNNBetaBuffer(capacity=0)


def test_replay_no_op_empty() -> None:
    clf = _make_clf()
    w0 = clf.W_c1.copy()
    _replay_step(clf, [], lr=0.1, n_steps=1)
    np.testing.assert_array_equal(clf.W_c1, w0)


def test_downscale_one_noop_then_half() -> None:
    clf = _make_clf()
    w0 = clf.W_c1.copy()
    _downscale_step(clf, factor=1.0)
    np.testing.assert_array_equal(clf.W_c1, w0)
    _downscale_step(clf, factor=0.5)
    np.testing.assert_allclose(clf.W_c1, 0.5 * w0)


def test_downscale_invalid_bounds() -> None:
    clf = _make_clf()
    with pytest.raises(ValueError):
        _downscale_step(clf, factor=0.0)
    with pytest.raises(ValueError):
        _downscale_step(clf, factor=1.5)


def test_restructure_zero_noop_else_W_c2_only() -> None:
    clf = _make_clf()
    w_c1_0, w_c2_0, w_out_0 = (
        clf.W_c1.copy(),
        clf.W_c2.copy(),
        clf.W_out.copy(),
    )
    _restructure_step(clf, factor=0.0, seed=0)
    np.testing.assert_array_equal(clf.W_c2, w_c2_0)
    _restructure_step(clf, factor=0.1, seed=42)
    assert not np.allclose(clf.W_c2, w_c2_0)
    np.testing.assert_array_equal(clf.W_c1, w_c1_0)
    np.testing.assert_array_equal(clf.W_out, w_out_0)


def test_recombine_empty_or_single_class_no_op() -> None:
    clf = _make_clf()
    w0 = clf.W_out.copy()
    _recombine_step(clf, latents=[], n_synthetic=4, lr=0.1, seed=0)
    np.testing.assert_array_equal(clf.W_out, w0)
    _recombine_step(
        clf,
        latents=[([0.0] * 64, 0), ([0.1] * 64, 0)],
        n_synthetic=4,
        lr=0.1,
        seed=0,
    )
    np.testing.assert_array_equal(clf.W_out, w0)


def test_dream_episode_logs_one_entry_per_call() -> None:
    clf = _make_clf()
    profile = build_esnn_cnn_profile("P_equ", seed=0)
    buf = EsnnCNNBetaBuffer(capacity=4)
    rng = np.random.default_rng(0)
    for _ in range(4):
        buf.push(
            x=rng.standard_normal((32, 32, 3)).astype(np.float32),
            y=int(rng.integers(0, 2)),
            latent=rng.standard_normal(64).astype(np.float32),
        )
    n0 = len(profile.runtime.log)
    dream_episode_cnn_esnn(
        clf,
        profile,
        seed=0,
        beta_buffer=buf,
        replay_n_records=2,
        replay_n_steps=1,
        replay_lr=0.01,
        downscale_factor=0.99,
        restructure_factor=0.01,
        recombine_n_synthetic=4,
        recombine_lr=0.01,
    )
    assert len(profile.runtime.log) == n0 + 1
