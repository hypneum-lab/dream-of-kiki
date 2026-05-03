"""Unit tests for EsnnG5TerSpikingCNN."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g5_ter_spiking_cnn.spiking_cnn import (
    EsnnG5TerSpikingCNN,
    avg_pool_4x4,
    avg_pool_4x4_backward,
    conv2d_backward,
    conv2d_forward,
)


def _make_clf(seed: int = 0) -> EsnnG5TerSpikingCNN:
    return EsnnG5TerSpikingCNN(n_classes=2, seed=seed, n_steps=4)


def test_init_shapes() -> None:
    clf = _make_clf()
    assert clf.W_c1.shape == (3, 3, 3, 16)
    assert clf.W_c2.shape == (3, 3, 16, 32)
    assert clf.W_fc1.shape == (2048, 64)
    assert clf.W_out.shape == (64, 2)
    for b, n in (
        (clf.b_c1, 16),
        (clf.b_c2, 32),
        (clf.b_fc1, 64),
        (clf.b_out, 2),
    ):
        assert b.shape == (n,) and b.dtype == np.float32


def test_predict_logits_shape_and_empty() -> None:
    clf = _make_clf()
    x = np.random.default_rng(0).standard_normal(
        (5, 32, 32, 3)
    ).astype(np.float32)
    assert clf.predict_logits(x).shape == (5, 2)
    assert clf.predict_logits(
        np.zeros((0, 32, 32, 3), np.float32)
    ).shape == (0, 2)


def test_latent_shape() -> None:
    clf = _make_clf()
    x = np.random.default_rng(1).standard_normal(
        (3, 32, 32, 3)
    ).astype(np.float32)
    assert clf.latent(x).shape == (3, 64)


def test_eval_accuracy_determinism() -> None:
    clf = _make_clf()
    x = np.zeros((4, 32, 32, 3), dtype=np.float32)
    y = np.zeros((4,), dtype=np.int64)
    assert clf.eval_accuracy(x, y) == clf.eval_accuracy(x, y)


def test_train_task_changes_weights() -> None:
    clf = _make_clf()
    rng = np.random.default_rng(7)
    x = rng.standard_normal((16, 32, 32, 3)).astype(np.float32)
    y = rng.integers(0, 2, size=16, dtype=np.int64)
    w0 = clf.W_out.copy()
    clf.train_task(
        {"x_train": x, "y_train": y},
        epochs=1,
        batch_size=8,
        lr=0.01,
    )
    assert not np.allclose(w0, clf.W_out)


def test_seed_determinism() -> None:
    a, b = _make_clf(seed=42), _make_clf(seed=42)
    np.testing.assert_array_equal(a.W_c1, b.W_c1)
    np.testing.assert_array_equal(a.W_out, b.W_out)


def test_avg_pool_4x4_shape_and_backward() -> None:
    x = np.ones((2, 32, 32, 4), dtype=np.float32)
    p = avg_pool_4x4(x)
    assert p.shape == (2, 8, 8, 4) and np.allclose(p, 1.0)
    grad = avg_pool_4x4_backward(np.ones_like(p), x.shape)
    assert grad.shape == x.shape and np.allclose(grad, 1.0 / 16.0)


def test_conv2d_forward_backward_shapes() -> None:
    rng = np.random.default_rng(0)
    x = rng.standard_normal((1, 8, 8, 3)).astype(np.float32)
    W = rng.standard_normal((3, 3, 3, 4)).astype(np.float32)
    b = np.zeros(4, dtype=np.float32)
    y = conv2d_forward(x, W, b, padding=1)
    assert y.shape == (1, 8, 8, 4)
    dx, dW, db = conv2d_backward(np.ones_like(y), x, W, padding=1)
    assert dx.shape == x.shape
    assert dW.shape == W.shape
    assert db.shape == b.shape


def test_invalid_n_classes() -> None:
    with pytest.raises(ValueError):
        EsnnG5TerSpikingCNN(n_classes=1, seed=0)


def test_lif_population_rates_matches_unvectorised() -> None:
    """Vectorised LIF rates must match per-sample reference."""
    clf = EsnnG5TerSpikingCNN(n_classes=2, seed=0, n_steps=5)
    rng = np.random.default_rng(0)
    currents = rng.standard_normal((4, 16)).astype(np.float32)
    fast = clf._lif_population_rates(currents)
    ref = clf._lif_population_rates_unvectorised(currents)
    np.testing.assert_allclose(fast, ref, rtol=1e-6, atol=1e-6)
