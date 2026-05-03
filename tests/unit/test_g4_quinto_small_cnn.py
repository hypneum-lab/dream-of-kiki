"""Unit tests for G4SmallCNN (Conv2d x2 + MaxPool2d x2 + Linear x2)."""
from __future__ import annotations

import numpy as np
import pytest

from experiments.g4_quinto_test.small_cnn import G4SmallCNN


def _cnn() -> G4SmallCNN:
    return G4SmallCNN(latent_dim=64, n_classes=2, seed=0)


def _batch() -> np.ndarray:
    return (
        np.random.default_rng(0)
        .standard_normal((4, 32, 32, 3))
        .astype(np.float32)
    )


def test_forward_shape_2_classes() -> None:
    assert _cnn().predict_logits(_batch()).shape == (4, 2)


def test_latent_shape() -> None:
    assert _cnn().latent(_batch()).shape == (4, 64)


def test_restructure_step_perturbs_conv2_only() -> None:
    cnn = _cnn()
    w1 = np.asarray(cnn._conv1.weight).copy()
    w2 = np.asarray(cnn._conv2.weight).copy()
    cnn.restructure_step(factor=0.05, seed=42)
    np.testing.assert_array_equal(np.asarray(cnn._conv1.weight), w1)
    assert not np.allclose(np.asarray(cnn._conv2.weight), w2)


def test_downscale_multiplies_all_weights() -> None:
    cnn = _cnn()
    w = np.asarray(cnn._fc2.weight).copy()
    cnn.downscale_step(factor=0.5)
    np.testing.assert_allclose(
        np.asarray(cnn._fc2.weight), 0.5 * w, rtol=1e-6
    )


def test_downscale_bounds_reject() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\]"):
        _cnn().downscale_step(factor=0.0)


def test_recombine_step_with_two_classes_runs() -> None:
    rng = np.random.default_rng(0)
    latents = [
        (rng.standard_normal(64).astype(np.float32).tolist(), c)
        for c in (0, 0, 1, 1)
    ]
    _cnn().recombine_step(
        latents=latents, n_synthetic=8, lr=0.01, seed=0
    )


def test_recombine_step_empty_is_noop() -> None:
    cnn = _cnn()
    w = np.asarray(cnn._fc2.weight).copy()
    cnn.recombine_step(latents=[], n_synthetic=8, lr=0.01, seed=0)
    np.testing.assert_array_equal(np.asarray(cnn._fc2.weight), w)
