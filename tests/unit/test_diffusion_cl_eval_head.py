"""Small CL classifier head for diffusion-substrate delta_acc (M7 D2)."""

from __future__ import annotations

import mlx.core as mx

from kiki_oniric.substrates._diffusion.cl_eval_head import (
    ClEvalHead,
    eval_head_accuracy,
)


def test_cl_eval_head_shapes() -> None:
    head = ClEvalHead(d_latent=64, n_classes=20)
    z = mx.zeros((8, 64))
    logits = head(z)
    assert logits.shape == (8, 20)


def test_eval_head_accuracy_in_unit_range() -> None:
    head = ClEvalHead(d_latent=64, n_classes=20)
    z = mx.zeros((16, 64))
    y = mx.zeros((16,), dtype=mx.int32)
    acc = eval_head_accuracy(head, z, y)
    assert 0.0 <= acc <= 1.0


def test_denoiser_feature_shape_and_determinism() -> None:
    """denoiser_feature returns (batch, d_latent) and is deterministic."""
    from kiki_oniric.substrates._diffusion.cl_eval_head import (
        denoiser_feature,
    )
    from kiki_oniric.substrates._diffusion.model import MLPDenoiser

    d = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    z = mx.zeros((5, 4))
    out_a = denoiser_feature(d, z, t_fixed=5)
    out_b = denoiser_feature(d, z, t_fixed=5)

    assert out_a.shape == (5, 4)
    import numpy as np
    np.testing.assert_allclose(
        np.asarray(out_a), np.asarray(out_b), atol=0.0,
    )
