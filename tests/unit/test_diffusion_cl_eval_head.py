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
