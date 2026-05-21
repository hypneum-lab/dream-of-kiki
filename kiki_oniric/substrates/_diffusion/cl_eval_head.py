"""1-layer classifier head + train/eval helpers for delta_acc.

The M7 substrate measures per-cell ``delta_acc`` = accuracy on the
task's val split AFTER the dream cycle minus accuracy BEFORE the
dream cycle, on a tiny 1-layer Linear head trained on the
substrate's latent space. Hyper-parameters pinned here so the
bench is byte-deterministic.

See docs/superpowers/specs/2026-05-21-m7-substrate-dr3-design.md
§3 D2.
"""
from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import mlx.nn as _nn
import mlx.optimizers as optim

# Mirror the Encoder's import pattern: pin a local ``nn`` alias typed
# as ``Any`` to avoid per-line ``type: ignore`` annotations (mlx.nn
# re-exports confuse mypy's star-import resolution).
nn: Any = cast(Any, _nn)

# Pinned hyper-parameters — do not vary per cell.
_HEAD_STEPS = 50
_HEAD_LR = 1e-2


class ClEvalHead(nn.Module):  # type: ignore[misc]
    """One ``nn.Linear`` from latent to class logits.

    Args:
        d_latent: Input latent dimensionality (must match Encoder.d_latent).
        n_classes: Number of output classes (20 for CIFAR-100 task window).
    """

    def __init__(self, d_latent: int, n_classes: int) -> None:
        super().__init__()
        if d_latent <= 0:
            raise ValueError(f"d_latent must be positive, got {d_latent}")
        if n_classes <= 0:
            raise ValueError(f"n_classes must be positive, got {n_classes}")
        self.d_latent = d_latent
        self.n_classes = n_classes
        self.linear = nn.Linear(d_latent, n_classes)

    def __call__(self, z: mx.array) -> mx.array:
        """Project ``z`` (batch, d_latent) → logits (batch, n_classes)."""
        out: mx.array = self.linear(z)
        return out


def train_head_inplace(
    head: ClEvalHead,
    latents: mx.array,
    labels: mx.array,
    *,
    steps: int = _HEAD_STEPS,
    lr: float = _HEAD_LR,
) -> None:
    """Train the head in-place on ``(latents, labels)``.

    Mutates ``head``; deterministic given a fixed MLX seed upstream.
    50-step SGD on cross-entropy loss, hyper-parameters pinned in-module.
    """
    optimizer = optim.SGD(learning_rate=lr)

    def loss_fn(
        h: ClEvalHead, z: mx.array, y: mx.array
    ) -> mx.array:
        logits = h(z)
        loss: mx.array = nn.losses.cross_entropy(logits, y, reduction="mean")
        return loss

    loss_and_grad = nn.value_and_grad(head, loss_fn)
    for _ in range(steps):
        _, grads = loss_and_grad(head, latents, labels)
        optimizer.update(head, grads)
        mx.eval(head.parameters())


def eval_head_accuracy(
    head: ClEvalHead, latents: mx.array, labels: mx.array
) -> float:
    """Evaluate classification accuracy of ``head`` on ``(latents, labels)``.

    Returns:
        Accuracy in [0.0, 1.0].
    """
    logits = head(latents)
    preds = mx.argmax(logits, axis=-1)
    eq: mx.array = cast(mx.array, preds == labels)
    correct = eq.astype(mx.float32)
    return float(mx.mean(correct).item())
