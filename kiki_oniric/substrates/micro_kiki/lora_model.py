"""LoRA-adapter model abstraction (B1a, issue #15).

A `LoRALinear` layer carries a frozen base weight `W0` and a
trainable low-rank adapter pair `(A, B)`; its effective weight is
`W0 + (alpha/rank) * (B @ A)`. `LoRAModel` stacks named
`LoRALinear` layers and exposes only the adapters as the
trainable surface — so a downstream gradient step (sub-project
B1b: `replay`) touches the adapters and nothing else.

B1a builds the model only. No training, no channel output.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from numpy.typing import NDArray

__all__ = ["LoRALinear", "LoRAModel", "adapter_delta"]


class LoRALinear(nn.Module):  # type: ignore[name-defined,misc]  # mlx.nn dynamic
    """A LoRA-adapted linear layer.

    Base weight `W0` (and optional bias) are frozen; the rank-`r`
    adapters `lora_a` (r, in) and `lora_b` (out, r) are trainable.
    Standard LoRA init: `A` seeded-random, `B` zeros — so the
    initial effective weight equals `W0`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        *,
        bias: bool = True,
        key: mx.array | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.use_bias = bias

        if key is None:
            key = mx.random.key(0)
        k_base, k_a = mx.random.split(key, 2)

        bound = 1.0 / (in_features ** 0.5)
        self.base_weight = mx.random.uniform(
            low=-bound,
            high=bound,
            shape=(out_features, in_features),
            key=k_base,
        )
        frozen = ["base_weight"]
        if bias:
            self.bias = mx.zeros((out_features,))
            frozen.append("bias")

        self.lora_a = mx.random.normal(
            shape=(rank, in_features), key=k_a
        ) * bound
        self.lora_b = mx.zeros((out_features, rank))

        self.freeze(recurse=False, keys=frozen)

    def __call__(self, x: mx.array) -> mx.array:
        delta = self.scale * (self.lora_b @ self.lora_a)
        y = x @ (self.base_weight + delta).T
        if self.use_bias:
            y = y + self.bias
        return y  # type: ignore[no-any-return]  # mlx.nn dynamic


class LoRAModel(nn.Module):  # type: ignore[name-defined,misc]  # mlx.nn dynamic
    """A feed-forward stack of named `LoRALinear` layers.

    `layer_sizes` is the sequence of widths, e.g. `(4, 8, 2)` →
    two layers `layer0` (4→8) and `layer1` (8→2). ReLU is applied
    between layers, not after the last.
    """

    def __init__(
        self,
        layer_sizes: tuple[int, ...],
        rank: int,
        alpha: float,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError(
                f"layer_sizes needs >= 2 widths, got {layer_sizes}"
            )
        n = len(layer_sizes) - 1
        keys = mx.random.split(mx.random.key(seed), n)
        self.layers = [
            LoRALinear(
                layer_sizes[i],
                layer_sizes[i + 1],
                rank,
                alpha,
                key=keys[i],
            )
            for i in range(n)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < last:
                x = nn.relu(x)  # type: ignore[attr-defined]  # mlx.nn dynamic
        return x

    def adapter_parameters(self) -> dict[str, mx.array]:
        """Return ONLY the trainable adapter arrays, keyed by
        `layer<i>.lora_a` / `layer<i>.lora_b`. Base weights are
        excluded — this is the surface a B1b gradient step trains.

        The returned arrays are the layer attributes themselves,
        not copies — in-place mutations are visible to the model.
        """
        out: dict[str, mx.array] = {}
        for i, layer in enumerate(self.layers):
            out[f"layer{i}.lora_a"] = layer.lora_a
            out[f"layer{i}.lora_b"] = layer.lora_b
        return out


def adapter_delta(
    before: dict[str, mx.array],
    after: dict[str, mx.array],
) -> dict[str, NDArray[np.float32]]:
    """Per-adapter delta ``after - before`` as float32 numpy arrays.

    ``before`` / ``after`` are `LoRAModel.adapter_parameters()`
    snapshots taken around a gradient step; their key sets must
    match. The MLX→numpy conversion lives here (a `substrates/`
    module) so MLX-only op modules need not import numpy.
    """
    if before.keys() != after.keys():
        raise ValueError(
            "adapter_delta: before/after adapter keys differ"
        )
    return {
        k: np.asarray(after[k] - before[k], dtype=np.float32)
        for k in before
    }
