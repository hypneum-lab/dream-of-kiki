"""Random-init MLP decoder for the recombine VAE-shape contract.

The diffusion substrate's denoiser is *not* a VAE decoder — it
predicts noise. The recombine handler's Protocol (defined in
``kiki_oniric/dream/operations/recombine_real.py``) expects a
``VAEDecoder.__call__(z) -> mx.array`` mapping latents back to
raw features. This module provides a minimal random-init MLP that
satisfies the Protocol *shape* without claiming any reconstruction
fidelity — see ``docs/superpowers/specs/2026-05-21-m7-substrate-
dr3-design.md`` §3 D7.
"""
from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import mlx.nn as _nn

# Mirror the Encoder's import pattern: pin a local ``nn`` alias typed
# as ``Any`` to avoid per-line ``type: ignore`` annotations (mlx.nn
# re-exports confuse mypy's star-import resolution).
nn: Any = cast(Any, _nn)


class Decoder(nn.Module):  # type: ignore[misc]
    """``d_latent -> d_out`` MLP. Two hidden layers, GELU.

    Args:
        d_latent: Input latent dimensionality (must match Encoder.d_latent).
        d_out: Output dimensionality (3072 for CIFAR-100 flat features).
        d_hidden: Hidden layer width. Default 256.
    """

    def __init__(
        self, d_latent: int, d_out: int, d_hidden: int = 256
    ) -> None:
        super().__init__()
        if d_latent <= 0:
            raise ValueError(f"d_latent must be positive, got {d_latent}")
        if d_out <= 0:
            raise ValueError(f"d_out must be positive, got {d_out}")
        if d_hidden <= 0:
            raise ValueError(f"d_hidden must be positive, got {d_hidden}")
        self.d_latent = d_latent
        self.d_out = d_out
        self.d_hidden = d_hidden
        self.up = nn.Linear(d_latent, d_hidden)
        self.mid = nn.Linear(d_hidden, d_hidden)
        self.out_proj = nn.Linear(d_hidden, d_out)

    def __call__(self, z: mx.array) -> mx.array:
        """Project ``z`` (batch, d_latent) → (batch, d_out)."""
        if z.ndim < 1:
            raise ValueError(
                f"Decoder expects at least 1-D input, got ndim={z.ndim}"
            )
        if z.ndim == 1:
            z = z[None, :]
        h: mx.array = nn.gelu(self.up(z))
        h = nn.gelu(self.mid(h))
        out: mx.array = self.out_proj(h)
        return out
