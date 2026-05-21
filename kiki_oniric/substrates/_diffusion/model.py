"""MLX latent-diffusion building blocks — skeleton (Wave 3b M2).

This module defines the three core building blocks of the
latent-diffusion substrate at **skeleton level**: typed signatures
only, no training, no learned weights.

- ``Encoder`` (E) — maps awake activations ``x`` to a latent
  vector ``z``. Skeleton implementation returns a deterministic
  zero-projection of the correct shape (no learned parameters
  yet — those land in Wave 3b M3 alongside the trainer).
- ``MLPDenoiser`` (U) — denoises a noisy latent ``z_t`` at
  diffusion step ``t``. Skeleton implementation returns
  ``zeros_like(z)`` to honour the typed signature without
  committing to a training recipe.
- ``NoiseSchedule`` (σ) — analytic linear-β schedule. ``sigma(t)``
  and ``alpha_bar(t)`` ARE implemented (no learned weights
  involved) so the §3.2 typing-table claim "7/8 typed"
  is testable on real arrays in M3+.

Methods that depend on M3 artefacts (training, sampling)
raise ``NotImplementedError`` with a pointer to the milestone
where they land. This is a deliberate contract: callers can
import the classes and verify Protocol conformance today, but
any attempt to *train* or *sample* surfaces the missing
milestone explicitly rather than silently returning placeholder
numbers.

References:
- ``docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md``
  §3.1 (module layout) + §3.2 (typing table)
- ``kiki_oniric/core/primitives.py`` (typed Protocols this
  skeleton will eventually satisfy)
"""
from __future__ import annotations

import mlx.core as mx


class Encoder:
    """E — Awake-side encoder: ``x`` (input batch) → ``z`` (latent).

    Wave 3b M2 skeleton: the forward pass returns a deterministic
    zero-projection of shape ``(batch, d_latent)``. No learned
    weights, no training. The class exists so M3 can layer the
    real linear+nonlinearity weights on top while keeping the
    public signature ``__call__(x: mx.array) -> mx.array`` stable.

    Args:
        d_in: Input activation dimensionality.
        d_latent: Output latent dimensionality. The plan's
            D2 default is ``64`` and is enforced by the public
            ``MLXLatentDiffusionSubstrate`` adapter.
    """

    def __init__(self, d_in: int, d_latent: int) -> None:
        if d_in <= 0:
            raise ValueError(f"d_in must be positive, got {d_in}")
        if d_latent <= 0:
            raise ValueError(f"d_latent must be positive, got {d_latent}")
        self.d_in = d_in
        self.d_latent = d_latent

    def __call__(self, x: mx.array) -> mx.array:
        """Return a zero latent of shape ``(batch, d_latent)``.

        Skeleton-only: training, real linear projection, and
        nonlinearity land in Wave 3b M3 (`_diffusion/trainer.py`).
        """
        if x.ndim < 1:
            raise ValueError(
                f"Encoder expects at least 1-D input, got ndim={x.ndim}"
            )
        batch = x.shape[0] if x.ndim > 1 else 1
        return mx.zeros((batch, self.d_latent), dtype=mx.float32)

    def train_step(self, *args: object, **kwargs: object) -> mx.array:
        """Not implemented in M2.

        Training lands in Wave 3b M3 alongside
        ``_diffusion/trainer.py`` (plan §4 M3).
        """
        raise NotImplementedError(
            "Encoder.train_step is deferred to Wave 3b M3 "
            "(_diffusion/trainer.py). M2 is skeleton-only."
        )


class MLPDenoiser:
    """U — MLP denoiser network ``(z_t, t) → ẑ_0``.

    Wave 3b M2 skeleton: the forward pass returns
    ``mx.zeros_like(z_t)``. The class records ``d_latent``,
    ``d_hidden`` and ``n_layers`` so M3 can materialize the
    real MLP weights without changing the public signature
    ``__call__(z: mx.array, t: mx.array) -> mx.array``.

    Args:
        d_latent: Latent dimensionality (must match ``Encoder.d_latent``).
        d_hidden: Hidden layer width.
        n_layers: Number of hidden layers. Plan D2 default = 3.
    """

    def __init__(
        self,
        d_latent: int,
        d_hidden: int,
        n_layers: int = 3,
    ) -> None:
        if d_latent <= 0:
            raise ValueError(f"d_latent must be positive, got {d_latent}")
        if d_hidden <= 0:
            raise ValueError(f"d_hidden must be positive, got {d_hidden}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.n_layers = n_layers

    def __call__(self, z: mx.array, t: mx.array) -> mx.array:
        """Return a zero denoised latent of shape ``z.shape``.

        Skeleton-only: real weights + denoising trajectory land
        in Wave 3b M3 (`_diffusion/trainer.py` +
        `_diffusion/sampler.py`).
        """
        if z.ndim < 1:
            raise ValueError(
                f"MLPDenoiser expects at least 1-D z, got ndim={z.ndim}"
            )
        if t.ndim < 1:
            raise ValueError(
                f"MLPDenoiser expects at least 1-D t, got ndim={t.ndim}"
            )
        return mx.zeros_like(z)

    def train_step(self, *args: object, **kwargs: object) -> mx.array:
        """Not implemented in M2.

        Training lands in Wave 3b M3 alongside
        ``_diffusion/trainer.py`` (plan §4 M3).
        """
        raise NotImplementedError(
            "MLPDenoiser.train_step is deferred to Wave 3b M3 "
            "(_diffusion/trainer.py). M2 is skeleton-only."
        )


class NoiseSchedule:
    """σ — Linear-β forward noise schedule (DDPM-style).

    Implements the analytic ``β(t)``, ``α(t)``, and ``ᾱ(t)`` of
    a linear-β DDPM forward process. No learned weights are
    involved, so this is fully implemented in M2 — the typing
    table (plan §3.2) needs at least one working primitive
    on real arrays to keep the DR-3 "signature typing" angle
    testable end-to-end before M3.

    Args:
        t_steps: Number of discrete diffusion timesteps.
        beta_min: Starting β at ``t=0``.
        beta_max: Ending β at ``t=t_steps-1``.
    """

    def __init__(
        self,
        t_steps: int,
        beta_min: float,
        beta_max: float,
    ) -> None:
        if t_steps < 2:
            raise ValueError(f"t_steps must be >= 2, got {t_steps}")
        if not (0.0 < beta_min < beta_max < 1.0):
            raise ValueError(
                "Require 0 < beta_min < beta_max < 1, got "
                f"beta_min={beta_min}, beta_max={beta_max}"
            )
        self.t_steps = t_steps
        self.beta_min = beta_min
        self.beta_max = beta_max
        # Pre-compute the linear-β schedule once. Shape (t_steps,).
        self._betas: mx.array = mx.linspace(beta_min, beta_max, t_steps)
        # ᾱ(t) = ∏_{s<=t} (1 - β_s). Cumulative product.
        self._alpha_bars: mx.array = mx.cumprod(1.0 - self._betas)

    def _gather(self, table: mx.array, t: mx.array) -> mx.array:
        """Gather schedule values at integer timesteps ``t``."""
        if t.ndim < 1:
            raise ValueError(
                f"NoiseSchedule expects at least 1-D t, got ndim={t.ndim}"
            )
        idx = t.astype(mx.int32)
        return table[idx]

    def sigma(self, t: mx.array) -> mx.array:
        """Return ``σ(t) = sqrt(β(t))`` at integer timesteps ``t``."""
        beta_t = self._gather(self._betas, t)
        return mx.sqrt(beta_t)

    def alpha_bar(self, t: mx.array) -> mx.array:
        """Return ``ᾱ(t) = ∏_{s<=t} (1 - β_s)`` at integer timesteps ``t``."""
        return self._gather(self._alpha_bars, t)
