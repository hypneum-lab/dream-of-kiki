"""MLX latent-diffusion building blocks (Wave 3b M3).

The three core building blocks of the latent-diffusion substrate :

- :class:`Encoder` (E) — maps awake activations ``x`` to a latent
  vector ``z`` via a single ``Linear`` + ReLU. ``train_step``
  performs one SGD step on a reconstruction MSE objective against
  the input (a deliberately weak self-supervised objective ; the
  full encoder is M5-bench territory, this is the M3 R1-bootstrap
  version).
- :class:`MLPDenoiser` (U) — n-layer MLP that predicts the noise
  added at a given diffusion timestep. ``train_step`` performs one
  SGD step on a noise-prediction MSE objective.
- :class:`NoiseSchedule` (σ) — analytic linear-β DDPM schedule.
  No learned weights ; pre-computes β and ᾱ tables at construction.

These classes are intentionally minimal. The R1 contract (plan
§2.3 D3) is that *given the same MLX seed*, training and sampling
are bit-identical on a fixed (MLX version, chip family) pair.

The M2 skeleton returned ``zeros`` from the forward passes — the M3
upgrade swaps in real ``nn.Linear`` weights so the training loop
and the M3 R1 hashes have something non-trivial to bite into. The
public ``__call__`` shape and dtype guarantees are unchanged.

References:
- ``docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md``
  §3.1 (module layout) + §3.2 (typing table) + §4 M3
- ``kiki_oniric/core/primitives.py`` (typed Protocols this
  substrate satisfies via the M3 wiring)
"""
from __future__ import annotations

from typing import Any, cast

import mlx.core as mx
import mlx.nn as _nn

# mlx.nn re-exports under a star pattern that mypy cannot resolve
# (cf. ``tests/reproducibility/test_r1_bit_exact.py`` for the
# precedent). We pin a local ``nn`` alias typed as ``Any`` so the
# substrate code does not need per-line ``type: ignore`` annotations.
nn: Any = cast(Any, _nn)


class Encoder(nn.Module):  # type: ignore[misc]
    """E — Awake-side encoder ``x → z``.

    Single ``Linear(d_in → d_latent)`` + ReLU. A symmetric
    reconstruction head ``Linear(d_latent → d_in)`` is used only
    by :meth:`train_step` ; the public forward pass exposes only
    the latent ``z``.

    Args:
        d_in: Input activation dimensionality.
        d_latent: Output latent dimensionality. Plan D2 default ``64``.
    """

    def __init__(self, d_in: int, d_latent: int) -> None:
        super().__init__()
        if d_in <= 0:
            raise ValueError(f"d_in must be positive, got {d_in}")
        if d_latent <= 0:
            raise ValueError(f"d_latent must be positive, got {d_latent}")
        self.d_in = d_in
        self.d_latent = d_latent
        self.fc_enc = nn.Linear(d_in, d_latent)
        self.fc_dec = nn.Linear(d_latent, d_in)

    def __call__(self, x: mx.array) -> mx.array:
        """Project ``x`` (batch, d_in) → ``z`` (batch, d_latent)."""
        if x.ndim < 1:
            raise ValueError(
                f"Encoder expects at least 1-D input, got ndim={x.ndim}"
            )
        if x.ndim == 1:
            x = x[None, :]
        out: mx.array = nn.relu(self.fc_enc(x))
        return out

    def reconstruct(self, z: mx.array) -> mx.array:
        """Decode the latent back to input space (training only)."""
        out: mx.array = self.fc_dec(z)
        return out

    def train_step(
        self,
        x: mx.array,
        lr: float = 1e-3,
    ) -> float:
        """One SGD step on the reconstruction MSE ``|| dec(enc(x)) - x ||²``.

        Returns the scalar loss (Python float) for logging /
        deterministic R1 hashing. Reproducibility contract :
        bit-identical given the same ``x``, ``lr`` and prior
        weights on a fixed MLX version + chip family.
        """
        if x.ndim < 1:
            raise ValueError(
                f"Encoder.train_step expects at least 1-D x, got ndim={x.ndim}"
            )
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")

        def loss_fn(model: "Encoder", batch: mx.array) -> mx.array:
            z = model(batch)
            x_hat = model.reconstruct(z)
            target = batch[None, :] if batch.ndim == 1 else batch
            return mx.mean((x_hat - target) ** 2)

        loss_and_grad = nn.value_and_grad(self, loss_fn)
        loss, grads = loss_and_grad(self, x)
        new_params = _sgd_apply(self.parameters(), grads, lr)
        self.update(new_params)
        return float(loss.item())


class MLPDenoiser(nn.Module):  # type: ignore[misc]
    """U — MLP denoiser network ``(z_t, t) → ε̂``.

    n-layer MLP that predicts the noise added at diffusion step
    ``t``. The timestep is broadcast as a scalar feature appended
    to the latent on input.

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
        super().__init__()
        if d_latent <= 0:
            raise ValueError(f"d_latent must be positive, got {d_latent}")
        if d_hidden <= 0:
            raise ValueError(f"d_hidden must be positive, got {d_hidden}")
        if n_layers < 1:
            raise ValueError(f"n_layers must be >= 1, got {n_layers}")
        self.d_latent = d_latent
        self.d_hidden = d_hidden
        self.n_layers = n_layers
        # Input concatenates the latent (d_latent) with the scalar
        # timestep feature (1) ; output predicts the per-feature noise.
        layers: list[Any] = [nn.Linear(d_latent + 1, d_hidden)]
        for _ in range(n_layers - 1):
            layers.append(nn.Linear(d_hidden, d_hidden))
        layers.append(nn.Linear(d_hidden, d_latent))
        self.layers = layers

    def __call__(self, z: mx.array, t: mx.array) -> mx.array:
        """Predict the noise added to ``z`` at step ``t``."""
        if z.ndim < 1:
            raise ValueError(
                f"MLPDenoiser expects at least 1-D z, got ndim={z.ndim}"
            )
        if t.ndim < 1:
            raise ValueError(
                f"MLPDenoiser expects at least 1-D t, got ndim={t.ndim}"
            )
        if z.ndim == 1:
            z = z[None, :]
        batch = z.shape[0]
        # Broadcast t to (batch, 1) and concatenate to z along the
        # feature axis. Accept either a scalar timestep (shape (1,))
        # or a per-batch timestep (shape (batch,)).
        if t.shape[0] == 1 and batch != 1:
            t_feat = mx.broadcast_to(
                t.astype(mx.float32)[:, None], (batch, 1)
            )
        else:
            t_feat = t.astype(mx.float32).reshape(batch, 1)
        h = mx.concatenate([z, t_feat], axis=-1)
        for layer in self.layers[:-1]:
            h = nn.relu(layer(h))
        out: mx.array = self.layers[-1](h)
        return out

    def train_step(
        self,
        z_clean: mx.array,
        t: mx.array,
        noise: mx.array,
        schedule: "NoiseSchedule",
        lr: float = 1e-3,
    ) -> float:
        """One SGD step on noise-prediction MSE ``|| ε̂(z_t, t) - ε ||²``.

        Args:
            z_clean: Clean latent batch ``(batch, d_latent)``.
            t: Per-batch timesteps ``(batch,)`` ``int32``.
            noise: Standard normal noise of shape ``z_clean.shape``.
            schedule: :class:`NoiseSchedule` providing ᾱ.
            lr: SGD learning rate.

        Returns:
            Scalar loss (Python float).
        """
        if lr <= 0.0:
            raise ValueError(f"lr must be positive, got {lr}")

        alpha_bar_t = schedule.alpha_bar(t)  # shape (batch,)
        sqrt_ab = mx.sqrt(alpha_bar_t)[:, None]
        sqrt_one_minus_ab = mx.sqrt(1.0 - alpha_bar_t)[:, None]
        z_noisy = sqrt_ab * z_clean + sqrt_one_minus_ab * noise

        def loss_fn(
            model: "MLPDenoiser",
            zn: mx.array,
            tt: mx.array,
            target: mx.array,
        ) -> mx.array:
            eps_hat = model(zn, tt)
            return mx.mean((eps_hat - target) ** 2)

        loss_and_grad = nn.value_and_grad(self, loss_fn)
        loss, grads = loss_and_grad(self, z_noisy, t, noise)
        new_params = _sgd_apply(self.parameters(), grads, lr)
        self.update(new_params)
        return float(loss.item())


class NoiseSchedule:
    """σ — Linear-β forward noise schedule (DDPM-style).

    Implements the analytic ``β(t)``, ``α(t)``, and ``ᾱ(t)`` of
    a linear-β DDPM forward process. No learned weights.

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
        self._betas: mx.array = mx.linspace(beta_min, beta_max, t_steps)
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


def _sgd_apply(
    params: object,
    grads: object,
    lr: float,
) -> object:
    """Apply a plain SGD update ``θ ← θ − lr · g`` element-wise.

    Mirrors the tree structure of ``params`` / ``grads`` (both
    nested ``dict`` / ``list`` of ``mx.array``). Uses
    :func:`mlx.utils.tree_map` when available, with a manual
    fallback for older MLX builds.
    """
    try:
        from mlx.utils import tree_map
    except ImportError:  # pragma: no cover — fallback for very old MLX
        return _sgd_apply_manual(params, grads, lr)

    def _step(p: mx.array, g: mx.array) -> mx.array:
        return p - lr * g

    return tree_map(_step, params, grads)


def _sgd_apply_manual(  # pragma: no cover — fallback only
    params: object,
    grads: object,
    lr: float,
) -> object:
    if isinstance(params, dict) and isinstance(grads, dict):
        return {k: _sgd_apply_manual(params[k], grads[k], lr) for k in params}
    if isinstance(params, list) and isinstance(grads, list):
        return [_sgd_apply_manual(p, g, lr) for p, g in zip(params, grads)]
    if hasattr(params, "shape"):
        return params - lr * grads  # type: ignore[operator]
    return params
