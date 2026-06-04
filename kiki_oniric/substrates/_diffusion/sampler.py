"""DDPM reverse-process sampler with an R1-clean key derivation tree.

Wave 3b M3 (per plan §4 M3 + §2.3 D3).

The sampler implements the standard DDPM reverse update :

    x_{t-1} = (1 / sqrt(alpha_t)) * (x_t - (beta_t / sqrt(1 - alpha_bar_t)) * eps_theta(x_t, t))
              + sigma_t * z,        z ~ N(0, I)

where ``eps_theta`` is the trained :class:`MLPDenoiser` and the
schedule values come from :class:`NoiseSchedule`. The skeleton
:class:`MLPDenoiser` returns ``zeros_like(z)`` so the sampler is
exercisable end-to-end against the M2 weights without committing
to a particular trained model. M3+ swaps in real denoiser weights
without changing the public surface.

R1 contract (per plan §2.3 D3).
================================

The sampler MUST consume only ``mx.random.split``-derived keys, and
NEVER a raw ``mx.random.key(seed)`` passed straight to
``mx.random.normal``. The cross-machine probe (2026-05-20,
``docs/milestones/r1-cross-machine-m5-vs-m1-2026-05-20.md``) showed
that raw-key consumption diverges on M1 Max while split-derived
keys reproduce across all Apple Silicon variants tested.

Key derivation is therefore :

    root_key                 = caller-provided ``mx.array``
    subkeys (n_steps + 1)    = ``mx.random.split(root_key, num=n_steps + 1)``
    subkeys[0]               = reserved (do not consume) — separates
                               the root-key shape from the first
                               step-key draw so the per-step
                               consumption pattern stays uniform.
    subkeys[t + 1]           = key for reverse step ``t`` (``t`` in
                               ``[0, n_steps)``)

Determinism guarantees (given a fixed MLX version + chip family) :

* same ``root_key`` + ``n_steps`` + ``shape`` → bit-identical
  output ``mx.array``.
* different ``root_key`` (any flip) → different output (with
  overwhelming probability ; the test asserts inequality on a
  non-zero element of the diff).
* ``shape`` invariance : the output has exactly the requested
  ``shape``.

These guarantees are exercised by
``tests/unit/test_diffusion_sampler_keys.py``.

References
----------
- ``docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md``
  §2.3 D3 (R1 key derivation), §3.1 (module layout), §4 M3
  (this file's milestone).
- Ho, Jain, Abbeel (2020) — *Denoising Diffusion Probabilistic
  Models*, arXiv 2006.11239, §3.2 (reverse process), Algorithm 2
  (sampling).
"""
from __future__ import annotations

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
else:
    import mlx.core as mx  # noqa: F401 — re-imported at runtime

from kiki_oniric.substrates._diffusion.model import (
    MLPDenoiser,
    NoiseSchedule,
)


class Sampler:
    """DDPM reverse-process sampler bound to a denoiser + schedule.

    The sampler is intentionally a thin object : the denoiser and
    schedule own their respective state (weights, β/ᾱ tables) and
    the sampler is responsible only for the reverse-update loop
    and the R1-clean key-derivation tree (§2.3 D3).

    Args:
        model: Trained (or skeleton) :class:`MLPDenoiser`. Must
            satisfy ``__call__(z: mx.array, t: mx.array) -> mx.array``
            returning an estimate of the noise added to ``z``.
        schedule: :class:`NoiseSchedule` with at least as many
            timesteps as the ``n_steps`` requested at sample time.
    """

    def __init__(self, model: MLPDenoiser, schedule: NoiseSchedule) -> None:
        self.model = model
        self.schedule = schedule

    def sample(
        self,
        key: "mx.array",
        n_steps: int,
        shape: tuple[int, ...],
    ) -> "mx.array":
        """Run ``n_steps`` of the DDPM reverse process from pure noise.

        Args:
            key: Root ``mx.array`` key (typically
                ``mx.random.key(cell_seed)``). Consumed only via
                :func:`mx.random.split` per the R1 contract.
            n_steps: Number of reverse steps. Must be in
                ``[1, schedule.t_steps]``.
            shape: Output shape (e.g. ``(batch, d_latent)``).

        Returns:
            A sample ``mx.array`` of the requested ``shape``.

        Raises:
            ValueError: if ``n_steps`` is non-positive or exceeds
                the schedule's ``t_steps``, or if ``shape`` is empty.
        """
        if n_steps <= 0:
            raise ValueError(f"n_steps must be positive, got {n_steps}")
        if n_steps > self.schedule.t_steps:
            raise ValueError(
                f"n_steps={n_steps} exceeds schedule.t_steps="
                f"{self.schedule.t_steps}"
            )
        if len(shape) == 0:
            raise ValueError("shape must be non-empty")

        # Derive a per-step subkey tree using split — never a raw key
        # straight to mx.random.normal (per R1 §2.3 D3).
        # We request n_steps + 1 subkeys ; subkeys[0] is reserved
        # (initial-noise draw) and subkeys[1..n_steps] feed the
        # reverse-step stochastic draws.
        subkeys = mx.random.split(key, num=n_steps + 1)

        # Sample x_T ~ N(0, I) using the reserved subkey.
        x = mx.random.normal(shape=shape, key=subkeys[0])

        # Reverse loop : t goes n_steps-1 → 0 (the standard DDPM
        # iteration order). Each step pulls its noise draw from the
        # corresponding split-derived subkey to keep the key tree
        # deterministic and cross-machine-stable.
        for step in range(n_steps):
            t_value = n_steps - 1 - step
            t_arr = mx.array([t_value], dtype=mx.int32)

            # Denoiser estimate of the noise added at step t.
            eps_hat = self.model(x, t_arr)

            # Schedule lookups : alpha_bar_t and beta_t. The
            # NoiseSchedule API returns alpha_bar(t) ; we recover
            # alpha_t and beta_t from the pre-computed tables via
            # the public accessors only.
            alpha_bar_t = self.schedule.alpha_bar(t_arr)
            beta_t = self.schedule.sigma(t_arr) ** 2  # sigma = sqrt(beta)
            alpha_t = 1.0 - beta_t

            # DDPM reverse mean.
            coef_eps = beta_t / mx.sqrt(1.0 - alpha_bar_t)
            mean = (x - coef_eps * eps_hat) / mx.sqrt(alpha_t)

            if t_value > 0:
                noise = mx.random.normal(shape=shape, key=subkeys[step + 1])
                sigma_t = self.schedule.sigma(t_arr)
                x = mean + sigma_t * noise
            else:
                # At t=0 the standard DDPM update drops the stochastic
                # tail, returning the mean as the final sample.
                x = mean

        # Defensive shape check : the reverse loop must not change
        # the requested shape (it does not, but assert it cheaply).
        if tuple(x.shape) != tuple(shape):
            raise RuntimeError(
                f"Sampler produced shape {tuple(x.shape)} "
                f"!= requested shape {tuple(shape)}"
            )
        return x


__all__ = ["Sampler"]
