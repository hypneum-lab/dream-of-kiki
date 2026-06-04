"""Training loop for the MLX latent-diffusion substrate (Wave 3b M3).

Provides :class:`Trainer`, a thin wrapper that runs noise-prediction
SGD on :class:`MLPDenoiser` against samples drawn from a
caller-provided latent dataset. The trainer owns the per-step
random-key tree but never owns model weights — those live on the
:class:`MLPDenoiser` instance the caller passes in.

R1 contract (per plan §2.3 D3).
================================

Training MUST consume only ``mx.random.split``-derived keys, never
a raw ``mx.random.key(seed)`` value passed straight to
``mx.random.normal``. The per-step subkey tree is :

    seed_key                = caller-provided ``mx.array``
    subkeys (n_total + 1)   = ``mx.random.split(seed_key, num=n_total + 1)``
                              with ``n_total = n_epochs * steps_per_epoch``
    subkeys[0]              = reserved (matches sampler's reservation)
    subkeys[i + 1]          = key for the (i+1)-th training step
                              (epoch * steps_per_epoch + step)

Determinism guarantees (given a fixed MLX version + chip family) :

* same ``dataset`` (identical bytes, identical iteration order),
  same ``n_epochs``, same ``seed_key`` → bit-identical loss history.
* the loss history is the contract surface for the R1 hash ; weight
  hashes are an additional layer captured in
  ``tests/reproducibility/test_r1_diffusion.py``.

References
----------
- ``docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md``
  §2.3 D3 (R1 key derivation), §3.1 (module layout), §4 M3
  (this file's milestone).
- Ho, Jain, Abbeel (2020) — *Denoising Diffusion Probabilistic
  Models*, arXiv 2006.11239, §3.2 (training objective),
  Algorithm 1 (training loop).
"""
from __future__ import annotations

from typing import Iterable

import mlx.core as mx

from kiki_oniric.substrates._diffusion.model import (
    MLPDenoiser,
    NoiseSchedule,
)


class Trainer:
    """Noise-prediction SGD trainer for :class:`MLPDenoiser`.

    Args:
        model: :class:`MLPDenoiser` to train (in-place updates).
        schedule: :class:`NoiseSchedule` providing ᾱ for the
            forward-noising step in the DDPM training objective.
        optimizer_kwargs: ``{"lr": float}`` ; reserved for future
            optimizer choices (currently SGD-only).
    """

    def __init__(
        self,
        model: MLPDenoiser,
        schedule: NoiseSchedule,
        optimizer_kwargs: dict[str, float] | None = None,
    ) -> None:
        self.model = model
        self.schedule = schedule
        opts = dict(optimizer_kwargs or {})
        self.lr: float = float(opts.pop("lr", 1e-3))
        if self.lr <= 0.0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if opts:
            raise ValueError(
                f"Unsupported optimizer_kwargs keys : {sorted(opts)}"
            )

    def train_step(
        self,
        batch: mx.array,
        key: mx.array,
    ) -> dict[str, float]:
        """Run one SGD step on ``batch`` using ``key`` for noise sampling.

        Args:
            batch: Clean latents ``(batch, d_latent)``.
            key: Split-derived MLX key for the noise + timestep
                draws (per the R1 contract).

        Returns:
            ``{"loss": float}`` dict for logging / R1 hashing.
        """
        if batch.ndim < 2:
            raise ValueError(
                f"Trainer.train_step expects 2-D batch, got ndim={batch.ndim}"
            )

        # Two sub-keys : one for timestep sampling, one for noise.
        noise_key, t_key = mx.random.split(key, num=2)

        n = batch.shape[0]
        t = mx.random.randint(
            low=0, high=self.schedule.t_steps, shape=(n,), key=t_key
        ).astype(mx.int32)
        noise = mx.random.normal(shape=batch.shape, key=noise_key)

        loss = self.model.train_step(
            batch, t, noise, self.schedule, lr=self.lr
        )
        return {"loss": loss}

    def fit(
        self,
        dataset: Iterable[mx.array],
        n_epochs: int,
        seed_key: mx.array,
    ) -> dict[str, list[float]]:
        """Train for ``n_epochs`` over ``dataset`` using ``seed_key``.

        The dataset MUST be re-iterable (list-like or generator
        factory result) — the trainer iterates it once per epoch.
        The number of steps per epoch is inferred as the length of
        the materialized list of batches on the first epoch.

        Args:
            dataset: Iterable of batches ``(batch, d_latent)``.
            n_epochs: Number of epochs.
            seed_key: Root MLX key. The trainer derives all per-step
                sub-keys via :func:`mx.random.split` per §2.3 D3.

        Returns:
            ``{"loss": [...]}`` list of per-step losses across all
            epochs (length ``n_epochs * steps_per_epoch``).
        """
        if n_epochs <= 0:
            raise ValueError(f"n_epochs must be positive, got {n_epochs}")

        # Materialize once : we need a deterministic ordering and
        # steps_per_epoch for the subkey tree.
        batches: list[mx.array] = list(dataset)
        if len(batches) == 0:
            raise ValueError("dataset must yield at least one batch")
        steps_per_epoch = len(batches)
        n_total = n_epochs * steps_per_epoch

        # Derive the per-step subkey tree. We request ``n_total + 1``
        # subkeys ; subkeys[0] is reserved (matches the sampler
        # reservation pattern) and subkeys[i+1] feeds the i-th step.
        subkeys = mx.random.split(seed_key, num=n_total + 1)

        losses: list[float] = []
        step_index = 0
        for _ in range(n_epochs):
            for batch in batches:
                step_key = subkeys[step_index + 1]
                metrics = self.train_step(batch, step_key)
                losses.append(metrics["loss"])
                step_index += 1
        return {"loss": losses}


__all__ = ["Trainer"]
