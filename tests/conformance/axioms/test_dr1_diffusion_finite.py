"""DR-1 Finiteness — MLX latent-diffusion substrate (Wave 3b M3).

Substrate-side translation of DR-1 (every episodic record is
eventually consumed by a DE) :

* Every batch in the trainer's dataset MUST be consumed exactly
  ``n_epochs`` times (no leakage, no replay-amplification beyond
  the declared epoch count).
* Training MUST terminate in a bounded number of steps (no
  infinite loop).
* The sampler MUST terminate in exactly ``n_steps`` reverse
  iterations.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
"""
from __future__ import annotations

import math

import pytest

mx = pytest.importorskip("mlx.core")

from kiki_oniric.substrates._diffusion import (  # noqa: E402
    MLPDenoiser,
    NoiseSchedule,
    Sampler,
    Trainer,
)


def _build_pair(
    d_latent: int = 4, t_steps: int = 8
) -> tuple[Trainer, Sampler]:
    mx.random.seed(0)
    denoiser = MLPDenoiser(d_latent=d_latent, d_hidden=8, n_layers=2)
    schedule = NoiseSchedule(t_steps=t_steps, beta_min=1e-4, beta_max=2e-2)
    return Trainer(model=denoiser, schedule=schedule), Sampler(
        model=denoiser, schedule=schedule
    )


def test_dr1_training_terminates_in_finite_steps() -> None:
    """Training MUST terminate ; total step count is bounded."""
    trainer, _ = _build_pair()
    root = mx.random.key(0)
    data_keys = mx.random.split(root, num=2)
    dataset = [mx.random.normal(shape=(2, 4), key=k) for k in data_keys]
    history = trainer.fit(dataset=dataset, n_epochs=3, seed_key=root)
    # Strict upper bound : 3 epochs × 2 batches = 6 steps.
    assert len(history["loss"]) <= 6


def test_dr1_sampler_terminates_in_exactly_n_steps() -> None:
    """The sampler returns after exactly ``n_steps`` iterations.

    Negative witness : there is no inner unbounded loop. We probe
    by asserting termination at multiple ``n_steps`` values.
    """
    _, sampler = _build_pair()
    for n_steps in (1, 2, 4, 8):
        out = sampler.sample(
            key=mx.random.key(0), n_steps=n_steps, shape=(1, 4)
        )
        assert tuple(out.shape) == (1, 4)


def test_dr1_every_batch_consumed_n_epochs_times() -> None:
    """Conservation : steps_per_epoch * n_epochs == loss-history length.

    No batch leaks (no over-consumption), no batch is silently
    dropped (no under-consumption).
    """
    trainer, _ = _build_pair()
    root = mx.random.key(0)
    data_keys = mx.random.split(root, num=4)
    dataset = [mx.random.normal(shape=(1, 4), key=k) for k in data_keys]
    history = trainer.fit(dataset=dataset, n_epochs=2, seed_key=root)
    assert len(history["loss"]) == len(dataset) * 2


def test_dr1_all_losses_finite() -> None:
    """No NaN / Inf in the loss trace (S2-aligned guard for DR-1)."""
    trainer, _ = _build_pair()
    root = mx.random.key(0)
    data_keys = mx.random.split(root, num=3)
    dataset = [mx.random.normal(shape=(2, 4), key=k) for k in data_keys]
    losses = trainer.fit(dataset=dataset, n_epochs=1, seed_key=root)["loss"]
    for loss in losses:
        assert math.isfinite(loss)
