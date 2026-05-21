"""DR-0 Accountability — MLX latent-diffusion substrate (Wave 3b M3).

Substrate-side translation of DR-0 (every executed DE appears in
the runtime log with a finite budget) :

* Every training step counts as one DE. The Trainer's ``fit``
  produces exactly ``n_epochs * steps_per_epoch`` loss entries
  (accountable record).
* The per-DE budget (loss is finite, runtime is bounded by the
  caller-supplied ``n_epochs``) is enforced.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
"""
from __future__ import annotations

import math

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

mx = pytest.importorskip("mlx.core")

from kiki_oniric.substrates._diffusion import (  # noqa: E402
    MLPDenoiser,
    NoiseSchedule,
    Trainer,
)


def _build_trainer(d_latent: int = 4) -> Trainer:
    mx.random.seed(0)
    denoiser = MLPDenoiser(d_latent=d_latent, d_hidden=8, n_layers=2)
    schedule = NoiseSchedule(t_steps=16, beta_min=1e-4, beta_max=2e-2)
    return Trainer(model=denoiser, schedule=schedule)


@given(
    n_epochs=st.integers(min_value=1, max_value=4),
    n_batches=st.integers(min_value=1, max_value=4),
    batch_size=st.integers(min_value=1, max_value=4),
)
@settings(max_examples=10, deadline=None, derandomize=True)
def test_dr0_loss_history_length_matches_de_count(
    n_epochs: int, n_batches: int, batch_size: int
) -> None:
    """Loss history length == ``n_epochs * n_batches`` (DR-0 accounting)."""
    trainer = _build_trainer()
    root = mx.random.key(11)
    data_keys = mx.random.split(root, num=n_batches)
    dataset = [
        mx.random.normal(shape=(batch_size, 4), key=k) for k in data_keys
    ]
    history = trainer.fit(dataset=dataset, n_epochs=n_epochs, seed_key=root)
    assert len(history["loss"]) == n_epochs * n_batches


def test_dr0_each_de_loss_is_finite_and_non_negative() -> None:
    """Every DE in the log carries a finite, non-negative loss (S2 + DR-0)."""
    trainer = _build_trainer()
    root = mx.random.key(13)
    data_keys = mx.random.split(root, num=3)
    dataset = [
        mx.random.normal(shape=(2, 4), key=k) for k in data_keys
    ]
    losses = trainer.fit(
        dataset=dataset, n_epochs=2, seed_key=root
    )["loss"]
    assert len(losses) == 6
    for loss in losses:
        assert math.isfinite(loss)
        assert loss >= 0.0


def test_dr0_trainer_rejects_zero_epochs() -> None:
    """DR-0 budget cap : ``n_epochs`` MUST be positive."""
    trainer = _build_trainer()
    with pytest.raises(ValueError, match="n_epochs"):
        trainer.fit(
            dataset=[mx.zeros((1, 4), dtype=mx.float32)],
            n_epochs=0,
            seed_key=mx.random.key(0),
        )


def test_dr0_trainer_rejects_empty_dataset() -> None:
    """DR-0 budget cap : the dataset MUST yield at least one batch."""
    trainer = _build_trainer()
    with pytest.raises(ValueError, match="at least one batch"):
        trainer.fit(
            dataset=[], n_epochs=1, seed_key=mx.random.key(0)
        )
