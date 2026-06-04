"""R1 bit-exact reproducibility — MLX latent-diffusion substrate (M3).

Adds three R1 entries per plan §2.3 D3 + §4 M3 :

* ``test_r1_diffusion_train`` — hashes the per-step loss history of
  a fixed-seed training run.
* ``test_r1_diffusion_sample`` — hashes the sampler output of a
  fixed-seed reverse-process draw against a deterministically-
  initialized denoiser.
* ``test_r1_diffusion_full_pipeline`` — hashes the end-to-end
  encoder-train → denoiser-train → sample trace, with all three
  building blocks seeded from a single root key (split-derived,
  per the R1 contract).

Per the §2.3 D3 contract, every ``mx.random.normal`` consumer
inside the substrate stack receives a ``mx.random.split``-derived
sub-key, never a raw ``mx.random.key(seed)`` value — the
cross-machine probe (2026-05-20) showed raw-key consumption
diverges on M1 Max while split-derived keys reproduce.

The golden hashes are captured locally per chip family via
:func:`tests.reproducibility._r1_helpers.compare_or_bootstrap` (the
existing pattern) and start with ``"status": "pending_review"`` ;
promotion to ``"accepted"`` happens manually after user review
(2026-05-10 N2 rebaseline discipline).
"""
from __future__ import annotations

from typing import Any, cast

import pytest

mx = pytest.importorskip("mlx.core")

from kiki_oniric.substrates._diffusion import (  # noqa: E402
    Encoder,
    MLPDenoiser,
    NoiseSchedule,
    Sampler,
    Trainer,
)
from tests.reproducibility._r1_helpers import (  # noqa: E402
    CANONICAL_SEED,
    canonical_hash,
    compare_or_bootstrap,
    tensor_to_list,
)


# Canonical sub-scenario parameters for the diffusion R1 entries.
# These are pinned constants — never reorder, never re-seed in-place.
_D_LATENT = 4
_D_HIDDEN = 8
_N_LAYERS = 2
_T_STEPS = 8
_BETA_MIN = 1e-4
_BETA_MAX = 2e-2
_BATCH_SIZE = 2
_N_BATCHES = 2
_N_EPOCHS = 1
_N_SAMPLE_STEPS = 4


def _make_models() -> tuple[Encoder, MLPDenoiser, NoiseSchedule]:
    """Build E / U / σ under the canonical MLX seed.

    Seeding is process-wide via ``mx.random.seed`` BEFORE any
    ``nn.Linear`` construction so the initial weights are
    deterministic across runs on a fixed (mlx version, chip family).
    """
    mx.random.seed(CANONICAL_SEED)
    encoder = Encoder(d_in=_D_LATENT, d_latent=_D_LATENT)
    mx.random.seed(CANONICAL_SEED + 1)
    denoiser = MLPDenoiser(
        d_latent=_D_LATENT, d_hidden=_D_HIDDEN, n_layers=_N_LAYERS,
    )
    schedule = NoiseSchedule(
        t_steps=_T_STEPS, beta_min=_BETA_MIN, beta_max=_BETA_MAX,
    )
    return encoder, denoiser, schedule


def _model_param_payload(model: Any) -> Any:
    """Serialize a nn.Module's parameters to a nested float list."""
    from mlx.utils import tree_flatten
    flat = cast(list[tuple[str, Any]], tree_flatten(model.parameters()))
    return [(key, tensor_to_list(val)) for key, val in flat]


def test_r1_diffusion_train() -> None:
    """Hash the loss history of a fixed-seed training run."""
    _, denoiser, schedule = _make_models()
    trainer = Trainer(
        model=denoiser, schedule=schedule, optimizer_kwargs={"lr": 1e-3},
    )
    root = mx.random.key(CANONICAL_SEED)
    train_root, data_root = mx.random.split(root, num=2)
    data_keys = mx.random.split(data_root, num=_N_BATCHES)
    dataset = [
        mx.random.normal(shape=(_BATCH_SIZE, _D_LATENT), key=k)
        for k in data_keys
    ]
    losses = trainer.fit(
        dataset=dataset, n_epochs=_N_EPOCHS, seed_key=train_root,
    )["loss"]

    payload = {
        "op": "diffusion_train",
        "losses": [float(loss) for loss in losses],
        "final_params": _model_param_payload(denoiser),
        "seed": CANONICAL_SEED,
        "n_epochs": _N_EPOCHS,
        "n_batches": _N_BATCHES,
        "batch_size": _BATCH_SIZE,
        "d_latent": _D_LATENT,
        "t_steps": _T_STEPS,
    }
    digest = canonical_hash(payload)
    compare_or_bootstrap("test_r1_diffusion_train", digest)


def test_r1_diffusion_sample() -> None:
    """Hash the sampler output of a fixed-seed reverse draw."""
    _, denoiser, schedule = _make_models()
    sampler = Sampler(model=denoiser, schedule=schedule)
    root = mx.random.key(CANONICAL_SEED)
    sample = sampler.sample(
        key=root,
        n_steps=_N_SAMPLE_STEPS,
        shape=(_BATCH_SIZE, _D_LATENT),
    )

    payload = {
        "op": "diffusion_sample",
        "sample": tensor_to_list(sample),
        "seed": CANONICAL_SEED,
        "n_steps": _N_SAMPLE_STEPS,
        "shape": [_BATCH_SIZE, _D_LATENT],
        "t_steps": _T_STEPS,
    }
    digest = canonical_hash(payload)
    compare_or_bootstrap("test_r1_diffusion_sample", digest)


def test_r1_diffusion_full_pipeline() -> None:
    """Hash the end-to-end encoder-train → denoiser-train → sample trace.

    Single-DE pipeline mirroring the existing R1 full-pipeline
    pattern (one DE through all the wired ops, final state
    hashed).
    """
    encoder, denoiser, schedule = _make_models()
    root = mx.random.key(CANONICAL_SEED)
    enc_root, train_root, sample_root, data_root = mx.random.split(
        root, num=4
    )

    # 1. One encoder train step on a fixed batch.
    enc_batch_key = mx.random.split(enc_root, num=1)[0]
    enc_batch = mx.random.normal(
        shape=(_BATCH_SIZE, _D_LATENT), key=enc_batch_key,
    )
    enc_loss = encoder.train_step(enc_batch, lr=1e-3)

    # 2. One denoiser fit pass on a small synthetic dataset.
    trainer = Trainer(
        model=denoiser, schedule=schedule, optimizer_kwargs={"lr": 1e-3},
    )
    data_keys = mx.random.split(data_root, num=_N_BATCHES)
    dataset = [
        mx.random.normal(shape=(_BATCH_SIZE, _D_LATENT), key=k)
        for k in data_keys
    ]
    losses = trainer.fit(
        dataset=dataset, n_epochs=_N_EPOCHS, seed_key=train_root,
    )["loss"]

    # 3. One reverse-process draw.
    sampler = Sampler(model=denoiser, schedule=schedule)
    sample = sampler.sample(
        key=sample_root,
        n_steps=_N_SAMPLE_STEPS,
        shape=(_BATCH_SIZE, _D_LATENT),
    )

    payload = {
        "op": "diffusion_full_pipeline",
        "encoder_loss": float(enc_loss),
        "denoiser_losses": [float(loss) for loss in losses],
        "encoder_params": _model_param_payload(encoder),
        "denoiser_params": _model_param_payload(denoiser),
        "final_sample": tensor_to_list(sample),
        "seed": CANONICAL_SEED,
    }
    digest = canonical_hash(payload)
    compare_or_bootstrap("test_r1_diffusion_full_pipeline", digest)
