"""Unit tests for the Wave 3b M2 latent-diffusion substrate skeleton.

Scope (per plan §4 M2):

1. ``MLXLatentDiffusionSubstrate`` satisfies the
   ``SubstrateAdapter`` Protocol (runtime_checkable structural).
2. ``build_substrate_adapter("mlx_latent_diffusion")`` returns
   an instance of the substrate.
3. The three building blocks (``Encoder``, ``MLPDenoiser``,
   ``NoiseSchedule``) instantiate with the D2 defaults.
4. Forward passes on dummy ``mx.array`` inputs return the
   expected shapes.
5. Skeleton methods raise ``NotImplementedError`` where the
   public contract documents the M3+ deferral.

This file is **Track-neutral** (per the M2 ship instructions):
no DR-3 conformance tests live here. Those land under
``tests/conformance/axioms/`` in Wave 3b M3+ if Track S is chosen.
"""
from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

if TYPE_CHECKING:
    import mlx.core as mx
else:
    mx = pytest.importorskip("mlx.core")

from kiki_oniric.substrates._diffusion import (
    Encoder,
    MLPDenoiser,
    NoiseSchedule,
)
from kiki_oniric.substrates.factory import (
    SUBSTRATE_NAMES,
    SubstrateAdapter,
    build_substrate_adapter,
)
from kiki_oniric.substrates.mlx_latent_diffusion import (
    MLX_LATENT_DIFFUSION_SUBSTRATE_NAME,
    MLXLatentDiffusionConfig,
    MLXLatentDiffusionSubstrate,
)


# ----------------------------------------------------------------------
# Test 1 — Protocol conformance (runtime_checkable structural check).
# ----------------------------------------------------------------------


def test_substrate_satisfies_substrate_adapter_protocol() -> None:
    """``MLXLatentDiffusionSubstrate`` exposes the Protocol surface.

    The ``SubstrateAdapter`` Protocol declares
    ``execute_profile(request) -> dict`` and ``teardown() -> None``.
    A structural ``isinstance`` check against a non-runtime_checkable
    Protocol is not possible directly, so we verify the surface
    by attribute + callable inspection (mirrors the pattern in
    ``tests/unit/test_substrate_factory.py``).
    """
    adapter = MLXLatentDiffusionSubstrate()
    assert callable(getattr(adapter, "execute_profile"))
    assert callable(getattr(adapter, "teardown"))
    # The Protocol itself must be importable and identifiable.
    assert SubstrateAdapter is not None
    # Identity claim: the substrate name is registered.
    assert MLX_LATENT_DIFFUSION_SUBSTRATE_NAME == "mlx_latent_diffusion"
    assert MLX_LATENT_DIFFUSION_SUBSTRATE_NAME in SUBSTRATE_NAMES


# ----------------------------------------------------------------------
# Test 2 — Factory dispatch returns the substrate by literal name.
# ----------------------------------------------------------------------


def test_factory_returns_substrate_for_mlx_latent_diffusion() -> None:
    """``build_substrate_adapter`` wires the M2 substrate."""
    adapter = build_substrate_adapter("mlx_latent_diffusion")
    assert isinstance(adapter, MLXLatentDiffusionSubstrate)
    # And the name is in the canonical tuple.
    assert "mlx_latent_diffusion" in SUBSTRATE_NAMES


def test_factory_rejects_unknown_substrate_name() -> None:
    """Unknown names raise ``ValueError`` with the canonical tuple."""
    with pytest.raises(ValueError, match="Unknown substrate name"):
        build_substrate_adapter("does_not_exist")  # type: ignore[arg-type]


# ----------------------------------------------------------------------
# Test 3 — Building blocks instantiate with the D2 defaults.
# ----------------------------------------------------------------------


def test_building_blocks_instantiate_with_defaults() -> None:
    """E / U / σ instantiate with the plan D2 defaults via the adapter."""
    cfg = MLXLatentDiffusionConfig()
    assert cfg.d_latent == 64
    assert cfg.n_layers == 3

    adapter = MLXLatentDiffusionSubstrate()
    assert isinstance(adapter.encoder, Encoder)
    assert isinstance(adapter.denoiser, MLPDenoiser)
    assert isinstance(adapter.schedule, NoiseSchedule)
    assert adapter.encoder.d_latent == 64
    assert adapter.denoiser.d_latent == 64
    assert adapter.denoiser.n_layers == 3
    assert adapter.schedule.t_steps == 1000

    # Standalone constructors also work with explicit args.
    e = Encoder(d_in=128, d_latent=32)
    assert e.d_in == 128
    u = MLPDenoiser(d_latent=32, d_hidden=64, n_layers=2)
    assert u.d_hidden == 64
    s = NoiseSchedule(t_steps=10, beta_min=1e-4, beta_max=2e-2)
    assert s.t_steps == 10

    # Components map returns the canonical dotted paths.
    comps = adapter.components()
    assert "encoder" in comps and "denoiser" in comps and "schedule" in comps


def test_building_blocks_reject_invalid_hyperparameters() -> None:
    """Constructors reject pathological hyper-parameters early."""
    with pytest.raises(ValueError):
        Encoder(d_in=0, d_latent=8)
    with pytest.raises(ValueError):
        Encoder(d_in=4, d_latent=0)
    with pytest.raises(ValueError):
        MLPDenoiser(d_latent=0, d_hidden=16, n_layers=2)
    with pytest.raises(ValueError):
        MLPDenoiser(d_latent=8, d_hidden=0, n_layers=2)
    with pytest.raises(ValueError):
        MLPDenoiser(d_latent=8, d_hidden=16, n_layers=0)
    with pytest.raises(ValueError):
        NoiseSchedule(t_steps=1, beta_min=1e-4, beta_max=2e-2)
    with pytest.raises(ValueError):
        NoiseSchedule(t_steps=10, beta_min=2e-2, beta_max=1e-4)


def test_building_blocks_reject_zero_dim_inputs() -> None:
    """Forward passes reject zero-dim ``mx.array`` inputs (defensive guards)."""
    enc = Encoder(d_in=4, d_latent=2)
    with pytest.raises(ValueError, match="at least 1-D"):
        enc(mx.array(0.0, dtype=mx.float32))

    denoiser = MLPDenoiser(d_latent=2, d_hidden=4, n_layers=1)
    z_ok = mx.zeros((1, 2), dtype=mx.float32)
    t_ok = mx.array([0], dtype=mx.int32)
    with pytest.raises(ValueError, match="at least 1-D z"):
        denoiser(mx.array(0.0, dtype=mx.float32), t_ok)
    with pytest.raises(ValueError, match="at least 1-D t"):
        denoiser(z_ok, mx.array(0, dtype=mx.int32))

    schedule = NoiseSchedule(t_steps=10, beta_min=1e-4, beta_max=2e-2)
    with pytest.raises(ValueError, match="at least 1-D t"):
        schedule.sigma(mx.array(0, dtype=mx.int32))


# ----------------------------------------------------------------------
# Test 4 — Forward passes on dummy mx.array return correct shapes.
# ----------------------------------------------------------------------


def test_forward_passes_return_correct_shapes() -> None:
    """Skeleton forward passes honour the typed signatures.

    Encoder: ``(batch, d_in)`` → ``(batch, d_latent)``.
    Denoiser: ``(batch, d_latent)``, ``(batch,)`` → ``(batch, d_latent)``.
    Schedule: ``sigma(t)`` and ``alpha_bar(t)`` → ``t.shape``.
    """
    batch = 4
    d_in = 16
    d_latent = 8
    t_steps = 10

    enc = Encoder(d_in=d_in, d_latent=d_latent)
    x = mx.zeros((batch, d_in), dtype=mx.float32)
    z = enc(x)
    assert z.shape == (batch, d_latent)
    assert z.dtype == mx.float32

    denoiser = MLPDenoiser(d_latent=d_latent, d_hidden=16, n_layers=2)
    t = mx.array([0, 1, 2, 3], dtype=mx.int32)
    z_hat = denoiser(z, t)
    assert z_hat.shape == z.shape

    schedule = NoiseSchedule(t_steps=t_steps, beta_min=1e-4, beta_max=2e-2)
    sigma_t = schedule.sigma(t)
    assert sigma_t.shape == t.shape
    alpha_bar_t = schedule.alpha_bar(t)
    assert alpha_bar_t.shape == t.shape
    # σ and ᾱ are non-negative and ᾱ is monotone decreasing on [0..t_steps-1].
    # mx.array.tolist() returns int | float | list — cast through numpy for
    # a clean typed list[float] without nesting issues.
    import numpy as np
    sigma_list: list[float] = np.asarray(sigma_t).astype(float).tolist()
    assert all(s >= 0.0 for s in sigma_list)
    ab_full: list[float] = np.asarray(
        schedule.alpha_bar(mx.arange(t_steps, dtype=mx.int32))
    ).astype(float).tolist()
    assert all(ab_full[i + 1] <= ab_full[i] for i in range(t_steps - 1))


# ----------------------------------------------------------------------
# Test 5 — Skeleton methods raise NotImplementedError where deferred.
# ----------------------------------------------------------------------


def test_execute_profile_runs_minimal_smoke_cycle() -> None:
    """Wave 3b M3 wires a minimal train + sample driver.

    The smoke cycle returns the standard metrics-dict shape used
    by the sibling substrates and carries a ``"synthetic": True``
    flag per CLAUDE.md §Working rules item 3 (no synthetic
    placeholder may be reported as an empirical claim).
    """
    from dataclasses import dataclass

    @dataclass
    class _StubRequest:
        seed: int = 7
        profile: str = "p_min"

    adapter = MLXLatentDiffusionSubstrate()
    out = adapter.execute_profile(_StubRequest())

    # Standard sibling-substrate keys.
    for key in (
        "replay_rate", "downscale_norm", "restructure_sum",
        "recombine_rate", "delta_acc", "wall_time_s",
    ):
        assert key in out, f"missing metric key {key!r}"
        assert isinstance(out[key], float), f"{key!r} must be float"

    # M3-specific provenance keys (honesty marker + identity).
    assert out["synthetic"] is True
    assert out["profile"] == "p_min"
    assert out["seed"] == 7
    assert out["substrate"] == "mlx_latent_diffusion"
    assert isinstance(out["substrate_version"], str)
    assert out["substrate_version"].startswith("C-v")
    assert out["substrate_version"].endswith("+PARTIAL")

    # teardown is a real no-op : it must be callable without raising.
    adapter.teardown()


def test_encoder_and_denoiser_train_step_smoke() -> None:
    """``train_step`` runs without raising and returns a finite loss."""
    import math

    enc = Encoder(d_in=4, d_latent=2)
    x = mx.zeros((3, 4), dtype=mx.float32)
    loss = enc.train_step(x, lr=1e-3)
    assert isinstance(loss, float)
    assert math.isfinite(loss)
    assert loss >= 0.0

    schedule = NoiseSchedule(t_steps=10, beta_min=1e-4, beta_max=2e-2)
    denoiser = MLPDenoiser(d_latent=2, d_hidden=4, n_layers=1)
    z = mx.zeros((3, 2), dtype=mx.float32)
    t = mx.array([0, 1, 2], dtype=mx.int32)
    noise = mx.zeros((3, 2), dtype=mx.float32)
    loss_d = denoiser.train_step(z, t, noise, schedule, lr=1e-3)
    assert isinstance(loss_d, float)
    assert math.isfinite(loss_d)
    assert loss_d >= 0.0


def test_execute_profile_consumes_loader_batches() -> None:
    """execute_profile uses real batches when loader_batches is non-empty.

    When request.loader_batches carries at least one batch, the substrate
    must encode the raw features into latents (synthetic=False) instead of
    generating synthetic normal latents (synthetic=True).  The standard
    metrics dict must still be returned with the same keys, and
    ``replay_rate`` / ``downscale_norm`` must be present.
    """

    class _Req:
        seed = 0
        profile = "p_min"
        task_idx = 0
        loader_batches = (
            type("B", (), {
                "features": mx.zeros((8, 3072)),
                "labels": mx.zeros((8,), dtype=mx.int32),
                "task_idx": 0,
            })(),
        )

    substrate = MLXLatentDiffusionSubstrate()
    metrics = substrate.execute_profile(_Req())
    assert metrics["synthetic"] is False
    assert "replay_rate" in metrics
