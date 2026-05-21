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


def test_skeleton_methods_raise_not_implemented_where_deferred() -> None:
    """Documented M3+ deferrals surface as ``NotImplementedError``.

    This pins the public contract: callers must not silently get
    placeholder numbers from training or from
    ``execute_profile`` until the M3 trainer and ablation_cycle3
    wiring land.
    """
    adapter = MLXLatentDiffusionSubstrate()

    with pytest.raises(NotImplementedError, match="Wave 3b M3"):
        adapter.execute_profile(request=None)

    enc = Encoder(d_in=4, d_latent=2)
    with pytest.raises(NotImplementedError, match="Wave 3b M3"):
        enc.train_step()

    denoiser = MLPDenoiser(d_latent=2, d_hidden=4, n_layers=1)
    with pytest.raises(NotImplementedError, match="Wave 3b M3"):
        denoiser.train_step()

    # teardown is a real no-op in M2 (not deferred): it must be callable
    # without raising. The return annotation is ``-> None`` so we just
    # invoke it; mypy disallows asserting on a None-returning callable.
    adapter.teardown()
