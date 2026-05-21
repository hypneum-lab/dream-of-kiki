"""Unit tests for DenoiserWeightDeltaChannel (M7 delta_acc wiring)."""

from __future__ import annotations

import numpy as np
import pytest

from kiki_oniric.substrates._diffusion.model import MLPDenoiser
from kiki_oniric.substrates._diffusion.denoiser_weight_channel import (
    DenoiserWeightDeltaChannel,
)


def _denoiser() -> MLPDenoiser:
    return MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)


def test_apply_adds_delta_to_named_layer_weight() -> None:
    """apply() must add the per-layer delta to the matching weight."""
    d = _denoiser()
    layer0_w_before = np.asarray(d.layers[0].weight).copy()
    delta = np.ones_like(layer0_w_before, dtype=np.float32)
    channel = DenoiserWeightDeltaChannel(d)

    channel.apply({"layer_0_weight": delta}, fisher_bump=None)

    layer0_w_after = np.asarray(d.layers[0].weight)
    np.testing.assert_allclose(
        layer0_w_after, layer0_w_before + delta, atol=1e-6,
    )


def test_apply_adds_delta_to_named_layer_bias() -> None:
    """apply() must add the per-layer delta to the matching bias."""
    d = _denoiser()
    layer0_b_before = np.asarray(d.layers[0].bias).copy()
    delta = np.ones_like(layer0_b_before, dtype=np.float32) * 0.5
    channel = DenoiserWeightDeltaChannel(d)

    channel.apply({"layer_0_bias": delta}, fisher_bump=None)

    layer0_b_after = np.asarray(d.layers[0].bias)
    np.testing.assert_allclose(
        layer0_b_after, layer0_b_before + delta, atol=1e-6,
    )


def test_apply_unknown_key_raises() -> None:
    """An unknown layer key must raise — silent skip would hide bugs."""
    d = _denoiser()
    delta = np.zeros((2, 2), dtype=np.float32)
    channel = DenoiserWeightDeltaChannel(d)

    with pytest.raises(KeyError, match="bogus_key"):
        channel.apply({"bogus_key": delta}, fisher_bump=None)


def test_apply_finite_guard_raises_on_non_finite_post() -> None:
    """If the apply would produce NaN/inf, raise with S2 in the message."""
    d = _denoiser()
    layer0_w = np.asarray(d.layers[0].weight)
    bad = np.full_like(layer0_w, np.inf, dtype=np.float32)
    channel = DenoiserWeightDeltaChannel(d)

    # WeightUpdate.__post_init__ already rejects non-finite input,
    # so we bypass it by feeding the raw dict directly to apply.
    with pytest.raises(ValueError, match="S2"):
        channel.apply({"layer_0_weight": bad}, fisher_bump=None)


def test_apply_accepts_fisher_bump_without_using_it() -> None:
    """fisher_bump is recorded for traceability but does not gate apply."""
    d = _denoiser()
    delta = np.zeros_like(np.asarray(d.layers[0].weight), dtype=np.float32)
    fisher = {"layer_0_weight": np.ones_like(delta)}
    channel = DenoiserWeightDeltaChannel(d)

    channel.apply({"layer_0_weight": delta}, fisher_bump=fisher)
    # No exception; fisher_bump captured on the channel for traceability.
    assert channel.last_fisher_bump is fisher
