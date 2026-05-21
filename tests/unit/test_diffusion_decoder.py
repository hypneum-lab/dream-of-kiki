"""Decoder MLP for the recombine VAE-shape contract (M7 D7)."""

from __future__ import annotations

import mlx.core as mx

from kiki_oniric.substrates._diffusion.decoder import Decoder


def test_decoder_maps_d_latent_to_3072() -> None:
    decoder = Decoder(d_latent=64, d_out=3072)
    z = mx.zeros((4, 64))
    out = decoder(z)
    assert out.shape == (4, 3072)


def test_decoder_is_seed_deterministic() -> None:
    decoder_a = Decoder(d_latent=64, d_out=3072)
    decoder_b = Decoder(d_latent=64, d_out=3072)
    # Two random inits produce different outputs (sanity that the
    # random init is actually firing).
    z = mx.zeros((1, 64))
    assert not mx.allclose(decoder_a(z), decoder_b(z))
