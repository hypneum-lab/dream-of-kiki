"""dream_ops_adapter: bind real handlers onto a profile runtime (M7 D3)."""

from __future__ import annotations

from kiki_oniric.dream.episode import Operation
from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_min import PMinProfile
from kiki_oniric.substrates._diffusion.decoder import Decoder
from kiki_oniric.substrates._diffusion.dream_ops_adapter import (
    bind_real_handlers,
)
from kiki_oniric.substrates._diffusion import Encoder


def test_bind_real_handlers_overrides_p_min_replay_downscale() -> None:
    profile = PMinProfile()
    model = type("Stub", (), {"layers": [], "parameters": lambda self: {}})()
    encoder = Encoder(d_in=3072, d_latent=64)
    decoder = Decoder(d_latent=64, d_out=3072)

    bound = bind_real_handlers(
        profile, model=model, encoder=encoder, decoder=decoder, seed=0
    )

    assert Operation.REPLAY in bound
    assert Operation.DOWNSCALE in bound
    assert Operation.RESTRUCTURE not in bound  # not activated for p_min
    assert Operation.RECOMBINE not in bound    # not activated for p_min


def test_bind_real_handlers_overrides_p_equ_all_four() -> None:
    profile = PEquProfile()
    model = type("Stub", (), {"layers": [], "parameters": lambda self: {}})()
    encoder = Encoder(d_in=3072, d_latent=64)
    decoder = Decoder(d_latent=64, d_out=3072)

    bound = bind_real_handlers(
        profile, model=model, encoder=encoder, decoder=decoder, seed=0
    )

    assert {
        Operation.REPLAY, Operation.DOWNSCALE,
        Operation.RESTRUCTURE, Operation.RECOMBINE,
    } <= bound
