# tests/conformance/axioms/test_dr3_diffusion_profile.py
"""DR-3 Conformance: mlx_latent_diffusion activates exactly the
framework-C §3.1 primitive set per profile.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md
§3.1 (normative table) + §3.2 (monotonic inclusion).
"""

from __future__ import annotations

from typing import Any, Type

import pytest

from kiki_oniric.dream.episode import Operation
from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_max import PMaxProfile
from kiki_oniric.profiles.p_min import PMinProfile
from kiki_oniric.substrates._diffusion.decoder import Decoder
from kiki_oniric.substrates._diffusion import Encoder
from kiki_oniric.substrates._diffusion.dream_ops_adapter import (
    bind_real_handlers,
)


def _model_stub() -> Any:
    # `model` is only used by bind_real_handlers to close over the
    # handler factories; the handlers are never executed in this
    # activation-wiring test, so a bare `layers` attribute suffices.
    # No `parameters` shim: if a future change executes a handler, an
    # AttributeError should fire loudly rather than silently yield {}.
    return type("Stub", (), {"layers": []})()


def _enc_dec() -> tuple[Encoder, Decoder]:
    return Encoder(d_in=3072, d_latent=64), Decoder(d_latent=64, d_out=3072)


@pytest.mark.parametrize(
    "profile_ctor, expected",
    [
        (PMinProfile, {Operation.REPLAY, Operation.DOWNSCALE}),
        (PEquProfile, {
            Operation.REPLAY, Operation.DOWNSCALE,
            Operation.RESTRUCTURE, Operation.RECOMBINE,
        }),
        (PMaxProfile, {
            Operation.REPLAY, Operation.DOWNSCALE,
            Operation.RESTRUCTURE, Operation.RECOMBINE,
        }),
    ],
)
def test_diffusion_profile_activates_exact_set(
    profile_ctor: Type[Any], expected: set[Operation],
) -> None:
    """Each profile activates EXACTLY the §3.1 op set on the substrate."""
    profile = profile_ctor()
    encoder, decoder = _enc_dec()
    activated = bind_real_handlers(
        profile, model=_model_stub(), encoder=encoder, decoder=decoder,
        seed=0,
    )
    assert activated == expected


def test_diffusion_profile_inclusion_chain_is_monotonic() -> None:
    """§3.2: ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)."""
    enc, dec = _enc_dec()
    a = bind_real_handlers(
        PMinProfile(), model=_model_stub(),
        encoder=enc, decoder=dec, seed=0,
    )
    enc, dec = _enc_dec()
    b = bind_real_handlers(
        PEquProfile(), model=_model_stub(),
        encoder=enc, decoder=dec, seed=0,
    )
    enc, dec = _enc_dec()
    c = bind_real_handlers(
        PMaxProfile(), model=_model_stub(),
        encoder=enc, decoder=dec, seed=0,
    )
    assert a <= b <= c
