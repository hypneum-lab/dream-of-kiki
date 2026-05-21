"""Bind ``_real.py`` handlers onto a profile's ``DreamRuntime``.

Each profile (``PMinProfile`` / ``PEquProfile`` / ``PMaxProfile``)
auto-registers *skeleton* handlers in ``__post_init__``. M7 takes
that profile, inspects which states it carries, and **overrides**
those registrations with the real MLX-backed handlers bound to the
diffusion substrate's denoiser + encoder + decoder.

This is the cleanest way to honour framework-C §3.1 normative
activation without duplicating the activation logic — the profile
stays the source of truth for "which ops are active".

See docs/superpowers/specs/2026-05-21-m7-substrate-dr3-design.md
§3 D3.
"""
from __future__ import annotations

from typing import Any

from kiki_oniric.dream.episode import Operation
from kiki_oniric.dream.operations.downscale_real import (
    DownscaleRealState,
    downscale_real_handler,
)
from kiki_oniric.dream.operations.recombine_real import (
    RecombineRealState,
    recombine_real_handler,
)
from kiki_oniric.dream.operations.replay_real import (
    ReplayRealState,
    replay_real_handler,
)
from kiki_oniric.dream.operations.restructure_real import (
    RestructureRealState,
    restructure_real_handler,
)


def bind_real_handlers(
    profile: Any,
    *,
    model: Any,
    encoder: Any,
    decoder: Any,
    seed: int,
) -> set[Operation]:
    """Override skeleton handlers on ``profile.runtime`` with real ones.

    Returns the set of Operations actually re-registered, derived
    from which ``*_state`` attributes the profile carries (the
    profile's own __post_init__ already picked the activation set
    per framework-C §3.1).

    Mutates ``profile.runtime`` in-place ; also swaps the profile's
    state fields to the ``*RealState`` variants so the handlers can
    write their K1 / metrics to the same address the substrate will
    snapshot later.
    """
    overridden: set[Operation] = set()

    if hasattr(profile, "replay_state"):
        profile.replay_state = ReplayRealState()
        profile.runtime.register_handler(
            Operation.REPLAY,
            replay_real_handler(profile.replay_state, model=model),
        )
        overridden.add(Operation.REPLAY)

    if hasattr(profile, "downscale_state"):
        profile.downscale_state = DownscaleRealState()
        profile.runtime.register_handler(
            Operation.DOWNSCALE,
            downscale_real_handler(profile.downscale_state, model=model),
        )
        overridden.add(Operation.DOWNSCALE)

    if hasattr(profile, "restructure_state"):
        profile.restructure_state = RestructureRealState()
        profile.runtime.register_handler(
            Operation.RESTRUCTURE,
            restructure_real_handler(
                profile.restructure_state, model=model
            ),
        )
        overridden.add(Operation.RESTRUCTURE)

    if hasattr(profile, "recombine_state"):
        profile.recombine_state = RecombineRealState()
        profile.runtime.register_handler(
            Operation.RECOMBINE,
            recombine_real_handler(
                profile.recombine_state,
                encoder=encoder, decoder=decoder, seed=seed,
            ),
        )
        overridden.add(Operation.RECOMBINE)

    return overridden


__all__ = ["bind_real_handlers"]
