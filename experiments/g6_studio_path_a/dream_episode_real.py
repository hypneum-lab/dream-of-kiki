"""Live-tensor dream-episode runner for G6-Studio Path A.

Fixes the G6 Path B spectator pattern : the four
:class:`kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate` handlers
operate on the **live** LoRA delta dict (passed by reference), not on
a synthesised payload disjoint from the eval surface. ``P_max``
runs the same four ops as ``P_equ`` but with TIES-Merge
``alpha=2.0`` (paper §3 amplified contribution ; matches G4-ter HP
grid C5).

DR-0 / DR-1 stamps land on
``substrate.{restructure,recombine}_state`` via the handler closures.

Reference :
- Plan : ``docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md`` Task 4
- Pre-reg : ``docs/osf-prereg-g6-studio-path-a.md`` §5
- Path B template (the spectator pattern this fixes) :
  ``experiments/g6_mmlu_stream/dream_wrap.py`` and
  ``experiments/g6_mmlu_stream/run_g6.py:539-578``
"""
from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray

from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate


# Profile → ordered tuple of dream operations to run on the live
# LoRA delta. ``baseline`` is the no-op control. ``P_min`` runs
# REPLAY + DOWNSCALE only (G4-ter strongest positive signal, +2.77
# on MLX richer head). ``P_equ`` and ``P_max`` add RESTRUCTURE
# (OPLoRA) + RECOMBINE (TIES-Merge) ; the difference between them
# is the recombine alpha (1.0 vs 2.0).
PROFILE_OPS_REAL: dict[str, tuple[str, ...]] = {
    "baseline": (),
    "P_min": ("replay", "downscale"),
    "P_equ": ("replay", "downscale", "restructure", "recombine"),
    "P_max": ("replay", "downscale", "restructure", "recombine"),
}

# Per-profile TIES-Merge alpha. P_equ uses the paper default
# (1.0) ; P_max amplifies the merged contribution (2.0) per G4-ter
# HP grid C5.
PROFILE_RECOMBINE_ALPHA: dict[str, float] = {
    "P_equ": 1.0,
    "P_max": 2.0,
}

# Multiplicative shrink applied by the DOWNSCALE op (B-Tononi SHY
# proxy). Same value as the Path B synthetic payload to preserve
# direct cross-path comparability.
DEFAULT_SHRINK_FACTOR: float = 0.99

# LoRA tensor key the dream runtime mutates by reference. Single
# source of truth ; do not duplicate as a string literal in the
# driver.
DEFAULT_PRIMARY_KEY: str = "layer_0_lora_B"


def dream_episode_real(
    *,
    substrate: MicroKikiSubstrate,
    profile: str,
    live_delta: dict[str, NDArray[np.float32]],
    seed: int,
    subdomain: str,
    prior_deltas: list[NDArray],
    sibling_deltas: list[NDArray],
    primary_key: str = DEFAULT_PRIMARY_KEY,
) -> dict[str, NDArray[np.float32]]:
    """Run one dream episode against the LIVE LoRA delta dict (in place).

    Parameters
    ----------
    substrate
        Loaded :class:`MicroKikiSubstrate` (real backend or stub).
        The four handler factories are read from this instance ;
        DR-0 / DR-1 stamps land on its read-only state objects.
    profile
        One of ``baseline`` / ``P_min`` / ``P_equ`` / ``P_max``.
    live_delta
        LoRA delta dict produced by
        :func:`experiments.g6_studio_path_a.lora_train_step.train_subdomain_lora`.
        **Mutated in place** : the entry at ``primary_key`` is
        rewritten by the DOWNSCALE / RESTRUCTURE / RECOMBINE
        handlers so the next subdomain's eval sees the dream-mutated
        adapter (this is the load-bearing fix vs Path B).
    seed
        Cell seed (forwarded into the deterministic RNG seeding the
        REPLAY beta records).
    subdomain
        Subdomain label (used to construct the DR-1 episode_id stamp
        ``g6s-<profile>-<subdomain>-seed<seed>``).
    prior_deltas
        List of prior-stack adapter deltas, fed into the OPLoRA
        projector via the RESTRUCTURE handler. Empty for the first
        subdomain in the CL stream.
    sibling_deltas
        List of sibling-stack deltas merged by RECOMBINE
        (TIES-Merge). Includes the in-flight ``live_delta`` plus
        any prior-subdomain deltas the driver retains.
    primary_key
        LoRA tensor name to mutate. Defaults to
        :data:`DEFAULT_PRIMARY_KEY`.

    Returns
    -------
    dict[str, NDArray[np.float32]]
        The same ``live_delta`` reference passed in (mutated). Callers
        may discard the return value and rely on the in-place update.

    Raises
    ------
    ValueError
        ``profile`` is not in :data:`PROFILE_OPS_REAL`.
    KeyError
        ``primary_key`` is missing from ``live_delta`` while a
        non-empty op list is requested.
    """
    if profile not in PROFILE_OPS_REAL:
        raise ValueError(
            f"unknown profile {profile!r} ; expected one of "
            f"{sorted(PROFILE_OPS_REAL)}"
        )
    ops = PROFILE_OPS_REAL[profile]
    if not ops:
        # baseline arm — return live_delta unchanged.
        return live_delta
    if primary_key not in live_delta:
        raise KeyError(
            f"live_delta missing primary_key {primary_key!r} "
            f"(present keys : {sorted(live_delta)})"
        )

    episode_id = f"g6s-{profile}-{subdomain}-seed{seed}"
    rng = np.random.default_rng(seed)

    # 1. REPLAY → A-Walker/Stickgold replay vector aggregate. The
    # handler is pure (returns a vector, does not mutate state) ;
    # we run it for DR-0 accountability + side-effect-free symmetry
    # with Path B.
    if "replay" in ops:
        primary_shape = live_delta[primary_key].shape
        beta_records = [
            {
                "input": rng.standard_normal(primary_shape[0])
                .astype(np.float32)
                .tolist(),
            }
            for _ in range(4)
        ]
        substrate.replay_handler_factory()(beta_records, 20)

    # 2. DOWNSCALE → B-Tononi SHY multiplicative shrink. Mutates
    # live_delta[primary_key] in place (the load-bearing fix vs
    # Path B's synthetic-payload spectator).
    if "downscale" in ops:
        downscale = substrate.downscale_handler_factory()
        live_delta[primary_key] = downscale(
            live_delta[primary_key], DEFAULT_SHRINK_FACTOR,
        )

    # 3. RESTRUCTURE → D-Friston FEP / OPLoRA projection. Builds
    # ``P = I - U U^T`` from prior_deltas and replaces
    # live_delta[primary_key] with ``P @ live_delta[primary_key]``.
    # DR-0 stamp lands on substrate._restructure_state.
    if "restructure" in ops:
        adapter_view: dict[str, Any] = {
            primary_key: live_delta[primary_key],
            "prior_deltas": list(prior_deltas),
            "episode_id": episode_id,
        }
        substrate.restructure_handler_factory()(
            adapter_view, "oplora", primary_key,
        )
        live_delta[primary_key] = np.asarray(
            adapter_view[primary_key], dtype=np.float32,
        )

    # 4. RECOMBINE → C-Hobson recombine / TIES-Merge over the
    # sibling stack (the live delta + sibling_deltas accumulated by
    # the driver across subdomains). DR-0 stamp lands on
    # substrate._recombine_state.
    if "recombine" in ops:
        recombine = substrate.recombine_handler_factory(
            alpha=PROFILE_RECOMBINE_ALPHA.get(profile, 1.0),
        )
        merged = recombine(
            {
                "deltas": [live_delta[primary_key]] + list(sibling_deltas),
                "episode_id": episode_id,
            },
            "ties",
        )
        live_delta[primary_key] = np.asarray(merged, dtype=np.float32)

    return live_delta


__all__ = [
    "DEFAULT_PRIMARY_KEY",
    "DEFAULT_SHRINK_FACTOR",
    "PROFILE_OPS_REAL",
    "PROFILE_RECOMBINE_ALPHA",
    "dream_episode_real",
]
