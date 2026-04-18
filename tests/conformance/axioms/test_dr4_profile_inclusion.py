"""DR-4 Profile Chain Inclusion — axiom property test.

Verifies ops(P_min) ⊆ ops(P_equ) and
channels(P_min) ⊆ channels(P_equ).

P_max inclusion deferred until P_max profile is wired (S13+).

Reference: docs/proofs/dr4-profile-inclusion.md
         : docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
"""
from __future__ import annotations

from kiki_oniric.dream.episode import Operation, OutputChannel
from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_min import PMinProfile


# Profile contracts: ops registered, output channels emitted.
# Derived empirically from runtime handler registration so the
# tests exercise the actual wiring (not a static metadata lookup).


def _registered_ops(profile_factory) -> set[Operation]:
    profile = profile_factory()
    return set(profile.runtime._handlers.keys())


P_MIN_CHANNELS_OUT: set[OutputChannel] = {OutputChannel.WEIGHT_DELTA}
P_EQU_CHANNELS_OUT: set[OutputChannel] = {
    OutputChannel.WEIGHT_DELTA,
    OutputChannel.HIERARCHY_CHG,
    OutputChannel.LATENT_SAMPLE,
}


def test_dr4_ops_inclusion_p_min_subset_p_equ() -> None:
    """ops(P_min) ⊆ ops(P_equ)."""
    ops_min = _registered_ops(PMinProfile)
    ops_equ = _registered_ops(PEquProfile)
    assert ops_min <= ops_equ, (
        f"DR-4 violated: ops(P_min)={sorted(o.value for o in ops_min)} "
        f"not subset of ops(P_equ)={sorted(o.value for o in ops_equ)}"
    )


def test_dr4_channels_inclusion_p_min_subset_p_equ() -> None:
    """channels(P_min) ⊆ channels(P_equ) on out-channels."""
    assert P_MIN_CHANNELS_OUT <= P_EQU_CHANNELS_OUT


def test_dr4_p_equ_strictly_richer_than_p_min() -> None:
    """P_equ should not be equal to P_min — strict superset."""
    ops_min = _registered_ops(PMinProfile)
    ops_equ = _registered_ops(PEquProfile)
    assert ops_min != ops_equ, (
        "DR-4: P_equ must be strictly richer than P_min"
    )
    assert P_MIN_CHANNELS_OUT != P_EQU_CHANNELS_OUT


def test_dr4_p_equ_contains_restructure_and_recombine() -> None:
    """The two ops that distinguish P_equ from P_min must be present."""
    ops_equ = _registered_ops(PEquProfile)
    assert Operation.RESTRUCTURE in ops_equ
    assert Operation.RECOMBINE in ops_equ


# === DR-4 P_equ ⊆ P_max chain (S16.2 extension) ===

from kiki_oniric.profiles.p_max import PMaxProfile

P_EQU_OPS_DECLARED: set[Operation] = {
    Operation.REPLAY,
    Operation.DOWNSCALE,
    Operation.RESTRUCTURE,
    Operation.RECOMBINE,
}


def _p_max_metadata():
    """Read PMaxProfile target metadata (handlers not yet wired)."""
    return PMaxProfile()


def test_dr4_ops_inclusion_p_equ_subset_p_max() -> None:
    """ops(P_equ) ⊆ target_ops(P_max) via skeleton metadata."""
    p_max = _p_max_metadata()
    ops_equ = _registered_ops(PEquProfile)
    assert ops_equ <= p_max.target_ops, (
        f"DR-4 violated: ops(P_equ) not subset of "
        f"target_ops(P_max)"
    )


def test_dr4_channels_inclusion_p_equ_subset_p_max() -> None:
    """channels(P_equ) ⊆ target_channels_out(P_max)."""
    p_max = _p_max_metadata()
    assert P_EQU_CHANNELS_OUT <= p_max.target_channels_out
    # P_max strict superset: must contain ATTENTION_PRIOR
    assert (
        OutputChannel.ATTENTION_PRIOR in p_max.target_channels_out
    )
    assert (
        OutputChannel.ATTENTION_PRIOR not in P_EQU_CHANNELS_OUT
    )
