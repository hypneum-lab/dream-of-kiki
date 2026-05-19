"""DR-4 Profile Chain Inclusion — axiom property test.

Verifies ops(P_min) ⊆ ops(P_equ) ⊆ target_ops(P_max) and
channels(P_min) ⊆ channels(P_equ) ⊆ target_channels_out(P_max).

The P_max half is exercised against the cycle-1 skeleton's
target metadata (handlers wired cycle 2) — see the second
section of the file.

Reference: docs/proofs/dr4-profile-inclusion.md
         : docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
"""
from __future__ import annotations

from kiki_oniric.dream.episode import Operation, OutputChannel
from kiki_oniric.profiles.p_max import PMaxProfile
from tests.conformance.axioms._dsl import profile_channels, registered_ops


def test_dr4_ops_inclusion_p_min_subset_p_equ() -> None:
    """ops(P_min) ⊆ ops(P_equ)."""
    ops_min = registered_ops("P_min")
    ops_equ = registered_ops("P_equ")
    assert ops_min <= ops_equ, (
        f"DR-4 violated: ops(P_min)={sorted(o.value for o in ops_min)} "
        f"not subset of ops(P_equ)={sorted(o.value for o in ops_equ)}"
    )


def test_dr4_channels_inclusion_p_min_subset_p_equ() -> None:
    """channels(P_min) ⊆ channels(P_equ) on out-channels."""
    assert set(profile_channels("P_min")) <= set(profile_channels("P_equ"))


def test_dr4_p_equ_strictly_richer_than_p_min() -> None:
    """P_equ should not be equal to P_min — strict superset."""
    ops_min = registered_ops("P_min")
    ops_equ = registered_ops("P_equ")
    assert ops_min != ops_equ, (
        "DR-4: P_equ must be strictly richer than P_min"
    )
    assert set(profile_channels("P_min")) != set(profile_channels("P_equ"))


def test_dr4_p_equ_contains_restructure_and_recombine() -> None:
    """The two ops that distinguish P_equ from P_min must be present."""
    ops_equ = registered_ops("P_equ")
    assert Operation.RESTRUCTURE in ops_equ
    assert Operation.RECOMBINE in ops_equ


# === DR-4 P_equ ⊆ P_max chain (S16.2 extension) ===


def _p_max_metadata() -> PMaxProfile:
    """Read PMaxProfile target metadata (handlers not yet wired)."""
    return PMaxProfile()


def test_dr4_ops_inclusion_p_equ_subset_p_max() -> None:
    """ops(P_equ) ⊆ target_ops(P_max) via skeleton metadata."""
    p_max = _p_max_metadata()
    ops_equ = registered_ops("P_equ")
    assert ops_equ <= p_max.target_ops, (
        "DR-4 violated: ops(P_equ) not subset of "
        "target_ops(P_max)"
    )


def test_dr4_channels_inclusion_p_equ_subset_p_max() -> None:
    """channels(P_equ) ⊆ target_channels_out(P_max)."""
    p_max = _p_max_metadata()
    assert set(profile_channels("P_equ")) <= p_max.target_channels_out
    # P_max strict superset: must contain ATTENTION_PRIOR
    assert (
        OutputChannel.ATTENTION_PRIOR in p_max.target_channels_out
    )
    assert (
        OutputChannel.ATTENTION_PRIOR not in set(profile_channels("P_equ"))
    )
