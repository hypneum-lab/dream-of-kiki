"""Stable public API for the GENIAL framework axioms (DR-0..DR-4, DR-2').

The 6 axioms of the framework C design spec §6.2 are exposed as frozen
`Axiom` dataclass instances so runtime code, harness scripts, and
downstream projects (nerve-wml, micro-kiki) can query axiom metadata
and invoke executable predicates without importing test-only code.

Each axiom carries its formal statement (verbatim from spec §6.2),
spec cross-references, the DualVer string of its last-modification,
pointers to its executable conformance tests, amendment pointers, and
an optional `predicate` callable. Axioms with no simple Python
predicate (DR-0, DR-1, DR-3 — runtime invariants validated via
property tests) have `predicate=None`; their `test_references` are the
source of truth.

Version: C-v0.8.0+PARTIAL (FC-MINOR bump, issue #5).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

from kiki_oniric.dream.episode import Operation


@dataclass(frozen=True)
class Axiom:
    """Stable API surface for a GENIAL axiom.

    Attributes
    ----------
    name : str
        Short axiom identifier (e.g. ``"DR-0"``, ``"DR-2'"``).
    title : str
        Human-readable axiom title.
    formal_statement : str
        Mathematical statement copied verbatim from spec §6.2.
    spec_section : str
        Section reference in the framework C design spec.
    test_references : tuple[str, ...]
        Relative paths to the canonical conformance test files.
    version : str
        DualVer string of the most recent modification to this axiom.
    amendments : tuple[str, ...]
        Relative paths to amendment docs (empty for unchanged axioms).
    predicate : Callable[..., bool] | None
        Optional Python callable evaluating the axiom on a concrete
        input. None for axioms without a simple predicate form.
    """

    name: str
    title: str
    formal_statement: str
    spec_section: str
    test_references: tuple[str, ...]
    version: str
    amendments: tuple[str, ...] = field(default_factory=tuple)
    predicate: Callable[..., bool] | None = None


_CURRENT_VERSION: str = "C-v0.8.0+PARTIAL"


# --- Predicate helpers ---


def _dr2_precondition(permutation: tuple[Operation, ...]) -> bool:
    """Return True iff the permutation satisfies weakened DR-2's precondition.

    The weakened DR-2 (2026-04-21, commit db35e1b) rejects any
    permutation π such that ∃ i<j with π_i = RESTRUCTURE ∧ π_j = REPLAY.
    This function returns True exactly when no such (i, j) pair exists
    — i.e. the permutation is safe to compose under the weakened axiom.
    """
    try:
        idx_restructure = permutation.index(Operation.RESTRUCTURE)
        idx_replay = permutation.index(Operation.REPLAY)
    except ValueError:
        return True
    return not (idx_restructure < idx_replay)


_CANONICAL_ORDER: tuple[Operation, ...] = (
    Operation.REPLAY,
    Operation.DOWNSCALE,
    Operation.RESTRUCTURE,
    Operation.RECOMBINE,
)


def _dr2_prime_canonical_order(permutation: tuple[Operation, ...]) -> bool:
    """Return True iff the permutation equals the canonical DR-2' order."""
    return tuple(permutation) == _CANONICAL_ORDER


def _dr4_inclusion(
    smaller_ops: set[Operation], larger_ops: set[Operation]
) -> bool:
    """Return True iff ``smaller_ops`` is a subset of ``larger_ops``.

    Encodes the DR-4 profile-chain inclusion check. Caller provides the
    two op-sets; the predicate is substrate-agnostic and does not
    introspect any profile implementation directly.
    """
    return smaller_ops <= larger_ops


# --- Canonical axiom registry ---


DR0 = Axiom(
    name="DR-0",
    title="Accountability",
    formal_statement=(
        "∀ δ ∈ dream_output_channels,\n"
        "∃ DE ∈ History : budget(DE) < ∞ ∧ δ ∈ outputs(DE)"
    ),
    spec_section="§6.2",
    test_references=(
        "tests/conformance/axioms/test_dr0_accountability.py",
    ),
    version=_CURRENT_VERSION,
    predicate=None,
)


DR1 = Axiom(
    name="DR-1",
    title="Episodic conservation",
    formal_statement=(
        "∀ e ∈ β_t, ∃ t' ∈ [t, t + τ_max] : e ∈ inputs(DE_{t'})"
    ),
    spec_section="§6.2",
    test_references=(
        "tests/conformance/axioms/test_dr1_episodic_conservation.py",
    ),
    version=_CURRENT_VERSION,
    predicate=None,
)


DR2 = Axiom(
    name="DR-2",
    title="Compositionality (weakened with precondition)",
    formal_statement=(
        "∀ permutation π = (op_0, ..., op_{n-1}) over Op such that\n"
        "  ¬∃ i < j : (π_i = RESTRUCTURE ∧ π_j = REPLAY),\n"
        "  π is composable into Op\n"
        "  ∧ budget(π) = Σ_k budget(π_k)\n"
        "  ∧ effect(π, s) = effect(π_{n-1}, ..., effect(π_0, s))"
    ),
    spec_section="§6.2",
    test_references=(
        "tests/conformance/axioms/test_dr2_compositionality_empirical.py",
    ),
    version=_CURRENT_VERSION,
    amendments=(
        "docs/specs/amendments/"
        "2026-04-21-dr2-empirical-falsification.md",
    ),
    predicate=_dr2_precondition,
)


DR2_PRIME = Axiom(
    name="DR-2'",
    title="Canonical-order compositionality (fallback)",
    formal_statement=(
        "∀ op_1, op_2 ∈ Op_canonical_order,\n"
        "  op_2 ∘ op_1 ∈ Op\n"
        "  (canonical order = replay < downscale < restructure < recombine)"
    ),
    spec_section="§6.2",
    test_references=(
        "tests/conformance/axioms/test_dr2_prime_canonical_order.py",
    ),
    version=_CURRENT_VERSION,
    predicate=_dr2_prime_canonical_order,
)


DR3 = Axiom(
    name="DR-3",
    title="Substrate-agnosticism (operational criterion)",
    formal_statement=(
        "∀ substrate S, if S satisfies the Conformance Criterion below,\n"
        "then DR-0, DR-1, DR-2 (or DR-2') are empirically validated on S\n"
        "(operational sense, see §6.2 'Operational statement' ; not a\n"
        "formal implication — DR-2 itself remains a weakened axiom\n"
        "with precondition, §6.2)."
    ),
    spec_section="§6.2",
    test_references=(
        "tests/conformance/axioms/test_dr3_substrate.py",
        "tests/conformance/axioms/test_dr3_esnn_substrate.py",
    ),
    version=_CURRENT_VERSION,
    predicate=None,
)


DR4 = Axiom(
    name="DR-4",
    title="Profile chain inclusion",
    formal_statement=(
        "ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)\n"
        "∧ channels(P_min) ⊆ channels(P_equ) ⊆ channels(P_max)"
    ),
    spec_section="§6.2",
    test_references=(
        "tests/conformance/axioms/test_dr4_profile_inclusion.py",
    ),
    version=_CURRENT_VERSION,
    predicate=_dr4_inclusion,
)


__all__ = [
    "Axiom",
    "DR0",
    "DR1",
    "DR2",
    "DR2_PRIME",
    "DR3",
    "DR4",
]
