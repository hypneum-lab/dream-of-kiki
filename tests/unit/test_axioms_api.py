"""Unit tests for kiki_oniric.axioms public API (issue #5)."""
from __future__ import annotations

import pytest

from kiki_oniric.axioms import (
    DR0,
    DR1,
    DR2,
    DR2_PRIME,
    DR3,
    DR4,
    Axiom,
)
from kiki_oniric.dream.episode import Operation


def test_all_six_axioms_exposed() -> None:
    axioms = (DR0, DR1, DR2, DR2_PRIME, DR3, DR4)
    names = {a.name for a in axioms}
    assert names == {"DR-0", "DR-1", "DR-2", "DR-2'", "DR-3", "DR-4"}


def test_axiom_is_frozen_dataclass() -> None:
    with pytest.raises(Exception):
        DR0.name = "mutated"  # type: ignore[misc]


def test_axiom_is_hashable() -> None:
    {DR0, DR1, DR2, DR2_PRIME, DR3, DR4}


def test_every_axiom_has_nonempty_formal_statement() -> None:
    for axiom in (DR0, DR1, DR2, DR2_PRIME, DR3, DR4):
        assert len(axiom.formal_statement) > 20, axiom.name


def test_every_axiom_points_to_at_least_one_test_file() -> None:
    for axiom in (DR0, DR1, DR2, DR2_PRIME, DR3, DR4):
        assert len(axiom.test_references) >= 1
        for ref in axiom.test_references:
            assert ref.startswith("tests/conformance/axioms/")
            assert ref.endswith(".py")


def test_every_axiom_uses_current_dualver() -> None:
    for axiom in (DR0, DR1, DR2, DR2_PRIME, DR3, DR4):
        assert axiom.version == "C-v0.8.0+PARTIAL"


def test_every_axiom_references_section_6_2() -> None:
    for axiom in (DR0, DR1, DR2, DR2_PRIME, DR3, DR4):
        assert axiom.spec_section == "§6.2"


def test_dr2_amendment_pointer_present() -> None:
    assert any(
        "dr2-empirical-falsification" in path for path in DR2.amendments
    )


def test_dr2_predicate_accepts_safe_permutation() -> None:
    assert DR2.predicate is not None
    canonical = (
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    )
    assert DR2.predicate(canonical) is True


def test_dr2_predicate_rejects_restructure_immediately_before_replay() -> None:
    assert DR2.predicate is not None
    unsafe = (
        Operation.RESTRUCTURE,
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RECOMBINE,
    )
    assert DR2.predicate(unsafe) is False


def test_dr2_predicate_rejects_restructure_distantly_before_replay() -> None:
    assert DR2.predicate is not None
    unsafe = (
        Operation.RESTRUCTURE,
        Operation.DOWNSCALE,
        Operation.RECOMBINE,
        Operation.REPLAY,
    )
    assert DR2.predicate(unsafe) is False


def test_dr2_predicate_accepts_replay_before_restructure() -> None:
    assert DR2.predicate is not None
    safe = (
        Operation.REPLAY,
        Operation.RESTRUCTURE,
        Operation.DOWNSCALE,
        Operation.RECOMBINE,
    )
    assert DR2.predicate(safe) is True


def test_dr2_prime_predicate_accepts_canonical_order() -> None:
    assert DR2_PRIME.predicate is not None
    canonical = (
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    )
    assert DR2_PRIME.predicate(canonical) is True


def test_dr2_prime_predicate_rejects_non_canonical() -> None:
    assert DR2_PRIME.predicate is not None
    non_canonical = (
        Operation.DOWNSCALE,
        Operation.REPLAY,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    )
    assert DR2_PRIME.predicate(non_canonical) is False


def test_dr4_inclusion_predicate_subset_holds() -> None:
    assert DR4.predicate is not None
    p_min = {Operation.REPLAY, Operation.DOWNSCALE}
    p_equ = {
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    }
    assert DR4.predicate(p_min, p_equ) is True


def test_dr4_inclusion_predicate_subset_violated() -> None:
    assert DR4.predicate is not None
    p_min = {Operation.REPLAY, Operation.DOWNSCALE}
    p_equ = {
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    }
    assert DR4.predicate(p_equ, p_min) is False


def test_non_predicate_axioms_have_none() -> None:
    for axiom in (DR0, DR1, DR3):
        assert axiom.predicate is None


def test_axiom_dataclass_is_public_export() -> None:
    assert Axiom.__module__ == "kiki_oniric.axioms"
