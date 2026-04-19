"""Unit tests for the combined Bonferroni family helper (cycle-3 C3.5).

Cycle 1 used a 4-test family {H1, H2, H3, H4} with α_per_test = 0.0125.
Cycle 3 extends this to an 8-test family
{H1, H2, H3, H4, H5-I, H5-II, H5-III, H6} with α_per_test = 0.00625.
The BonferroniFamily dataclass and apply_bonferroni_family helper
expose both families under a uniform surface so the eval reporter
can stay generic.
"""
from __future__ import annotations

from kiki_oniric.eval.statistics import (
    CYCLE1_FAMILY,
    CYCLE3_FAMILY,
    BonferroniFamily,
    apply_bonferroni_family,
)


def test_bonferroni_family_computes_per_test_alpha() -> None:
    """α_per_test = α_global / family_size for both cycle-1 and cycle-3."""
    cycle1 = BonferroniFamily(family_size=4)
    cycle3 = BonferroniFamily(family_size=8)
    assert cycle1.alpha_global == 0.05
    assert cycle3.alpha_global == 0.05
    assert cycle1.alpha_per_test == 0.0125
    assert cycle3.alpha_per_test == 0.00625
    # Module-level constants match the spec values.
    assert CYCLE1_FAMILY.alpha_per_test == 0.0125
    assert CYCLE3_FAMILY.alpha_per_test == 0.00625


def test_apply_bonferroni_family_rejects_per_test() -> None:
    """Each p-value is compared against family.alpha_per_test."""
    family = BonferroniFamily(family_size=8)  # α_per = 0.00625
    p_values = [0.001, 0.005, 0.00624, 0.00626, 0.05, 0.1, 0.5, 0.9]
    rejects = apply_bonferroni_family(p_values, family)
    assert rejects == [True, True, True, False, False, False, False, False]
    assert len(rejects) == len(p_values)


def test_bonferroni_family_cycle1_baseline_consistency() -> None:
    """Cycle-1 family_size=4 preserves the original α=0.0125 baseline.

    Integration-style check : a p-value grid that lies at the cycle-1
    threshold must flip rejection decisions exactly at α=0.0125 when
    run through the generic helper. This protects the existing H1-H4
    tests from a regression introduced by the combined 8-test API.
    """
    p_values = [0.005, 0.010, 0.0124, 0.0125, 0.02, 0.05, 0.1]
    # Cycle-1 : α_per_test = 0.0125 → strictly-less-than is the
    # convention used across statistics.py, so 0.0125 itself does not
    # reject.
    cycle1_rejects = apply_bonferroni_family(p_values, CYCLE1_FAMILY)
    assert cycle1_rejects == [True, True, True, False, False, False, False]
    # Cycle-3 : α_per_test = 0.00625 → only the tightest p-values survive,
    # confirming the combined 8-test family is strictly more conservative
    # than the cycle-1 baseline.
    cycle3_rejects = apply_bonferroni_family(p_values, CYCLE3_FAMILY)
    assert cycle3_rejects == [True, False, False, False, False, False, False]
    # Family size ordering matches rejection conservatism : every cycle-3
    # rejection is also a cycle-1 rejection, never the reverse.
    for c1, c3 in zip(cycle1_rejects, cycle3_rejects, strict=True):
        if c3:
            assert c1, "cycle-3 rejection must imply cycle-1 rejection"
