"""K-coupling invariant — SO x fast-spindle phase-coupling measurement.

Reference: docs/invariants/registry.md (K2),
docs/proofs/k2-coupling-evidence.md, framework-C spec §5,
empirical anchor `elife2025bayesian` (eLife 2025,
coupling strength 0.33 [0.27, 0.39]).
"""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import pytest

from kiki_oniric.core.observables import PhaseCouplingObservable
from kiki_oniric.dream.guards.coupling import (
    CouplingGuardError,
    check_coupling_in_window,
)


def _is_protocol(cls: type) -> bool:
    return Protocol in getattr(cls, "__mro__", ())


def test_phase_coupling_observable_is_runtime_checkable() -> None:
    """Protocol must be @runtime_checkable so isinstance() works."""
    assert runtime_checkable(PhaseCouplingObservable) is PhaseCouplingObservable


def test_phase_coupling_observable_is_protocol() -> None:
    """Structural test mirrors test_dr3_substrate.test_all_8_protocols_declared."""
    assert _is_protocol(PhaseCouplingObservable)


def test_k2_guard_passes_inside_window() -> None:
    """Mid-window value must pass silently."""
    check_coupling_in_window(0.33, ci_low=0.27, ci_high=0.39)


def test_k2_guard_rejects_below_window() -> None:
    """Value < ci_low must raise."""
    with pytest.raises(CouplingGuardError, match="below"):
        check_coupling_in_window(0.20, ci_low=0.27, ci_high=0.39)


def test_k2_guard_rejects_above_window() -> None:
    """Value > ci_high must raise."""
    with pytest.raises(CouplingGuardError, match="above"):
        check_coupling_in_window(0.50, ci_low=0.27, ci_high=0.39)


def test_k2_guard_rejects_nan() -> None:
    """NaN slips through naive comparisons; explicit guard required."""
    with pytest.raises(CouplingGuardError, match="NaN"):
        check_coupling_in_window(math.nan, ci_low=0.27, ci_high=0.39)


def test_k2_guard_rejects_inverted_window() -> None:
    """ci_low > ci_high is a programmer error, must raise."""
    with pytest.raises(ValueError, match="ci_low"):
        check_coupling_in_window(0.33, ci_low=0.50, ci_high=0.10)
