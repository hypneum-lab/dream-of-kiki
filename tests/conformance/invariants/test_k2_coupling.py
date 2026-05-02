"""K-coupling invariant — SO x fast-spindle phase-coupling measurement.

Reference: docs/invariants/registry.md (K2),
docs/proofs/k2-coupling-evidence.md, framework-C spec §5,
empirical anchor `elife2025bayesian` (eLife 2025,
coupling strength 0.33 [0.27, 0.39]).
"""
from __future__ import annotations

import math
from typing import Protocol, runtime_checkable

import numpy as np
import pytest

from kiki_oniric.core.observables import PhaseCouplingObservable
from kiki_oniric.dream.guards.coupling import (
    CouplingGuardError,
    check_coupling_in_window,
)
from tests.conformance.invariants._synthetic_phase_coupling import (
    SyntheticPhaseCouplingSubstrate,
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


def test_synthetic_substrate_satisfies_protocol() -> None:
    """Synthetic fixture must structurally implement the Protocol."""
    sub = SyntheticPhaseCouplingSubstrate()
    assert isinstance(sub, PhaseCouplingObservable)


def test_synthetic_substrate_returns_aligned_arrays() -> None:
    """Phase + amplitude arrays must have the requested length and fs > 0."""
    sub = SyntheticPhaseCouplingSubstrate()
    phase, amp, fs = sub.emit_phase_coupling_signal(n_samples=2048, seed=7)
    assert phase.shape == (2048,)
    assert amp.shape == (2048,)
    assert phase.dtype.name == "float32"
    assert amp.dtype.name == "float32"
    assert fs > 0.0


def test_synthetic_substrate_is_deterministic() -> None:
    """Same seed -> bit-identical output (R1 reproducibility, parent rule)."""
    sub = SyntheticPhaseCouplingSubstrate()
    p1, a1, _ = sub.emit_phase_coupling_signal(n_samples=512, seed=42)
    p2, a2, _ = sub.emit_phase_coupling_signal(n_samples=512, seed=42)
    np.testing.assert_array_equal(p1, p2)
    np.testing.assert_array_equal(a1, a2)


def test_synthetic_substrate_seeds_are_independent() -> None:
    """Distinct seeds produce distinct realisations (no global state)."""
    sub = SyntheticPhaseCouplingSubstrate()
    _, a1, _ = sub.emit_phase_coupling_signal(n_samples=512, seed=1)
    _, a2, _ = sub.emit_phase_coupling_signal(n_samples=512, seed=2)
    assert not np.array_equal(a1, a2)


def _mean_vector_length(
    phase: np.ndarray, amplitude: np.ndarray
) -> float:
    """Tort 2010-style mean vector length (PAC strength).

    MVL = | mean_t [ amplitude(t) * exp(i * phase(t)) ] | / mean_t amplitude(t)

    Returns a float in [0, 1]. Pure numpy, no SciPy needed; SciPy
    is reserved for any future Hilbert-transform based estimator.
    """
    if phase.shape != amplitude.shape:
        raise ValueError("phase and amplitude must have identical shapes")
    z = amplitude.astype(np.float64) * np.exp(1j * phase.astype(np.float64))
    num = float(np.abs(z.mean()))
    denom = float(np.abs(amplitude.astype(np.float64)).mean())
    if denom == 0.0:
        return 0.0
    return num / denom


def test_estimator_zero_for_random_phase() -> None:
    """No coupling: random uniform phase yields MVL ~= 0 (large N)."""
    rng = np.random.default_rng(0)
    n = 8192
    phase = rng.uniform(-np.pi, np.pi, size=n).astype(np.float32)
    amp = (0.5 + rng.normal(0.0, 0.05, size=n)).astype(np.float32)
    mvl = _mean_vector_length(phase, amp)
    assert mvl < 0.05, f"expected near-zero MVL on random phase, got {mvl}"


def test_estimator_one_for_perfect_coupling() -> None:
    """Perfect coupling: amplitude = 1 only at phase 0 -> MVL = 1.0."""
    n = 1024
    phase = np.zeros(n, dtype=np.float32)
    amp = np.ones(n, dtype=np.float32)
    mvl = _mean_vector_length(phase, amp)
    assert abs(mvl - 1.0) < 1e-6
