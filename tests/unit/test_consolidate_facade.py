"""Unit tests for the public `kiki_oniric.consolidate()` facade.

The facade is composition-only over already-tested primitives
(`DreamRuntime`, the 4 op handlers, the 3 profiles). These tests
pin the public contract : output shape / dtype, DR-4 chain
inclusion (P_min delta subset of P_equ delta), and R1 bit-exact
determinism.

Reference: GitHub issue #14 (hypneum-lab/dream-of-kiki).
"""
from __future__ import annotations

import numpy as np
import pytest

from kiki_oniric import consolidate
from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_max import PMaxProfile
from kiki_oniric.profiles.p_min import PMinProfile


def _synthetic_trace(n: int, seed: int) -> np.ndarray:
    """Build a deterministic [n, 4] int32 event trace.

    Columns: (src_wml, dst_wml, code, phase_clock).
    """
    rng = np.random.default_rng(seed)
    src = rng.integers(0, 5, size=n)
    dst = rng.integers(0, 5, size=n)
    code = rng.integers(0, 8, size=n)
    phase = rng.integers(0, 6, size=n)
    return np.stack([src, dst, code, phase], axis=1).astype(np.int32)


def test_consolidate_zero_trace_returns_zero_delta() -> None:
    """Empty trace -> no ops fire -> zero delta of correct shape."""
    trace = np.zeros((0, 4), dtype=np.int32)
    delta = consolidate(
        trace, PEquProfile(), n_transducers=3, alphabet_size=8
    )
    assert delta.shape == (3, 8, 8)
    assert delta.dtype == np.float32
    assert (delta == 0).all()


def test_consolidate_default_shape_and_dtype() -> None:
    """Default n_transducers=12, alphabet_size=64 honoured."""
    trace = _synthetic_trace(n=20, seed=1)
    delta = consolidate(trace, PEquProfile())
    assert delta.shape == (12, 64, 64)
    assert delta.dtype == np.float32


def test_consolidate_p_min_subset_of_p_equ() -> None:
    """DR-4 chain inclusion at the facade level.

    Every non-zero cell of the P_min delta must also be non-zero
    in the P_equ delta (P_equ activates a superset of P_min ops).
    """
    trace = _synthetic_trace(n=100, seed=0)
    delta_min = consolidate(
        trace, PMinProfile(), n_transducers=3, alphabet_size=8
    )
    delta_equ = consolidate(
        trace, PEquProfile(), n_transducers=3, alphabet_size=8
    )
    assert ((delta_min != 0) <= (delta_equ != 0)).all()


def test_consolidate_p_equ_subset_of_p_max() -> None:
    """DR-4 chain inclusion continues P_equ subset of P_max."""
    trace = _synthetic_trace(n=100, seed=3)
    delta_equ = consolidate(
        trace, PEquProfile(), n_transducers=3, alphabet_size=8
    )
    delta_max = consolidate(
        trace, PMaxProfile(), n_transducers=3, alphabet_size=8
    )
    assert ((delta_equ != 0) <= (delta_max != 0)).all()


def test_consolidate_bit_exact_determinism_dr4() -> None:
    """Same trace + profile -> bit-exact delta (R1 contract)."""
    trace = _synthetic_trace(n=50, seed=42)
    d1 = consolidate(trace, PEquProfile())
    d2 = consolidate(trace, PEquProfile())
    np.testing.assert_array_equal(d1, d2)


def test_consolidate_p_min_nonzero_when_trace_nonempty() -> None:
    """A non-empty trace produces a non-zero P_min delta."""
    trace = _synthetic_trace(n=40, seed=7)
    delta = consolidate(
        trace, PMinProfile(), n_transducers=3, alphabet_size=8
    )
    assert (delta != 0).any()


def test_consolidate_rejects_wrong_trace_ndim() -> None:
    """Trace must be 2-D [n_events, 4]."""
    with pytest.raises(ValueError, match="shape"):
        consolidate(np.zeros((4,), dtype=np.int32), PEquProfile())


def test_consolidate_rejects_wrong_trace_columns() -> None:
    """Trace must have exactly 4 columns."""
    with pytest.raises(ValueError, match="4 columns"):
        consolidate(np.zeros((3, 5), dtype=np.int32), PEquProfile())


def test_consolidate_rejects_non_finite_trace() -> None:
    """S2 finite: a trace with NaN/Inf must be rejected."""
    trace = np.zeros((2, 4), dtype=np.float64)
    trace[0, 0] = np.inf
    with pytest.raises(ValueError, match="S2"):
        consolidate(trace, PEquProfile())


def test_consolidate_rejects_non_positive_dims() -> None:
    """n_transducers and alphabet_size must be positive."""
    trace = _synthetic_trace(n=5, seed=2)
    with pytest.raises(ValueError, match="n_transducers"):
        consolidate(trace, PEquProfile(), n_transducers=0)
    with pytest.raises(ValueError, match="alphabet_size"):
        consolidate(trace, PEquProfile(), alphabet_size=-1)


def test_consolidate_codes_clamped_into_alphabet() -> None:
    """Codes outside [0, alphabet_size) do not raise; they are
    clamped so the delta tensor never indexes out of bounds."""
    trace = _synthetic_trace(n=30, seed=9)
    # codes range up to 7; alphabet_size=4 forces clamping
    delta = consolidate(
        trace, PEquProfile(), n_transducers=3, alphabet_size=4
    )
    assert delta.shape == (3, 4, 4)
    assert np.isfinite(delta).all()
