"""Unit tests for α-stream raw traces ring buffer (C2.5)."""
from __future__ import annotations

import numpy as np
import pytest

from kiki_oniric.dream.channels.alpha_stream import (
    AlphaStreamBuffer,
    AlphaStreamError,
    TraceRecord,
)


def test_ring_buffer_wraps_at_capacity() -> None:
    """Buffer capacity N : appending N+k records evicts k oldest."""
    buf = AlphaStreamBuffer(capacity=3, order="fifo")
    for i in range(5):
        buf.append(TraceRecord(
            tokens=np.array([i], dtype=np.int32),
            activations=np.array([float(i)], dtype=np.float32),
            attention=np.zeros(1, dtype=np.float32),
            errors=np.array([0.1 * i], dtype=np.float32),
        ))
    snapshot = buf.snapshot()
    assert len(snapshot) == 3
    # FIFO: oldest kept are the 3 most recent
    assert [r.tokens[0] for r in snapshot] == [2, 3, 4]


def test_ring_buffer_fifo_vs_lifo_order() -> None:
    """FIFO reads in insertion order ; LIFO reads reversed."""
    buf_fifo = AlphaStreamBuffer(capacity=4, order="fifo")
    buf_lifo = AlphaStreamBuffer(capacity=4, order="lifo")
    records = [
        TraceRecord(
            tokens=np.array([i], dtype=np.int32),
            activations=np.zeros(1, dtype=np.float32),
            attention=np.zeros(1, dtype=np.float32),
            errors=np.zeros(1, dtype=np.float32),
        )
        for i in range(3)
    ]
    for r in records:
        buf_fifo.append(r)
        buf_lifo.append(r)
    fifo_ids = [r.tokens[0] for r in buf_fifo.snapshot()]
    lifo_ids = [r.tokens[0] for r in buf_lifo.snapshot()]
    assert fifo_ids == [0, 1, 2]
    assert lifo_ids == [2, 1, 0]


def test_ring_buffer_rejects_non_finite_retention() -> None:
    """S2 invariant propagation : NaN/Inf in activations rejected."""
    buf = AlphaStreamBuffer(capacity=2, order="fifo")
    bad_record = TraceRecord(
        tokens=np.array([0], dtype=np.int32),
        activations=np.array([float("nan")], dtype=np.float32),
        attention=np.zeros(1, dtype=np.float32),
        errors=np.zeros(1, dtype=np.float32),
    )
    with pytest.raises(AlphaStreamError, match="finite"):
        buf.append(bad_record)


def test_ring_buffer_respects_capacity_integrity() -> None:
    """Capacity must be > 0 ; integrity check on invalid size."""
    with pytest.raises(ValueError, match="capacity"):
        AlphaStreamBuffer(capacity=0, order="fifo")
    with pytest.raises(ValueError, match="capacity"):
        AlphaStreamBuffer(capacity=-1, order="fifo")


def test_ring_buffer_order_validation() -> None:
    """Order must be 'fifo' or 'lifo'."""
    with pytest.raises(ValueError, match="order"):
        # Intentionally invalid literal to exercise the guard.
        AlphaStreamBuffer(capacity=4, order="invalid")  # type: ignore[arg-type]


def test_ring_buffer_len_reflects_live_count() -> None:
    """len() returns count of records currently held."""
    buf = AlphaStreamBuffer(capacity=3, order="fifo")
    assert len(buf) == 0
    for i in range(5):
        buf.append(TraceRecord(
            tokens=np.array([i], dtype=np.int32),
            activations=np.zeros(1, dtype=np.float32),
            attention=np.zeros(1, dtype=np.float32),
            errors=np.zeros(1, dtype=np.float32),
        ))
    assert len(buf) == 3  # capacity cap
