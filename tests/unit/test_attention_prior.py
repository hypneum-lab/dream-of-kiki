"""Unit tests for ATTENTION_PRIOR canal-4 emission (C2.7)."""
from __future__ import annotations

import numpy as np
import pytest

from kiki_oniric.dream.channels.attention_prior import (
    AttentionPriorChannel,
    AttentionPriorError,
)


def test_emit_valid_prior_stored() -> None:
    """A valid attention prior is stored and retrievable."""
    channel = AttentionPriorChannel(budget_attention=1.5)
    prior = np.array([0.3, 0.4, 0.5, 0.3])  # sum=1.5, all in [0,1]
    channel.emit(prior)
    retrieved = channel.get_prior()
    np.testing.assert_array_equal(retrieved, prior)


def test_emit_rejects_components_outside_unit_interval() -> None:
    """S4 invariant: each component must be in [0, 1]."""
    channel = AttentionPriorChannel(budget_attention=2.0)
    bad_negative = np.array([0.3, -0.1, 0.5])
    with pytest.raises(AttentionPriorError, match="\\[0, 1\\]"):
        channel.emit(bad_negative)
    bad_above = np.array([0.3, 1.5, 0.2])
    with pytest.raises(AttentionPriorError, match="\\[0, 1\\]"):
        channel.emit(bad_above)


def test_emit_rejects_sum_above_budget() -> None:
    """S4 invariant: sum(prior) <= budget_attention."""
    channel = AttentionPriorChannel(budget_attention=1.0)
    bad_sum = np.array([0.5, 0.6, 0.3])  # sum=1.4 > 1.0
    with pytest.raises(AttentionPriorError, match="budget"):
        channel.emit(bad_sum)


def test_clear_resets_prior() -> None:
    """Clearing returns to None state."""
    channel = AttentionPriorChannel(budget_attention=2.0)
    channel.emit(np.array([0.4, 0.5]))
    assert channel.get_prior() is not None
    channel.clear()
    assert channel.get_prior() is None


def test_get_prior_returns_readonly_view() -> None:
    """S4 contract: get_prior() must not let callers mutate
    internal state and bypass emit() validation.
    """
    channel = AttentionPriorChannel(budget_attention=1.5)
    channel.emit(np.array([0.3, 0.4, 0.5]))
    view = channel.get_prior()
    assert view is not None
    assert view.flags.writeable is False
    # Direct in-place assignment must fail.
    with pytest.raises(ValueError):
        view[0] = 99.0
    # np.copyto must also fail (would otherwise overwrite the buffer).
    with pytest.raises(ValueError):
        np.copyto(view, np.array([99.0, 99.0, 99.0]))


def test_get_prior_returns_none_when_uninitialized() -> None:
    """get_prior() returns None on a fresh channel (no view error)."""
    channel = AttentionPriorChannel(budget_attention=1.5)
    assert channel.get_prior() is None
