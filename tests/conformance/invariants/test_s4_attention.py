"""Conformance test for S4 invariant — attention prior bounded."""
from __future__ import annotations

import numpy as np
import pytest

from kiki_oniric.dream.guards.attention import (
    AttentionGuardError,
    check_attention_bounded,
)


def test_s4_passes_valid_prior() -> None:
    """Valid prior (each in [0,1], sum <= budget) passes silently."""
    prior = np.array([0.3, 0.4, 0.2])
    check_attention_bounded(prior, budget=1.0)


def test_s4_rejects_out_of_unit_interval() -> None:
    """S4 must reject any component outside [0, 1]."""
    bad = np.array([0.5, -0.2, 0.4])
    with pytest.raises(AttentionGuardError):
        check_attention_bounded(bad, budget=1.5)


def test_s4_rejects_nan_components() -> None:
    """S4 must reject NaN — comparisons against NaN return False
    so range/budget checks would silently miss it without the
    explicit NaN guard.
    """
    bad = np.array([0.3, float("nan"), 0.4])
    with pytest.raises(AttentionGuardError, match="NaN"):
        check_attention_bounded(bad, budget=1.5)


def test_s4_rejects_all_nan() -> None:
    """An all-NaN prior must also be rejected (sum is NaN, not >budget)."""
    bad = np.array([float("nan"), float("nan")])
    with pytest.raises(AttentionGuardError, match="NaN"):
        check_attention_bounded(bad, budget=1.5)
