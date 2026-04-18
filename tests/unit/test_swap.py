"""Unit tests for swap protocol skeleton (S1 + S2 guards)."""
from __future__ import annotations

import math

import numpy as np
import pytest

from kiki_oniric.dream.swap import (
    SwapAborted,
    SwapResult,
    swap_atomic,
)


def test_swap_succeeds_with_clean_scratch() -> None:
    w_awake = np.array([0.1, 0.2, 0.3])
    w_scratch = np.array([0.11, 0.21, 0.31])
    result = swap_atomic(
        _w_awake=w_awake,
        w_scratch=w_scratch,
        retained_eval=lambda w: 0.95,
        retained_pre_acc=0.95,
        delta_regression=0.02,
    )
    assert isinstance(result, SwapResult)
    assert result.committed is True
    assert np.array_equal(result.w_new, w_scratch)


def test_swap_aborts_on_nan() -> None:
    w_awake = np.array([0.1, 0.2])
    w_scratch = np.array([math.nan, 0.2])
    with pytest.raises(SwapAborted, match="S2"):
        swap_atomic(
            _w_awake=w_awake,
            w_scratch=w_scratch,
            retained_eval=lambda w: 0.95,
            retained_pre_acc=0.95,
            delta_regression=0.02,
        )


def test_swap_aborts_on_retained_regression() -> None:
    w_awake = np.array([0.1, 0.2])
    w_scratch = np.array([0.5, 0.6])
    with pytest.raises(SwapAborted, match="S1"):
        swap_atomic(
            _w_awake=w_awake,
            w_scratch=w_scratch,
            retained_eval=lambda w: 0.50,  # huge regression
            retained_pre_acc=0.95,
            delta_regression=0.02,
        )


def test_swap_passes_marginal_regression_within_threshold() -> None:
    """Regression within delta_regression is acceptable."""
    w_awake = np.array([0.1])
    w_scratch = np.array([0.1])
    result = swap_atomic(
        _w_awake=w_awake,
        w_scratch=w_scratch,
        retained_eval=lambda w: 0.94,  # 1pp below pre, within 2%
        retained_pre_acc=0.95,
        delta_regression=0.02,
    )
    assert result.committed is True
