"""Swap protocol skeleton — atomic W_awake ← W_scratch promotion.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §7

Skeleton (S7.3): enforces S2 (finite guard) and S1 (retained
non-regression). S3 (hierarchy guard) is a no-op stub here (real
topology validation lands S9+). K3 swap latency monitoring lands
S9+ with concurrent runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np
from numpy.typing import NDArray

from kiki_oniric.dream.guards.finite import (
    FiniteGuardError,
    check_finite,
)


class SwapAborted(Exception):
    """Raised when a swap guard rejects W_scratch."""


@dataclass(frozen=True)
class SwapResult:
    """Immutable record of a successful swap."""

    w_new: NDArray
    retained_post_acc: float
    committed: bool


def swap_atomic(
    _w_awake: NDArray,  # Reserved for S3 hierarchy guard (S9+)
    w_scratch: NDArray,
    retained_eval: Callable[[NDArray], float],
    retained_pre_acc: float,
    delta_regression: float = 0.02,
) -> SwapResult:
    """Atomic swap W_awake ← W_scratch with S1+S2 guards.

    1. S2 guard: finite + bounded check on W_scratch.
    2. S1 guard: retained_eval(W_scratch) >=
       retained_pre_acc - delta_regression.
    3. Commit: promote W_scratch to W_awake.

    Raises SwapAborted with the violated invariant code (S1 or S2)
    when a guard rejects.
    """
    try:
        check_finite(w_scratch)
    except FiniteGuardError as exc:
        raise SwapAborted(f"S2 guard failed: {exc}") from exc

    retained_post = retained_eval(w_scratch)
    if retained_post < retained_pre_acc - delta_regression:
        raise SwapAborted(
            f"S1 guard failed: retained_post={retained_post} < "
            f"retained_pre={retained_pre_acc} - "
            f"delta={delta_regression}"
        )

    return SwapResult(
        w_new=w_scratch,
        retained_post_acc=retained_post,
        committed=True,
    )
