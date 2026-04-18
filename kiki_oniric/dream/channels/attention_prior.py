"""ATTENTION_PRIOR canal-4 — meta-cognitive guidance (P_max only).

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §2.1
    Dream → Awake channel 4 (ATTENTION_PRIOR).

Each emission validated by S4 guard before storage. Live read-only
access via get_prior() ; clear() resets to None.

Cycle 2 C2.7 : skeleton implementation for P_max wiring.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kiki_oniric.dream.guards.attention import (
    DEFAULT_BUDGET_ATTENTION,
    AttentionGuardError,
    check_attention_bounded,
)


class AttentionPriorError(Exception):
    """Raised when an emission violates S4 invariant or schema."""


class AttentionPriorChannel:
    """Canal-4 emitter for attention priors (P_max profile).

    Stores the most recent prior, validated against S4 (each
    component in [0, 1], sum <= budget_attention). Awake reader
    consumes via get_prior() at swap or live.
    """

    def __init__(
        self, budget_attention: float = DEFAULT_BUDGET_ATTENTION
    ) -> None:
        self._budget = budget_attention
        self._prior: NDArray | None = None

    def emit(self, prior: NDArray) -> None:
        """Emit a new attention prior.

        Raises AttentionPriorError on S4 violation (component out
        of [0, 1] or sum > budget).
        """
        try:
            check_attention_bounded(prior, budget=self._budget)
        except AttentionGuardError as exc:
            raise AttentionPriorError(str(exc)) from exc
        self._prior = np.asarray(prior).copy()

    def get_prior(self) -> NDArray | None:
        """Return the current prior or None if cleared.

        The returned array is a read-only numpy view : callers cannot
        mutate the channel's internal state by writing into it. Any
        in-place modification (assignment, ``np.copyto``, ufunc with
        ``out=``) raises ``ValueError`` because ``flags.writeable``
        is forced to ``False``. This preserves the S4 validation
        contract enforced by ``emit()`` — every prior visible to
        downstream readers has been bounds-checked.
        """
        if self._prior is None:
            return None
        view = self._prior.view()
        view.flags.writeable = False
        return view

    def clear(self) -> None:
        """Reset to no prior (None)."""
        self._prior = None

    @property
    def budget(self) -> float:
        return self._budget
