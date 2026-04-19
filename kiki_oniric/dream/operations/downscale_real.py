"""Real-weight downscale op — MLX weight shrinkage + S2 finite guard.

Cycle-3 C3.3 counterpart to
:mod:`kiki_oniric.dream.operations.downscale` that actually shrinks
every weight + bias tensor in ``model.layers`` by ``shrink_factor``
and then immediately runs the S2 finite guard — catching NaN / Inf
that a bad factor or a poisoned checkpoint could inject.

Contract :

- ``shrink_factor`` read from ``episode.input_slice`` ; must lie in
  ``(0, 1]``.
- Every ``layer.weight`` and ``layer.bias`` in ``model.layers`` is
  multiplied by the factor in-place, then ``mx.eval`` is forced.
- After the multiplication, :func:`check_finite` is invoked over the
  updated weights ; an S2 violation raises
  :class:`FiniteGuardError` (test 6).
- ``state.compound_factor *= shrink_factor`` accumulates the
  multiplicative drift across calls.
- ``state.last_compute_flops`` is tagged with a rough cost estimate
  so K1 budget accounting works downstream.

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kiki_oniric.dream.episode import DreamEpisode
from kiki_oniric.dream.guards.finite import FiniteGuardError, check_finite


@dataclass
class DownscaleRealState:
    """K1-tagged downscale state across multiple episodes."""

    compound_factor: float = 1.0
    last_compute_flops: int = 0


def _param_count(model) -> int:
    """Total scalar parameter count across ``model.layers``."""
    total = 0
    for layer in model.layers:
        w = getattr(layer, "weight", None)
        b = getattr(layer, "bias", None)
        if w is not None:
            total += int(w.size)
        if b is not None:
            total += int(b.size)
    return total


def downscale_real_handler(
    state: DownscaleRealState,
    *,
    model,
) -> Callable[[DreamEpisode], None]:
    """Build a real-weight downscale handler bound to ``state``.

    Imports MLX + numpy lazily so pure-synthetic code paths don't
    pay the cost.
    """
    import mlx.core as mx
    import numpy as np

    def handler(episode: DreamEpisode) -> None:
        factor = episode.input_slice.get("shrink_factor", 1.0)
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )

        for layer in model.layers:
            w = getattr(layer, "weight", None)
            b = getattr(layer, "bias", None)
            if w is not None:
                layer.weight = w * factor
            if b is not None:
                layer.bias = b * factor

        mx.eval(*[layer.weight for layer in model.layers])

        # S2 finite guard — check every weight + bias after shrink.
        # Convert via np.asarray (mx.array supports buffer protocol)
        # so `check_finite` can apply its NaN / Inf / |w|>w_max rules.
        weights_by_layer: dict[str, np.ndarray] = {}
        for idx, layer in enumerate(model.layers):
            if getattr(layer, "weight", None) is not None:
                weights_by_layer[f"layer_{idx}_weight"] = np.asarray(
                    layer.weight
                )
            if getattr(layer, "bias", None) is not None:
                weights_by_layer[f"layer_{idx}_bias"] = np.asarray(
                    layer.bias
                )
        # Let FiniteGuardError propagate verbatim — the handler
        # contract is "raise on S2 violation".
        check_finite(weights_by_layer)

        state.compound_factor *= factor
        # K1 tag : ~1 multiply per scalar parameter.
        state.last_compute_flops = max(_param_count(model), 1)

    return handler


__all__ = [
    "DownscaleRealState",
    "downscale_real_handler",
    # Re-export FiniteGuardError so test imports read naturally.
    "FiniteGuardError",
]
