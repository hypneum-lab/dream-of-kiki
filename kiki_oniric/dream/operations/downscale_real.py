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
from typing import TYPE_CHECKING, Callable

from kiki_oniric.dream.episode import DreamEpisode
from kiki_oniric.dream.guards.finite import FiniteGuardError, check_finite

if TYPE_CHECKING:
    from kiki_oniric.dream.channels import WeightUpdate


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


def _flop_estimate_downscale_lora(model) -> int:
    """Rough FLOP count for a LoRA-adapter shrinkage step.

    Dominated by the per-layer elementwise scale of the A/B
    adapters: ``rank * (in + out)`` ops per layer, times two
    (forward + scratch). Smaller than a full-weight shrinkage.
    """
    per_layer = sum(
        2 * layer.rank * (layer.in_features + layer.out_features)
        for layer in model.layers
    )
    return max(per_layer, 1)


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

        # Collect every mutated tensor (weight + bias) across layers,
        # skipping None entries so we never pass a missing attribute to
        # ``mx.eval`` — which would raise AttributeError / TypeError.
        tensors_to_eval = []
        for layer in model.layers:
            if getattr(layer, "weight", None) is not None:
                tensors_to_eval.append(layer.weight)
            if getattr(layer, "bias", None) is not None:
                tensors_to_eval.append(layer.bias)
        if tensors_to_eval:
            mx.eval(*tensors_to_eval)

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


def downscale_lora_handler(
    state: DownscaleRealState,
    *,
    model,  # LoRAModel — typed loosely for lazy MLX import
) -> Callable[[DreamEpisode], "WeightUpdate | None"]:
    """Build a LoRA-only downscale handler that emits a ``WeightUpdate``.

    Reads ``shrink_factor`` from the episode (default ``1.0``),
    validates ``0 < factor <= 1``, then multiplies every
    ``LoRALinear``'s ``lora_a`` / ``lora_b`` by the factor in place.
    The frozen base weight is not touched — SHY shrinkage on the
    LoRA substrate targets the adaptation only. ``factor == 1.0``
    is an S1 no-op (returns ``None``, FLOPs 0). Returns
    ``WeightUpdate(lora_delta=..., fisher_bump=None)`` with the
    per-adapter low-rank deltas keyed as ``adapter_parameters()``.

    Reference:
      docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
    """
    import mlx.core as mx

    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.substrates.micro_kiki.lora_model import adapter_delta

    def handler(episode: DreamEpisode) -> "WeightUpdate | None":
        factor = episode.input_slice.get("shrink_factor", 1.0)
        # Validate BEFORE any mutation — S2 / operations/CLAUDE.md.
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )

        if factor == 1.0:
            # S1 no-op branch : zero compute, no emission.
            state.last_compute_flops = 0
            return None

        # Snapshot adapters before shrinkage. MLX arrays are
        # immutable values; the optimizer rebinds them, so
        # ``mx.array(v)`` is a defensive detach.
        before = {
            k: mx.array(v)
            for k, v in model.adapter_parameters().items()
        }

        # Per-layer shrinkage of A/B adapters. Base weight + bias
        # are frozen by LoRALinear and intentionally untouched.
        for layer in model.layers:
            layer.lora_a = layer.lora_a * factor
            layer.lora_b = layer.lora_b * factor

        mx.eval(model.parameters())
        after = model.adapter_parameters()

        flops = _flop_estimate_downscale_lora(model)
        state.compound_factor *= factor
        state.last_compute_flops = flops

        return WeightUpdate(
            lora_delta=adapter_delta(before, after),
            fisher_bump=None,
        )

    return handler


__all__ = [
    "DownscaleRealState",
    "downscale_real_handler",
    "downscale_lora_handler",
    # Re-export FiniteGuardError so test imports read naturally.
    "FiniteGuardError",
]
