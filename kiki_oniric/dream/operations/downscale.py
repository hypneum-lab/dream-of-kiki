"""Downscale operation — B-Tononi SHY synaptic homeostasis source.

Skeleton version (S7.1): records shrink factor + compound product
across episodes. Real weight shrinkage on np.ndarray lands S9+ with
MLX integration.

Mathematical properties (per docs/proofs/op-pair-analysis.md):
- Commutative: downscale_f ∘ downscale_g = downscale_g ∘ downscale_f
  (scalar multiplication commutes).
- NOT idempotent: downscale_f ∘ downscale_f gives weights × f²,
  not weights × f. Repeated shrinkage compounds multiplicatively.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kiki_oniric.dream.episode import DreamEpisode


@dataclass
class DownscaleOpState:
    """Mutable counter state for downscale op across episodes."""

    total_episodes_handled: int = 0
    last_factor_applied: float = 1.0
    compound_factor: float = 1.0


def downscale_handler(
    state: DownscaleOpState,
) -> Callable[[DreamEpisode], None]:
    """Build a downscale handler bound to a state instance.

    Handler reads `shrink_factor` from input_slice (must be in (0, 1]),
    updates state. Real `W *= factor` lands S9+ with MLX.
    """

    def handler(episode: DreamEpisode) -> None:
        factor = episode.input_slice.get("shrink_factor", 1.0)
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )
        state.total_episodes_handled += 1
        state.last_factor_applied = factor
        state.compound_factor *= factor

    return handler


def downscale_handler_mlx(
    state: DownscaleOpState,
    model,  # mlx.nn.Module — typed loosely for lazy import
) -> Callable[[DreamEpisode], None]:
    """Build a downscale handler with real MLX weight shrinkage.

    Walks the model parameter tree, multiplies each leaf array by
    `shrink_factor`, then forces evaluation. State counters updated
    consistently with skeleton handler.

    The skeleton `downscale_handler` remains for tests / contexts
    not requiring MLX. This MLX variant is the production path for
    the G2 GO-FULL gate (real weight shrinkage).

    Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
    """
    import mlx.core as mx

    def _shrink(node, factor: float):
        if isinstance(node, dict):
            return {k: _shrink(v, factor) for k, v in node.items()}
        if isinstance(node, (list, tuple)):
            shrunk = [_shrink(v, factor) for v in node]
            # Handle namedtuples (have _fields attr) vs regular tuples
            if hasattr(node, '_fields'):
                return type(node)(*shrunk)
            return type(node)(shrunk)
        if isinstance(node, mx.array):
            return node * factor
        return node

    def handler(episode: DreamEpisode) -> None:
        factor = episode.input_slice.get("shrink_factor", 1.0)
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )
        new_params = _shrink(model.parameters(), factor)
        model.update(new_params)
        mx.eval(model.parameters())
        state.total_episodes_handled += 1
        state.last_factor_applied = factor
        state.compound_factor *= factor

    return handler
