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
