"""SNN-substrate recombine op — rate-domain interpolation (cycle-3 C3.12).

Norse-substrate counterpart to
:mod:`kiki_oniric.dream.operations.recombine_real`. Instead of
driving a VAE encoder/decoder pair, this variant performs
VAE-style latent interpolation directly in the spike-rate domain :

    mixed_rate = alpha * rate_a + (1 - alpha) * rate_b

with ``alpha`` drawn from a per-episode isolated PRNG (mirrors the
cycle-2 MLX fix — the process-wide RNG is never mutated, so
concurrent dream workers can run multiple SNN-proxy recombine
handlers without interfering).

Contract :

- ``delta_latents`` read from ``episode.input_slice`` ; must be a
  list of at least 2 latent vectors with consistent dimensions
  (I3 guard, matches skeleton + MLX variants).
- Latents are interpreted as **weight logits** — converted to
  spike rates, interpolated, converted back.
- ``state.last_sample`` stores the decoded sample as a list[float].
- ``state.last_compute_flops`` is tagged at ~3 multiply-adds per
  latent entry.

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2, §6.2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from kiki_oniric.dream.episode import DreamEpisode
from kiki_oniric.dream.operations.replay_snn import (
    spike_rates_to_weights,
    weights_to_spike_rates,
)


@dataclass
class RecombineSNNState:
    """K1-tagged SNN-proxy recombine state across multiple episodes.

    ``last_sample`` is typed as ``list | None`` (matches
    :class:`RecombineRealState`) — callers can assume ``list[float]``
    when populated.
    """

    last_sample: list | None = None
    last_compute_flops: int = 0
    total_compute_flops: int = 0
    # Episode-counter drives per-call PRNG derivation so the same
    # seed + same episode index reproduces a byte-identical
    # ``alpha`` (and therefore the same mixed sample).
    _episode_count: int = 0


def recombine_snn_handler(
    state: RecombineSNNState,
    *,
    seed: int,
    max_rate: float = 100.0,
) -> Callable[[DreamEpisode], None]:
    """Build an SNN-proxy recombine handler bound to ``state``.

    ``seed`` is combined with ``state._episode_count`` at each
    invocation so two handlers built from the same seed and fed
    the same episodes produce identical samples.

    ``max_rate`` must be safely above the ``1e-6`` clipping epsilon
    used downstream — otherwise the ``[1e-6, max_rate - 1e-6]``
    interval inverts and the clipping logic produces non-monotone
    bounds. We refuse any value below ``1e-5`` rather than coerce
    it silently.
    """
    if max_rate <= 1e-5:
        raise ValueError(
            f"max_rate must be > 1e-5 to keep the rate-clipping "
            f"interval [1e-6, max_rate - 1e-6] non-degenerate ; "
            f"got {max_rate!r}"
        )

    def handler(episode: DreamEpisode) -> None:
        latents = episode.input_slice.get("delta_latents", [])
        if not latents:
            raise ValueError(
                "I3: delta_latents must not be empty for recombine_snn"
            )
        if len(latents) < 2:
            raise ValueError(
                f"I3: delta_latents must contain at least 2 "
                f"latents, got {len(latents)}"
            )
        if not all(len(lat) == len(latents[0]) for lat in latents):
            raise ValueError(
                "I3: delta_latents must all have the same "
                "dimensionality"
            )

        # Per-episode isolated PRNG — does not touch global numpy
        # state. Matches the cycle-2 MLX fix pattern.
        key_seed = seed + state._episode_count
        rng = np.random.default_rng(key_seed)
        alpha = float(rng.random())

        w_a = np.asarray(latents[0], dtype=float)
        w_b = np.asarray(latents[1], dtype=float)
        rate_a = weights_to_spike_rates(w_a, max_rate=max_rate)
        rate_b = weights_to_spike_rates(w_b, max_rate=max_rate)
        mixed_rate = alpha * rate_a + (1.0 - alpha) * rate_b
        mixed_rate = np.clip(mixed_rate, 1e-6, max_rate - 1e-6)
        sample_w = spike_rates_to_weights(
            mixed_rate, max_rate=max_rate
        )

        state.last_sample = [float(v) for v in sample_w.ravel()]
        flops = max(3 * int(w_a.size), 1)
        state.last_compute_flops = flops
        state.total_compute_flops += flops
        state._episode_count += 1

    return handler


__all__ = [
    "RecombineSNNState",
    "recombine_snn_handler",
]
