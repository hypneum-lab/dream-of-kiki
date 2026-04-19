"""Real-weight recombine op — VAE reparameterization over MLX.

Cycle-3 C3.3 counterpart to the light recombine op in
:mod:`kiki_oniric.dream.operations.recombine`. This variant drives a
real encoder / decoder pair through the reparameterization trick
(``z = mu + sigma * epsilon``) and emits a decoded latent sample on
canal 2 (LATENT_SAMPLE).

Determinism contract (mirrors the cycle-2 MLX fix) :

- The handler keeps a per-state episode counter so ``seed +
  episode_count`` drives an isolated :func:`mlx.core.random.key` for
  ``epsilon``. The process-wide MLX RNG is *never* mutated, so
  concurrent dream workers can run multiple recombine handlers
  without interfering — and two handlers built with the same seed
  produce identical samples under identical input (test 8).

Contract :

- ``delta_latents`` read from ``episode.input_slice`` ; must be a
  non-empty list of list[float]. Only ``latents[0]`` is consumed —
  diversity comes from sampling ``z``, not latent selection.
- ``state.last_sample`` stores the decoder output as a list[float]
  (length 4 in the _TinyDecoder test fixture).
- ``state.last_compute_flops`` is tagged with a rough cost estimate.

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable


@dataclass
class RecombineRealState:
    """K1-tagged recombine state across multiple episodes.

    ``last_sample`` is typed as ``list | None`` per the adapter
    spec ; callers can assume ``list[float]`` when populated.
    """

    last_sample: list | None = None
    last_compute_flops: int = 0
    # Episode-counter drives per-call RNG key derivation so same
    # seed + same episode index reproduces byte-identical samples.
    _episode_count: int = 0


def recombine_real_handler(
    state: RecombineRealState,
    *,
    encoder,
    decoder,
    seed: int,
):
    """Build a real-weight recombine handler bound to ``state``.

    Imports MLX lazily so pure-synthetic callers don't pay the cost.
    """
    import mlx.core as mx

    def handler(episode) -> None:
        latents = episode.input_slice.get("delta_latents", [])
        if not latents:
            raise ValueError(
                "I3: delta_latents must not be empty for recombine_real"
            )

        # Per-episode isolated PRNG key — does not touch the
        # process-wide MLX RNG.
        key_seed = seed + state._episode_count
        key = mx.random.key(key_seed)
        _, sample_key = mx.random.split(key)

        x = mx.array(latents[0])
        mu, log_var = encoder(x)
        sigma = mx.exp(0.5 * log_var)
        epsilon = mx.random.normal(shape=mu.shape, key=sample_key)
        z = mu + sigma * epsilon
        sample_arr = decoder(z)
        mx.eval(sample_arr)

        state.last_sample = [float(v) for v in sample_arr.tolist()]
        # K1 tag : encoder + decoder fwd passes over a tiny latent.
        state.last_compute_flops = max(
            2 * (mu.size + sample_arr.size), 1
        )
        state._episode_count += 1

    return handler


__all__ = [
    "RecombineRealState",
    "recombine_real_handler",
]
