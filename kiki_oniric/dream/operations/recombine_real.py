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
- B4 (issue #15) widens the return type to ``LatentSample | None`` —
  the handler builds and returns a channel-2 ``LatentSample`` whose
  ``latent_vector`` is the sampled ``z``. ``species`` comes from
  ``input_slice`` (default ``"default"``); ``provenance`` is auto-
  derived from ``(episode_id, episode_count, key_seed)``.

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable, Protocol, runtime_checkable

import numpy as np

if TYPE_CHECKING:
    import mlx.core as mx

    from kiki_oniric.dream.channels import LatentSample
    from kiki_oniric.dream.episode import DreamEpisode


@runtime_checkable
class VAEEncoder(Protocol):
    """Callable interface for a VAE encoder used by ``recombine_real_handler``.

    The encoder takes an input tensor and returns a ``(mu, log_var)``
    pair. MLX ``nn.Module`` subclasses with the matching ``__call__``
    signature satisfy this Protocol structurally. The Protocol is
    ``runtime_checkable`` for ad-hoc validation, but note that runtime
    ``isinstance`` checks on Protocols only verify method names, not
    signatures — mypy / pyright will catch signature mismatches at
    static-check time.
    """

    def __call__(
        self, x: "mx.array",
    ) -> "tuple[mx.array, mx.array]": ...


@runtime_checkable
class VAEDecoder(Protocol):
    """Callable interface for a VAE decoder used by ``recombine_real_handler``.

    The decoder takes a latent sample ``z`` and returns a
    reconstruction tensor. See ``VAEEncoder`` for Protocol caveats.
    """

    def __call__(self, z: "mx.array") -> "mx.array": ...


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
    encoder: "VAEEncoder",
    decoder: "VAEDecoder",
    seed: int,
) -> Callable[["DreamEpisode"], "LatentSample | None"]:
    """Build a real-weight recombine handler bound to ``state``.

    Imports MLX lazily so pure-synthetic callers don't pay the cost.
    Returns a ``LatentSample`` (channel 2) carrying the sampled
    latent ``z`` as ``latent_vector``. The decoded output continues
    to live in ``state.last_sample`` for backwards compatibility.
    """
    import mlx.core as mx

    from kiki_oniric.dream.channels import LatentSample

    def handler(episode: "DreamEpisode") -> "LatentSample | None":
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

        # Flatten to a 1-D list[float] — decoder may return a
        # multi-dimensional array (batch / feature axes) and calling
        # float(v) on a nested list would raise TypeError.
        state.last_sample = [
            float(v) for v in np.asarray(sample_arr).ravel().tolist()
        ]
        # K1 tag : encoder + decoder fwd passes over a tiny latent.
        state.last_compute_flops = max(
            2 * (mu.size + sample_arr.size), 1
        )

        # B4 — build the LatentSample BEFORE bumping the counter so
        # `provenance` reads `ep=<state._episode_count>` directly
        # (no off-by-one). If LatentSample.__post_init__ raises (non-
        # finite z), the counter does not advance — sound semantics
        # for "this episode didn't actually emit".
        latent_vector = (
            np.asarray(z, dtype=np.float32).ravel().copy()
        )
        species = episode.input_slice.get("species", "default")
        if not isinstance(species, str):
            raise ValueError(
                f"recombine: species must be str, "
                f"got {type(species).__name__}"
            )
        provenance = (
            f"recombine:de={episode.episode_id}:"
            f"ep={state._episode_count}:seed={key_seed}"
        )
        sample = LatentSample(
            species=species,
            latent_vector=latent_vector,
            provenance=provenance,
        )

        state._episode_count += 1
        return sample

    return handler


__all__ = [
    "RecombineRealState",
    "VAEDecoder",
    "VAEEncoder",
    "recombine_real_handler",
]
