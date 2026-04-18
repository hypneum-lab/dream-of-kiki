"""Recombine operation — C-Hobson VAE light source (creative branch).

Skeleton "light" version (S11.1): linear interpolation between two
randomly-sampled latents from `delta_latents` input. Real VAE
sampling (encoder/decoder pair) lands S13+ alongside concurrent
dream worker.

Mathematical role (per docs/proofs/op-pair-analysis.md): canonical
parallel branch (§4.3) — recombine runs in parallel with the
serial A-B-D branch to preserve generative diversity. Sampling is
non-deterministic by design; rng injection enables reproducible
tests.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Callable

from kiki_oniric.dream.episode import DreamEpisode


@dataclass
class RecombineOpState:
    """Mutable state for recombine op across episodes."""

    total_episodes_handled: int = 0
    total_samples_emitted: int = 0
    last_sample: list[float] | None = None
    sample_history: list[list[float]] = field(default_factory=list)
    # New in C2.6 for full VAE variant — populated only by the
    # `recombine_handler_full_mlx` factory. Light variants keep it
    # as None for backward compatibility.
    last_kl_divergence: float | None = None


@dataclass(frozen=True)
class RecombineFullResult:
    """Output of a full VAE recombine op (sample + KL divergence).

    Returned conceptually by `recombine_handler_full_mlx` via the
    state object (`state.last_sample` + `state.last_kl_divergence`).
    This dataclass exposes the pair in an immutable form for code
    paths that prefer value-level plumbing over state mutation.
    """

    sample: list[float]
    kl_divergence: float


def _interpolate(
    a: list[float], b: list[float], alpha: float
) -> list[float]:
    """Linear interpolation: alpha*a + (1-alpha)*b component-wise.

    Raises `ValueError` annotated with invariant I3 (latent
    distributional drift bounded — requires consistent latent
    dimensions) when the two latents have mismatched length.
    """
    if len(a) != len(b):
        raise ValueError(
            f"I3: latent dimensions mismatch: {len(a)} vs {len(b)}"
        )
    return [alpha * x + (1.0 - alpha) * y for x, y in zip(a, b)]


def recombine_handler(
    state: RecombineOpState,
    rng: random.Random | None = None,
) -> Callable[[DreamEpisode], None]:
    """Build a recombine handler bound to a state instance.

    Handler reads `delta_latents` from input_slice (must contain
    >= 2 latents), samples 2 distinct indices via rng, interpolates
    with alpha ~ U(0, 1), updates state. Real VAE sampling lands
    S13+ with MLX integration.

    Preserves : DR-0 (every call bumps `total_episodes_handled`
    and appends to `sample_history`), I3 (latent dimension
    consistency — guarded by `_interpolate`), and DR-4 (recombine
    op is part of the P_equ/P_max chain).

    Reproducibility : when `rng` is None, a deterministic seed
    `random.Random(0)` is used to honour the R1 contract (same
    inputs → same outputs). Production callers that need genuine
    stochastic sampling MUST inject their own `random.Random`
    instance seeded from the harness run-registry.
    """
    if rng is None:
        # Deterministic default for R1 reproducibility contract.
        # Production callers should inject their own rng for
        # genuine stochastic sampling.
        rng = random.Random(0)

    def handler(episode: DreamEpisode) -> None:
        latents = episode.input_slice.get("delta_latents", [])
        # Invariant I3 (input shape): need >= 2 latents to
        # interpolate.
        if len(latents) < 2:
            raise ValueError(
                f"I3: delta_latents must contain at least 2 "
                f"latents, got {len(latents)}"
            )
        idx_a, idx_b = rng.sample(range(len(latents)), 2)
        alpha = rng.random()
        sample = _interpolate(latents[idx_a], latents[idx_b], alpha)
        state.total_episodes_handled += 1
        state.total_samples_emitted += 1
        state.last_sample = sample
        state.sample_history.append(sample)

    return handler


def recombine_handler_mlx(
    state: RecombineOpState,
    encoder,
    decoder,
    seed: int = 0,
) -> Callable[[DreamEpisode], None]:
    """Build a recombine handler with real MLX VAE-light sampling.

    Encoder is expected to return ``(mu, log_var)`` — i.e. the
    second tensor is the log of the *variance* of the latent
    distribution (the sigma used in reparameterization is then
    ``exp(0.5 * log_var)``). If your encoder instead returns
    ``(mu, log_std)``, wrap it in an adapter that applies
    ``log_var = 2 * log_std`` before calling this handler.

    Sampling uses the MLX reparameterization trick : ``z = mu +
    sigma * epsilon`` with ``epsilon ~ N(0, I)``. The decoder
    maps ``z`` to an output sample.

    Reproducibility is delivered by deriving epsilon from a local
    PRNG key (``mx.random.key`` + ``mx.random.split``) keyed by
    ``seed + state.total_episodes_handled``. The global MLX RNG
    is *not* mutated — concurrent dream workers can therefore run
    multiple recombine handlers without interfering.

    Skeleton handler preserved for tests / contexts not requiring
    MLX. This MLX variant produces real generative samples for the
    G4 GO-FULL gate (canal 2 output).

    Preserves : DR-0, DR-4, I3 (latent shape).

    Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
    """
    import mlx.core as mx

    def handler(episode: DreamEpisode) -> None:
        latents = episode.input_slice.get("delta_latents", [])
        # Invariant I3 (input shape): keep API compatibility with
        # the skeleton handler that requires >= 2 latents to
        # interpolate ; the MLX variant only consumes `latents[0]`
        # (diversity comes from sampling z, not latent selection)
        # but the >= 2 contract is preserved so callers can swap
        # handlers without changing their input_slice.
        if len(latents) < 2:
            raise ValueError(
                f"I3: delta_latents must contain at least 2 "
                f"latents, got {len(latents)}"
            )
        # I3 dimensionality consistency : even though only
        # latents[0] is consumed, reject mismatched-dim batches
        # so the contract matches the skeleton handler exactly.
        if not all(len(lat) == len(latents[0]) for lat in latents):
            raise ValueError(
                "I3: delta_latents must all have the same "
                "dimensionality"
            )

        # Local PRNG key for isolated reproducibility — does not
        # touch the process-wide MLX RNG.
        key_seed = seed + state.total_episodes_handled
        key = mx.random.key(key_seed)
        _, sample_key = mx.random.split(key)

        # Pick first latent as encoder input (deterministic choice
        # — diversity comes from sampling z, not from latent
        # selection).
        x = mx.array(latents[0])
        mu, log_var = encoder(x)
        sigma = mx.exp(0.5 * log_var)
        epsilon = mx.random.normal(shape=mu.shape, key=sample_key)
        z = mu + sigma * epsilon
        sample_arr = decoder(z)
        mx.eval(sample_arr)

        sample = [float(v) for v in sample_arr.tolist()]
        state.total_episodes_handled += 1
        state.total_samples_emitted += 1
        state.last_sample = sample
        state.sample_history.append(sample)

    return handler


def recombine_handler_full_mlx(
    state: RecombineOpState,
    encoder,
    decoder,
    seed: int = 0,
) -> Callable[[DreamEpisode], None]:
    """Build a full VAE recombine handler (C-Hobson full, C2.6).

    Upgrade from `recombine_handler_mlx` light variant (C2.3)
    with proper full VAE semantics :
    - encoder : input -> (mu, log_sigma)
    - reparameterization : z = mu + sigma * epsilon (eps ~ N(0,I))
    - decoder : z -> output sample
    - KL divergence computed vs standard Gaussian prior :
      KL = -0.5 * mean(1 + log_sigma - mu**2 - exp(log_sigma))
      (non-negative by construction of KL(q||p) with p = N(0,I))

    Seed-based per-episode determinism for reproducibility (R1
    contract). `state.last_sample` + `state.last_kl_divergence`
    populated after each call.

    Preserves : DR-0 (episode count + history), DR-4 (recombine is
    part of P_equ/P_max chain), I3 (latent shape consistency).

    Reference:
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
    """
    import mlx.core as mx

    def handler(episode: DreamEpisode) -> None:
        latents = episode.input_slice.get("delta_latents", [])
        # I3 input shape : keep API compatibility with light
        # handlers that require >= 2 latents. Only latents[0] is
        # consumed — diversity comes from sampling z.
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

        # Per-episode local PRNG key for R1 reproducibility — does
        # not touch the process-wide MLX RNG, so concurrent dream
        # workers can run multiple full-VAE recombine handlers
        # without interfering. Mirrors the light variant's pattern
        # (`recombine_handler_mlx`) above.
        key_seed = seed + state.total_episodes_handled
        key = mx.random.key(key_seed)
        _, sample_key = mx.random.split(key)

        x = mx.array(latents[0])
        mu, log_sigma = encoder(x)
        sigma = mx.exp(0.5 * log_sigma)
        epsilon = mx.random.normal(shape=mu.shape, key=sample_key)
        z = mu + sigma * epsilon
        sample_arr = decoder(z)
        # KL(q(z|x) || N(0, I)) with q(z|x) = N(mu, exp(log_sigma))
        # under the convention log_sigma = log(sigma**2) (log-var).
        # KL = -0.5 * mean(1 + log_sigma - mu**2 - exp(log_sigma))
        kl = -0.5 * mx.mean(
            1.0 + log_sigma - mu * mu - mx.exp(log_sigma)
        )
        mx.eval(sample_arr, kl)

        sample = [float(v) for v in sample_arr.tolist()]
        kl_value = float(kl.item())
        state.total_episodes_handled += 1
        state.total_samples_emitted += 1
        state.last_sample = sample
        state.sample_history.append(sample)
        state.last_kl_divergence = kl_value

    return handler
