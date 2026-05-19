"""Unit tests for recombine_full MLX VAE variant (C2.6 cycle 2)."""
from __future__ import annotations

import pytest

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
    import mlx.nn as nn
else:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.recombine import (
    RecombineOpState,
    recombine_handler_full_mlx,
)


class TinyVAEEncoder(nn.Module):  # type: ignore[misc,name-defined]  # mlx.nn dynamic
    """Encoder: input_dim -> 2 * latent_dim (mu + log_sigma)."""

    def __init__(self, input_dim: int, latent_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(input_dim, 2 * latent_dim)  # type: ignore[attr-defined]  # mlx dynamic
        self.latent_dim = latent_dim

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        out = self.fc(x)
        mu = out[..., : self.latent_dim]
        log_sigma = out[..., self.latent_dim :]
        return mu, log_sigma


class TinyVAEDecoder(nn.Module):  # type: ignore[misc,name-defined]  # mlx.nn dynamic
    """Decoder: latent_dim -> output_dim."""

    def __init__(self, latent_dim: int, output_dim: int) -> None:
        super().__init__()
        self.fc = nn.Linear(latent_dim, output_dim)  # type: ignore[attr-defined]  # mlx dynamic

    def __call__(self, z: mx.array) -> mx.array:
        return self.fc(z)  # type: ignore[no-any-return]  # mlx dynamic


def make_recombine_episode(
    ep_id: str, latents: list[list[float]]
) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"delta_latents": latents},
        operation_set=(Operation.RECOMBINE,),
        output_channels=(OutputChannel.LATENT_SAMPLE,),
        budget=BudgetCap(flops=30_000, wall_time_s=2.0, energy_j=0.2),
        episode_id=ep_id,
    )


def test_recombine_full_round_trip_dim_correct() -> None:
    """Full VAE : encode → sample → decode yields output_dim."""
    state = RecombineOpState()
    encoder = TinyVAEEncoder(input_dim=3, latent_dim=4)
    decoder = TinyVAEDecoder(latent_dim=4, output_dim=5)
    handler = recombine_handler_full_mlx(
        state=state, encoder=encoder, decoder=decoder, seed=42
    )
    handler(make_recombine_episode(
        "de-full0", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    ))
    assert state.total_episodes_handled == 1
    assert state.last_sample is not None
    assert len(state.last_sample) == 5  # output_dim


def test_recombine_full_kl_divergence_non_negative() -> None:
    """KL divergence of VAE prior should be >= 0 (standard VAE)."""
    state = RecombineOpState()
    encoder = TinyVAEEncoder(input_dim=3, latent_dim=4)
    decoder = TinyVAEDecoder(latent_dim=4, output_dim=5)
    handler = recombine_handler_full_mlx(
        state=state, encoder=encoder, decoder=decoder, seed=42
    )
    # Handler should store last KL via state.last_kl_divergence
    handler(make_recombine_episode(
        "de-full1", [[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]]
    ))
    # KL divergence is non-negative by definition
    assert hasattr(state, "last_kl_divergence")
    assert state.last_kl_divergence is not None
    assert state.last_kl_divergence >= 0.0


def test_recombine_full_sample_diversity_over_seeds() -> None:
    """Different seeds produce different samples — non-determinism."""
    encoder = TinyVAEEncoder(input_dim=3, latent_dim=4)
    decoder = TinyVAEDecoder(latent_dim=4, output_dim=5)
    samples = []
    for seed in [1, 2, 3, 4, 5]:
        state = RecombineOpState()
        handler = recombine_handler_full_mlx(
            state=state, encoder=encoder, decoder=decoder, seed=seed
        )
        handler(make_recombine_episode(
            f"de-full-div-{seed}",
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ))
        assert state.last_sample is not None
        samples.append(tuple(state.last_sample))
    # At least 3 distinct samples across 5 seeds = diversity
    assert len(set(samples)) >= 3
