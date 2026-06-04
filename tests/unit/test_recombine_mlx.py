"""Unit tests for recombine operation MLX-native VAE backend (S13.2)."""
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
    recombine_handler_mlx,
)


class TinyEncoder(nn.Module):  # type: ignore[misc,name-defined]  # mlx.nn dynamic
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


class TinyDecoder(nn.Module):  # type: ignore[misc,name-defined]  # mlx.nn dynamic
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
        budget=BudgetCap(flops=15_000, wall_time_s=1.5, energy_j=0.15),
        episode_id=ep_id,
    )


def test_recombine_mlx_sample_has_correct_output_dim() -> None:
    state = RecombineOpState()
    encoder = TinyEncoder(input_dim=3, latent_dim=4)
    decoder = TinyDecoder(latent_dim=4, output_dim=5)
    handler = recombine_handler_mlx(
        state=state, encoder=encoder, decoder=decoder, seed=42
    )
    handler(make_recombine_episode(
        "de-mlx-rc0", [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]
    ))
    assert state.total_episodes_handled == 1
    assert state.total_samples_emitted == 1
    assert state.last_sample is not None
    assert len(state.last_sample) == 5  # output_dim


def test_recombine_mlx_sampling_diversity_over_runs() -> None:
    """Different seeds should produce different samples."""
    encoder = TinyEncoder(input_dim=3, latent_dim=4)
    decoder = TinyDecoder(latent_dim=4, output_dim=5)
    samples = []
    for seed in [1, 2, 3, 4, 5]:
        state = RecombineOpState()
        handler = recombine_handler_mlx(
            state=state, encoder=encoder, decoder=decoder, seed=seed
        )
        handler(make_recombine_episode(
            f"de-mlx-rc-div-{seed}",
            [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]],
        ))
        assert state.last_sample is not None
        samples.append(tuple(state.last_sample))
    # All samples should be distinct (diversity)
    assert len(set(samples)) >= 3


def test_recombine_mlx_rejects_mismatched_latent_dimensions() -> None:
    """I3 dimensionality check : even though only latents[0] is
    consumed, the MLX handler rejects mismatched-dim batches so the
    contract matches the skeleton handler exactly.
    """
    state = RecombineOpState()
    encoder = TinyEncoder(input_dim=3, latent_dim=4)
    decoder = TinyDecoder(latent_dim=4, output_dim=5)
    handler = recombine_handler_mlx(
        state=state, encoder=encoder, decoder=decoder, seed=42
    )
    with pytest.raises(ValueError, match="I3"):
        handler(make_recombine_episode(
            "de-mlx-dim-mismatch",
            [[1.0, 0.0, 0.0], [0.0, 1.0]],  # mismatched dims
        ))


def test_recombine_mlx_deterministic_with_same_seed() -> None:
    """Same seed should produce identical samples."""
    encoder = TinyEncoder(input_dim=3, latent_dim=4)
    decoder = TinyDecoder(latent_dim=4, output_dim=5)
    state_a = RecombineOpState()
    state_b = RecombineOpState()
    handler_a = recombine_handler_mlx(
        state=state_a, encoder=encoder, decoder=decoder, seed=42
    )
    handler_b = recombine_handler_mlx(
        state=state_b, encoder=encoder, decoder=decoder, seed=42
    )
    latents = [[0.5, 0.5, 0.5], [0.1, 0.1, 0.1]]
    handler_a(make_recombine_episode("de-a", latents))
    handler_b(make_recombine_episode("de-b", latents))
    # last_sample is `list[float]`; compare element-wise with a
    # single boolean assertion (the previous `==` against an mx
    # array silently produced an array of booleans).
    assert state_a.last_sample is not None
    assert state_b.last_sample is not None
    assert list(state_a.last_sample) == list(state_b.last_sample)
