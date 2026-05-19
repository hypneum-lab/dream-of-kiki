"""Unit tests for recombine_real_handler emitting LatentSample (B4)."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import mlx.core as mx
    import mlx.nn as nn
else:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

from kiki_oniric.dream.channels import LatentSample
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.recombine_real import (
    RecombineRealState,
    recombine_real_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime


LATENT_DIM = 4
INPUT_DIM = 4


class _TinyEncoder(nn.Module):  # type: ignore[misc,name-defined]  # mlx.nn dynamic
    """Linear encoder → (mu, log_var) with fixed weights (R1)."""

    def __init__(self) -> None:
        super().__init__()
        self.mu_w = mx.ones((LATENT_DIM, INPUT_DIM)) * 0.1
        self.lv_w = mx.ones((LATENT_DIM, INPUT_DIM)) * -0.5

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return x @ self.mu_w.T, x @ self.lv_w.T


class _TinyDecoder(nn.Module):  # type: ignore[misc,name-defined]  # mlx.nn dynamic
    """Linear decoder z → output."""

    def __init__(self) -> None:
        super().__init__()
        self.w = mx.ones((INPUT_DIM, LATENT_DIM)) * 0.2

    def __call__(self, z: mx.array) -> mx.array:
        return z @ self.w.T


class _InfLogVarEncoder(nn.Module):  # type: ignore[misc,name-defined]  # mlx.nn dynamic
    """Pathological encoder: log_var = +inf → z = +inf via sigma * eps."""

    def __init__(self) -> None:
        super().__init__()
        self.w = mx.ones((LATENT_DIM, INPUT_DIM)) * 0.1

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        mu = x @ self.w.T
        log_var = mx.array([float("inf")] * LATENT_DIM)
        return mu, log_var


def _episode(
    delta_latents: list[list[float]] | None = None,
    *,
    species: object = None,
    episode_id: str = "de-rcb",
) -> DreamEpisode:
    if delta_latents is None:
        delta_latents = [[0.1, 0.2, 0.3, 0.4]]
    slice_d: dict[str, object] = {"delta_latents": delta_latents}
    if species is not None:
        slice_d["species"] = species
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice=slice_d,
        operation_set=(Operation.RECOMBINE,),
        output_channels=(OutputChannel.LATENT_SAMPLE,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id=episode_id,
    )


def _make_runtime(
    encoder: nn.Module,  # type: ignore[name-defined]  # mlx.nn dynamic
    decoder: nn.Module,  # type: ignore[name-defined]  # mlx.nn dynamic
    *,
    seed: int = 0,
) -> tuple[RecombineRealState, DreamRuntime]:
    state = RecombineRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.RECOMBINE,
        recombine_real_handler(
            state, encoder=encoder, decoder=decoder, seed=seed,
        ),
    )
    return state, runtime


def test_recombine_emits_latent_sample() -> None:
    state, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder())
    runtime.execute(_episode())
    assert isinstance(runtime.log[-1].channel_outputs[0], LatentSample)


def test_recombine_latent_vector_dtype_shape() -> None:
    _, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder())
    runtime.execute(_episode())
    out = runtime.log[-1].channel_outputs[0]
    assert isinstance(out, LatentSample)
    assert out.latent_vector.dtype == np.float32
    assert out.latent_vector.shape == (LATENT_DIM,)


def test_recombine_finite_propagation_via_inf_log_var() -> None:
    """Pathological log_var=inf → z=inf → LatentSample raises S2."""
    _, runtime = _make_runtime(_InfLogVarEncoder(), _TinyDecoder())
    with pytest.raises(ValueError, match=r"^S2:"):
        runtime.execute(_episode())


def test_recombine_species_default() -> None:
    _, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder())
    runtime.execute(_episode())
    out = runtime.log[-1].channel_outputs[0]
    assert isinstance(out, LatentSample)
    assert out.species == "default"


def test_recombine_species_from_input() -> None:
    _, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder())
    runtime.execute(_episode(species="replay-mix"))
    out = runtime.log[-1].channel_outputs[0]
    assert isinstance(out, LatentSample)
    assert out.species == "replay-mix"


def test_recombine_species_non_str_rejected() -> None:
    _, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder())
    with pytest.raises(ValueError, match=r"^recombine: species must be str"):
        runtime.execute(_episode(species=42))


def test_recombine_provenance_format() -> None:
    _, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder(), seed=11)
    runtime.execute(_episode(episode_id="de-fmt"))
    out = runtime.log[-1].channel_outputs[0]
    assert isinstance(out, LatentSample)
    assert re.match(
        r"^recombine:de=de-fmt:ep=\d+:seed=\d+$", out.provenance,
    )


def test_recombine_provenance_count_increments() -> None:
    _, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder(), seed=11)
    runtime.execute(_episode(episode_id="de-inc"))
    runtime.execute(_episode(episode_id="de-inc"))
    first = runtime.log[0].channel_outputs[0]
    second = runtime.log[1].channel_outputs[0]
    assert isinstance(first, LatentSample) and isinstance(second, LatentSample)
    assert "ep=0:" in first.provenance
    assert "ep=1:" in second.provenance


def test_recombine_is_deterministic() -> None:
    _, r1 = _make_runtime(_TinyEncoder(), _TinyDecoder(), seed=7)
    _, r2 = _make_runtime(_TinyEncoder(), _TinyDecoder(), seed=7)
    r1.execute(_episode(episode_id="de-det"))
    r2.execute(_episode(episode_id="de-det"))
    a = r1.log[-1].channel_outputs[0]
    b = r2.log[-1].channel_outputs[0]
    assert isinstance(a, LatentSample) and isinstance(b, LatentSample)
    np.testing.assert_array_equal(a.latent_vector, b.latent_vector)
    assert a.provenance == b.provenance


def test_recombine_empty_delta_latents_raises() -> None:
    _, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder())
    with pytest.raises(ValueError, match=r"^I3:"):
        runtime.execute(_episode(delta_latents=[]))


def test_recombine_state_last_sample_preserved() -> None:
    state, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder())
    runtime.execute(_episode())
    assert state.last_sample is not None
    assert isinstance(state.last_sample, list)
    assert all(isinstance(v, float) for v in state.last_sample)
    assert state._episode_count == 1
    assert state.last_compute_flops > 0
