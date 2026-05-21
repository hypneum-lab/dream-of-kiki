"""Unit tests for the diffusion-substrate emitting handlers."""

from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as _nn_inner
import numpy as np

from kiki_oniric.dream.channels import WeightUpdate
from kiki_oniric.dream.episode import (
    BudgetCap, DreamEpisode, EpisodeTrigger, Operation, OutputChannel,
)
from kiki_oniric.dream.operations.replay_real import ReplayRealState
from kiki_oniric.substrates._diffusion.handlers_emit import (
    replay_diffusion_handler,
)
from kiki_oniric.substrates._diffusion.model import MLPDenoiser

# mlx.nn re-exports confuse mypy's star-import resolution; pin an
# Any-typed alias to keep the test free of per-line type: ignores.
_nn: Any = _nn_inner


def _model_adapter(denoiser: MLPDenoiser) -> Any:
    # Thin nn.Module holder exposing .layers (mirroring
    # _DenoiserSingleArgAdapter surface that bind_real_handlers
    # binds to). Must subclass mlx.nn.Module so nn.value_and_grad
    # can traverse the parameter tree.

    class _A(_nn.Module):  # type: ignore[misc]
        def __init__(self, d: MLPDenoiser) -> None:
            super().__init__()
            self.denoiser = d

        @property
        def layers(self) -> list[Any]:
            return self.denoiser.layers

        def __call__(self, x: mx.array) -> mx.array:
            t = mx.zeros((x.shape[0],), dtype=mx.int32)
            return self.denoiser(x, t)

    return _A(denoiser)


def _episode_with_records(records: list[dict[str, Any]]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"beta_records": records},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10**9, wall_time_s=10.0, energy_j=1.0),
        episode_id="t/2/diff",
    )


def test_replay_handler_emits_weight_update_on_records() -> None:
    """Non-empty beta_records must yield a WeightUpdate with all denoiser
    layer keys populated."""
    d = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    adapter = _model_adapter(d)
    state = ReplayRealState()
    handler = replay_diffusion_handler(state, model=adapter, lr=1e-2)

    records = [
        {"x": mx.zeros((4,)), "y": mx.zeros((4,))} for _ in range(3)
    ]
    out = handler(_episode_with_records(records))

    assert isinstance(out, WeightUpdate)
    # Three layers (n_layers=2 hidden + 1 output) → 6 keys
    # (weight + bias each), all present.
    expected = {f"layer_{i}_{a}" for i in range(3) for a in ("weight", "bias")}
    assert set(out.lora_delta.keys()) == expected


def test_replay_handler_returns_none_on_empty_records() -> None:
    """Empty beta_records is the S1 no-op branch — no emission."""
    d = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    adapter = _model_adapter(d)
    state = ReplayRealState()
    handler = replay_diffusion_handler(state, model=adapter, lr=1e-2)

    out = handler(_episode_with_records([]))

    assert out is None
    assert state.last_loss is None


def test_replay_handler_delta_matches_post_minus_pre() -> None:
    """The emitted delta must equal (post_param − pre_param) per layer."""
    d = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    pre = {
        f"layer_{i}_{a}": np.asarray(getattr(d.layers[i], a)).copy()
        for i in range(3) for a in ("weight", "bias")
    }
    adapter = _model_adapter(d)
    state = ReplayRealState()
    handler = replay_diffusion_handler(state, model=adapter, lr=1e-2)

    records = [
        {"x": mx.ones((4,)), "y": mx.ones((4,))} for _ in range(3)
    ]
    out = handler(_episode_with_records(records))
    assert out is not None

    for key, pre_arr in pre.items():
        # _i / attr inferred from key for assertions
        i = int(key.split("_")[1])
        a = key.split("_")[2]
        post = np.asarray(getattr(d.layers[i], a))
        np.testing.assert_allclose(
            out.lora_delta[key], post - pre_arr, atol=1e-6,
        )
