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
from kiki_oniric.dream.operations.downscale_real import DownscaleRealState
from kiki_oniric.dream.operations.replay_real import ReplayRealState
from kiki_oniric.substrates._diffusion.handlers_emit import (
    downscale_diffusion_handler,
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


def _pre_snapshot(d: MLPDenoiser) -> dict[str, np.ndarray]:
    """Capture a numpy snapshot of all layer.weight / layer.bias."""
    return {
        f"layer_{i}_{a}": np.asarray(getattr(d.layers[i], a)).copy()
        for i in range(len(d.layers)) for a in ("weight", "bias")
    }


def _assert_model_unchanged(
    d: MLPDenoiser, pre: dict[str, np.ndarray]
) -> None:
    """Assert that handler is compute-only: model must be unchanged (B5)."""
    post_state = {
        f"layer_{i}_{a}": np.asarray(getattr(d.layers[i], a)).copy()
        for i in range(len(d.layers)) for a in ("weight", "bias")
    }
    for key in pre:
        np.testing.assert_allclose(
            pre[key], post_state[key], atol=1e-6,
            err_msg=f"handler mutated {key} in place",
        )


def test_replay_handler_emits_weight_update_on_records() -> None:
    """Non-empty beta_records must yield a WeightUpdate with all denoiser
    layer keys populated."""
    d = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    pre = _pre_snapshot(d)
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

    # After the handler returns, the model must be unchanged (B5: apply
    # is the sole applier; the handler is compute-only).
    _assert_model_unchanged(d, pre)


def test_replay_handler_returns_none_on_empty_records() -> None:
    """Empty beta_records is the S1 no-op branch — no emission."""
    d = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    adapter = _model_adapter(d)
    state = ReplayRealState()
    handler = replay_diffusion_handler(state, model=adapter, lr=1e-2)

    out = handler(_episode_with_records([]))

    assert out is None
    assert state.last_loss is None


def test_replay_handler_compute_only_model_unchanged() -> None:
    """The handler must not mutate the model in place (B5 contract).

    After the handler returns, all layer params must equal their
    pre-call values. The delta is emitted but not applied here —
    apply_channel_outputs is the sole applier.
    """
    d = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    pre = _pre_snapshot(d)
    adapter = _model_adapter(d)
    state = ReplayRealState()
    handler = replay_diffusion_handler(state, model=adapter, lr=1e-2)

    records = [
        {"x": mx.ones((4,)), "y": mx.ones((4,))} for _ in range(3)
    ]
    out = handler(_episode_with_records(records))
    assert out is not None

    _assert_model_unchanged(d, pre)


def test_replay_handler_delta_nonzero_model_unchanged() -> None:
    """The emitted delta must be non-zero and the model must be unchanged.

    The previous test (delta_matches_post_minus_pre) no longer holds
    with the rollback design — post == pre after the handler returns,
    so the correct assertion is: delta != 0 everywhere (the SGD step
    would have moved the weights) and model is compute-only.
    """
    d = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    pre = _pre_snapshot(d)
    adapter = _model_adapter(d)
    state = ReplayRealState()
    handler = replay_diffusion_handler(state, model=adapter, lr=1e-2)

    records = [
        {"x": mx.ones((4,)), "y": mx.ones((4,))} for _ in range(3)
    ]
    out = handler(_episode_with_records(records))
    assert out is not None

    # Delta is non-trivial (SGD moved something).
    total_abs = sum(np.abs(v).sum() for v in out.lora_delta.values())
    assert total_abs > 0.0, "expected non-zero delta"

    # Model unchanged after the compute-only handler (B5).
    _assert_model_unchanged(d, pre)


def test_downscale_handler_compute_only_delta_correct() -> None:
    """Downscale handler must be compute-only and emit correct delta.

    The emitted delta must equal (factor - 1) * pre for each param,
    and the model must be unchanged after the call (B5 contract).
    """
    d = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    pre = _pre_snapshot(d)
    adapter = _model_adapter(d)
    state = DownscaleRealState()
    handler = downscale_diffusion_handler(state, model=adapter)

    factor = 0.9
    episode = DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"shrink_factor": factor},
        operation_set=(Operation.DOWNSCALE,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10**9, wall_time_s=10.0, energy_j=1.0),
        episode_id="t/ds/diff",
    )
    out = handler(episode)
    assert isinstance(out, WeightUpdate)

    # Delta == (factor - 1) * pre_param for each key.
    for key, pre_arr in pre.items():
        expected = ((factor - 1.0) * pre_arr).astype(np.float32)
        np.testing.assert_allclose(
            out.lora_delta[key], expected, atol=1e-6,
            err_msg=f"delta mismatch for {key}",
        )

    # Model must be unchanged — handler is compute-only (B5).
    _assert_model_unchanged(d, pre)
