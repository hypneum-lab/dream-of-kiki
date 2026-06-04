"""Unit tests for the LoRA replay handler (B1b)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import mlx.core as mx
else:
    mx = pytest.importorskip("mlx.core")

from kiki_oniric.dream.channels import WeightUpdate
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.replay_real import (
    ReplayRealState,
    replay_lora_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime, EpisodeLogEntry
from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


def _records() -> list[dict[str, list[float]]]:
    return [
        {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
        {"x": [0.5, 0.6, 0.7, 0.8], "y": [0.0, 1.0]},
    ]


def _episode(records: list[dict[str, list[float]]]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"beta_records": records},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-lora",
    )


def _run(
    model: LoRAModel, records: list[dict[str, list[float]]]
) -> tuple[ReplayRealState, EpisodeLogEntry]:
    state = ReplayRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.REPLAY,
        replay_lora_handler(state, model=model, lr=0.05),
    )
    runtime.execute(_episode(records))
    return state, runtime.log[-1]


def test_replay_lora_emits_weight_update() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _, entry = _run(model, _records())
    assert isinstance(entry.channel_outputs[0], WeightUpdate)


def test_replay_lora_delta_keys_match_adapters() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _, entry = _run(model, _records())
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    assert set(out.lora_delta) == set(model.adapter_parameters())


def test_replay_lora_delta_is_finite_float32() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _, entry = _run(model, _records())
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    for arr in out.lora_delta.values():
        assert arr.dtype == np.float32
        assert bool(np.isfinite(arr).all())
    # B is zero-init: first step moves lora_b but not lora_a;
    # at least one adapter must have changed.
    assert any(bool(np.any(arr != 0.0)) for arr in out.lora_delta.values())


def test_replay_lora_deltas_match_adapter_transition() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    before = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    _, entry = _run(model, _records())
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    after = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    for k in before:
        np.testing.assert_allclose(
            before[k] + out.lora_delta[k], after[k],
            rtol=1e-5, atol=1e-6,
        )


def test_replay_lora_composed_effective_delta() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    layer0 = model.layers[0]
    a_before = np.asarray(layer0.lora_a, dtype=np.float32)
    b_before = np.asarray(layer0.lora_b, dtype=np.float32)
    scale = layer0.scale
    _, entry = _run(model, _records())
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    a_after = a_before + out.lora_delta["layer0.lora_a"]
    b_after = b_before + out.lora_delta["layer0.lora_b"]
    composed = scale * (b_after @ a_after - b_before @ a_before)
    actual = scale * (
        np.asarray(model.layers[0].lora_b, dtype=np.float32)
        @ np.asarray(model.layers[0].lora_a, dtype=np.float32)
        - b_before @ a_before
    )
    np.testing.assert_allclose(composed, actual, rtol=1e-5, atol=1e-6)


def test_replay_lora_empty_records_returns_none() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    state, entry = _run(model, [])
    assert entry.channel_outputs[0] is None
    assert state.last_compute_flops == 0
    assert state.last_loss is None


def test_replay_lora_tags_k1_flops() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    state, _ = _run(model, _records())
    assert state.last_compute_flops > 0
    assert state.total_compute_flops == state.last_compute_flops
    assert state.total_records_consumed == 2


def test_replay_lora_is_deterministic() -> None:
    m1 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=3)
    m2 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=3)
    _, e1 = _run(m1, _records())
    _, e2 = _run(m2, _records())
    o1, o2 = e1.channel_outputs[0], e2.channel_outputs[0]
    assert isinstance(o1, WeightUpdate) and isinstance(o2, WeightUpdate)
    for k in o1.lora_delta:
        np.testing.assert_array_equal(o1.lora_delta[k], o2.lora_delta[k])


def test_replay_lora_rejects_malformed_record() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    before = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    state = ReplayRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.REPLAY, replay_lora_handler(state, model=model),
    )
    with pytest.raises(ValueError, match="missing"):
        runtime.execute(_episode([{"x": [0.1, 0.2, 0.3, 0.4]}]))
    # Validation must run BEFORE any adapter mutation.
    after = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    for k in before:
        np.testing.assert_array_equal(before[k], after[k])
