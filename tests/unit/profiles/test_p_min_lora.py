"""Unit tests for PMinLoRAProfile and the DreamRuntime.reset_log helper (B6a)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import mlx.core as mx
else:
    mx = pytest.importorskip("mlx.core")

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.runtime import DreamRuntime
from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


def _clones(seed: int = 0) -> tuple[LoRAModel, LoRAModel]:
    """Two bit-identical LoRAModels at the same seed."""
    return (
        LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=seed),
        LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=seed),
    )


def _assert_lora_models_equal(a: LoRAModel, b: LoRAModel) -> None:
    """Assert bit-equality of every layer's base + adapters."""
    assert len(a.layers) == len(b.layers)
    for la, lb in zip(a.layers, b.layers):
        np.testing.assert_array_equal(
            np.asarray(la.base_weight), np.asarray(lb.base_weight),
        )
        np.testing.assert_array_equal(
            np.asarray(la.lora_a), np.asarray(lb.lora_a),
        )
        np.testing.assert_array_equal(
            np.asarray(la.lora_b), np.asarray(lb.lora_b),
        )


def _replay_episode(records: "list[dict[str, object]]" | None = None) -> DreamEpisode:
    if records is None:
        records = [
            {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
            {"x": [0.5, 0.6, 0.7, 0.8], "y": [0.0, 1.0]},
        ]
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"beta_records": records},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pmin-lora",
    )


def _downscale_episode(factor: float = 0.5) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"shrink_factor": factor},
        operation_set=(Operation.DOWNSCALE,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pmin-lora-dn",
    )


def test_dream_runtime_reset_log_clears() -> None:
    """DreamRuntime.reset_log() drops every entry."""
    from kiki_oniric.dream.operations.replay_real import (
        ReplayRealState,
        replay_lora_handler,
    )

    dream, _ = _clones(seed=0)
    runtime = DreamRuntime()
    state = ReplayRealState()
    runtime.register_handler(
        Operation.REPLAY,
        replay_lora_handler(state, model=dream, lr=0.01),
    )
    runtime.execute(_replay_episode())
    runtime.execute(_replay_episode())
    assert len(runtime.log) == 2
    runtime.reset_log()
    assert len(runtime.log) == 0
