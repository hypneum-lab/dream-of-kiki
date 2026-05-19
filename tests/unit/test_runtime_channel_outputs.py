"""Unit tests for the channel_outputs log field and collection."""
from __future__ import annotations

import numpy as np
import pytest

from kiki_oniric.dream.channels import WeightUpdate
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.runtime import DreamRuntime, EpisodeLogEntry


def test_log_entry_channel_outputs_defaults_empty() -> None:
    entry = EpisodeLogEntry(
        episode_id="de-x",
        operations_executed=(Operation.REPLAY,),
        completed=True,
    )
    assert entry.channel_outputs == ()


def test_log_entry_accepts_channel_outputs() -> None:
    wu = WeightUpdate(lora_delta={"l0": np.zeros(2, dtype=np.float32)})
    entry = EpisodeLogEntry(
        episode_id="de-y",
        operations_executed=(Operation.REPLAY,),
        completed=True,
        channel_outputs=(wu,),
    )
    assert entry.channel_outputs == (wu,)


def _episode(ops: tuple[Operation, ...]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={},
        operation_set=ops,
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=1, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-exec",
    )


def test_execute_collects_handler_returns() -> None:
    wu = WeightUpdate(lora_delta={"l0": np.zeros(2, dtype=np.float32)})
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, lambda ep: wu)
    runtime.register_handler(Operation.DOWNSCALE, lambda ep: None)
    runtime.execute(_episode((Operation.REPLAY, Operation.DOWNSCALE)))
    entry = runtime.log[-1]
    assert entry.channel_outputs == (wu, None)
    assert len(entry.channel_outputs) == len(entry.operations_executed)


def test_execute_channel_outputs_parallel_on_error() -> None:
    def boom(ep: DreamEpisode) -> None:
        raise RuntimeError("handler failed")

    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, lambda ep: None)
    runtime.register_handler(Operation.DOWNSCALE, boom)
    with pytest.raises(RuntimeError, match="handler failed"):
        runtime.execute(_episode((Operation.REPLAY, Operation.DOWNSCALE)))
    entry = runtime.log[-1]
    assert entry.completed is False
    assert len(entry.channel_outputs) == len(entry.operations_executed)
    assert entry.channel_outputs == (None, None)


def test_log_entry_rejects_mismatched_channel_outputs() -> None:
    wu = WeightUpdate(lora_delta={"l0": np.zeros(2, dtype=np.float32)})
    with pytest.raises(ValueError, match="channel_outputs"):
        EpisodeLogEntry(
            episode_id="de-bad",
            operations_executed=(Operation.REPLAY, Operation.DOWNSCALE),
            completed=True,
            channel_outputs=(wu,),
        )
