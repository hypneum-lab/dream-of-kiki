"""Unit tests for replay operation (P_min op 1/2, A-Walker source)."""
from __future__ import annotations

from typing import Any

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.replay import (
    ReplayOpState,
    replay_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime


def make_replay_episode(ep_id: str, records: list[dict[str, Any]]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"beta_records": records},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000, wall_time_s=1.0, energy_j=0.1),
        episode_id=ep_id,
    )


def test_replay_consumes_all_records() -> None:
    state = ReplayOpState()
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, replay_handler(state))

    records = [{"id": i, "context": f"ctx-{i}"} for i in range(5)]
    runtime.execute(make_replay_episode("de-r0", records))

    assert state.total_records_consumed == 5


def test_replay_handles_empty_records() -> None:
    state = ReplayOpState()
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, replay_handler(state))
    runtime.execute(make_replay_episode("de-r1", []))
    assert state.total_records_consumed == 0


def test_replay_across_multiple_episodes_accumulates() -> None:
    state = ReplayOpState()
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, replay_handler(state))

    runtime.execute(make_replay_episode(
        "de-r2", [{"id": 0}, {"id": 1}]
    ))
    runtime.execute(make_replay_episode(
        "de-r3", [{"id": 2}, {"id": 3}, {"id": 4}]
    ))

    assert state.total_records_consumed == 5
    assert state.total_episodes_handled == 2
