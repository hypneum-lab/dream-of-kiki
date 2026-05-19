"""Unit tests for P_min profile wiring (replay + downscale).

Swap protocol E2E tests are in test_p_min_e2e.py (S9.4+).
"""
from __future__ import annotations

from typing import Any

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.profiles.p_min import PMinProfile


def make_replay_de(ep_id: str, records: list[dict[str, Any]]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"beta_records": records},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=1000, wall_time_s=0.1, energy_j=0.01),
        episode_id=ep_id,
    )


def make_downscale_de(ep_id: str, factor: float) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"shrink_factor": factor},
        operation_set=(Operation.DOWNSCALE,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=500, wall_time_s=0.05, energy_j=0.005),
        episode_id=ep_id,
    )


def test_p_min_registers_replay_and_downscale() -> None:
    profile = PMinProfile()
    assert Operation.REPLAY in profile.runtime._handlers
    assert Operation.DOWNSCALE in profile.runtime._handlers


def test_p_min_executes_replay_then_downscale() -> None:
    profile = PMinProfile()
    profile.runtime.execute(
        make_replay_de("de-min0", [{"id": 1}, {"id": 2}])
    )
    profile.runtime.execute(make_downscale_de("de-min1", 0.95))
    assert profile.replay_state.total_episodes_handled == 1
    assert profile.replay_state.total_records_consumed == 2
    assert profile.downscale_state.total_episodes_handled == 1
    assert profile.downscale_state.compound_factor == 0.95


def test_p_min_log_contains_both_episodes() -> None:
    profile = PMinProfile()
    profile.runtime.execute(make_replay_de("de-min2", []))
    profile.runtime.execute(make_downscale_de("de-min3", 0.99))
    ids = [e.episode_id for e in profile.runtime.log]
    assert ids == ["de-min2", "de-min3"]
    assert all(e.completed for e in profile.runtime.log)
