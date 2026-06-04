"""Unit tests for dream-episode 5-tuple dataclass (DR-0, DR-1 contract)."""
from __future__ import annotations

import pytest

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)


def test_episode_is_frozen_dataclass() -> None:
    ep = DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=1_000_000, wall_time_s=1.0, energy_j=0.5),
        episode_id="de-0000",
    )
    with pytest.raises((AttributeError, TypeError)):
        # Intentional write to a read-only property to assert it
        # is immutable; mypy correctly flags the assignment.
        ep.episode_id = "de-mutated"  # type: ignore[misc]


def test_episode_has_required_5_tuple_fields() -> None:
    fields = {f.name for f in DreamEpisode.__dataclass_fields__.values()}
    # 5-tuple per framework spec §4.1 + episode_id for traceability
    assert {"trigger", "input_slice", "operation_set",
            "output_channels", "budget", "episode_id"} <= fields


def test_budget_cap_rejects_negative_flops() -> None:
    with pytest.raises(ValueError, match="flops"):
        BudgetCap(flops=-1, wall_time_s=1.0, energy_j=0.5)


def test_operation_set_is_non_empty() -> None:
    with pytest.raises(ValueError, match="operation_set"):
        DreamEpisode(
            trigger=EpisodeTrigger.SCHEDULED,
            input_slice={},
            operation_set=(),
            output_channels=(OutputChannel.WEIGHT_DELTA,),
            budget=BudgetCap(flops=100, wall_time_s=0.1, energy_j=0.01),
            episode_id="de-empty",
        )


def test_episode_trigger_and_operation_enums() -> None:
    assert EpisodeTrigger.SCHEDULED.value == "scheduled"
    assert Operation.REPLAY.value == "replay"
    assert Operation.DOWNSCALE.value == "downscale"
    assert Operation.RESTRUCTURE.value == "restructure"
    assert Operation.RECOMBINE.value == "recombine"
    assert OutputChannel.WEIGHT_DELTA.value == "weight_delta"
