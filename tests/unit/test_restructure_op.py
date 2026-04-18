"""Unit tests for restructure operation (D-Friston FEP source)."""
from __future__ import annotations

import pytest

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.restructure import (
    RestructureOpState,
    restructure_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime


def make_restructure_episode(
    ep_id: str, topo_op: str
) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"topo_op": topo_op, "target_layer": "fc1"},
        operation_set=(Operation.RESTRUCTURE,),
        output_channels=(OutputChannel.HIERARCHY_CHG,),
        budget=BudgetCap(flops=20_000, wall_time_s=2.0, energy_j=0.2),
        episode_id=ep_id,
    )


def test_restructure_records_diff() -> None:
    state = RestructureOpState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.RESTRUCTURE, restructure_handler(state)
    )
    runtime.execute(make_restructure_episode("de-r0", "add"))
    assert state.total_episodes_handled == 1
    assert state.total_diffs_emitted == 1
    assert state.last_diff_type == "add"


def test_restructure_rejects_unknown_topo_op() -> None:
    state = RestructureOpState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.RESTRUCTURE, restructure_handler(state)
    )
    with pytest.raises(ValueError, match="topo_op"):
        runtime.execute(make_restructure_episode("de-r1", "INVALID"))


def test_restructure_accumulates_across_episodes() -> None:
    state = RestructureOpState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.RESTRUCTURE, restructure_handler(state)
    )
    runtime.execute(make_restructure_episode("de-r2", "add"))
    runtime.execute(make_restructure_episode("de-r3", "remove"))
    runtime.execute(make_restructure_episode("de-r4", "reroute"))
    assert state.total_episodes_handled == 3
    assert state.total_diffs_emitted == 3
    assert state.last_diff_type == "reroute"
    # Diff history should preserve order
    assert state.diff_history == ["add", "remove", "reroute"]
