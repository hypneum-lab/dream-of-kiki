"""Unit tests for restructure operation MLX-native backend (S13.1)."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.restructure import (
    RestructureOpState,
    restructure_handler_mlx,
)


class StackedMLP:
    """Simple list-of-Linear model for restructure tests."""

    def __init__(self, dims: list[int]) -> None:
        self.layers: list[nn.Linear] = [
            nn.Linear(dims[i], dims[i + 1])
            for i in range(len(dims) - 1)
        ]

    @property
    def num_layers(self) -> int:
        return len(self.layers)


def make_restructure_episode(
    ep_id: str, topo_op: str, **extra: object
) -> DreamEpisode:
    slice_data: dict = {"topo_op": topo_op}
    slice_data.update(extra)
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice=slice_data,
        operation_set=(Operation.RESTRUCTURE,),
        output_channels=(OutputChannel.HIERARCHY_CHG,),
        budget=BudgetCap(flops=20_000, wall_time_s=2.0, energy_j=0.2),
        episode_id=ep_id,
    )


def test_restructure_mlx_add_layer_increases_count() -> None:
    state = RestructureOpState()
    model = StackedMLP([4, 8, 2])
    handler = restructure_handler_mlx(state=state, model=model)
    handler(make_restructure_episode(
        "de-mlx-r0", "add", new_dim=4
    ))
    assert model.num_layers == 3
    assert state.total_diffs_emitted == 1
    assert state.last_diff_type == "add"


def test_restructure_mlx_remove_layer_decreases_count() -> None:
    state = RestructureOpState()
    model = StackedMLP([4, 8, 16, 2])
    handler = restructure_handler_mlx(state=state, model=model)
    handler(make_restructure_episode(
        "de-mlx-r1", "remove", layer_index=1
    ))
    assert model.num_layers == 2
    assert state.total_diffs_emitted == 1
    assert state.last_diff_type == "remove"


def test_restructure_mlx_reroute_swaps_layers() -> None:
    state = RestructureOpState()
    model = StackedMLP([4, 8, 16, 2])
    layer_0_id = id(model.layers[0])
    layer_2_id = id(model.layers[2])
    handler = restructure_handler_mlx(state=state, model=model)
    handler(make_restructure_episode(
        "de-mlx-r2", "reroute", swap_indices=[0, 2]
    ))
    assert id(model.layers[0]) == layer_2_id
    assert id(model.layers[2]) == layer_0_id
    assert state.total_diffs_emitted == 1
    assert state.last_diff_type == "reroute"
