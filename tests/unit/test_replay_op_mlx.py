"""Unit tests for replay operation MLX backend (S9.1)."""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

# Deterministic seed for MLX random init across all tests
mx.random.seed(42)

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.replay import (
    ReplayOpState,
    replay_handler_mlx,
)


class TinyMLP(nn.Module):
    """Minimal 2-layer MLP for replay tests (seeded via mx.random.seed)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


def make_replay_episode(ep_id: str, records: list[dict]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"beta_records": records},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000, wall_time_s=1.0, energy_j=0.1),
        episode_id=ep_id,
    )


def test_replay_mlx_updates_loss() -> None:
    state = ReplayOpState()
    model = TinyMLP()
    handler = replay_handler_mlx(state=state, model=model, lr=0.01)
    records = [
        {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
        {"x": [0.5, 0.6, 0.7, 0.8], "y": [0.0, 1.0]},
    ]
    handler(make_replay_episode("de-mlx0", records))
    assert state.total_records_consumed == 2
    assert state.total_episodes_handled == 1
    assert state.last_loss is not None
    assert state.last_loss >= 0.0


def test_replay_mlx_handles_empty_records() -> None:
    state = ReplayOpState()
    model = TinyMLP()
    handler = replay_handler_mlx(state=state, model=model, lr=0.01)
    handler(make_replay_episode("de-mlx1", []))
    assert state.total_records_consumed == 0
    assert state.last_loss is None  # no batch -> no loss
