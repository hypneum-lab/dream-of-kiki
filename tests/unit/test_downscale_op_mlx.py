"""Unit tests for downscale operation MLX backend (S9.2)."""
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
from kiki_oniric.dream.operations.downscale import (
    DownscaleOpState,
    downscale_handler_mlx,
)


class TinyMLP(nn.Module):
    """Minimal 2-layer MLP for downscale tests (seeded via mx.random.seed)."""

    def __init__(self) -> None:
        super().__init__()
        self.fc1 = nn.Linear(4, 8)
        self.fc2 = nn.Linear(8, 2)

    def __call__(self, x):
        return self.fc2(nn.relu(self.fc1(x)))


def make_downscale_episode(
    ep_id: str, factor: float
) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"shrink_factor": factor},
        operation_set=(Operation.DOWNSCALE,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=5_000, wall_time_s=0.5, energy_j=0.05),
        episode_id=ep_id,
    )


def _flat_weight_norm(model: nn.Module) -> float:
    """Compute the Frobenius norm across all model parameters."""
    leaves: list = []

    def collect(node) -> None:
        if isinstance(node, dict):
            for v in node.values():
                collect(v)
        elif isinstance(node, (list, tuple)):
            for v in node:
                collect(v)
        elif isinstance(node, mx.array):
            leaves.append((node.flatten() ** 2).sum())

    collect(model.parameters())
    if not leaves:
        return 0.0
    total = leaves[0]
    for leaf in leaves[1:]:
        total = total + leaf
    return float(mx.sqrt(total).item())


def test_downscale_mlx_shrinks_weights() -> None:
    """Apply shrink_factor=0.5 should halve the weight norm."""
    state = DownscaleOpState()
    model = TinyMLP()
    norm_before = _flat_weight_norm(model)

    handler = downscale_handler_mlx(state=state, model=model)
    handler(make_downscale_episode("de-mlx-d0", 0.5))

    norm_after = _flat_weight_norm(model)
    assert norm_after == pytest.approx(0.5 * norm_before, rel=1e-5)
    assert state.compound_factor == pytest.approx(0.5)


def test_downscale_mlx_not_idempotent() -> None:
    """Two shrinks with same factor f compound to f² (not f)."""
    state = DownscaleOpState()
    model = TinyMLP()
    norm_initial = _flat_weight_norm(model)

    handler = downscale_handler_mlx(state=state, model=model)
    handler(make_downscale_episode("de-mlx-d1", 0.9))
    handler(make_downscale_episode("de-mlx-d2", 0.9))

    norm_final = _flat_weight_norm(model)
    # Empirical proof of non-idempotence: norm shrunk by 0.9² = 0.81
    assert norm_final == pytest.approx(0.81 * norm_initial, rel=1e-5)
    assert state.compound_factor == pytest.approx(0.81)
    assert state.total_episodes_handled == 2
