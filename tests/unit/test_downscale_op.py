"""Unit tests for downscale operation (P_min op 2/2, B-Tononi SHY)."""
from __future__ import annotations

import pytest

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.downscale import (
    DownscaleOpState,
    downscale_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime


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


def test_downscale_records_factor() -> None:
    state = DownscaleOpState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.DOWNSCALE, downscale_handler(state)
    )
    runtime.execute(make_downscale_episode("de-d0", 0.95))
    assert state.total_episodes_handled == 1
    assert state.last_factor_applied == 0.95


def test_downscale_rejects_factor_out_of_range() -> None:
    state = DownscaleOpState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.DOWNSCALE, downscale_handler(state)
    )
    # SHY shrinkage: factor must be in (0, 1] — values outside
    # this range are nonsensical (no shrinkage or amplification).
    with pytest.raises(ValueError, match="shrink_factor"):
        runtime.execute(make_downscale_episode("de-d1", 1.5))
    with pytest.raises(ValueError, match="shrink_factor"):
        runtime.execute(make_downscale_episode("de-d2", 0.0))


def test_downscale_accumulates_compound_factor() -> None:
    """Compound factor across multiple episodes (factor1 * factor2).

    Reflects the corrected understanding (post CodeRabbit fix):
    downscale is NOT idempotent — repeated shrink_f compounds
    multiplicatively (shrink^2 = factor^2, not factor).
    """
    state = DownscaleOpState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.DOWNSCALE, downscale_handler(state)
    )
    runtime.execute(make_downscale_episode("de-d3", 0.9))
    runtime.execute(make_downscale_episode("de-d4", 0.8))
    assert state.total_episodes_handled == 2
    assert state.compound_factor == pytest.approx(0.9 * 0.8)
