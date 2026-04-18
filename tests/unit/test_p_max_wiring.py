"""Unit tests for P_max profile fully wired (4 ops + 4 channels)."""
from __future__ import annotations

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.profiles.p_max import PMaxProfile


def _make_de(ep_id: str, op: Operation, slice_data: dict) -> DreamEpisode:
    channel_map = {
        Operation.REPLAY: OutputChannel.WEIGHT_DELTA,
        Operation.DOWNSCALE: OutputChannel.WEIGHT_DELTA,
        Operation.RESTRUCTURE: OutputChannel.HIERARCHY_CHG,
        Operation.RECOMBINE: OutputChannel.LATENT_SAMPLE,
    }
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice=slice_data,
        operation_set=(op,),
        output_channels=(channel_map[op],),
        budget=BudgetCap(flops=2_000, wall_time_s=0.2, energy_j=0.02),
        episode_id=ep_id,
    )


def test_p_max_status_wired() -> None:
    profile = PMaxProfile()
    assert profile.status == "wired"
    assert profile.unimplemented_ops == []


def test_p_max_registers_4_ops() -> None:
    profile = PMaxProfile()
    handlers = profile.runtime._handlers
    assert Operation.REPLAY in handlers
    assert Operation.DOWNSCALE in handlers
    assert Operation.RESTRUCTURE in handlers
    assert Operation.RECOMBINE in handlers
    assert len(handlers) == 4


def test_p_max_executes_4_op_sequence() -> None:
    profile = PMaxProfile()
    profile.runtime.execute(
        _make_de("de-mx0", Operation.REPLAY,
                 {"beta_records": [{"id": 1}]})
    )
    profile.runtime.execute(
        _make_de("de-mx1", Operation.DOWNSCALE,
                 {"shrink_factor": 0.95})
    )
    profile.runtime.execute(
        _make_de("de-mx2", Operation.RESTRUCTURE,
                 {"topo_op": "add"})
    )
    profile.runtime.execute(
        _make_de("de-mx3", Operation.RECOMBINE,
                 {"delta_latents": [[1.0, 0.0], [0.0, 1.0]]})
    )
    assert profile.replay_state.total_episodes_handled == 1
    assert profile.downscale_state.total_episodes_handled == 1
    assert profile.restructure_state.total_episodes_handled == 1
    assert profile.recombine_state.total_episodes_handled == 1


def test_p_max_alpha_stream_buffer_present() -> None:
    """P_max declares alpha_stream channel for raw traces."""
    profile = PMaxProfile()
    assert hasattr(profile, "alpha_stream")
    assert profile.alpha_stream.capacity > 0


def test_p_max_attention_prior_channel_present() -> None:
    """P_max declares attention_prior channel-4 channel."""
    profile = PMaxProfile()
    assert hasattr(profile, "attention_prior")
    assert profile.attention_prior.budget > 0
