"""Unit tests for P_equ profile fully wired (4 ops + 3 channels)."""
from __future__ import annotations

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.profiles.p_equ import PEquProfile


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


def test_p_equ_registers_4_ops() -> None:
    profile = PEquProfile()
    handlers = profile.runtime._handlers
    assert Operation.REPLAY in handlers
    assert Operation.DOWNSCALE in handlers
    assert Operation.RESTRUCTURE in handlers
    assert Operation.RECOMBINE in handlers
    assert len(handlers) == 4


def test_p_equ_executes_4_op_sequence() -> None:
    profile = PEquProfile()
    profile.runtime.execute(
        _make_de("de-eq0", Operation.REPLAY,
                 {"beta_records": [{"id": 1}]})
    )
    profile.runtime.execute(
        _make_de("de-eq1", Operation.DOWNSCALE,
                 {"shrink_factor": 0.95})
    )
    profile.runtime.execute(
        _make_de("de-eq2", Operation.RESTRUCTURE,
                 {"topo_op": "add"})
    )
    profile.runtime.execute(
        _make_de("de-eq3", Operation.RECOMBINE,
                 {"delta_latents": [[1.0, 0.0], [0.0, 1.0]]})
    )
    assert profile.replay_state.total_episodes_handled == 1
    assert profile.downscale_state.total_episodes_handled == 1
    assert profile.restructure_state.total_episodes_handled == 1
    assert profile.recombine_state.total_episodes_handled == 1


def test_p_equ_status_marker_updated() -> None:
    profile = PEquProfile()
    assert profile.status == "wired"
    assert profile.unimplemented_ops == []


def test_p_equ_log_contains_all_4_episodes() -> None:
    profile = PEquProfile()
    profile.runtime.execute(
        _make_de("de-eq4", Operation.REPLAY,
                 {"beta_records": []})
    )
    profile.runtime.execute(
        _make_de("de-eq5", Operation.DOWNSCALE,
                 {"shrink_factor": 0.99})
    )
    profile.runtime.execute(
        _make_de("de-eq6", Operation.RESTRUCTURE,
                 {"topo_op": "reroute"})
    )
    profile.runtime.execute(
        _make_de("de-eq7", Operation.RECOMBINE,
                 {"delta_latents": [[1.0], [0.0]]})
    )
    ids = [e.episode_id for e in profile.runtime.log]
    assert ids == ["de-eq4", "de-eq5", "de-eq6", "de-eq7"]
    assert all(e.completed for e in profile.runtime.log)
