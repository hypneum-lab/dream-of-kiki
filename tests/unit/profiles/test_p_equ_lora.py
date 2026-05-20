"""Unit tests for PEquLoRAProfile (B6b)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import mlx.core as mx
else:
    mx = pytest.importorskip("mlx.core")

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from tests.unit.profiles._lora_helpers import (
    assert_lora_models_equal,
    lora_clones,
)


def _replay_episode(records: list[dict[str, object]] | None = None) -> DreamEpisode:
    if records is None:
        records = [
            {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
            {"x": [0.5, 0.6, 0.7, 0.8], "y": [0.0, 1.0]},
        ]
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"beta_records": records},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pequ-lora-replay",
    )


def _downscale_episode(factor: float = 0.5) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"shrink_factor": factor},
        operation_set=(Operation.DOWNSCALE,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pequ-lora-dn",
    )


def _restructure_episode(
    topo_ops: list[dict[str, object]],
) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"topo_ops": topo_ops},
        operation_set=(Operation.RESTRUCTURE,),
        output_channels=(OutputChannel.HIERARCHY_CHG,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pequ-lora-restr",
    )


def _recombine_light_episode() -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"delta_latents": [[1.0, 2.0], [3.0, 4.0]]},
        operation_set=(Operation.RECOMBINE,),
        output_channels=(),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pequ-lora-rec",
    )


def test_pequ_lora_construction_happy_path() -> None:
    from kiki_oniric.dream.channels.attention_prior import (
        AttentionPriorChannel,
    )
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = lora_clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    assert isinstance(profile.weight_channel, LoRAWeightDeltaChannel)
    assert isinstance(profile.hierarchy_channel, LoRAHierarchyChangeChannel)
    assert isinstance(profile.attention_prior, AttentionPriorChannel)
    # All four LoRA / skeleton handlers must be registered.
    for op in (
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    ):
        assert op in profile.runtime._handlers


def test_pequ_lora_construction_missing_dream_raises() -> None:
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    _, awake = lora_clones(seed=0)
    with pytest.raises(TypeError):
        PEquLoRAProfile(awake_model=awake)  # type: ignore[call-arg]


def test_pequ_lora_construction_missing_awake_raises() -> None:
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, _ = lora_clones(seed=0)
    with pytest.raises(TypeError):
        PEquLoRAProfile(dream_model=dream)  # type: ignore[call-arg]


def test_pequ_lora_replay_emits_weight_update_in_log() -> None:
    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = lora_clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, WeightUpdate)


def test_pequ_lora_downscale_emits_weight_update_in_log() -> None:
    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = lora_clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_downscale_episode(factor=0.5))
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, WeightUpdate)


def test_pequ_lora_restructure_emits_topology_diff_in_log() -> None:
    from kiki_oniric.dream.channels import TopologyDiff
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = lora_clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, TopologyDiff)


def test_pequ_lora_recombine_light_returns_none_in_log() -> None:
    """Skeleton recombine_handler returns None — no ch2 emission."""
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = lora_clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_recombine_light_episode())
    out = profile.runtime.log[-1].channel_outputs[0]
    assert out is None


def test_pequ_lora_consolidate_log_applies_weight_to_awake_bit_equal() -> None:
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = lora_clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())
    # Sanity: dream mutated, awake untouched yet.
    assert not np.array_equal(
        np.asarray(dream.layers[0].lora_b),
        np.asarray(awake.layers[0].lora_b),
    )
    profile.consolidate_log()
    assert_lora_models_equal(dream, awake)


def test_pequ_lora_consolidate_log_applies_hierarchy_to_awake() -> None:
    """add via restructure → consolidate → awake gets new layer bit-equal."""
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = lora_clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    pre_len = len(dream.layers)
    add_op = {
        "op": "add",
        "index": pre_len,
        "in_features": 4,
        "out_features": 8,
        "rank": 2,
        "alpha": 4.0,
    }
    profile.runtime.execute(_restructure_episode([add_op]))
    # Dream gained a layer; awake hasn't yet.
    assert len(dream.layers) == pre_len + 1
    assert len(awake.layers) == pre_len
    profile.consolidate_log()
    # After consolidation, awake matches dream bit-for-bit.
    assert_lora_models_equal(dream, awake)


def test_pequ_lora_consolidate_log_mixed_emits_count() -> None:
    """3 episodes (replay, restructure, downscale) → 3 dispatched outputs."""
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = lora_clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())
    profile.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    profile.runtime.execute(_downscale_episode(factor=0.7))
    assert profile.consolidate_log() == 3


def test_pequ_lora_consolidate_log_clears_and_idempotent() -> None:
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = lora_clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())
    assert len(profile.runtime.log) == 1
    profile.consolidate_log()
    assert len(profile.runtime.log) == 0
    # Second call without further episodes is idempotent no-op.
    assert profile.consolidate_log() == 0


def test_pequ_lora_attention_prior_settable_and_readable() -> None:
    """ch4 is a state surface: set via .set_prior, read via .get_prior."""
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = lora_clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    prior = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    profile.attention_prior.set_prior(prior)
    got = profile.attention_prior.get_prior()
    assert got is not None
    np.testing.assert_array_equal(got, prior)


def test_pequ_lora_no_latent_in_log_after_recombine_light() -> None:
    """No LatentSample ever appears — channel 2 not in P_equ."""
    from kiki_oniric.dream.channels import LatentSample
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = lora_clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())
    profile.runtime.execute(_recombine_light_episode())
    profile.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    for entry in profile.runtime.log:
        for output in entry.channel_outputs:
            assert not isinstance(output, LatentSample)


def test_pequ_lora_dr4_chain_inclusion_with_pmin_lora() -> None:
    """DR-4: ops(PMinLoRA) ⊆ ops(PEquLoRA); channel emitters likewise."""
    from kiki_oniric.dream.channels import TopologyDiff, WeightUpdate
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile
    from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile

    dream_min, awake_min = lora_clones(seed=0)
    pmin = PMinLoRAProfile(dream_model=dream_min, awake_model=awake_min)
    dream_equ, awake_equ = lora_clones(seed=0)
    pequ = PEquLoRAProfile(dream_model=dream_equ, awake_model=awake_equ)

    pmin_ops = set(pmin.runtime._handlers.keys())
    pequ_ops = set(pequ.runtime._handlers.keys())
    assert pmin_ops <= pequ_ops, "ops(PMin) must be subset of ops(PEqu)"

    # Channel emitter set inclusion: PMin emits ch1 only; PEqu emits
    # ch1 + ch3. We confirm by sampling one of each from each profile.
    pmin.runtime.execute(_replay_episode())
    pmin_emitted = {
        type(o) for entry in pmin.runtime.log
        for o in entry.channel_outputs if o is not None
    }
    assert pmin_emitted == {WeightUpdate}

    pequ.runtime.execute(_replay_episode())
    pequ.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    pequ_emitted = {
        type(o) for entry in pequ.runtime.log
        for o in entry.channel_outputs if o is not None
    }
    assert pmin_emitted <= pequ_emitted
    assert pequ_emitted == {WeightUpdate, TopologyDiff}
