"""Unit tests for PMaxLoRAProfile (B6c)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    import mlx.core as mx
    import mlx.nn as nn
else:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

from kiki_oniric.dream.channels.alpha_stream import TraceRecord
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


LATENT_DIM = 4
INPUT_DIM = 4


class _TinyEncoder(nn.Module):  # type: ignore[misc,name-defined]
    """Deterministic linear encoder x → (mu, log_var).

    Matches the fixture in tests/unit/test_recombine_latent_sample.py.
    Copied here to avoid a cross-test-file import (pytest collection
    fragility) — small enough that duplication is cheaper than
    extracting into a third helpers module.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mu_w = mx.ones((LATENT_DIM, INPUT_DIM)) * 0.1
        self.lv_w = mx.ones((LATENT_DIM, INPUT_DIM)) * -0.5

    def __call__(self, x: Any) -> Any:  # noqa: ANN401
        return x @ self.mu_w.T, x @ self.lv_w.T


class _TinyDecoder(nn.Module):  # type: ignore[misc,name-defined]
    """Deterministic linear decoder z → output."""

    def __init__(self) -> None:
        super().__init__()
        self.w = mx.ones((INPUT_DIM, LATENT_DIM)) * 0.2

    def __call__(self, z: Any) -> Any:  # noqa: ANN401
        return z @ self.w.T


def _replay_episode() -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={
            "beta_records": [
                {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
                {"x": [0.5, 0.6, 0.7, 0.8], "y": [0.0, 1.0]},
            ],
        },
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pmax-lora-replay",
    )


def _downscale_episode(factor: float = 0.5) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"shrink_factor": factor},
        operation_set=(Operation.DOWNSCALE,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pmax-lora-dn",
    )


def _restructure_episode(topo_ops: list[dict[str, object]]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"topo_ops": topo_ops},
        operation_set=(Operation.RESTRUCTURE,),
        output_channels=(OutputChannel.HIERARCHY_CHG,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pmax-lora-restr",
    )


def _recombine_episode() -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"delta_latents": [[1.0, 2.0, 3.0, 4.0]]},
        operation_set=(Operation.RECOMBINE,),
        output_channels=(OutputChannel.LATENT_SAMPLE,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pmax-lora-rec",
    )


def _build_profile(seed: int = 0) -> Any:  # noqa: ANN401
    """Build a PMaxLoRAProfile with bit-identical dream/awake LoRAModels."""
    from kiki_oniric.profiles.p_max_lora import PMaxLoRAProfile

    dream, awake = lora_clones(seed=seed)
    return PMaxLoRAProfile(
        dream_model=dream,
        awake_model=awake,
        encoder=_TinyEncoder(),
        decoder=_TinyDecoder(),
        seed=42,
    ), dream, awake


def test_pmax_lora_construction_happy_path() -> None:
    from kiki_oniric.dream.channels.alpha_stream import AlphaStreamBuffer
    from kiki_oniric.dream.channels.attention_prior import (
        AttentionPriorChannel,
    )
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )

    profile, _, _ = _build_profile()
    assert isinstance(profile.weight_channel, LoRAWeightDeltaChannel)
    assert isinstance(profile.hierarchy_channel, LoRAHierarchyChangeChannel)
    assert isinstance(profile.latent_channel, LatentSampleQueue)
    assert isinstance(profile.attention_prior, AttentionPriorChannel)
    assert isinstance(profile.alpha_stream, AlphaStreamBuffer)
    for op in (
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    ):
        assert op in profile.runtime._handlers


def test_pmax_lora_construction_missing_dream_raises() -> None:
    from kiki_oniric.profiles.p_max_lora import PMaxLoRAProfile

    _, awake = lora_clones(seed=0)
    with pytest.raises(TypeError):
        PMaxLoRAProfile(  # type: ignore[call-arg]
            awake_model=awake,
            encoder=_TinyEncoder(),
            decoder=_TinyDecoder(),
        )


def test_pmax_lora_construction_missing_awake_raises() -> None:
    from kiki_oniric.profiles.p_max_lora import PMaxLoRAProfile

    dream, _ = lora_clones(seed=0)
    with pytest.raises(TypeError):
        PMaxLoRAProfile(  # type: ignore[call-arg]
            dream_model=dream,
            encoder=_TinyEncoder(),
            decoder=_TinyDecoder(),
        )


def test_pmax_lora_construction_missing_encoder_raises() -> None:
    from kiki_oniric.profiles.p_max_lora import PMaxLoRAProfile

    dream, awake = lora_clones(seed=0)
    with pytest.raises(TypeError):
        PMaxLoRAProfile(  # type: ignore[call-arg]
            dream_model=dream,
            awake_model=awake,
            decoder=_TinyDecoder(),
        )


def test_pmax_lora_construction_missing_decoder_raises() -> None:
    from kiki_oniric.profiles.p_max_lora import PMaxLoRAProfile

    dream, awake = lora_clones(seed=0)
    with pytest.raises(TypeError):
        PMaxLoRAProfile(  # type: ignore[call-arg]
            dream_model=dream,
            awake_model=awake,
            encoder=_TinyEncoder(),
        )


def test_pmax_lora_replay_emits_weight_update() -> None:
    from kiki_oniric.dream.channels import WeightUpdate

    profile, _, _ = _build_profile()
    profile.runtime.execute(_replay_episode())
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, WeightUpdate)


def test_pmax_lora_downscale_emits_weight_update() -> None:
    from kiki_oniric.dream.channels import WeightUpdate

    profile, _, _ = _build_profile()
    profile.runtime.execute(_downscale_episode(factor=0.5))
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, WeightUpdate)


def test_pmax_lora_restructure_emits_topology_diff() -> None:
    from kiki_oniric.dream.channels import TopologyDiff

    profile, _, _ = _build_profile()
    profile.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, TopologyDiff)


def test_pmax_lora_recombine_emits_latent_sample() -> None:
    """First profile to emit ch2 — recombine_real_handler VAE."""
    from kiki_oniric.dream.channels import LatentSample

    profile, _, _ = _build_profile()
    profile.runtime.execute(_recombine_episode())
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, LatentSample)


def test_pmax_lora_consolidate_log_applies_weight_to_awake_bit_equal() -> None:
    profile, dream, awake = _build_profile()
    profile.runtime.execute(_replay_episode())
    assert not np.array_equal(
        np.asarray(dream.layers[0].lora_b),
        np.asarray(awake.layers[0].lora_b),
    )
    profile.consolidate_log()
    assert_lora_models_equal(dream, awake)


def test_pmax_lora_consolidate_log_applies_hierarchy_to_awake() -> None:
    profile, dream, awake = _build_profile()
    pre_len = len(dream.layers)
    profile.runtime.execute(
        _restructure_episode([{
            "op": "add",
            "index": pre_len,
            "in_features": 4,
            "out_features": 8,
            "rank": 2,
            "alpha": 4.0,
        }]),
    )
    assert len(dream.layers) == pre_len + 1
    assert len(awake.layers) == pre_len
    profile.consolidate_log()
    assert_lora_models_equal(dream, awake)


def test_pmax_lora_consolidate_log_enqueues_latent_sample() -> None:
    """recombine emits LatentSample → consolidate_log enqueues it."""
    profile, _, _ = _build_profile()
    profile.runtime.execute(_recombine_episode())
    profile.consolidate_log()
    assert len(profile.latent_channel) == 1
    sample = profile.latent_channel.dequeue()
    assert sample is not None
    assert "species" in sample
    assert "latent_vector" in sample
    assert "provenance" in sample


def test_pmax_lora_consolidate_log_mixed_emits_count_4() -> None:
    profile, dream, awake = _build_profile()
    profile.runtime.execute(_replay_episode())
    profile.runtime.execute(_downscale_episode(factor=0.7))
    profile.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    profile.runtime.execute(_recombine_episode())
    assert profile.consolidate_log() == 4


def test_pmax_lora_consolidate_log_clears_and_idempotent() -> None:
    profile, _, _ = _build_profile()
    profile.runtime.execute(_replay_episode())
    profile.consolidate_log()
    assert len(profile.runtime.log) == 0
    assert profile.consolidate_log() == 0


def test_pmax_lora_attention_prior_settable_and_readable() -> None:
    profile, _, _ = _build_profile()
    prior = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    profile.attention_prior.set_prior(prior)
    got = profile.attention_prior.get_prior()
    assert got is not None
    np.testing.assert_array_equal(got, prior)


def test_pmax_lora_alpha_stream_append_and_read() -> None:
    """α input channel inherited from PMaxProfile — append + read FIFO."""
    profile, _, _ = _build_profile()
    rec = TraceRecord(
        tokens=np.zeros(2, dtype=np.int32),
        activations=np.zeros(2, dtype=np.float32),
        attention=np.zeros(2, dtype=np.float32),
        errors=np.zeros(2, dtype=np.float32),
    )
    profile.alpha_stream.append(rec)
    out = profile.alpha_stream.snapshot()
    assert len(out) == 1


def test_pmax_lora_dr4_chain_inclusion_full_triple() -> None:
    """DR-4 across all three LoRA profiles: ops + channel emitters."""
    from kiki_oniric.dream.channels import (
        LatentSample,
        TopologyDiff,
        WeightUpdate,
    )
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile
    from kiki_oniric.profiles.p_max_lora import PMaxLoRAProfile
    from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile

    dream_min, awake_min = lora_clones(seed=0)
    pmin = PMinLoRAProfile(dream_model=dream_min, awake_model=awake_min)

    dream_equ, awake_equ = lora_clones(seed=0)
    pequ = PEquLoRAProfile(dream_model=dream_equ, awake_model=awake_equ)

    dream_max, awake_max = lora_clones(seed=0)
    pmax = PMaxLoRAProfile(
        dream_model=dream_max,
        awake_model=awake_max,
        encoder=_TinyEncoder(),
        decoder=_TinyDecoder(),
        seed=7,
    )

    pmin_ops = set(pmin.runtime._handlers.keys())
    pequ_ops = set(pequ.runtime._handlers.keys())
    pmax_ops = set(pmax.runtime._handlers.keys())
    assert pmin_ops <= pequ_ops, "ops(PMin) ⊆ ops(PEqu)"
    assert pequ_ops <= pmax_ops, "ops(PEqu) ⊆ ops(PMax)"

    pmin.runtime.execute(_replay_episode())
    pmin_emitted = {
        type(o) for entry in pmin.runtime.log
        for o in entry.channel_outputs if o is not None
    }

    pequ.runtime.execute(_replay_episode())
    pequ.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    pequ_emitted = {
        type(o) for entry in pequ.runtime.log
        for o in entry.channel_outputs if o is not None
    }

    pmax.runtime.execute(_replay_episode())
    pmax.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    pmax.runtime.execute(_recombine_episode())
    pmax_emitted = {
        type(o) for entry in pmax.runtime.log
        for o in entry.channel_outputs if o is not None
    }

    assert pmin_emitted == {WeightUpdate}
    assert pequ_emitted == {WeightUpdate, TopologyDiff}
    assert pmax_emitted == {WeightUpdate, TopologyDiff, LatentSample}
    assert pmin_emitted <= pequ_emitted <= pmax_emitted
