"""Tests for the G6 dream-episode wrapper over MicroKikiSubstrate."""
from __future__ import annotations

import pytest

from experiments.g6_mmlu_stream.dream_wrap import (
    G6DreamRunner,
    build_episode_payload,
)
from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate


def _seeded_substrate() -> MicroKikiSubstrate:
    return MicroKikiSubstrate(num_layers=4, rank=8, seed=0)


def test_build_episode_payload_shape() -> None:
    payload = build_episode_payload(
        seed=0,
        adapter_keys=("layer_0_lora_B",),
        out_dim=8,
        rank=4,
    )
    assert "beta_records" in payload
    assert "deltas" in payload
    assert "prior_deltas" in payload
    assert "shrink_factor" in payload
    assert payload["shrink_factor"] == pytest.approx(0.99)
    assert all("input" in r for r in payload["beta_records"])
    assert all("episode_id" not in r for r in payload["beta_records"])


def test_g6_dream_runner_baseline_is_no_op() -> None:
    sub = _seeded_substrate()
    runner = G6DreamRunner(substrate=sub, profile_name="baseline")
    before_recombine = sub.recombine_state.total_episodes_handled
    before_restructure = sub.restructure_state.total_episodes_handled
    runner.run_episode(seed=0, subdomain="anatomy")
    # baseline arm runs no DE — substrate state unchanged.
    assert sub.recombine_state.total_episodes_handled == before_recombine
    assert sub.restructure_state.total_episodes_handled == before_restructure


def test_g6_dream_runner_p_min_invokes_replay_downscale_only() -> None:
    sub = _seeded_substrate()
    runner = G6DreamRunner(substrate=sub, profile_name="P_min")
    runner.run_episode(seed=0, subdomain="anatomy")
    # P_min must NOT call restructure / recombine — they are not in
    # the P_min op set.
    assert sub.recombine_state.total_episodes_handled == 0
    assert sub.restructure_state.total_episodes_handled == 0
    # Replay + downscale handlers do not bump state on micro-kiki
    # substrate (they return tensors, not mutate _state). The wrapper
    # must therefore expose its own counter.
    assert runner.episodes_run == 1
    assert "P_min" in runner.last_episode_id


def test_g6_dream_runner_p_equ_invokes_all_four_handlers() -> None:
    sub = _seeded_substrate()
    runner = G6DreamRunner(substrate=sub, profile_name="P_equ")
    runner.run_episode(seed=0, subdomain="astronomy")
    # P_equ wires restructure + recombine — substrate state bumps.
    assert sub.recombine_state.total_episodes_handled == 1
    assert sub.restructure_state.total_episodes_handled == 1
    assert sub.recombine_state.last_episode_id is not None
    assert "astronomy" in sub.recombine_state.last_episode_id


def test_g6_dream_runner_p_max_invokes_all_four_handlers() -> None:
    sub = _seeded_substrate()
    runner = G6DreamRunner(substrate=sub, profile_name="P_max")
    runner.run_episode(seed=0, subdomain="business_ethics")
    assert sub.recombine_state.total_episodes_handled == 1
    assert sub.restructure_state.total_episodes_handled == 1


def test_g6_dream_runner_rejects_unknown_profile() -> None:
    sub = _seeded_substrate()
    with pytest.raises(ValueError, match="unknown profile"):
        G6DreamRunner(substrate=sub, profile_name="P_quantum")


def test_g6_dream_runner_episode_ids_are_unique_per_call() -> None:
    sub = _seeded_substrate()
    runner = G6DreamRunner(substrate=sub, profile_name="P_equ")
    runner.run_episode(seed=0, subdomain="anatomy")
    runner.run_episode(seed=0, subdomain="astronomy")
    # Subdomain enters the episode_id, so two distinct subjects with
    # the same seed must yield distinct ids.
    assert (
        sub.recombine_state.episode_ids[0]
        != sub.recombine_state.episode_ids[1]
    )
    assert "anatomy" in sub.recombine_state.episode_ids[0]
    assert "astronomy" in sub.recombine_state.episode_ids[1]
