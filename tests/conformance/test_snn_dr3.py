"""DR-3 Conformance Criterion — SNN-proxy ops (cycle-3 C3.12).

Validates that the 4 SNN-substrate dream ops
(:mod:`kiki_oniric.dream.operations.{replay,downscale,restructure,
recombine}_snn`) satisfy condition (1) of the DR-3 Conformance
Criterion : typed Protocol-compatible state + handler factory
signatures matching framework-C §2.1, plus the spike-rate-proxy
round-trip invariant (S1 information preservation).

The 3 TDD-mandated tests land first :
1. ``test_weights_to_spike_rates_proxy`` — real weights map to
   bounded spike rates (``[0, max_rate]``).
2. ``test_dream_op_on_spike_rates`` — each of the 4 ops exhibits
   the expected behaviour on spike rates.
3. ``test_project_back_preserves_information`` — round-trip
   ``weights → rates → weights`` is identity within tolerance.

Then 4 op-specific tests cover the per-op semantics in more
detail (7 tests total — above the ≥ 7 threshold in C3.12).

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
"""
from __future__ import annotations

from dataclasses import is_dataclass

import numpy as np
import pytest

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.guards.finite import FiniteGuardError
from kiki_oniric.dream.operations import (
    downscale_snn,
    recombine_snn,
    replay_snn,
    restructure_snn,
)


def _make_episode(input_slice: dict, op: Operation) -> DreamEpisode:
    """Minimal DreamEpisode fixture for the SNN-proxy ops."""
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice=input_slice,
        operation_set=(op,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="test-snn-de",
    )


# ===== Plan-mandated TDD tests =====


def test_weights_to_spike_rates_proxy() -> None:
    """Test 1 : Qwen-style real weights map to bounded spike rates.

    Any-sign real weights must land in the closed interval
    ``[0, max_rate]``. The mapping is sigmoid-scaled, so ``w = 0``
    lands at ``max_rate / 2`` (the neutral baseline) and the
    mapping is monotonically increasing.
    """
    w = np.array([-5.0, -1.0, 0.0, 1.0, 5.0])
    rates = replay_snn.weights_to_spike_rates(w, max_rate=100.0)

    # Bounded in [0, max_rate]
    assert (rates >= 0.0).all()
    assert (rates <= 100.0).all()
    # Monotone increasing
    assert np.all(np.diff(rates) > 0.0)
    # w = 0 → max_rate / 2
    assert abs(rates[2] - 50.0) < 1e-9


def test_dream_op_on_spike_rates() -> None:
    """Test 2 : dream-op applied on spike rates yields expected
    behaviour for each of the 4 SNN-proxy ops.

    - ``replay_snn``   : rates should move toward ``target_rates``
      (post-update rate-delta norm < pre-update delta norm).
    - ``downscale_snn``: rates should shrink (monotone ≤ previous).
    - ``restructure_snn``: rates along axis-0 are swapped.
    - ``recombine_snn``: last_sample length matches latent length.
    """
    # --- replay_snn : target attraction
    w_replay = np.zeros(4, dtype=float)  # rates = 50 each
    state_r = replay_snn.ReplaySNNState()
    handler_r = replay_snn.replay_snn_handler(
        state_r, weights=w_replay, lr=0.5, max_rate=100.0
    )
    target = np.array([80.0, 80.0, 20.0, 20.0])
    ep = _make_episode({"target_rates": target}, Operation.REPLAY)
    handler_r(ep)
    # After update, rates should be strictly closer to target
    new_rates = replay_snn.weights_to_spike_rates(w_replay)
    assert np.linalg.norm(target - new_rates) < np.linalg.norm(
        target - np.full(4, 50.0)
    )

    # --- downscale_snn : rate shrink
    w_down = np.array([1.0, 2.0, 3.0, 4.0], dtype=float)
    rates_before = replay_snn.weights_to_spike_rates(w_down)
    state_d = downscale_snn.DownscaleSNNState()
    handler_d = downscale_snn.downscale_snn_handler(
        state_d, weights=w_down
    )
    ep = _make_episode({"shrink_factor": 0.5}, Operation.DOWNSCALE)
    handler_d(ep)
    rates_after = replay_snn.weights_to_spike_rates(w_down)
    assert (rates_after <= rates_before + 1e-9).all()

    # --- restructure_snn : channel swap
    w_res = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    rates_before = replay_snn.weights_to_spike_rates(w_res).copy()
    state_s = restructure_snn.RestructureSNNState()
    handler_s = restructure_snn.restructure_snn_handler(
        state_s, weights=w_res
    )
    ep = _make_episode(
        {"topo_op": "reroute", "swap_indices": [0, 1]},
        Operation.RESTRUCTURE,
    )
    handler_s(ep)
    rates_after = replay_snn.weights_to_spike_rates(w_res)
    np.testing.assert_allclose(rates_after[0], rates_before[1])
    np.testing.assert_allclose(rates_after[1], rates_before[0])

    # --- recombine_snn : interpolation shape
    state_c = recombine_snn.RecombineSNNState()
    handler_c = recombine_snn.recombine_snn_handler(state_c, seed=42)
    latents = [[0.5, -0.5, 1.0], [-1.0, 1.0, 0.0]]
    ep = _make_episode(
        {"delta_latents": latents}, Operation.RECOMBINE
    )
    handler_c(ep)
    assert state_c.last_sample is not None
    assert len(state_c.last_sample) == 3


def test_project_back_preserves_information() -> None:
    """Test 3 : round-trip weights → rates → weights is identity
    (S1 information preservation within precision bounds).

    For weights in the well-conditioned band ``|w| <= 5``, the
    sigmoid + inverse-sigmoid round-trip preserves the vector norm
    within ``1e-6``. For extreme weights (``|w| > 10``) the sigmoid
    saturates and the inverse map loses precision — this is a
    documented concern of the proxy (see CLAUDE concerns below).
    """
    rng = np.random.default_rng(0)
    w = rng.uniform(-5.0, 5.0, size=(16,))
    rates = replay_snn.weights_to_spike_rates(w)
    w_back = replay_snn.spike_rates_to_weights(rates)

    # Element-wise identity within tolerance
    np.testing.assert_allclose(w, w_back, atol=1e-6, rtol=1e-6)
    # Norm preserved
    assert abs(
        float(np.linalg.norm(w)) - float(np.linalg.norm(w_back))
    ) < 1e-6


# ===== DR-3 conformance — typed state + factory signatures =====


def test_dr3_snn_ops_typed_protocol() -> None:
    """DR-3 condition (1) : all 4 _snn.py ops expose typed
    ``*State`` dataclasses + handler factories returning
    ``Callable[[DreamEpisode], None]``.

    Mirrors the E-SNN conformance suite pattern in
    :mod:`tests.conformance.axioms.test_dr3_esnn_substrate`.
    """
    # State dataclasses : must be @dataclass + expose the
    # K1 accounting fields.
    for state_cls in (
        replay_snn.ReplaySNNState,
        downscale_snn.DownscaleSNNState,
        restructure_snn.RestructureSNNState,
        recombine_snn.RecombineSNNState,
    ):
        assert is_dataclass(state_cls)
        instance = state_cls()
        assert hasattr(instance, "last_compute_flops")
        assert hasattr(instance, "total_compute_flops")

    # Factories : exposed + callable + return a 1-arg callable.
    for factory, kwargs, weights_shape in (
        (replay_snn.replay_snn_handler, {"lr": 0.01}, (4,)),
        (downscale_snn.downscale_snn_handler, {}, (4,)),
        (restructure_snn.restructure_snn_handler, {}, (4, 2)),
    ):
        state = {
            replay_snn.replay_snn_handler:
                replay_snn.ReplaySNNState(),
            downscale_snn.downscale_snn_handler:
                downscale_snn.DownscaleSNNState(),
            restructure_snn.restructure_snn_handler:
                restructure_snn.RestructureSNNState(),
        }[factory]
        w = np.zeros(weights_shape, dtype=float)
        handler = factory(state, weights=w, **kwargs)
        assert callable(handler)

    # recombine factory has a different signature (seed-based).
    rec_handler = recombine_snn.recombine_snn_handler(
        recombine_snn.RecombineSNNState(), seed=0
    )
    assert callable(rec_handler)


# ===== Per-op specific behaviour (S1/S2/K1 coverage) =====


def test_replay_snn_k1_tagged_and_deterministic() -> None:
    """``replay_snn`` : deterministic under seed (same input ⇒ same
    rate delta) + K1 FLOP tag populated.
    """
    w1 = np.array([0.0, 0.5, -0.5, 1.0], dtype=float)
    w2 = w1.copy()
    state1 = replay_snn.ReplaySNNState()
    state2 = replay_snn.ReplaySNNState()
    h1 = replay_snn.replay_snn_handler(state1, weights=w1, lr=0.1)
    h2 = replay_snn.replay_snn_handler(state2, weights=w2, lr=0.1)
    target = np.array([75.0, 75.0, 25.0, 25.0])
    ep = _make_episode({"target_rates": target}, Operation.REPLAY)
    h1(ep)
    h2(ep)

    np.testing.assert_array_equal(w1, w2)
    assert state1.last_compute_flops > 0
    assert state1.total_compute_flops == state1.last_compute_flops
    assert state1.last_spike_delta_norm is not None

    # No-op branch : no target_rates ⇒ no compute, no tag bump.
    ep_noop = _make_episode({}, Operation.REPLAY)
    h1(ep_noop)
    assert state1.last_compute_flops == 0
    assert state1.last_spike_delta_norm is None


def test_downscale_snn_compound_factor_and_s2_guard() -> None:
    """``downscale_snn`` : compound_factor accumulates across
    episodes + rejects factor outside (0, 1].
    """
    w = np.array([1.0, -1.0, 2.0], dtype=float)
    state = downscale_snn.DownscaleSNNState()
    handler = downscale_snn.downscale_snn_handler(state, weights=w)
    ep1 = _make_episode({"shrink_factor": 0.5}, Operation.DOWNSCALE)
    ep2 = _make_episode({"shrink_factor": 0.5}, Operation.DOWNSCALE)
    handler(ep1)
    handler(ep2)
    assert abs(state.compound_factor - 0.25) < 1e-9
    assert state.total_compute_flops == 2 * state.last_compute_flops

    # factor > 1 rejected
    with pytest.raises(ValueError, match="shrink_factor"):
        handler(_make_episode(
            {"shrink_factor": 1.5}, Operation.DOWNSCALE
        ))
    # factor <= 0 rejected
    with pytest.raises(ValueError, match="shrink_factor"):
        handler(_make_episode(
            {"shrink_factor": 0.0}, Operation.DOWNSCALE
        ))
    # FiniteGuardError re-exported for test convenience
    assert downscale_snn.FiniteGuardError is FiniteGuardError


def test_restructure_snn_s3_tag_on_unknown_op() -> None:
    """``restructure_snn`` : unknown ``topo_op`` raises a
    ValueError whose message contains the ``"S3"`` literal.
    """
    w = np.array([[1.0, 2.0], [3.0, 4.0]], dtype=float)
    state = restructure_snn.RestructureSNNState()
    handler = restructure_snn.restructure_snn_handler(
        state, weights=w
    )

    # Unknown op
    with pytest.raises(ValueError, match="S3"):
        handler(_make_episode(
            {"topo_op": "quantize"}, Operation.RESTRUCTURE
        ))
    # "add" is deliberately unsupported in the cycle-3 proxy
    with pytest.raises(ValueError, match="S3"):
        handler(_make_episode(
            {"topo_op": "add"}, Operation.RESTRUCTURE
        ))
    # swap_indices length != 2
    with pytest.raises(ValueError, match="S3"):
        handler(_make_episode(
            {"topo_op": "reroute", "swap_indices": [0]},
            Operation.RESTRUCTURE,
        ))
    # swap_indices out-of-bounds
    with pytest.raises(ValueError, match="S3"):
        handler(_make_episode(
            {"topo_op": "reroute", "swap_indices": [0, 99]},
            Operation.RESTRUCTURE,
        ))

    # Successful reroute populates diff_history
    handler(_make_episode(
        {"topo_op": "reroute", "swap_indices": [0, 1]},
        Operation.RESTRUCTURE,
    ))
    assert state.diff_history == ["reroute"]
    assert state.last_compute_flops > 0


def test_recombine_snn_seed_determinism_and_i3() -> None:
    """``recombine_snn`` : same seed + same input ⇒ same
    ``last_sample`` + I3 guard rejects empty / mismatched
    latent batches.
    """
    latents = [[0.5, -0.5, 1.0, 0.0], [1.0, 0.0, -1.0, 0.5]]
    state_a = recombine_snn.RecombineSNNState()
    state_b = recombine_snn.RecombineSNNState()
    h_a = recombine_snn.recombine_snn_handler(state_a, seed=7)
    h_b = recombine_snn.recombine_snn_handler(state_b, seed=7)
    ep = _make_episode(
        {"delta_latents": latents}, Operation.RECOMBINE
    )
    h_a(ep)
    h_b(ep)
    assert state_a.last_sample == state_b.last_sample
    assert state_a.total_compute_flops == state_a.last_compute_flops

    # Empty latents rejected
    with pytest.raises(ValueError, match="I3"):
        h_a(_make_episode(
            {"delta_latents": []}, Operation.RECOMBINE
        ))
    # Single latent rejected (need >= 2)
    with pytest.raises(ValueError, match="I3"):
        h_a(_make_episode(
            {"delta_latents": [[0.1, 0.2]]}, Operation.RECOMBINE
        ))
    # Mismatched latent dims rejected
    with pytest.raises(ValueError, match="I3"):
        h_a(_make_episode(
            {"delta_latents": [[0.1, 0.2], [0.3, 0.4, 0.5]]},
            Operation.RECOMBINE,
        ))

    # Episode counter advances ⇒ different sample next call
    h_a(ep)
    # state_a after 2 calls + counter increment ≠ state_b 1 call
    assert state_a._episode_count == 2
    assert state_b._episode_count == 1
