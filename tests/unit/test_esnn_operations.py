"""Unit tests for E-SNN operations wiring (C2.3 cycle 2)."""
from __future__ import annotations

import numpy as np

from kiki_oniric.substrates.esnn_thalamocortical import (
    EsnnBackend,
    EsnnSubstrate,
    LIFState,
    simulate_lif_step,
)


def test_lif_state_initializes_zeros() -> None:
    state = LIFState(n_neurons=8)
    assert state.v.shape == (8,)
    assert state.spikes.shape == (8,)
    assert np.all(state.v == 0.0)
    assert np.all(state.spikes == 0)


def test_lif_step_accumulates_membrane_potential() -> None:
    """Sub-threshold input integrates without spiking."""
    state = LIFState(n_neurons=4)
    # Input below threshold (threshold=1.0 default)
    input_current = np.array([0.2, 0.3, 0.1, 0.4])
    new_state = simulate_lif_step(state, input_current, dt=1.0,
                                  tau=10.0, threshold=1.0)
    assert np.all(new_state.v > 0.0)    # integrated input
    assert np.all(new_state.v < 1.0)    # no spike
    assert np.all(new_state.spikes == 0)


def test_lif_step_fires_on_threshold_crossing() -> None:
    """Super-threshold input fires spike and resets membrane."""
    state = LIFState(n_neurons=2)
    input_current = np.array([2.0, 2.0])  # above threshold
    new_state = simulate_lif_step(state, input_current, dt=1.0,
                                  tau=10.0, threshold=1.0)
    assert np.all(new_state.spikes == 1)
    assert np.all(new_state.v == 0.0)  # reset post-spike


def test_esnn_replay_handler_runs_spike_rate_sim() -> None:
    """Replay op uses LIF dynamics to compute retention gradient."""
    substrate = EsnnSubstrate(backend=EsnnBackend.NORSE)
    handler = substrate.replay_handler_factory()
    # Fake episode structure (reusing framework spec)
    beta_records = [
        {"input": [0.5, 0.7, 0.3, 0.8], "expected": [1.0, 0.0]},
    ]
    spike_rates = handler(beta_records, n_steps=20)
    assert isinstance(spike_rates, np.ndarray)
    assert spike_rates.shape[0] > 0


def test_esnn_downscale_handler_shrinks_synaptic_weights() -> None:
    """Downscale op applies multiplicative synaptic scaling."""
    substrate = EsnnSubstrate()
    handler = substrate.downscale_handler_factory()
    weights = np.array([0.8, 1.2, 0.5, 0.9])
    shrunk = handler(weights, factor=0.5)
    np.testing.assert_allclose(shrunk, weights * 0.5)


def test_esnn_restructure_handler_modifies_topology() -> None:
    """Restructure op returns modified connectivity matrix."""
    substrate = EsnnSubstrate()
    handler = substrate.restructure_handler_factory()
    # 4 neurons fully connected minus self-loops
    conn = np.array([
        [0, 1, 1, 1],
        [1, 0, 1, 1],
        [1, 1, 0, 1],
        [1, 1, 1, 0],
    ], dtype=float)
    # Add one connection (already fully connected so this tests remove)
    new_conn = handler(conn, op="remove", src=0, dst=1)
    assert new_conn[0, 1] == 0.0
    assert new_conn[1, 0] == 1.0  # untouched


def test_esnn_recombine_handler_samples_latent() -> None:
    """Recombine op samples latent via spike-train Poisson code."""
    substrate = EsnnSubstrate()
    handler = substrate.recombine_handler_factory()
    # Two input latents to recombine
    latents = np.array([
        [0.8, 0.2, 0.5, 0.1],
        [0.1, 0.9, 0.3, 0.7],
    ])
    sample = handler(latents, seed=42, n_steps=10)
    assert sample.shape == (4,)
    assert np.all(sample >= 0.0)  # spike rates non-negative
