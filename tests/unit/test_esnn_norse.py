"""Unit tests for Norse fallback wrapper (cycle-3 C3.11).

These tests must run WITHOUT `torch` or `norse` installed — the
module falls back to a numpy LIF path when Norse is unavailable,
so the unit tests cover the fallback leg deterministically. A
true Norse execution path is exercised only when `norse` is
importable (gated by ``pragma: no cover`` on the env-dependent
branches in the module).

Reference : docs/specs/2026-04-17-dreamofkiki-framework-C-design.md
§6.2 (DR-3 Conformance Criterion), K1 compute budget tagging.
"""
from __future__ import annotations

import numpy as np
import pytest

from kiki_oniric.substrates.esnn_norse import (
    ESNN_NORSE_SUBSTRATE_NAME,
    ESNN_NORSE_SUBSTRATE_VERSION,
    NorseSNNSubstrate,
    norse_substrate_components,
)


def test_norse_substrate_instantiates_with_configurable_neuron_count() -> None:
    """TDD-1 — ctor accepts configurable n_neurons (default 128) ;
    module identity + DualVer C-v0.7.0+PARTIAL locked. Zero /
    negative sizes rejected so downstream shape invariants hold.
    """
    assert ESNN_NORSE_SUBSTRATE_NAME == "esnn_norse"
    assert ESNN_NORSE_SUBSTRATE_VERSION == "C-v0.7.0+PARTIAL"

    default = NorseSNNSubstrate()
    assert default.n_neurons == 128

    tiny = NorseSNNSubstrate(n_neurons=8, seed=1)
    assert tiny.n_neurons == 8

    with pytest.raises(ValueError, match="n_neurons"):
        NorseSNNSubstrate(n_neurons=0)


def test_norse_forward_deterministic_under_seed() -> None:
    """TDD-2 — same seed + same input = same spike-rate output
    (R1 contract). Runs in the numpy-LIF fallback path.
    Also exercises input-validation (non-1D ; size mismatch)
    and the scalar-broadcast convenience branch.
    """
    sub_a = NorseSNNSubstrate(n_neurons=16, seed=42)
    sub_b = NorseSNNSubstrate(n_neurons=16, seed=42)
    spike_trains = np.full(16, 0.6, dtype=float)

    out_a = sub_a.forward(spike_trains, n_steps=5)
    out_b = sub_b.forward(spike_trains, n_steps=5)
    np.testing.assert_array_equal(out_a, out_b)
    assert out_a.shape == (16,)

    # Input validation : rejects 2-D + mismatched size
    with pytest.raises(ValueError, match="1-D"):
        sub_a.forward(np.zeros((16, 2), dtype=float))
    with pytest.raises(ValueError, match="n_neurons"):
        sub_a.forward(np.zeros(7, dtype=float))

    # Scalar-input broadcast branch
    scalar_sub = NorseSNNSubstrate(n_neurons=8, seed=0)
    broadcast_out = scalar_sub.forward(np.array([0.7]), n_steps=4)
    assert broadcast_out.shape == (8,)


def test_norse_substrate_protocol_contract_surface() -> None:
    """TDD-3 — 8-primitive Protocol surface satisfied.

    Mirrors ``esnn_thalamocortical`` 4 factories + shared guards /
    runtime / swap / profiles via the component registry. Each of
    the 4 ops is exercised : replay on empty + nominal records,
    downscale on invalid factors, restructure across all 3 ops +
    invalid op, recombine with determinism check + < 2 latents.
    """
    sub = NorseSNNSubstrate(n_neurons=8, seed=0)

    components = norse_substrate_components()
    core_keys = {
        "primitives",
        "replay", "downscale", "restructure", "recombine",
        "finite", "topology",
        "runtime", "swap",
        "p_min", "p_equ", "p_max",
    }
    assert core_keys <= set(components.keys())

    # replay — empty + malformed + nominal
    replay = sub.replay_handler_factory()
    assert replay([], n_steps=5).shape == (1,)
    assert replay([{"other": 1.0}], n_steps=5).shape == (1,)
    nominal = replay([{"input": [0.5, 0.6, 0.7, 0.4]}], n_steps=6)
    assert nominal.shape == (4,)

    # downscale — valid scale + invalid factor branches
    downscale = sub.downscale_handler_factory()
    np.testing.assert_allclose(
        downscale(np.array([1.0, 0.5]), factor=0.5),
        np.array([0.5, 0.25]),
    )
    with pytest.raises(ValueError, match="shrink_factor"):
        downscale(np.array([1.0]), factor=0.0)
    with pytest.raises(ValueError, match="shrink_factor"):
        downscale(np.array([1.0]), factor=1.5)

    # restructure — add / remove / reroute + invalid op
    restructure = sub.restructure_handler_factory()
    conn = np.zeros((3, 3), dtype=float)
    added = restructure(conn, "add", src=0, dst=1)
    assert added[0, 1] == 1.0
    assert not np.shares_memory(added, conn)
    removed = restructure(
        np.ones((3, 3), dtype=float), "remove", src=0, dst=2,
    )
    assert removed[0, 2] == 0.0
    rerouted = restructure(
        np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]], dtype=float),
        "reroute", src=0, dst=2,
    )
    np.testing.assert_array_equal(rerouted[0], [7, 8, 9])
    np.testing.assert_array_equal(rerouted[2], [1, 2, 3])
    with pytest.raises(ValueError, match="op must be"):
        restructure(conn, "bogus")

    # recombine — deterministic + < 2 latents branch
    recombine = sub.recombine_handler_factory()
    latents = np.array([[0.5, 0.1, 0.8], [0.2, 0.9, 0.3]])
    out_a = recombine(latents, seed=7, n_steps=8)
    out_b = recombine(latents, seed=7, n_steps=8)
    np.testing.assert_array_equal(out_a, out_b)
    assert out_a.shape == (3,)
    with pytest.raises(ValueError, match=">=2 latents"):
        recombine(latents[:1], seed=0)


def test_norse_forward_records_k1_flop_estimate() -> None:
    """TDD-4 — K1 compute-budget tagging : every forward records
    estimated FLOPs. Simple LIF ≈ 4 FLOPs / neuron / step
    (decay, add, compare, reset). Tag is refreshed per call
    (not accumulated) so callers read the current-call budget.
    """
    sub = NorseSNNSubstrate(n_neurons=16, seed=0)
    spike_trains = np.full(16, 0.3, dtype=float)

    assert sub.last_flops == 0  # before any forward
    sub.forward(spike_trains, n_steps=10)
    assert sub.last_flops == 4 * 16 * 10

    # Second forward replaces (not accumulates) the counter
    sub.forward(spike_trains, n_steps=5)
    assert sub.last_flops == 4 * 16 * 5


def test_norse_substrate_version_matches_dualver_partial() -> None:
    """TDD-5 — ``ESNN_NORSE_SUBSTRATE_VERSION`` constant must be
    exactly ``C-v0.7.0+PARTIAL`` (the cycle-3 Phase 1 DualVer
    bump). Aligned with E-SNN thalamocortical + MLX substrate
    versions. Also asserts the norse_available introspection
    flag is a bool (importability probe result).
    """
    assert ESNN_NORSE_SUBSTRATE_VERSION == "C-v0.7.0+PARTIAL"
    sub = NorseSNNSubstrate(n_neurons=4, seed=0)
    assert isinstance(sub.norse_available, bool)
