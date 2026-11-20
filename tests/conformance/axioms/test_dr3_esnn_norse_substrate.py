"""DR-3 Conformance Criterion — E-SNN Norse substrate (C3.11).

Validates that the Norse SNN substrate (numpy-fallback path) satisfies
the 3 conditions of DR-3 Conformance Criterion:

1. Signature typing: substrate exports the expected Protocol-compatible
   factories (callable + correct signatures), identity constants, and
   component registry.
2. Axiom property tests: DR-0 (observable output) and DR-1/DR-2
   (determinism, op semantics) on the numpy-fallback path that runs
   without ``torch``/``norse``.
3. BLOCKING invariants enforceable: S2 finite guard and S3 topology
   guard operate on Norse substrate state representations.

This file covers the **numpy fallback path only** (no ``torch``/``norse``
required). The True branch exercised when Norse is installed is tagged
``pragma: no cover`` in the substrate module itself.

On Linux CI, this file is excluded by the ``test_dr3_*.py`` glob in
root ``conftest.py`` (same treatment as other DR-3 suite files). That
is expected; note it here and in the PR.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
(DR-3 Conformance Criterion, folder rule).
"""
from __future__ import annotations

import numpy as np
import pytest

from kiki_oniric.dream.guards.finite import check_finite
from kiki_oniric.dream.guards.topology import validate_topology
from kiki_oniric.substrates.esnn_norse import (
    ESNN_NORSE_SUBSTRATE_NAME,
    NorseSNNSubstrate,
    norse_substrate_components,
)


# ===== Condition 1: signature typing =====


def test_c1_esnn_norse_exports_identity_constant() -> None:
    """Norse substrate module exports required name constant."""
    assert ESNN_NORSE_SUBSTRATE_NAME == "esnn_norse"


def test_c1_esnn_norse_substrate_instantiable() -> None:
    """NorseSNNSubstrate instantiates with defaults and custom size."""
    default = NorseSNNSubstrate()
    assert default.n_neurons == 128

    small = NorseSNNSubstrate(n_neurons=8, seed=42)
    assert small.n_neurons == 8


def test_c1_esnn_norse_provides_4_op_factory_methods() -> None:
    """All four op factories return callables."""
    substrate = NorseSNNSubstrate(n_neurons=8, seed=0)
    assert callable(substrate.replay_handler_factory())
    assert callable(substrate.downscale_handler_factory())
    assert callable(substrate.restructure_handler_factory())
    assert callable(substrate.recombine_handler_factory())


def test_c1_esnn_norse_components_registry_has_core_keys() -> None:
    """norse_substrate_components() shares the core keys with MLX."""
    from kiki_oniric.substrates.mlx_kiki_oniric import (
        mlx_substrate_components,
    )

    norse = norse_substrate_components()
    mlx = mlx_substrate_components()
    core_keys = {
        "primitives",
        "replay", "downscale", "restructure", "recombine",
        "finite", "topology",
        "runtime", "swap",
        "p_min", "p_equ", "p_max",
    }
    assert core_keys <= set(norse.keys())
    assert core_keys <= set(mlx.keys())


# ===== Condition 2: axiom property tests =====


def test_c2_esnn_norse_replay_dr0_produces_observable_output() -> None:
    """DR-0: every execution produces an observable output (spike rates).

    replay with non-empty records returns a non-None numpy array.
    Empty records return a zero-filled array. This is the Norse
    analogue of DR-0 'every DE produces a log entry'.
    """
    substrate = NorseSNNSubstrate(n_neurons=4, seed=7)
    replay = substrate.replay_handler_factory()

    empty_out = replay([], n_steps=5)
    assert isinstance(empty_out, np.ndarray)

    records = [{"input": [0.5, 0.7, 0.3, 0.2]}]
    non_empty_out = replay(records, n_steps=10)
    assert isinstance(non_empty_out, np.ndarray)
    assert non_empty_out.shape == (4,)


def test_c2_esnn_norse_downscale_non_idempotent() -> None:
    """DR-2 op-pair: downscale is commutative but NOT idempotent.

    shrink_f(shrink_f(w)) = w * factor^2 != w * factor.
    """
    substrate = NorseSNNSubstrate(n_neurons=4, seed=7)
    downscale = substrate.downscale_handler_factory()

    weights = np.array([0.8, 1.2, 0.5, 0.9])
    factor = 0.5
    once = downscale(weights, factor=factor)
    twice = downscale(once, factor=factor)

    np.testing.assert_allclose(once, weights * factor)
    np.testing.assert_allclose(twice, weights * (factor ** 2))
    assert not np.allclose(twice, once)


def test_c2_esnn_norse_recombine_deterministic_with_seed() -> None:
    """DR-2: same inputs + same seed → same output (R1 contract)."""
    substrate = NorseSNNSubstrate(n_neurons=4, seed=7)
    recombine = substrate.recombine_handler_factory()

    latents = np.array([[0.8, 0.2, 0.5], [0.1, 0.9, 0.3]])
    out_a = recombine(latents, seed=42, n_steps=10)
    out_b = recombine(latents, seed=42, n_steps=10)
    np.testing.assert_array_equal(out_a, out_b)


# ===== Condition 3: BLOCKING invariants enforceable =====


def test_c3_s2_finite_guard_accepts_valid_norse_weights() -> None:
    """S2: check_finite accepts a valid Norse spike-rate array."""
    substrate = NorseSNNSubstrate(n_neurons=8, seed=7)
    spike_trains = np.full(8, 0.5, dtype=float)
    output = substrate.forward(spike_trains, n_steps=5)
    # forward output must pass the finite guard
    check_finite(output)


def test_c3_s2_finite_guard_rejects_nan_in_norse_state() -> None:
    """S2: check_finite raises FiniteGuardError on NaN."""
    from kiki_oniric.dream.guards.finite import FiniteGuardError

    bad = np.array([0.1, float("nan"), 0.3])
    with pytest.raises(FiniteGuardError):
        check_finite(bad)


def test_c3_s2_finite_guard_rejects_inf_in_norse_state() -> None:
    """S2: check_finite raises FiniteGuardError on Inf."""
    from kiki_oniric.dream.guards.finite import FiniteGuardError

    bad = np.array([0.1, float("inf"), 0.3])
    with pytest.raises(FiniteGuardError):
        check_finite(bad)


def test_c3_s3_topology_guard_accepts_canonical_chain() -> None:
    """S3: validate_topology passes on the canonical ortho chain."""
    canonical = {
        "rho_phono": ["rho_lex"],
        "rho_lex": ["rho_syntax"],
        "rho_syntax": ["rho_sem"],
        "rho_sem": [],
    }
    validate_topology(canonical)  # must not raise


def test_c3_s3_topology_guard_rejects_self_loop() -> None:
    """S3: validate_topology rejects a self-loop."""
    from kiki_oniric.dream.guards.topology import TopologyGuardError

    bad = {
        "rho_phono": ["rho_phono"],  # self-loop
        "rho_lex": ["rho_syntax"],
        "rho_syntax": ["rho_sem"],
        "rho_sem": [],
    }
    with pytest.raises(TopologyGuardError):
        validate_topology(bad)
