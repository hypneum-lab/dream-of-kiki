"""DR-3 Conformance Criterion — micro-kiki LoRA substrate (C2.5).

Validates that the micro-kiki transformer-LoRA substrate satisfies
the 3 conditions of DR-3 Conformance Criterion :

1. Signature typing : substrate exports the expected Protocol-
   compatible factories (callable + correct signatures).
2. Axiom property tests : DR-0 accountability via restructure
   bookkeeping ; DR-1 episode_id propagation ; OPLoRA orthogonality
   property of the wired ``restructure`` op.
3. BLOCKING invariants enforceable : S2 finite + S3 topology guards
   operate on micro-kiki state representations (numpy LoRA deltas).

The phase-3 ``recombine`` stub is exercised at the negative path
(asserts the ``NotImplementedError`` surface stays honest) so the
matrix entry stays visible in the cycle-3 conformance report.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
"""
from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from kiki_oniric.dream.guards.finite import check_finite
from kiki_oniric.dream.guards.topology import validate_topology
from kiki_oniric.substrates.micro_kiki import (
    MICRO_KIKI_SUBSTRATE_NAME,
    MICRO_KIKI_SUBSTRATE_VERSION,
    MicroKikiRestructureState,
    MicroKikiSubstrate,
    micro_kiki_substrate_components,
)


# ===== Condition 1: signature typing =====


def test_c1_micro_kiki_exports_required_identity_constants() -> None:
    """Substrate module exposes the canonical identity constants
    expected by the substrate-agnostic harness.
    """
    assert MICRO_KIKI_SUBSTRATE_NAME == "micro_kiki"
    assert MICRO_KIKI_SUBSTRATE_VERSION.startswith("C-v")
    assert MICRO_KIKI_SUBSTRATE_VERSION.endswith("+PARTIAL")


def test_c1_micro_kiki_substrate_instantiable_in_stub_mode() -> None:
    """Instantiable on any host (CI Linux, Apple Silicon Mac).

    No ``base_model_path`` → stub mode ; no MLX wheel required.
    The conformance harness must be host-portable.
    """
    substrate = MicroKikiSubstrate()
    assert substrate.base_model_path is None
    assert substrate.restructure_state is not None
    assert isinstance(substrate.restructure_state, MicroKikiRestructureState)


def test_c1_micro_kiki_provides_4_op_factory_methods() -> None:
    """All 4 op factories return callable handlers (DR-3 cond 1)."""
    substrate = MicroKikiSubstrate()
    assert callable(substrate.replay_handler_factory())
    assert callable(substrate.downscale_handler_factory())
    assert callable(substrate.restructure_handler_factory())
    assert callable(substrate.recombine_handler_factory())


def test_c1_micro_kiki_components_registry_mirrors_siblings() -> None:
    """Components map shares the canonical core keys with the
    sibling substrates so the harness can parametrize uniformly.
    """
    micro = micro_kiki_substrate_components()
    core_keys = {
        "primitives",
        "replay", "downscale", "restructure", "recombine",
        "finite", "topology",
        "runtime", "swap",
    }
    assert core_keys <= set(micro.keys()), (
        f"micro-kiki components missing core keys: "
        f"{core_keys - set(micro.keys())}"
    )


# ===== Condition 2: axiom property tests =====


def test_c2_micro_kiki_restructure_oplora_orthogonality() -> None:
    """OPLoRA contract (paper §3.2) : the restructure op produces
    a new B-matrix orthogonal to every prior delta column.

    This is the substrate-side translation of DR-2 (compositionality
    on the canonical order) : restructure produces an output that
    safely composes with subsequent replay because the new adapter
    no longer overlaps the prior subspace.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.restructure_handler_factory()

    rng = np.random.default_rng(7)
    out_dim, rank = 8, 2
    priors = [
        rng.standard_normal((out_dim, 3)).astype(np.float32),
        rng.standard_normal((out_dim, 4)).astype(np.float32),
    ]
    new_B = rng.standard_normal((out_dim, rank)).astype(np.float32)

    adapter: dict = {
        "B": new_B.copy(),
        "prior_deltas": priors,
    }
    out = handler(adapter, "oplora", "B")
    projected_B = out["B"]

    for prior in priors:
        cross = prior.T @ projected_B
        np.testing.assert_allclose(
            cross, np.zeros_like(cross), atol=1e-4,
        )


def test_c2_micro_kiki_restructure_dr0_accountability() -> None:
    """DR-0 : every restructure execution increments the substrate's
    accountability counters and stamps ``completed=True``.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.restructure_handler_factory()
    state_before = substrate.restructure_state
    n_before = state_before.total_episodes_handled

    out_dim = 4
    prior = np.eye(out_dim, dtype=np.float32)[:, :2]
    adapter: dict = {
        "B": np.ones((out_dim, 2), dtype=np.float32),
        "prior_deltas": [prior],
    }
    handler(adapter, "oplora", "B")

    state_after = substrate.restructure_state
    assert state_after.total_episodes_handled == n_before + 1
    assert state_after.last_completed is True
    assert state_after.last_operation == "restructure"


def test_c2_micro_kiki_restructure_dr1_episode_stamp() -> None:
    """DR-1 : an ``episode_id`` carried on the adapter is propagated
    into the substrate's episodic state so the episode is
    reconstructable downstream.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.restructure_handler_factory()

    out_dim = 4
    prior = np.eye(out_dim, dtype=np.float32)[:, :2]
    adapter: dict = {
        "B": np.ones((out_dim, 2), dtype=np.float32),
        "prior_deltas": [prior],
        "episode_id": "ep-conformance-dr1",
    }
    handler(adapter, "oplora", "B")
    state = substrate.restructure_state
    assert "ep-conformance-dr1" in state.episode_ids
    assert state.last_episode_id == "ep-conformance-dr1"


def test_c2_micro_kiki_recombine_ties_merge_contract() -> None:
    """TIES-Merge contract (Yadav et al., arXiv 2306.01708, §3) :
    the recombine handler merges a list of per-task delta tensors
    via trim → elect-sign → disjoint-mean and returns a tensor of
    the input shape and dtype.

    This is the positive contract test that supersedes the
    earlier ``recombine`` stub-surface test once the TIES-merge
    backend landed (commit f27f745, ``feat(micro_kiki): TIES
    recombine handler``), mirroring the OPLoRA migration path
    used for ``restructure``.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.recombine_handler_factory()

    rng = np.random.default_rng(11)
    shape = (4, 3)
    deltas = [
        rng.standard_normal(shape).astype(np.float32),
        rng.standard_normal(shape).astype(np.float32),
        rng.standard_normal(shape).astype(np.float32),
    ]
    payload: dict[str, Any] = {"deltas": deltas}
    merged = handler(payload, "ties")

    assert merged.shape == shape
    assert merged.dtype == np.float32
    np.testing.assert_array_equal(np.isfinite(merged), True)
    # Single-element fast path : alpha * delta (alpha defaults to 1).
    single_payload: dict[str, Any] = {"deltas": [deltas[0]]}
    single_out = handler(single_payload, "ties")
    np.testing.assert_allclose(single_out, deltas[0], atol=1e-6)


def test_c2_micro_kiki_recombine_rejects_unknown_op() -> None:
    """DR-3 condition (1) : unsupported ops surface ``ValueError``
    rather than silently no-op'ing. Mirrors the strict op-check on
    :meth:`restructure_handler_factory`.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.recombine_handler_factory()
    payload: dict[str, Any] = {
        "deltas": [np.zeros((2, 4), dtype=np.float32)],
    }
    with pytest.raises(ValueError, match="unsupported"):
        handler(payload, "not-a-merge-op")


# ===== Condition 3: BLOCKING invariants enforceable =====


def test_c3_s2_finite_guard_works_on_micro_kiki_lora_delta() -> None:
    """S2 invariant accepts valid LoRA delta tensors and rejects
    NaN / Inf in the delta representation.
    """
    delta = np.zeros((8, 4), dtype=np.float32)
    check_finite(delta)

    bad = delta.copy()
    bad[2, 1] = float("nan")
    from kiki_oniric.dream.guards.finite import FiniteGuardError
    with pytest.raises(FiniteGuardError):
        check_finite(bad)


def test_c3_s3_topology_guard_works_on_micro_kiki_topology() -> None:
    """S3 topology validator (substrate-agnostic) accepts the
    canonical ortho chain and rejects a self-loop. Confirms the
    micro-kiki substrate participates in the same topology contract
    as the sibling substrates.
    """
    canonical = {
        "rho_phono": ["rho_lex"],
        "rho_lex": ["rho_syntax"],
        "rho_syntax": ["rho_sem"],
        "rho_sem": [],
    }
    validate_topology(canonical)

    bad = {
        "rho_phono": ["rho_phono"],
        "rho_lex": ["rho_syntax"],
        "rho_syntax": ["rho_sem"],
        "rho_sem": [],
    }
    from kiki_oniric.dream.guards.topology import TopologyGuardError
    with pytest.raises(TopologyGuardError):
        validate_topology(bad)
