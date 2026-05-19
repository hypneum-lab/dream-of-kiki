"""OPLoRA restructure tests — micro-kiki substrate, cycle-3 phase 2.

Covers the OPLoRA projection wired into
:meth:`MicroKikiSubstrate.restructure_handler_factory` (arXiv
2510.13003, Du et al.). The paper's key invariants and the dream
runtime's DR-0 / DR-1 axioms are asserted in parallel :

- *Algebra* (paper §3.2) : ``P @ v = 0`` for ``v`` in the prior
  subspace ; ``P @ w = w`` for ``w`` in its orthogonal
  complement ; ``P`` is symmetric idempotent (``P @ P ≈ P``).
- *DR-0* (accountability) : every handler call bumps the
  ``restructure_state`` counters ; ``completed=True`` and
  ``operation='restructure'`` are recorded.
- *DR-1* (episodic stamp) : an ``episode_id`` carried on the
  adapter dict is propagated to ``state.last_episode_id`` and
  appended to ``state.episode_ids``.

These tests are numpy-only and run on any host (no MLX / torch
dep). Reference :
``docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`` §6.2
(DR-0, DR-1, DR-3).
"""
from __future__ import annotations

import numpy as np
from typing import Any
import pytest

from kiki_oniric.substrates.micro_kiki import (
    MicroKikiRestructureState,
    MicroKikiSubstrate,
    _oplora_projector,
)


# ---------------------------------------------------------------
# _oplora_projector — pure-function algebra
# ---------------------------------------------------------------


def test_projector_orthogonality() -> None:
    """Projector annihilates every column of every prior delta.

    Paper §3.2 : ``P = I - U U^T`` where ``U`` spans the column
    space of the stacked priors ; therefore ``P @ v = 0`` for
    any ``v`` that lies in that span — in particular, every
    column of every prior delta.
    """
    rng = np.random.default_rng(7)
    out_dim = 8
    priors = [
        rng.standard_normal((out_dim, 3)).astype(np.float32),
        rng.standard_normal((out_dim, 2)).astype(np.float32),
    ]
    P = _oplora_projector(priors)
    assert P.shape == (out_dim, out_dim)
    for prior in priors:
        projected = P @ prior
        np.testing.assert_allclose(
            projected, np.zeros_like(projected), atol=1e-5,
        )


def test_projector_preserves_orthogonal_complement() -> None:
    """Vectors orthogonal to the prior subspace pass through unchanged.

    Construct a small prior whose column space is exactly
    ``span(e_0, e_1)`` ; pick ``w = e_3`` which is orthogonal to
    that span ; assert ``P @ w ≈ w``.
    """
    out_dim = 6
    prior = np.zeros((out_dim, 2), dtype=np.float32)
    prior[0, 0] = 1.0  # column 0 == e_0
    prior[1, 1] = 1.0  # column 1 == e_1
    P = _oplora_projector([prior])

    w = np.zeros(out_dim, dtype=np.float32)
    w[3] = 1.0
    np.testing.assert_allclose(P @ w, w, atol=1e-6)

    # ``P`` is symmetric.
    np.testing.assert_allclose(P, P.T, atol=1e-6)
    # Idempotent : ``P @ P == P``.
    np.testing.assert_allclose(P @ P, P, atol=1e-5)


def test_projector_shape_mismatch_raises() -> None:
    """Different ``out_dim`` across priors is a caller bug."""
    p1 = np.zeros((4, 2), dtype=np.float32)
    p2 = np.zeros((5, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="out_dim"):
        _oplora_projector([p1, p2])


def test_projector_empty_priors_rejected() -> None:
    """Empty prior list → explicit error (caller handles the no-op).

    The handler path treats empty priors as a DR-0-credited no-op,
    but the pure helper is stricter : it must see at least one
    prior so the fallback semantics stay visible to callers.
    """
    with pytest.raises(ValueError, match="at least one"):
        _oplora_projector([])


def test_projector_rank_collapse_falls_back_to_identity() -> None:
    """All singular values below ``rank_thresh`` → ``P = I``.

    Paper §3.3 robustness study : if the priors degenerate to
    numerical noise the projector collapses to identity (i.e.
    we refuse to over-prune a meaningful new adapter based on
    noise).
    """
    out_dim = 4
    # Near-zero prior ; every singular value below the default
    # threshold of 1e-4.
    prior = np.full((out_dim, 2), 1e-8, dtype=np.float32)
    P = _oplora_projector([prior], rank_thresh=1e-4)
    np.testing.assert_allclose(
        P, np.eye(out_dim, dtype=np.float32), atol=1e-6,
    )


# ---------------------------------------------------------------
# restructure_handler_factory — DR-0 / DR-1 contract
# ---------------------------------------------------------------


def test_restructure_handler_callable() -> None:
    """Factory returns a callable (DR-3 condition 1)."""
    substrate = MicroKikiSubstrate()
    handler = substrate.restructure_handler_factory()
    assert callable(handler)
    # Factory also returns a fresh state container.
    assert isinstance(substrate.restructure_state, MicroKikiRestructureState)


def test_restructure_consumes_episode_projects_new_B() -> None:
    """End-to-end : 2 priors + 1 new B-matrix ⇒ projected B is
    orthogonal to every prior range, adapter dict mutated in
    place, DR-0 counters bumped.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.restructure_handler_factory()

    rng = np.random.default_rng(42)
    out_dim = 8
    rank = 2
    priors = [
        rng.standard_normal((out_dim, 3)).astype(np.float32),
        rng.standard_normal((out_dim, 4)).astype(np.float32),
    ]
    # Intentionally pick a new B-matrix NOT orthogonal to priors —
    # the projection must move it.
    new_B_original = rng.standard_normal((out_dim, rank)).astype(np.float32)

    adapter: dict[str, Any] = {
        "layers.5.q_proj_B": new_B_original.copy(),
        "prior_deltas": priors,
        "episode_id": "ep-restruct-42",
    }
    out = handler(adapter, "oplora", "layers.5.q_proj_B")

    projected_B = out["layers.5.q_proj_B"]
    # Shape + dtype preserved.
    assert projected_B.shape == new_B_original.shape
    assert projected_B.dtype == new_B_original.dtype

    # Projected-B columns orthogonal to every prior-delta column.
    for prior in priors:
        cross = prior.T @ projected_B
        np.testing.assert_allclose(
            cross, np.zeros_like(cross), atol=1e-4,
        )

    # The projection was non-trivial (new_B wasn't already in the
    # orthogonal complement). Sanity check the test setup.
    assert not np.allclose(projected_B, new_B_original, atol=1e-3)

    # DR-0 bookkeeping — completed flag, operation label, counters.
    state = substrate.restructure_state
    assert state.total_episodes_handled == 1
    assert state.total_projections_applied == 1
    assert state.last_completed is True
    assert state.last_operation == "restructure"


def test_restructure_dr1_episode_id_stamping() -> None:
    """DR-1 — every episode_id carried on the adapter dict lands
    on the state in order ; unstamped calls skip the append but
    still bump the counter.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.restructure_handler_factory()

    rng = np.random.default_rng(1)
    out_dim = 4
    prior = rng.standard_normal((out_dim, 2)).astype(np.float32)
    B0 = rng.standard_normal((out_dim, 2)).astype(np.float32)

    for eid in ("e0", "e1", "e2"):
        adapter: dict[str, Any] = {
            "B": B0.copy(),
            "prior_deltas": [prior],
            "episode_id": eid,
        }
        handler(adapter, "oplora", "B")

    # Unstamped call — state.last_episode_id sticks at "e2",
    # episode_ids list not appended, counter bumped.
    handler(
        {"B": B0.copy(), "prior_deltas": [prior]},
        "oplora",
        "B",
    )

    state = substrate.restructure_state
    assert state.episode_ids == ["e0", "e1", "e2"]
    assert state.last_episode_id == "e2"
    assert state.total_episodes_handled == 4
    assert state.total_projections_applied == 4


def test_restructure_empty_priors_is_dr0_noop() -> None:
    """Empty ``prior_deltas`` — handler is a DR-0-credited no-op.

    The adapter dict is returned unchanged (no projection applied)
    but the episode is still recorded : DR-0 credits every
    handler invocation, even those with no net effect on the
    adapter tensors. ``total_projections_applied`` stays at 0 so
    callers can audit the no-op leg.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.restructure_handler_factory()

    B = np.ones((3, 2), dtype=np.float32)
    adapter: dict[str, Any] = {
        "B": B.copy(),
        "prior_deltas": [],
        "episode_id": "ep-noop",
    }
    out = handler(adapter, "oplora", "B")
    np.testing.assert_array_equal(out["B"], B)

    state = substrate.restructure_state
    assert state.total_episodes_handled == 1
    assert state.total_projections_applied == 0
    assert state.last_episode_id == "ep-noop"
    assert state.last_completed is True


def test_restructure_rejects_wrong_op() -> None:
    """DR-3 condition 1 : unknown op-names fail loud.

    Silent fallbacks would mask dispatcher bugs in the
    conformance harness ; the handler only honours the OPLoRA
    op vocabulary.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.restructure_handler_factory()
    adapter = {"B": np.zeros((2, 2), dtype=np.float32), "prior_deltas": []}
    with pytest.raises(ValueError, match="unsupported op"):
        handler(adapter, "merge", "B")


def test_restructure_rejects_missing_key() -> None:
    """Missing B-matrix key → explicit KeyError.

    Pairs with the op-vocabulary guard above : caller bugs surface
    at the handler boundary, not as a downstream shape mismatch.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.restructure_handler_factory()
    adapter: dict[str, Any] = {"prior_deltas": []}
    with pytest.raises(KeyError, match="missing entry"):
        handler(adapter, "oplora", "does_not_exist")


def test_restructure_rejects_projector_shape_mismatch() -> None:
    """Prior out_dim ≠ new-B out_dim is a caller bug ; raise."""
    substrate = MicroKikiSubstrate()
    handler = substrate.restructure_handler_factory()

    prior = np.ones((5, 2), dtype=np.float32)  # out_dim = 5
    new_B = np.ones((4, 2), dtype=np.float32)  # out_dim = 4
    adapter: dict[str, Any] = {
        "B": new_B,
        "prior_deltas": [prior],
    }
    with pytest.raises(ValueError, match="out_dim"):
        handler(adapter, "oplora", "B")


def test_projector_full_rank_saturation_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """When priors span the full output space the projector
    collapses to zero ; the helper must emit a warning so the
    silent zeroing of new adapters is auditable.

    This is the operational gotcha of long sequential curricula
    — each new expert adds rank to the prior subspace and after
    enough experts the union saturates ``out_dim``. OPLoRA does
    not claim to handle this case (the math is correct, ``P = 0``
    *is* the orthogonal projector onto the empty complement) but
    the missing warning would otherwise let the regression slip
    by undetected.
    """
    out_dim = 4
    # ``out_dim`` independent unit vectors → range == R^out_dim,
    # complement is empty, projector is the zero matrix.
    prior = np.eye(out_dim, dtype=np.float32)
    with caplog.at_level("WARNING", logger="kiki_oniric.substrates.micro_kiki"):
        P = _oplora_projector([prior])
    np.testing.assert_allclose(
        P, np.zeros((out_dim, out_dim), dtype=np.float32), atol=1e-6,
    )
    assert any(
        "projector is effectively zero" in rec.message
        for rec in caplog.records
    ), f"Expected saturation warning ; got {[r.message for r in caplog.records]}"
