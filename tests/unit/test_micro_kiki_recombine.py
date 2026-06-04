"""TIES-Merge recombine tests — micro-kiki substrate, cycle-3 phase 2.

Covers the TIES-Merge algorithm (Yadav et al., arXiv 2306.01708)
wired into :meth:`MicroKikiSubstrate.recombine_handler_factory`.
The paper's three-step procedure and the dream runtime's DR-0 /
DR-1 axioms are asserted in parallel :

- *Algebra* (paper §3) : trim → elect-sign → disjoint-merge on a
  list of per-task deltas. Single-delta fast path returns
  ``alpha * delta``. Majority-sign consensus drives the merged
  contribution ; parameters with no consensus remain at 0.
- *DR-0* (accountability) : every handler call bumps the
  ``recombine_state`` counters ; ``completed=True`` and
  ``operation='recombine'`` are recorded.
- *DR-1* (episodic stamp) : an ``episode_id`` carried on the
  payload dict is propagated to ``state.last_episode_id`` and
  appended to ``state.episode_ids``.

Numpy-only ; runs on any host (no MLX / torch dep). Reference :
``docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`` §6.2
(DR-0, DR-1, DR-3).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from kiki_oniric.substrates.micro_kiki import (
    MicroKikiRecombineState,
    MicroKikiSubstrate,
    _ties_merge,
)


# ---------------------------------------------------------------
# _ties_merge — pure-function algebra
# ---------------------------------------------------------------


def test_ties_merge_single_delta_is_alpha_scaled() -> None:
    """Single-task fast path : returns ``alpha * delta`` verbatim.

    No election / trim step when only one task contributes —
    there is no interference to resolve.
    """
    delta = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    merged = _ties_merge([delta], trim_fraction=0.2, alpha=1.0)
    np.testing.assert_allclose(merged, delta)
    # Alpha scaling.
    merged_scaled = _ties_merge([delta], trim_fraction=0.2, alpha=2.5)
    np.testing.assert_allclose(merged_scaled, 2.5 * delta, rtol=1e-6)
    # Dtype preserved from first input.
    assert merged.dtype == delta.dtype


def test_ties_merge_two_opposing_deltas_sign_consensus() -> None:
    """Two deltas disagreeing per-element : elected sign wins by
    magnitude ; disjoint merge keeps only the majority.

    With deltas ``d1 = [5, -1]`` and ``d2 = [-2, 3]`` :
    - position 0 : signed sum = 5 + (-2) = 3 > 0 → elect ``+``;
      only ``d1[0] = 5`` agrees → merged = 5.
    - position 1 : signed sum = -1 + 3 = 2 > 0 → elect ``+``;
      only ``d2[1] = 3`` agrees → merged = 3.

    Use ``trim_fraction=1.0`` so no entries are trimmed (purely
    tests the sign-election + disjoint-merge legs).
    """
    d1 = np.array([5.0, -1.0], dtype=np.float64)
    d2 = np.array([-2.0, 3.0], dtype=np.float64)
    merged = _ties_merge([d1, d2], trim_fraction=1.0, alpha=1.0)
    np.testing.assert_allclose(merged, [5.0, 3.0])


def test_ties_merge_sign_election_three_deltas() -> None:
    """Three deltas with signs [+, +, -] at every position : elect
    ``+`` ; merged = mean of the two ``+`` deltas.

    Paper §3 : election is by signed-magnitude sum, and the
    disjoint merge averages only over agreeing tasks.
    """
    d1 = np.array([1.0, 2.0], dtype=np.float64)
    d2 = np.array([3.0, 4.0], dtype=np.float64)
    d3 = np.array([-10.0, -0.5], dtype=np.float64)
    merged = _ties_merge([d1, d2, d3], trim_fraction=1.0, alpha=1.0)
    # Pos 0 : sum = 1+3-10 = -6 → elect ``-`` ; only d3 agrees → merged = -10.
    # Pos 1 : sum = 2+4-0.5 = 5.5 → elect ``+`` ; d1, d2 agree → mean = 3.
    np.testing.assert_allclose(merged, [-10.0, 3.0])


def test_ties_merge_known_input_regression() -> None:
    """Regression test against a hand-computed small example.

    Deltas (no trim — trim_fraction=1.0) :
        d1 = [ 2, -3,  0]
        d2 = [ 4,  1, -2]
        d3 = [-1,  2,  5]

    Signed sum : [5, 0, 3]
    Elected  :   [+, 0, +]
        pos 0 : d1=+, d2=+, d3=- → agree={d1,d2} → mean(2,4)=3
        pos 1 : elected=0 → merged=0
        pos 2 : d1=0(no sign), d2=-, d3=+ → agree={d3} (0 has
                sign 0, excluded since elected=+) → mean=5
    """
    d1 = np.array([2.0, -3.0, 0.0], dtype=np.float64)
    d2 = np.array([4.0, 1.0, -2.0], dtype=np.float64)
    d3 = np.array([-1.0, 2.0, 5.0], dtype=np.float64)
    merged = _ties_merge([d1, d2, d3], trim_fraction=1.0, alpha=1.0)
    np.testing.assert_allclose(merged, [3.0, 0.0, 5.0])


def test_ties_merge_trim_step_halves_entries() -> None:
    """``trim_fraction=0.5`` keeps the top 50 % magnitudes per
    task, zeroing the rest. Verified via a prepared delta where
    the half to keep is unambiguous.

    d1 = [10, 1, 10, 1]  →  trim keeps [10, _, 10, _]
    d2 = [10, 1, 10, 1]  →  trim keeps [10, _, 10, _]
    Merged (alpha=1.0)  =  [10, 0, 10, 0]
    """
    d1 = np.array([10.0, 1.0, 10.0, 1.0], dtype=np.float64)
    d2 = np.array([10.0, 1.0, 10.0, 1.0], dtype=np.float64)
    merged = _ties_merge([d1, d2], trim_fraction=0.5, alpha=1.0)
    np.testing.assert_allclose(merged, [10.0, 0.0, 10.0, 0.0])


def test_ties_merge_shape_mismatch_raises() -> None:
    """Shape mismatch across deltas is a caller bug."""
    d1 = np.zeros((4,), dtype=np.float32)
    d2 = np.zeros((5,), dtype=np.float32)
    with pytest.raises(ValueError, match="shape"):
        _ties_merge([d1, d2])


def test_ties_merge_empty_list_raises() -> None:
    """Empty delta list → explicit error (caller handles no-op)."""
    with pytest.raises(ValueError, match="at least one"):
        _ties_merge([])


def test_ties_merge_invalid_trim_fraction_raises() -> None:
    """``trim_fraction`` outside ``(0, 1]`` is rejected."""
    d = np.ones((3,), dtype=np.float32)
    with pytest.raises(ValueError, match="trim_fraction"):
        _ties_merge([d, d], trim_fraction=0.0)
    with pytest.raises(ValueError, match="trim_fraction"):
        _ties_merge([d, d], trim_fraction=1.5)


# ---------------------------------------------------------------
# recombine_handler_factory — DR-0 / DR-1 contract
# ---------------------------------------------------------------


def test_recombine_handler_callable_and_state() -> None:
    """Factory returns a callable ; the substrate exposes a fresh
    ``MicroKikiRecombineState`` via :attr:`recombine_state`.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.recombine_handler_factory()
    assert callable(handler)
    assert isinstance(substrate.recombine_state, MicroKikiRecombineState)
    # Fresh state : zero episodes handled.
    assert substrate.recombine_state.total_episodes_handled == 0
    assert substrate.recombine_state.total_merges_applied == 0
    assert substrate.recombine_state.last_completed is False


def test_recombine_handler_consumes_episode_and_stamps_dr1() -> None:
    """End-to-end : multi-delta payload + episode_id ⇒ merged
    tensor + DR-0 counter bump + DR-1 episode_id stamp + shape
    stamp preserved on state.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.recombine_handler_factory()

    rng = np.random.default_rng(7)
    shape = (4, 3)
    deltas = [
        rng.standard_normal(shape).astype(np.float32),
        rng.standard_normal(shape).astype(np.float32),
        rng.standard_normal(shape).astype(np.float32),
    ]
    payload = {"deltas": deltas, "episode_id": "ep-recomb-7"}
    merged = handler(payload, "ties")

    # Output contract.
    assert merged.shape == shape
    assert merged.dtype == np.float32

    # DR-0 state.
    state = substrate.recombine_state
    assert state.total_episodes_handled == 1
    assert state.total_merges_applied == 1
    assert state.last_completed is True
    assert state.last_operation == "recombine"
    assert state.last_k_deltas == 3
    assert state.last_input_shape == shape
    assert state.last_output_shape == shape

    # DR-1 stamp.
    assert state.last_episode_id == "ep-recomb-7"
    assert state.episode_ids == ["ep-recomb-7"]

    # Multi-call accumulates episode_ids.
    payload2 = {"deltas": deltas, "episode_id": "ep-recomb-8"}
    handler(payload2, "ties_merge")
    payload3 = {"deltas": deltas}  # no episode_id
    handler(payload3, "merge")
    assert state.total_episodes_handled == 3
    assert state.episode_ids == ["ep-recomb-7", "ep-recomb-8"]
    assert state.last_episode_id == "ep-recomb-8"


def test_recombine_handler_rejects_unknown_op() -> None:
    """DR-3 condition 1 : unknown op-names fail loud — silent
    fallbacks would mask dispatcher bugs in the conformance
    harness.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.recombine_handler_factory()
    d = np.ones((2, 2), dtype=np.float32)
    with pytest.raises(ValueError, match="unsupported op"):
        handler({"deltas": [d, d]}, "interpolate")


def test_recombine_handler_rejects_missing_deltas_key() -> None:
    """Payload without ``deltas`` key → explicit KeyError."""
    substrate = MicroKikiSubstrate()
    handler = substrate.recombine_handler_factory()
    with pytest.raises(KeyError, match="deltas"):
        handler({"other": [1, 2, 3]}, "ties")


def test_recombine_handler_propagates_empty_deltas_error() -> None:
    """Empty ``deltas`` list → handler surfaces the underlying
    ``_ties_merge`` ValueError rather than silently returning
    zeros.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.recombine_handler_factory()
    with pytest.raises(ValueError, match="at least one"):
        handler({"deltas": []}, "ties")


def test_recombine_snapshot_roundtrip_with_accumulator(
    tmp_path: Path,
) -> None:
    """Snapshot / load_snapshot round-trip survives recombine state.

    The ``.npz`` accumulator is separate from the recombine state
    dataclass, so the round-trip only needs to preserve
    ``_current_delta``. Run a merge, seed the accumulator with
    the merged delta, snapshot, reload into a fresh substrate,
    assert the accumulator matches bit-for-bit.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.recombine_handler_factory()

    rng = np.random.default_rng(42)
    shape = (3, 2)
    deltas = [
        rng.standard_normal(shape).astype(np.float32),
        rng.standard_normal(shape).astype(np.float32),
    ]
    merged = handler({"deltas": deltas, "episode_id": "ep-snap"}, "ties")
    substrate._current_delta = {"layers.0.recombined": merged}

    snap = substrate.snapshot(tmp_path / "recomb-delta")
    assert snap.exists()

    fresh = MicroKikiSubstrate()
    fresh.load_snapshot(snap)
    assert set(fresh._current_delta.keys()) == {"layers.0.recombined"}
    np.testing.assert_array_equal(
        fresh._current_delta["layers.0.recombined"], merged,
    )
    # The new substrate has its own fresh recombine_state.
    assert fresh.recombine_state.total_episodes_handled == 0
