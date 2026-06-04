"""Unit tests for the micro-kiki substrate (cycle-3 Phase 2 draft).

These tests must run WITHOUT ``mlx_lm`` / ``mlx`` installed — the
substrate falls back to a numpy path when the MLX wheel is not
importable, so the unit tests cover the fallback leg
deterministically. A true ``mlx_lm`` execution is env-gated to
Apple Silicon hosts (matches the ``esnn_norse`` pattern).

Reference : ``docs/specs/2026-04-17-dreamofkiki-framework-C-design.md``
§6.2 (DR-3 Conformance Criterion).
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pytest

from kiki_oniric.substrates.micro_kiki import (
    MICRO_KIKI_SUBSTRATE_NAME,
    MICRO_KIKI_SUBSTRATE_VERSION,
    MicroKikiSubstrate,
    micro_kiki_substrate_components,
)


def test_module_manifest_matches_substrate_pattern() -> None:
    """TDD-1 — module-level identity + manifest shape align with
    the sibling ``esnn_*`` substrates (DR-3 condition 1).
    """
    assert MICRO_KIKI_SUBSTRATE_NAME == "micro_kiki"
    assert MICRO_KIKI_SUBSTRATE_VERSION == "C-v0.9.1+PARTIAL"

    components = micro_kiki_substrate_components()
    expected = {
        "primitives",
        "replay", "downscale", "restructure", "recombine",
        "finite", "topology",
        "runtime", "swap",
        "p_min", "p_equ", "p_max",
    }
    assert set(components.keys()) == expected
    # All dotted paths live under kiki_oniric — no micro-kiki-side
    # module leaks into the upstream manifest.
    for value in components.values():
        assert value.startswith("kiki_oniric."), value


def test_replay_handler_is_callable_and_aggregates_inputs() -> None:
    """TDD-2 — replay factory returns a callable ; the callable
    aggregates beta-record ``input`` vectors (mean drive). Exercises
    the empty-records + malformed-records + nominal branches.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.replay_handler_factory()

    assert callable(handler)
    # Empty + malformed : return 1-element zero vector (stable shape
    # contract shared with esnn_norse for the DR-3 matrix).
    assert handler([], n_steps=5).shape == (1,)
    assert handler([{"other": 1.0}], n_steps=5).shape == (1,)

    # Nominal : mean of two 4-D inputs.
    records = [
        {"input": [1.0, 0.0, 0.0, 0.0]},
        {"input": [0.0, 1.0, 0.0, 0.0]},
    ]
    out = handler(records, n_steps=4)
    assert out.shape == (4,)
    np.testing.assert_allclose(out, [0.5, 0.5, 0.0, 0.0])
    # dtype contract : float32 aligns with LoRA adapter storage
    assert out.dtype == np.float32


def test_downscale_preserves_shape_and_rejects_invalid_factor() -> None:
    """TDD-3 — downscale returns ``weights * factor`` with the
    input dtype preserved (so the LoRA adapter round-trips without
    a float64 bloat). The factor must lie in (0, 1] — mirrors the
    esnn_thalamocortical / esnn_norse contract so the DR-3 test
    matrix can parametrize. Matrix-preservation is what episode-id
    stamping (DR-1) binds to downstream.
    """
    substrate = MicroKikiSubstrate(rank=8, num_layers=4, seed=0)
    handler = substrate.downscale_handler_factory()

    weights = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    shrunk = handler(weights, factor=0.99)
    # Shape + dtype preserved — caller can stamp episode_id on the
    # same tensor object without re-casting.
    assert shrunk.shape == weights.shape
    assert shrunk.dtype == weights.dtype
    np.testing.assert_allclose(shrunk, weights * 0.99, rtol=1e-6)

    # Out-of-range factors rejected so the S1 retained-benchmark
    # guard can trust the downscale arithmetic (a factor > 1 would
    # amplify, a factor <= 0 would zero / flip the adapter).
    with pytest.raises(ValueError, match="shrink_factor"):
        handler(weights, factor=0.0)
    with pytest.raises(ValueError, match="shrink_factor"):
        handler(weights, factor=1.5)


def test_restructure_handler_oplora_wired() -> None:
    """TDD-4 (phase-2) — restructure factory returns an OPLoRA-
    backed callable (arXiv 2510.13003). Empty ``prior_deltas``
    is a DR-0-credited no-op ; the full algebra is exercised in
    ``tests/unit/test_micro_kiki_restructure.py``.

    The phase-1 ``NotImplementedError`` gate is retired here in
    favour of an assertion that the op is *wired* ; the gate is
    now covered by the dedicated OPLoRA test module so the DR-3
    conformance harness sees the real surface, not a stub.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.restructure_handler_factory()
    assert callable(handler)

    adapter: dict[str, Any] = {
        "layers.0.q_proj_B": np.zeros((4, 2), dtype=np.float32),
        "prior_deltas": [],  # empty → no-op projection leg
        "episode_id": "ep-restruct-0",
    }
    out = handler(adapter, "oplora", "layers.0.q_proj_B")
    # No priors → adapter entry unchanged ; state still bumped.
    np.testing.assert_array_equal(
        out["layers.0.q_proj_B"], np.zeros((4, 2), dtype=np.float32),
    )
    state = substrate.restructure_state
    assert state.total_episodes_handled == 1
    assert state.total_projections_applied == 0
    assert state.last_completed is True
    assert state.last_operation == "restructure"
    assert state.last_episode_id == "ep-restruct-0"

    # Unsupported op still rejected — DR-3 visibility.
    with pytest.raises(ValueError, match="unsupported op"):
        handler(adapter, "activate", "layers.0.q_proj_B")


def test_recombine_handler_ties_wired() -> None:
    """TDD-5 (phase-2) — recombine factory returns a callable
    backed by TIES-Merge (arXiv 2306.01708, Yadav et al.). The
    phase-1 ``NotImplementedError`` gate is retired here ; the
    full algebra is exercised in
    ``tests/unit/test_micro_kiki_recombine.py``.

    This smoke test only asserts the factory is wired (not a
    stub) and that the handler produces a numpy array of the
    right shape for a trivial two-delta payload.
    """
    substrate = MicroKikiSubstrate()
    handler = substrate.recombine_handler_factory()
    assert callable(handler)

    d1 = np.array([1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    d2 = np.array([1.0, -2.0, 3.0, -4.0], dtype=np.float32)
    merged = handler({"deltas": [d1, d2], "episode_id": "ep-ties-0"}, "ties")
    assert merged.shape == d1.shape
    assert merged.dtype == d1.dtype

    state = substrate.recombine_state
    assert state.total_episodes_handled == 1
    assert state.last_completed is True
    assert state.last_operation == "recombine"
    assert state.last_episode_id == "ep-ties-0"

    # Unsupported op still rejected — DR-3 visibility.
    with pytest.raises(ValueError, match="unsupported op"):
        handler({"deltas": [d1, d2]}, "interpolate")


def test_stub_mode_without_mlx_lm(tmp_path: Path) -> None:
    """TDD-6 — the substrate builds + exposes its full Protocol
    surface without any ``mlx_lm`` / ``mlx`` wheel installed.

    This is the CI leg — dream-of-kiki ships an Apple-Silicon-only
    MLX dep ; the Linux runners exercise the fallback path. The
    test also confirms :meth:`load` is a safe no-op in stub mode
    and the snapshot / load_snapshot round-trip works over the
    numpy ``.npz`` accumulator.
    """
    substrate = MicroKikiSubstrate(base_model_path=None)
    # mlx_lm_available is whatever the module-level probe saw — we
    # don't assert a specific value, only that it's a bool so
    # callers can introspect without a try-import.
    assert isinstance(substrate.mlx_lm_available, bool)

    # Stub awake — deterministic, no model loaded.
    assert substrate.awake("hello") == "[stub awake] hello"

    # load() is a safe no-op when base_model_path is None.
    substrate.load()  # must not raise
    assert substrate._model is None
    assert substrate._tokenizer is None

    # snapshot / load_snapshot round-trip over an empty accumulator.
    snap_path = tmp_path / "micro-kiki-delta.npz"
    written = substrate.snapshot(snap_path)
    assert written.exists()
    assert written.suffix == ".npz"

    # Populate the accumulator, round-trip through disk.
    substrate._current_delta = {
        "layers.0.lora_A": np.ones((8, 4), dtype=np.float32),
        "layers.0.lora_B": np.zeros((4, 8), dtype=np.float32),
    }
    written = substrate.snapshot(tmp_path / "delta2")  # no .npz suffix
    assert written.exists()
    substrate._current_delta = {}
    substrate.load_snapshot(written)
    assert set(substrate._current_delta.keys()) == {
        "layers.0.lora_A", "layers.0.lora_B",
    }
    np.testing.assert_array_equal(
        substrate._current_delta["layers.0.lora_A"],
        np.ones((8, 4), dtype=np.float32),
    )


def test_invalid_config_rejected() -> None:
    """TDD-7 — guard ctor args : ``num_layers`` and ``rank`` must
    be positive. Zero / negative values are caught at construction
    so downstream shape / LoRA-rank invariants hold without extra
    checks in the handlers.
    """
    with pytest.raises(ValueError, match="num_layers"):
        MicroKikiSubstrate(num_layers=0)
    with pytest.raises(ValueError, match="rank"):
        MicroKikiSubstrate(rank=-1)
