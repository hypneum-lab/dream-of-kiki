"""Unit tests for cycle-3 C3.2 Qwen MLX multi-scale wrappers.

Covers :mod:`harness.real_models.qwen_mlx` with 5 TDD tests per the
cycle-3 plan :

1. forward determinism : 1.5B wrapper under a fixed seed returns
   bit-identical logits.
2. SHA-256 pin verification : wrapper refuses a mismatched weight
   digest against the registry.
3. typed Protocol matching : wrapper satisfies GammaSnapshotProtocol
   (γ-channel, DR-3 condition 1 ; cf. framework-C §2.1).
4. K1 compute-budget tagging : every forward call records a FLOP
   estimate on the returned trace.
5. multi-scale dispatch : ``load_qwen(scale)`` resolves to the right
   registry pin for all three scale slots.

All tests mock :func:`mlx_lm.load` with a tiny synthetic model so
the suite stays deterministic, network-free and <100 ms per the
cycle-3 discipline.
"""
from __future__ import annotations

import hashlib
from unittest.mock import patch
from typing import Any

import mlx.core as mx
import numpy as np
import pytest

from harness.real_models.base_model_registry import (
    REGISTRY,
    get_pin,
)
from harness.real_models.qwen_mlx import (
    QwenMLXWrapper,
    load_qwen,
)
from kiki_oniric.core.primitives import GammaSnapshotProtocol


# --------------------------------------------------------------------------
# Tiny synthetic MLX "model" that behaves like the mlx-lm return value
# (just enough for the wrapper to drive forward passes).
# --------------------------------------------------------------------------


class _TinyMLXModel:
    """Minimal stand-in for an mlx-lm model.

    Exposes a single linear projection parameterised by a seed so
    tests can pin the "weights" and their SHA digest deterministically.
    """

    def __init__(self, seed: int, vocab_size: int = 8, hidden: int = 4) -> None:
        rng = np.random.default_rng(seed)
        self.vocab_size = vocab_size
        self.hidden = hidden
        # Fixed weights (converted to MLX array for realism).
        self.w = mx.array(rng.standard_normal((hidden, vocab_size)).astype(np.float32))

    def weights_bytes(self) -> bytes:
        """Deterministic byte-representation of the weights for sha256."""
        arr = np.asarray(self.w)
        return arr.tobytes()

    def __call__(self, token_ids: mx.array) -> mx.array:
        # Embedding : treat token id as an index into rows of w.T.
        # This gives us a differentiable-looking forward without
        # needing mlx-lm internals.
        idx = np.asarray(token_ids).astype(np.int64) % self.hidden
        emb = self.w[mx.array(idx)]
        mx.eval(emb)
        return emb


class _TinyTokenizer:
    """Shim tokenizer with encode() returning a deterministic token sequence."""

    def encode(self, text: str) -> list[int]:
        # Hash into a small range so tests stay bounded.
        return [b % 8 for b in text.encode("utf-8")[:16]]


def _mock_mlx_load(_repo_id: str, *, seed: int = 0) -> tuple[_TinyMLXModel, _TinyTokenizer]:
    return _TinyMLXModel(seed=seed), _TinyTokenizer()


# --------------------------------------------------------------------------
# Test 1 — forward determinism on the 1.5B slot
# --------------------------------------------------------------------------


def test_forward_determinism_1p5b_fixed_seed() -> None:
    """Same (wrapper, seed) → bit-identical token logits.

    Verifies R1 at the wrapper level : reloading the 1.5B slot and
    running the same forward must hash to the same bytes.
    """
    pin = get_pin("qwen3p5-1p5b")
    tiny = _TinyMLXModel(seed=123)
    tok = _TinyTokenizer()
    with patch(
        "harness.real_models.qwen_mlx._load_checkpoint",
        return_value=(tiny, tok),
    ):
        w = QwenMLXWrapper(pin)
        tokens = mx.array([1, 2, 3, 4])
        a = w.forward(tokens, seed=7)
        b = w.forward(tokens, seed=7)
    assert np.asarray(a.logits).tobytes() == np.asarray(b.logits).tobytes()


# --------------------------------------------------------------------------
# Test 2 — SHA-256 pin verification
# --------------------------------------------------------------------------


def test_sha256_pin_verification_rejects_mismatch() -> None:
    """Wrapper computes the weight digest at construction and
    refuses when the registry pin disagrees.

    Implements the R1 check for the model substrate : a silent
    weight reshuffle upstream must fail loudly here.
    """
    pin = get_pin("qwen3p5-1p5b")
    tiny = _TinyMLXModel(seed=321)
    tok = _TinyTokenizer()
    with patch(
        "harness.real_models.qwen_mlx._load_checkpoint",
        return_value=(tiny, tok),
    ):
        # The tiny model's bytes won't match the registry-pinned
        # file_sha256 — the wrapper must detect this when the
        # caller explicitly enables pin enforcement.
        with pytest.raises(ValueError, match="sha256 mismatch"):
            QwenMLXWrapper(pin, enforce_pin=True)

        # Without enforcement, construction succeeds and the
        # wrapper still records the actual digest.
        w = QwenMLXWrapper(pin, enforce_pin=False)
        actual = w.weights_sha256()
        expected = hashlib.sha256(tiny.weights_bytes()).hexdigest()
        assert actual == expected


# --------------------------------------------------------------------------
# Test 3 — typed Protocol matching (GammaSnapshotProtocol, DR-3)
# --------------------------------------------------------------------------


def test_wrapper_satisfies_gamma_snapshot_protocol() -> None:
    """Wrapper is a structural match for GammaSnapshotProtocol.

    Framework-C §2.1 : γ-channel = weights-only snapshot. A real-
    weight substrate must expose get_checkpoint_path() +
    get_checkpoint_sha256() per the runtime-checkable Protocol.
    This is DR-3 Conformance Criterion condition (1) for cycle-3.
    """
    pin = get_pin("qwen3p5-1p5b")
    tiny = _TinyMLXModel(seed=9)
    tok = _TinyTokenizer()
    with patch(
        "harness.real_models.qwen_mlx._load_checkpoint",
        return_value=(tiny, tok),
    ):
        w = QwenMLXWrapper(pin)
    assert isinstance(w, GammaSnapshotProtocol)
    # Methods behave as expected.
    path = w.get_checkpoint_path()
    assert path.name  # non-empty pathlib stem
    sha = w.get_checkpoint_sha256()
    assert len(sha) == 64


# --------------------------------------------------------------------------
# Test 4 — K1 compute-budget tagging
# --------------------------------------------------------------------------


def test_forward_tags_k1_compute_budget() -> None:
    """Every forward call records a compute_budget_flops estimate.

    K1 invariant : dream-episodes must be budget-bounded. The
    wrapper computes a FLOP estimate per forward (using scale_params
    as a linear proxy) and stores it on both the returned trace and
    an internal counter so the scheduler can refuse over-budget DEs.
    """
    pin = get_pin("qwen3p5-7b")
    tiny = _TinyMLXModel(seed=9)
    tok = _TinyTokenizer()
    with patch(
        "harness.real_models.qwen_mlx._load_checkpoint",
        return_value=(tiny, tok),
    ):
        w = QwenMLXWrapper(pin)
        trace1 = w.forward(mx.array([1, 2, 3, 4]), seed=0)
        trace2 = w.forward(mx.array([1, 2, 3, 4, 5, 6]), seed=0)
    # FLOP tag present and strictly positive.
    assert trace1.compute_flops > 0
    # Longer token sequence ⇒ more FLOPs (monotonic in n_tokens).
    assert trace2.compute_flops > trace1.compute_flops
    # Cumulative counter aggregates across calls.
    assert w.total_compute_flops == trace1.compute_flops + trace2.compute_flops


# --------------------------------------------------------------------------
# Test 5 — multi-scale dispatch by (scale, quant) key
# --------------------------------------------------------------------------


def test_load_qwen_dispatches_by_scale_key() -> None:
    """``load_qwen(slot)`` returns a wrapper for each registry slot.

    Cycle-3 plan : ``load_qwen("qwen3p5-1p5b" / "qwen3p5-7b" /
    "qwen3p5-35b")`` all resolve via the registry with stable keys.
    """
    tiny = _TinyMLXModel(seed=9)
    tok = _TinyTokenizer()
    for slot, expected_pin in (
        ("qwen3p5-1p5b", REGISTRY["qwen3p5-1p5b"]),
        ("qwen3p5-7b", REGISTRY["qwen3p5-7b"]),
        ("qwen3p5-35b", REGISTRY["qwen3p5-35b"]),
    ):
        with patch(
            "harness.real_models.qwen_mlx._load_checkpoint",
            return_value=(tiny, tok),
        ):
            w = load_qwen(slot)
        assert isinstance(w, QwenMLXWrapper)
        assert w.pin.name == expected_pin.name
        assert w.pin is expected_pin


def test_load_qwen_unknown_slot_raises() -> None:
    """Unknown slot key propagates :class:`KeyError` from the registry."""
    with pytest.raises(KeyError):
        load_qwen("qwen3p5-999b")


# --------------------------------------------------------------------------
# Auxiliary coverage : real mlx-lm fallback path for _weights_bytes
# (a model without a weights_bytes() helper must still hash via the
#  parameters() walk — exercises lines 102-124 of qwen_mlx.py).
# --------------------------------------------------------------------------


class _RealLikeModel:
    """Stand-in that mimics the real mlx-lm model surface.

    Exposes ``parameters()`` returning a nested dict-of-arrays
    (dict / list / mx.array branches) but **does not** expose a
    ``weights_bytes()`` shortcut. Also defines ``__call__`` so the
    wrapper's forward() path still works.
    """

    def __init__(self) -> None:
        self._params = {
            "layer0": {
                "weight": mx.array(np.arange(8, dtype=np.float32)),
                "bias": mx.array(np.zeros(4, dtype=np.float32)),
            },
            "extras": [
                mx.array(np.ones(2, dtype=np.float32)),
                mx.array(np.full(2, 2.0, dtype=np.float32)),
            ],
        }

    def parameters(self) -> dict[str, Any]:
        return self._params

    def __call__(self, tokens: mx.array) -> mx.array:
        # Return a deterministic dummy embedding so forward() works.
        n = int(np.asarray(tokens).size)
        return mx.array(np.zeros((n, 4), dtype=np.float32))


def test_weights_bytes_fallback_uses_parameters_walk() -> None:
    """Wrapper hashes real mlx-lm models via ``parameters()`` walk.

    When the model does not expose ``weights_bytes()``, the helper
    walks the nested parameter tree (dict + list + mx.array) to
    produce a deterministic byte-string that feeds SHA-256.
    """
    pin = get_pin("qwen3p5-7b")
    realish = _RealLikeModel()
    tok = _TinyTokenizer()
    with patch(
        "harness.real_models.qwen_mlx._load_checkpoint",
        return_value=(realish, tok),
    ):
        w = QwenMLXWrapper(pin)
    sha_a = w.weights_sha256()
    # Hash is deterministic across reloads of the same parameters.
    realish2 = _RealLikeModel()
    with patch(
        "harness.real_models.qwen_mlx._load_checkpoint",
        return_value=(realish2, tok),
    ):
        w2 = QwenMLXWrapper(pin)
    assert sha_a == w2.weights_sha256()

    # model + tokenizer properties are callable/returned intact.
    assert w.model is realish
    assert w.tokenizer is tok
