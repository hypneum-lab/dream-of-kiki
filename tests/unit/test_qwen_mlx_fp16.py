"""Unit tests for cycle-3 C3.8 Phase A Qwen FP16 wrapper.

Covers :mod:`harness.real_models.qwen_mlx_fp16` with the 3 TDD tests
called out in the Phase A dispatch :

1. FP16 pin registered — all three bf16 scale slots resolve via the
   shared registry and carry the ``bf16-mlx`` quantization tag.
2. Wrapper instantiates with a mocked mlx-lm load, satisfies
   :class:`GammaSnapshotProtocol`, and refuses Q4 pins.
3. SGD step runs without error on a tiny synthetic module — the
   ``parameters`` / ``update_parameters`` surface is consistent with
   :class:`mlx.optimizers.SGD` via :func:`mlx.nn.value_and_grad`,
   i.e. the gradient-bearing dream-op contract.

All tests monkey-patch :func:`mlx_lm.load` through the
``_load_checkpoint_fp16`` seam so the suite stays network-free and
sub-second.
"""
from __future__ import annotations

import hashlib
from typing import Any
from unittest.mock import patch

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

from harness.real_models.base_model_registry import (
    REGISTRY,
    get_pin,
)
from harness.real_models.qwen_mlx_fp16 import (
    FP16ForwardTrace,
    QwenMLXFP16Wrapper,
    load_qwen_fp16,
)
from kiki_oniric.core.primitives import GammaSnapshotProtocol


# --------------------------------------------------------------------------
# Tiny trainable MLX module that looks enough like an mlx-lm model.
# --------------------------------------------------------------------------


class _TinyTrainableModel(nn.Module):  # type: ignore[misc,name-defined]
    """Minimal ``nn.Module`` exposing ``parameters()`` + ``__call__``.

    A single ``nn.Linear(hidden, vocab_size)`` is enough to drive a
    gradient step through :func:`mlx.nn.value_and_grad` — the SGD
    smoke test in :func:`test_sgd_step_runs_on_tiny_synthetic`
    updates the linear's weight + bias without error.
    """

    def __init__(self, hidden: int = 4, vocab_size: int = 8) -> None:
        super().__init__()
        self.hidden = hidden
        self.vocab_size = vocab_size
        self.linear = nn.Linear(hidden, vocab_size)  # type: ignore[attr-defined]

    def __call__(self, token_ids: mx.array) -> mx.array:
        # Token id → one-hot row in the input space so the linear
        # has something deterministic to project. Preserves the
        # leading batch shape of ``token_ids`` so forward passes
        # with (batch, seq) or (seq,) inputs both project cleanly.
        arr = np.asarray(token_ids).astype(np.int64) % self.hidden
        orig_shape = arr.shape
        flat = arr.reshape(-1)
        one_hot = np.zeros((flat.size, self.hidden), dtype=np.float32)
        for i, j in enumerate(flat.tolist()):
            one_hot[i, int(j)] = 1.0
        x = mx.array(one_hot)
        out = self.linear(x)
        # Reshape back to (*orig_shape, vocab_size) so the caller's
        # target tensor broadcasts cleanly against the forward
        # output. A (2, 4) input therefore yields (2, 4, vocab_size).
        return mx.reshape(
            out, list(orig_shape) + [self.vocab_size]
        )


class _TinyTokenizer:
    """Shim tokenizer matching the Q4 test's surface."""

    def encode(self, text: str) -> list[int]:
        return [b % 8 for b in text.encode("utf-8")[:16]]


def _mock_load_fp16(
    _repo_id: str,
) -> tuple[_TinyTrainableModel, _TinyTokenizer]:
    # Fresh module per load so the mocked loader behaves like a
    # real weight fetch (no shared state between tests).
    return _TinyTrainableModel(), _TinyTokenizer()


# --------------------------------------------------------------------------
# Test 1 — FP16 pins are registered + well-formed.
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "slot",
    ("qwen3p5-1p5b-fp16", "qwen3p5-7b-fp16", "qwen3p5-35b-fp16"),
)
def test_fp16_pin_registered(slot: str) -> None:
    """Every bf16 scale slot is resolvable via the shared registry.

    Phase A contract : the C3.8 real pilot needs a gradient-bearing
    model at each of the three scale slots. Missing a pin would
    silently force the pilot to fall back on the Q4 variant (no
    backprop) and mask the dream-ops contract.
    """
    assert slot in REGISTRY, f"missing FP16 slot : {slot}"
    pin = get_pin(slot)
    assert pin.quantization.lower().startswith("bf16")
    assert pin.framework == "mlx-lm"
    assert pin.scale_params > 0
    # Multi-shard 7B/35B still pin shard 1 — 64-char hex is the
    # minimum contract even on multi-shard sets.
    assert pin.file_sha256 is not None
    assert len(pin.file_sha256) == 64


# --------------------------------------------------------------------------
# Test 2 — wrapper instantiates, satisfies GammaSnapshotProtocol,
#          refuses Q4 pins (guardrail against silent non-trainable pilot).
# --------------------------------------------------------------------------


def test_wrapper_instantiates_and_satisfies_protocol() -> None:
    """Mocked bf16 wrapper exposes the γ-channel + FLOPs surface.

    Framework-C §2.1 : the bf16 wrapper is a GammaSnapshotProtocol
    implementer (DR-3 condition (1)). The test also verifies that
    passing a Q4 pin raises :class:`ValueError` so the real-pilot
    path cannot accidentally connect to a non-trainable scorer.
    """
    pin = get_pin("qwen3p5-1p5b-fp16")
    with patch(
        "harness.real_models.qwen_mlx_fp16._load_checkpoint_fp16",
        side_effect=_mock_load_fp16,
    ):
        w = QwenMLXFP16Wrapper(pin)
    assert isinstance(w, GammaSnapshotProtocol)
    assert w.get_checkpoint_path().name  # non-empty path stem
    assert len(w.get_checkpoint_sha256()) == 64
    # Forward pass returns the new FP16 trace type.
    with patch(
        "harness.real_models.qwen_mlx_fp16._load_checkpoint_fp16",
        side_effect=_mock_load_fp16,
    ):
        w_fwd = QwenMLXFP16Wrapper(pin)
    trace = w_fwd.forward(mx.array([1, 2, 3, 4]), seed=7)
    assert isinstance(trace, FP16ForwardTrace)
    assert trace.compute_flops > 0
    assert trace.n_tokens == 4

    # Q4 pin must be rejected — guards the gradient contract.
    q4_pin = get_pin("qwen3p5-1p5b")
    with patch(
        "harness.real_models.qwen_mlx_fp16._load_checkpoint_fp16",
        side_effect=_mock_load_fp16,
    ):
        with pytest.raises(ValueError, match="bf16/fp16"):
            QwenMLXFP16Wrapper(q4_pin)

    # load_qwen_fp16 accepts both naming conventions + propagates
    # KeyError for unknown slots.
    with patch(
        "harness.real_models.qwen_mlx_fp16._load_checkpoint_fp16",
        side_effect=_mock_load_fp16,
    ):
        w_bare = load_qwen_fp16("qwen3p5-1p5b")
        w_suffix = load_qwen_fp16("qwen3p5-1p5b-fp16")
    assert w_bare.pin.name == w_suffix.pin.name == "qwen3p5-1p5b-fp16"
    with pytest.raises(KeyError):
        load_qwen_fp16("qwen3p5-999b-fp16")


# --------------------------------------------------------------------------
# Test 3 — SGD step runs on tiny synthetic (gradient-bearing contract).
# --------------------------------------------------------------------------


def test_sgd_step_runs_on_tiny_synthetic() -> None:
    """One ``mlx.optimizers.SGD`` step mutates the wrapped weights.

    Phase A requirement : the FP16 wrapper must feed
    :func:`mlx.nn.value_and_grad` and ``optimizer.update(model, grads)``
    exactly as the ``replay_real`` handler does. A non-trivial
    weight delta across a single SGD step is the empirical evidence
    that the gradient path is live end-to-end.
    """
    import mlx.optimizers as optim

    pin = get_pin("qwen3p5-1p5b-fp16")
    with patch(
        "harness.real_models.qwen_mlx_fp16._load_checkpoint_fp16",
        side_effect=_mock_load_fp16,
    ):
        w = QwenMLXFP16Wrapper(pin)

    # Snapshot weights before the SGD step for the delta assertion.
    w0 = np.asarray(w.model.linear.weight).copy()
    b0 = np.asarray(w.model.linear.bias).copy()

    # One MSE step over a batch of 2 token-ids — mirrors replay_real's
    # loss_fn + grad_fn pattern. 1-D input gives a 2-D (batch,
    # vocab_size) output that broadcasts cleanly against the
    # target tensor.
    xs = mx.array(np.array([1, 2], dtype=np.int64))
    ys = mx.array(
        np.zeros((2, w.model.vocab_size), dtype=np.float32)
    )

    def loss_fn(model: Any, x: mx.array, y: mx.array) -> mx.array:
        pred = model(x)
        return mx.mean((pred - y) ** 2)

    grad_fn = nn.value_and_grad(w.model, loss_fn)  # type: ignore[attr-defined]
    loss, grads = grad_fn(w.model, xs, ys)
    optimizer = optim.SGD(learning_rate=0.1)
    optimizer.update(w.model, grads)
    mx.eval(w.model.parameters())

    # Weights moved (non-zero delta is the live-gradient signal) ;
    # shape preserved (S3 topology trivial precondition).
    w1 = np.asarray(w.model.linear.weight)
    b1 = np.asarray(w.model.linear.bias)
    assert w1.shape == w0.shape
    assert b1.shape == b0.shape
    # Loss is a finite scalar — S2 invariant applied in-flight.
    assert np.isfinite(float(loss.item()))
    # At least one parameter delta is non-zero (SGD actually fired).
    assert not np.allclose(w1, w0) or not np.allclose(b1, b0)

    # update_parameters accepts a direct dict — exercises the
    # downscale_real / recombine_real path.
    shrunk = {
        "linear": {
            "weight": w.model.linear.weight * 0.5,
            "bias": w.model.linear.bias * 0.5,
        }
    }
    w.update_parameters(shrunk)
    mx.eval(w.model.parameters())
    w2 = np.asarray(w.model.linear.weight)
    # Roughly halved (allowing for tiny float error) — manual
    # update_parameters reached the module.
    assert np.allclose(w2, w1 * 0.5, atol=1e-5)


# --------------------------------------------------------------------------
# Auxiliary coverage : pin-mismatch enforcement + SHA fallback.
# --------------------------------------------------------------------------


def test_pin_enforcement_rejects_sha_mismatch() -> None:
    """``enforce_pin=True`` refuses a model whose digest disagrees.

    The tiny mock model's weights never match the registry-pinned
    FP16 shard-1 hash, so the wrapper must raise when enforcement
    is on. Leaving ``enforce_pin`` off records the actual digest.
    """
    pin = get_pin("qwen3p5-1p5b-fp16")
    with patch(
        "harness.real_models.qwen_mlx_fp16._load_checkpoint_fp16",
        side_effect=_mock_load_fp16,
    ):
        with pytest.raises(ValueError, match="sha256 mismatch"):
            QwenMLXFP16Wrapper(pin, enforce_pin=True)
        w = QwenMLXFP16Wrapper(pin, enforce_pin=False)
    # digest is a well-formed sha256 of 64 lowercase hex chars.
    # Exact value depends on mlx's Linear Glorot init which differs
    # between instantiations ; assert format only.
    actual = w.weights_sha256()
    assert len(actual) == 64
    # sanity : the digest path works on a fresh model too.
    fresh = _TinyTrainableModel()
    fresh_bytes = (
        np.asarray(fresh.linear.weight).tobytes()
        + np.asarray(fresh.linear.bias).tobytes()
    )
    assert len(hashlib.sha256(fresh_bytes).hexdigest()) == 64


def test_flop_counter_aggregates_across_forwards() -> None:
    """Cumulative FLOPs match the sum of per-forward tags."""
    pin = get_pin("qwen3p5-7b-fp16")
    with patch(
        "harness.real_models.qwen_mlx_fp16._load_checkpoint_fp16",
        side_effect=_mock_load_fp16,
    ):
        w = QwenMLXFP16Wrapper(pin)
    t1 = w.forward(mx.array([1, 2, 3]), seed=0)
    t2 = w.forward(mx.array([1, 2, 3, 4, 5]), seed=0)
    assert t2.compute_flops > t1.compute_flops  # monotonic in n_tokens
    assert w.total_compute_flops == t1.compute_flops + t2.compute_flops
    w.zero_grad_compute_counter()
    assert w.total_compute_flops == 0
