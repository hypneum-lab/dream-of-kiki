"""4-layer pure-numpy spiking CNN with STE backward (G5-ter).

Architecture (NHWC, batched) ::

    (N, 32, 32, 3)
      -> Conv2d(3->16, 3x3, pad=1) -> LIF rates  (N, 32, 32, 16)
      -> Conv2d(16->32, 3x3, pad=1) -> LIF rates (N, 32, 32, 32)
      -> avg_pool 4x4 (deterministic)            (N, 8, 8, 32)
      -> flatten + Linear(2048, 64) -> LIF       (N, 64)
      -> Linear(64, 2) (no LIF)                  (N, 2) logits

LIF defaults : ``tau=10.0``, ``threshold=1.0``, ``n_steps=20``.
Backward pass uses a straight-through estimator (Wu et al. 2018) :
``d_currents = d_rates`` at every LIF stage. Loss is softmax CE.

Pure-numpy throughout : no MLX, no PyTorch ; Conv2d uses an
im2col-style numpy matmul (NHWC, square kernels). avg-pool 4x4 is
parameter-free and fully-differentiable through the STE branch
(replaces MLX MaxPool2d which has no clean numpy backward).

Public surface mirrors ``EsnnG5BisHierarchicalClassifier``
(``predict_logits``, ``latent``, ``eval_accuracy``, ``train_task``,
``_ste_backward``) so the dream-episode wrapper transposes the
G5-bis MLP coupling onto this substrate mechanically.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md sec 6.2
    docs/osf-prereg-g5-ter-spiking-cnn.md sec 3
    experiments/g5_bis_richer_esnn/esnn_hier_classifier.py (sister)
    experiments/g4_quinto_test/small_cnn.py (MLX sister)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kiki_oniric.substrates.esnn_thalamocortical import (
    LIFState,
    simulate_lif_step,
)

__all__ = [
    "EsnnG5TerSpikingCNN",
    "avg_pool_4x4",
    "avg_pool_4x4_backward",
    "conv2d_backward",
    "conv2d_forward",
]


# ----------------------- conv2d helpers (pure numpy) -----------------------


def _im2col_nhwc(
    x: NDArray[np.float32], k: int, padding: int
) -> NDArray[np.float32]:
    """Unfold ``x`` (NHWC) into ``(N*H_out*W_out, k*k*C_in)`` for matmul.

    Stride is 1 ; H_out = H, W_out = W when ``padding=k//2``.
    """
    n, h, w, c = x.shape
    if padding > 0:
        x_p = np.zeros(
            (n, h + 2 * padding, w + 2 * padding, c), dtype=x.dtype
        )
        x_p[:, padding : padding + h, padding : padding + w, :] = x
    else:
        x_p = x
    h_out = h
    w_out = w
    cols = np.empty(
        (n * h_out * w_out, k * k * c), dtype=x.dtype
    )
    for ki in range(k):
        for kj in range(k):
            patch = x_p[:, ki : ki + h_out, kj : kj + w_out, :]
            patch = patch.reshape(n * h_out * w_out, c)
            base = (ki * k + kj) * c
            cols[:, base : base + c] = patch
    return cols


def conv2d_forward(
    x: NDArray[np.float32],
    W: NDArray[np.float32],
    b: NDArray[np.float32],
    *,
    padding: int,
) -> NDArray[np.float32]:
    """Pure-numpy NHWC Conv2d, square kernel, stride 1.

    Parameters
    ----------
    x
        Input tensor shape ``(N, H, W, C_in)``.
    W
        Weight tensor shape ``(k, k, C_in, C_out)``.
    b
        Bias tensor shape ``(C_out,)``.
    padding
        Symmetric zero padding. ``padding=k//2`` keeps spatial dims.

    Returns
    -------
    NDArray[np.float32]
        Output tensor shape ``(N, H_out, W_out, C_out)`` with
        ``H_out = H + 2*padding - k + 1`` (= H when ``padding=k//2``).
    """
    n, h, w, c_in = x.shape
    k1, k2, c_in_w, c_out = W.shape
    if k1 != k2:
        raise ValueError(
            f"square kernel required, got ({k1}, {k2})"
        )
    if c_in != c_in_w:
        raise ValueError(
            f"in-channel mismatch : x has {c_in}, W expects {c_in_w}"
        )
    cols = _im2col_nhwc(x.astype(np.float32), k1, padding)
    w_mat = W.reshape(k1 * k1 * c_in, c_out).astype(np.float32)
    out = (cols @ w_mat).astype(np.float32)
    h_out = h + 2 * padding - k1 + 1
    w_out = w + 2 * padding - k1 + 1
    out = out.reshape(n, h_out, w_out, c_out)
    out = out + b.astype(np.float32)
    return out.astype(np.float32)


def conv2d_backward(
    d_out: NDArray[np.float32],
    x: NDArray[np.float32],
    W: NDArray[np.float32],
    *,
    padding: int,
) -> tuple[
    NDArray[np.float32], NDArray[np.float32], NDArray[np.float32]
]:
    """Pure-numpy backward for ``conv2d_forward``.

    Returns ``(dx, dW, db)`` with the same shapes as ``(x, W, bias)``.
    Stride 1, square kernel only.
    """
    n, h, w, c_in = x.shape
    k1, _k2, _c_in_w, c_out = W.shape
    h_out = h + 2 * padding - k1 + 1
    w_out = w + 2 * padding - k1 + 1
    cols = _im2col_nhwc(x.astype(np.float32), k1, padding)
    w_mat = W.reshape(k1 * k1 * c_in, c_out).astype(np.float32)
    d_out_mat = d_out.reshape(n * h_out * w_out, c_out).astype(
        np.float32
    )
    # dW
    dW_mat = (cols.T @ d_out_mat).astype(np.float32)
    dW = dW_mat.reshape(k1, k1, c_in, c_out)
    # db
    db = d_out.sum(axis=(0, 1, 2)).astype(np.float32)
    # dx via col2im
    d_cols = (d_out_mat @ w_mat.T).astype(np.float32)
    dx_padded = np.zeros(
        (n, h + 2 * padding, w + 2 * padding, c_in), dtype=np.float32
    )
    for ki in range(k1):
        for kj in range(k1):
            base = (ki * k1 + kj) * c_in
            patch = d_cols[:, base : base + c_in].reshape(
                n, h_out, w_out, c_in
            )
            dx_padded[:, ki : ki + h_out, kj : kj + w_out, :] += patch
    if padding > 0:
        dx = dx_padded[:, padding : padding + h, padding : padding + w, :]
    else:
        dx = dx_padded
    return (
        dx.astype(np.float32),
        dW.astype(np.float32),
        db.astype(np.float32),
    )


# ----------------------- avg-pool 4x4 -----------------------


def avg_pool_4x4(x: NDArray[np.float32]) -> NDArray[np.float32]:
    """Deterministic 4x4 average pool, NHWC, stride 4.

    Requires ``H % 4 == 0`` and ``W % 4 == 0``.
    """
    n, h, w, c = x.shape
    if h % 4 != 0 or w % 4 != 0:
        raise ValueError(
            f"avg_pool_4x4 requires H, W divisible by 4 ; got ({h}, {w})"
        )
    h_out, w_out = h // 4, w // 4
    out = x.reshape(n, h_out, 4, w_out, 4, c).mean(axis=(2, 4))
    return out.astype(np.float32)


def avg_pool_4x4_backward(
    d_out: NDArray[np.float32], x_shape: tuple[int, ...]
) -> NDArray[np.float32]:
    """Backward for ``avg_pool_4x4`` : broadcast / 16 over 4x4 windows."""
    n, h, w, c = x_shape
    h_out, w_out = h // 4, w // 4
    factor = np.float32(1.0 / 16.0)
    grad = (
        np.broadcast_to(
            (d_out * factor).reshape(n, h_out, 1, w_out, 1, c),
            (n, h_out, 4, w_out, 4, c),
        )
        .reshape(n, h, w, c)
        .copy()
    )
    return grad.astype(np.float32)


# ----------------------- spiking CNN -----------------------


@dataclass
class EsnnG5TerSpikingCNN:
    """4-layer spiking CNN classifier with STE backward.

    Parameters
    ----------
    n_classes
        Output dimensionality. Must be ``>= 2``.
    seed
        Numpy RNG seed — controls Kaiming init + minibatch order.
    n_steps
        LIF simulation horizon per forward pass (default 20).
    tau, threshold
        LIF dynamics parameters passed to ``simulate_lif_step``.
    """

    n_classes: int
    seed: int
    n_steps: int = 20
    tau: float = 10.0
    threshold: float = 1.0
    W_c1: NDArray[np.float32] = field(init=False, repr=False)
    b_c1: NDArray[np.float32] = field(init=False, repr=False)
    W_c2: NDArray[np.float32] = field(init=False, repr=False)
    b_c2: NDArray[np.float32] = field(init=False, repr=False)
    W_fc1: NDArray[np.float32] = field(init=False, repr=False)
    b_fc1: NDArray[np.float32] = field(init=False, repr=False)
    W_out: NDArray[np.float32] = field(init=False, repr=False)
    b_out: NDArray[np.float32] = field(init=False, repr=False)
    _rng: np.random.Generator = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.n_classes < 2:
            raise ValueError(
                f"n_classes must be >= 2, got {self.n_classes}"
            )
        self._rng = np.random.default_rng(self.seed)
        # Kaiming sqrt(2/fan_in) for each weight tensor
        # conv1 : fan_in = 3*3*3 = 27
        scale_c1 = float(np.sqrt(2.0 / (3 * 3 * 3)))
        self.W_c1 = (
            self._rng.standard_normal((3, 3, 3, 16)) * scale_c1
        ).astype(np.float32)
        self.b_c1 = np.zeros(16, dtype=np.float32)
        # conv2 : fan_in = 3*3*16 = 144
        scale_c2 = float(np.sqrt(2.0 / (3 * 3 * 16)))
        self.W_c2 = (
            self._rng.standard_normal((3, 3, 16, 32)) * scale_c2
        ).astype(np.float32)
        self.b_c2 = np.zeros(32, dtype=np.float32)
        # fc1 : fan_in = 2048
        scale_fc1 = float(np.sqrt(2.0 / 2048))
        self.W_fc1 = (
            self._rng.standard_normal((2048, 64)) * scale_fc1
        ).astype(np.float32)
        self.b_fc1 = np.zeros(64, dtype=np.float32)
        # out : fan_in = 64
        scale_out = float(np.sqrt(2.0 / 64))
        self.W_out = (
            self._rng.standard_normal((64, self.n_classes)) * scale_out
        ).astype(np.float32)
        self.b_out = np.zeros(self.n_classes, dtype=np.float32)

    # -------------------- LIF rates (flat) --------------------

    def _lif_population_rates(
        self,
        currents: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Drive flat ``(N, K)`` currents through a LIF pop, return rates.

        Vectorised across the batch axis with the same Euler dynamics
        as ``simulate_lif_step`` (used by the G5-bis sister) :
        ``v_t+1 = v_t * decay + I * dt`` ; spike when ``v_t+1 >=
        threshold`` ; reset to zero post-spike. Equivalent semantics
        to the per-sample loop but ~2 orders of magnitude faster on
        Apple-Silicon numpy. Sanity-checked against the per-sample
        path in ``test_lif_population_rates_matches_unvectorised``.
        """
        n, n_neurons = currents.shape
        if n == 0 or n_neurons == 0:
            return np.zeros((n, n_neurons), dtype=np.float32)
        decay = np.float32(max(0.0, 1.0 - 1.0 / float(self.tau)))
        currents_f = np.ascontiguousarray(
            currents, dtype=np.float32
        )
        v = np.zeros_like(currents_f)
        spike_count = np.zeros(currents_f.shape, dtype=np.int32)
        thr = np.float32(self.threshold)
        for _ in range(self.n_steps):
            # In-place: v = v * decay + currents_f
            np.multiply(v, decay, out=v)
            np.add(v, currents_f, out=v)
            # Spike mask : v >= threshold
            spike_mask = v >= thr
            spike_count += spike_mask  # int32 += bool, no cast
            # Reset where spike fired (in-place, no temp tensor)
            v[spike_mask] = 0.0
        denom = max(self.n_steps, 1)
        return (spike_count / np.float32(denom)).astype(np.float32)

    def _lif_population_rates_unvectorised(
        self,
        currents: NDArray[np.float32],
    ) -> NDArray[np.float32]:
        """Reference per-sample implementation for cross-checking.

        Matches the G5-bis ``_lif_population_rates`` byte-for-byte
        (same ``simulate_lif_step`` calls) ; kept for the equivalence
        unit test only — the hot path uses the vectorised version.
        """
        n, n_neurons = currents.shape
        rates = np.zeros((n, n_neurons), dtype=np.float32)
        for i in range(n):
            state = LIFState(n_neurons=n_neurons)
            spike_sum = np.zeros(n_neurons, dtype=float)
            for _ in range(self.n_steps):
                state = simulate_lif_step(
                    state,
                    currents[i],
                    dt=1.0,
                    tau=self.tau,
                    threshold=self.threshold,
                )
                spike_sum += state.spikes
            denom = max(self.n_steps, 1)
            rates[i] = (spike_sum / denom).astype(np.float32)
        return rates

    # -------------------- forward --------------------

    def _forward_with_caches(
        self, x: NDArray[np.float32]
    ) -> dict[str, NDArray[np.float32]]:
        """Return cached activations needed for STE backward."""
        n = x.shape[0]
        i_c1 = conv2d_forward(x, self.W_c1, self.b_c1, padding=1)
        # LIF over flattened (N, 32*32*16) view
        i_c1_flat = i_c1.reshape(n, -1)
        r_c1_flat = self._lif_population_rates(i_c1_flat)
        r_c1 = r_c1_flat.reshape(n, 32, 32, 16)
        i_c2 = conv2d_forward(r_c1, self.W_c2, self.b_c2, padding=1)
        i_c2_flat = i_c2.reshape(n, -1)
        r_c2_flat = self._lif_population_rates(i_c2_flat)
        r_c2 = r_c2_flat.reshape(n, 32, 32, 32)
        p2 = avg_pool_4x4(r_c2)  # (N, 8, 8, 32)
        flat = p2.reshape(n, -1)  # (N, 2048)
        i_h1 = (flat @ self.W_fc1 + self.b_fc1).astype(np.float32)
        r_h1 = self._lif_population_rates(i_h1)
        logits = (r_h1 @ self.W_out + self.b_out).astype(np.float32)
        return {
            "x": x,
            "r_c1": r_c1,
            "r_c2": r_c2,
            "p2": p2,
            "flat": flat,
            "r_h1": r_h1,
            "logits": logits,
        }

    def predict_logits(
        self, x: NDArray[np.float32]
    ) -> NDArray[np.float32]:
        """Return logits of shape ``(N, n_classes)``."""
        if x.shape[0] == 0:
            return np.zeros((0, self.n_classes), dtype=np.float32)
        cache = self._forward_with_caches(x.astype(np.float32))
        return cache["logits"]

    def latent(self, x: NDArray[np.float32]) -> NDArray[np.float32]:
        """Return post-fc1 LIF rates ``(N, 64)``.

        Used by the beta buffer at push time as the support set for
        the RECOMBINE Gaussian-MoG sampler.
        """
        if x.shape[0] == 0:
            return np.zeros((0, 64), dtype=np.float32)
        cache = self._forward_with_caches(x.astype(np.float32))
        return cache["r_h1"]

    def eval_accuracy(
        self, x: NDArray[np.float32], y: NDArray[np.int64]
    ) -> float:
        """Classification accuracy in ``[0, 1]``."""
        if len(x) == 0:
            return 0.0
        logits = self.predict_logits(x)
        pred = logits.argmax(axis=1)
        return float((pred == y).mean())

    # -------------------- backward / training --------------------

    def _ste_backward(
        self,
        x: NDArray[np.float32],
        y: NDArray[np.int64],
        lr: float,
    ) -> None:
        """One SGD step with the straight-through gradient.

        Loss : softmax cross-entropy on logits. STE :
        ``d_currents = d_rates`` for every LIF stage, so gradient
        flows through the linear / convolutional projections only.
        Bias gradients on every layer.
        """
        if x.shape[0] == 0:
            return
        x_f = x.astype(np.float32)
        cache: dict[str, Any] = self._forward_with_caches(x_f)
        logits = cache["logits"]
        n_batch = x_f.shape[0]
        # Stable softmax
        logits_shift = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits_shift)
        probs = exp / exp.sum(axis=1, keepdims=True)
        one_hot = np.zeros_like(probs)
        one_hot[np.arange(n_batch), y] = 1.0
        d_logits = ((probs - one_hot) / max(n_batch, 1)).astype(
            np.float32
        )
        # Output linear
        d_W_out = (cache["r_h1"].T @ d_logits).astype(np.float32)
        d_b_out = d_logits.sum(axis=0).astype(np.float32)
        d_r_h1 = (d_logits @ self.W_out.T).astype(np.float32)
        # STE through fc-LIF
        d_i_h1 = d_r_h1
        d_W_fc1 = (cache["flat"].T @ d_i_h1).astype(np.float32)
        d_b_fc1 = d_i_h1.sum(axis=0).astype(np.float32)
        d_flat = (d_i_h1 @ self.W_fc1.T).astype(np.float32)
        d_p2 = d_flat.reshape(cache["p2"].shape)
        # Backprop through avg-pool
        d_r_c2 = avg_pool_4x4_backward(
            d_p2, cache["r_c2"].shape
        )
        # STE through conv2-LIF
        d_i_c2 = d_r_c2
        d_r_c1, d_W_c2, d_b_c2 = conv2d_backward(
            d_i_c2, cache["r_c1"], self.W_c2, padding=1
        )
        # STE through conv1-LIF
        d_i_c1 = d_r_c1
        _dx, d_W_c1, d_b_c1 = conv2d_backward(
            d_i_c1, x_f, self.W_c1, padding=1
        )
        # SGD update
        lr32 = np.float32(lr)
        self.W_out = (self.W_out - lr32 * d_W_out).astype(np.float32)
        self.b_out = (self.b_out - lr32 * d_b_out).astype(np.float32)
        self.W_fc1 = (self.W_fc1 - lr32 * d_W_fc1).astype(np.float32)
        self.b_fc1 = (self.b_fc1 - lr32 * d_b_fc1).astype(np.float32)
        self.W_c2 = (self.W_c2 - lr32 * d_W_c2).astype(np.float32)
        self.b_c2 = (self.b_c2 - lr32 * d_b_c2).astype(np.float32)
        self.W_c1 = (self.W_c1 - lr32 * d_W_c1).astype(np.float32)
        self.b_c1 = (self.b_c1 - lr32 * d_b_c1).astype(np.float32)

    def train_task(
        self,
        task: dict,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        """Per-epoch seeded permutation + minibatch SGD with STE.

        ``task`` is a dict with keys ``x_train`` (NHWC) and
        ``y_train``. Determinism : minibatch order is drawn from a
        numpy RNG seeded at ``self.seed`` so two classifiers built
        with the same seed converge to the same weights bit-exactly.
        """
        x_train = np.asarray(task["x_train"], dtype=np.float32)
        y_train = np.asarray(task["y_train"], dtype=np.int64)
        n = x_train.shape[0]
        if n == 0:
            return
        rng = np.random.default_rng(self.seed)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if len(idx) == 0:
                    continue
                xb = x_train[idx]
                yb = y_train[idx]
                self._ste_backward(xb, yb, lr)
