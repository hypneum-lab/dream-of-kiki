"""Small MLX CNN substrate for G4-quinto Steps 2 + 3 — NHWC layout.

Architecture (NHWC, MLX convention) ::

    (N, 32, 32, 3)
        -> Conv2d(3, 16, 3, pad=1) ReLU MaxPool2d(2, 2)  -> (N, 16, 16, 16)
        -> Conv2d(16, 32, 3, pad=1) ReLU MaxPool2d(2, 2) -> (N, 8, 8, 32)
        -> Flatten -> Linear(2048, latent_dim) ReLU
        -> Linear(latent_dim, n_classes)

Op site mapping :

- **REPLAY** — full-model SGD on a batch from the beta buffer
  (records may carry flat 3072 floats or NHWC arrays ; the
  optimizer reshapes flat-shaped inputs to NHWC).
- **DOWNSCALE** — multiply every weight + bias of {conv1, conv2,
  fc1, fc2} by ``factor``. Bound ``(0, 1]`` identical to MLP.
- **RESTRUCTURE** — perturb ``conv2.weight`` only (analogue of the
  MLP middle layer ``_l3`` ; preserves input projection ``conv1``
  and output classifier ``fc2``).
- **RECOMBINE** — synthetic latents (dim = ``latent_dim``) per the
  active strategy ∈ {mog, ae, none} ; one CE-loss SGD step on
  ``fc2``.

DR-0 accountability is provided by the dream-episode wrapper —
this module exposes only the model + train/eval primitives.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md sec 3.1
    docs/osf-prereg-g4-quinto-pilot.md sec 2 (H5-B, H5-C)
    docs/superpowers/plans/2026-05-03-g4-quinto-cifar10-escalation.md
"""
from __future__ import annotations

from dataclasses import dataclass, field

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


class _Flatten(nn.Module):
    """Reshape ``(N, H, W, C)`` -> ``(N, H * W * C)``.

    MLX has no built-in ``nn.Flatten`` ; this wrapper preserves
    seedability and integrates with ``nn.Sequential``.
    """

    def __call__(self, x: mx.array) -> mx.array:
        return mx.reshape(x, (x.shape[0], -1))


@dataclass
class G4SmallCNN:
    """Small CNN classifier for Split-CIFAR-10 2-class tasks.

    Layers : two ``nn.Conv2d`` + two ``nn.MaxPool2d`` (NHWC) +
    two ``nn.Linear``. Deterministic under a fixed ``seed`` via
    ``mx.random.seed`` at construction.
    """

    latent_dim: int
    n_classes: int
    seed: int
    _conv1: nn.Conv2d = field(init=False, repr=False)
    _conv2: nn.Conv2d = field(init=False, repr=False)
    _pool1: nn.MaxPool2d = field(init=False, repr=False)
    _pool2: nn.MaxPool2d = field(init=False, repr=False)
    _fc1: nn.Linear = field(init=False, repr=False)
    _fc2: nn.Linear = field(init=False, repr=False)
    _model: nn.Module = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mx.random.seed(self.seed)
        np.random.seed(self.seed)
        self._conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1)
        self._conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self._pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self._fc1 = nn.Linear(2048, self.latent_dim)
        self._fc2 = nn.Linear(self.latent_dim, self.n_classes)
        self._model = nn.Sequential(
            self._conv1, nn.ReLU(), self._pool1,
            self._conv2, nn.ReLU(), self._pool2,
            _Flatten(), self._fc1, nn.ReLU(), self._fc2,
        )
        mx.eval(self._model.parameters())

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """Return raw logits as a numpy array shape ``(N, n_classes)``."""
        out = self._model(mx.array(x))
        mx.eval(out)
        return np.asarray(out)

    def latent(self, x: np.ndarray) -> np.ndarray:
        """Return post-relu fc1 activations shape ``(N, latent_dim)``.

        These are the activations *after the third ReLU* (the post-
        ``fc1`` ReLU) — the RECOMBINE Gaussian-MoG sampling site.
        """
        h = self._pool1(nn.relu(self._conv1(mx.array(x))))
        h = self._pool2(nn.relu(self._conv2(h)))
        h = mx.reshape(h, (h.shape[0], -1))
        h = nn.relu(self._fc1(h))
        mx.eval(h)
        return np.asarray(h)

    def eval_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) == 0:
            return 0.0
        logits = self.predict_logits(x)
        pred = logits.argmax(axis=1)
        return float((pred == y).mean())

    def train_task(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        *,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        """Train one task on NHWC inputs.

        Accepts ``(x_train, y_train)`` arrays directly (CNN consumes
        NHWC ; no flat-dict wrapper needed unlike the MLP path).
        """
        x = mx.array(x_train)
        y = mx.array(y_train)
        n = x.shape[0]
        opt = optim.SGD(learning_rate=lr)
        rng = np.random.default_rng(self.seed)

        def loss_fn(model: nn.Module, xb: mx.array, yb: mx.array) -> mx.array:
            return nn.losses.cross_entropy(model(xb), yb, reduction="mean")

        loss_and_grad = nn.value_and_grad(self._model, loss_fn)
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if len(idx) == 0:
                    continue
                xb = x[mx.array(idx)]
                yb = y[mx.array(idx)]
                _loss, grads = loss_and_grad(self._model, xb, yb)
                opt.update(self._model, grads)
                mx.eval(self._model.parameters(), opt.state)

    def restructure_step(self, *, factor: float, seed: int) -> None:
        """Add ``factor * sigma * N(0, 1)`` to ``_conv2.weight`` only.

        ``sigma`` is the per-tensor std of ``self._conv2.weight`` at
        call time. ``factor=0`` is a no-op ; ``factor < 0`` raises.
        """
        if factor < 0.0:
            raise ValueError(
                f"factor must be non-negative, got {factor}"
            )
        if factor == 0.0:
            return
        w = np.asarray(self._conv2.weight)
        sigma = float(w.std()) if w.size > 0 else 0.0
        if sigma == 0.0:
            return
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(size=w.shape).astype(np.float32)
        new_w = w + factor * sigma * noise
        self._conv2.weight = mx.array(new_w)
        mx.eval(self._conv2.weight)

    def downscale_step(self, *, factor: float) -> None:
        """Multiply every weight + bias in ``self._model`` by ``factor``.

        Bounds : ``factor`` must lie in ``(0, 1]``.
        """
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )
        for layer in (self._conv1, self._conv2, self._fc1, self._fc2):
            layer.weight = layer.weight * factor
            if getattr(layer, "bias", None) is not None:
                layer.bias = layer.bias * factor
        mx.eval(self._model.parameters())

    def replay_optimizer_step(
        self,
        records: list[dict[str, list[float] | int]],
        *,
        lr: float,
        n_steps: int,
    ) -> None:
        """Replay-buffer SGD pass on the full CNN.

        ``records`` may carry ``"x"`` as a flat 3072-element list
        (FMNIST-style) or a nested NHWC structure. Flat-shaped
        inputs are reshaped to ``(N, 32, 32, 3)`` ; NHWC inputs are
        passed through unchanged.
        """
        if not records:
            return
        x_np = np.asarray([r["x"] for r in records], dtype=np.float32)
        if x_np.ndim == 2:
            # Flat 3072-d records reshape into NHWC for CNN consumption.
            x_np = x_np.reshape(-1, 32, 32, 3)
        y_np = np.asarray([r["y"] for r in records], dtype=np.int64)
        x = mx.array(x_np)
        y = mx.array(y_np)
        opt = optim.SGD(learning_rate=lr)

        def loss_fn(model: nn.Module, xb: mx.array, yb: mx.array) -> mx.array:
            return nn.losses.cross_entropy(model(xb), yb, reduction="mean")

        loss_and_grad = nn.value_and_grad(self._model, loss_fn)
        for _ in range(n_steps):
            _loss, grads = loss_and_grad(self._model, x, y)
            opt.update(self._model, grads)
            mx.eval(self._model.parameters(), opt.state)

    def recombine_step(
        self,
        *,
        latents: list[tuple[list[float], int]],
        n_synthetic: int,
        lr: float,
        seed: int,
    ) -> None:
        """Sample ``n_synthetic`` synthetic latents from a per-class
        Gaussian-MoG and run one CE-loss SGD pass through ``_fc2`` only.

        Empty / single-class ``latents`` -> no-op (S1-trivial).
        """
        if not latents:
            return
        classes = sorted({lbl for _, lbl in latents})
        if len(classes) < 2:
            return

        rng = np.random.default_rng(seed)
        components: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for c in classes:
            arr = np.asarray(
                [lat for lat, lbl in latents if lbl == c],
                dtype=np.float32,
            )
            mean = arr.mean(axis=0)
            std = arr.std(axis=0) + 1e-6
            components[c] = (mean, std)

        per_class = max(1, n_synthetic // len(classes))
        synth_x: list[np.ndarray] = []
        synth_y: list[int] = []
        for c in classes:
            mean, std = components[c]
            for _ in range(per_class):
                synth_x.append(
                    mean + std * rng.standard_normal(size=mean.shape).astype(
                        np.float32
                    )
                )
                synth_y.append(c)

        # Synthetic latents have latent_dim ; feed _fc2 directly
        # (no _l4 ReLU intermediate, unlike the 5-layer MLP).
        x_lat = mx.array(np.stack(synth_x).astype(np.float32))
        y = mx.array(np.asarray(synth_y, dtype=np.int32))

        opt = optim.SGD(learning_rate=lr)

        def loss_fn(layer: nn.Linear, xb: mx.array, yb: mx.array) -> mx.array:
            return nn.losses.cross_entropy(layer(xb), yb, reduction="mean")

        loss_and_grad = nn.value_and_grad(self._fc2, loss_fn)
        _loss, grads = loss_and_grad(self._fc2, x_lat, y)
        opt.update(self._fc2, grads)
        mx.eval(self._fc2.parameters(), opt.state)
