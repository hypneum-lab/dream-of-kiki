"""G4-ter hierarchical MLP with a TTT fast-weight layer (NH-03).

Forward : ``x -> l1 -> ReLU -> h1 -> h1 + h1 @ W_f.T -> l2 -> ReLU
-> l3``. With ``W_f = 0`` (its initial value) the network is the
G4-ter ``G4HierarchicalClassifier`` : same layer order, same
``mx.random.seed(seed)`` initialisation, same wake SGD. ``W_f`` is
excluded from the wake optimiser and written only through the
:class:`~kiki_oniric.substrates.ttt_fast_weights.TTTSubstrate`
handlers (TTT descent / forgetting gate), every write metered by the
substrate ledger.

Slow-weight consolidation writes (G4 replay step, G4 downscale,
matched-SGD control step) are metered on the same ledger so that the
write budget is comparable across arms.

Reference : docs/milestones/2026-10-25-nh03-ttt-vs-episodes.md
"""
from __future__ import annotations

from typing import Any

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np

from kiki_oniric.dream.guards.finite import check_finite
from kiki_oniric.substrates.ttt_fast_weights import (
    OP_DOWNSCALE,
    OP_REPLAY,
    TTTFastState,
    TTTSubstrate,
    WriteLedger,
    apply_fast,
)

OP_SGD_CONTROL = "sgd_control"


class _SlowNet(nn.Module):
    def __init__(self, in_dim: int, h1: int, h2: int, n_classes: int) -> None:
        super().__init__()
        self.l1 = nn.Linear(in_dim, h1)
        self.l2 = nn.Linear(h1, h2)
        self.l3 = nn.Linear(h2, n_classes)

    def hidden1(self, x: mx.array) -> mx.array:
        return nn.relu(self.l1(x))

    def head(self, h1: mx.array) -> mx.array:
        return self.l3(nn.relu(self.l2(h1)))

    def __call__(self, x: mx.array, w_fast: mx.array) -> mx.array:
        return self.head(apply_fast(w_fast, self.hidden1(x)))


class NH03Classifier:
    """Hierarchical classifier + fast weights + write ledger."""

    def __init__(
        self,
        *,
        in_dim: int,
        hidden_1: int,
        hidden_2: int,
        n_classes: int,
        seed: int,
        ledger: WriteLedger | None = None,
    ) -> None:
        self.seed = seed
        mx.random.seed(seed)
        np.random.seed(seed)
        self.net = _SlowNet(in_dim, hidden_1, hidden_2, n_classes)
        mx.eval(self.net.parameters())
        self.fast = TTTFastState.zeros(hidden_1)
        self.substrate = TTTSubstrate(ledger=ledger or WriteLedger())
        self._ttt_replay = self.substrate.replay_handler_factory()
        self._ttt_decay = self.substrate.downscale_handler_factory()
        self._n_slow = sum(
            int(v.size) for _, v in nn.utils.tree_flatten(self.net.parameters())
        )

    # ------------------------------------------------------------ eval
    @property
    def ledger(self) -> WriteLedger:
        return self.substrate.ledger

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        out = self.net(mx.array(x), self.fast.weights)
        mx.eval(out)
        return np.asarray(out)

    def eval_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) == 0:
            return 0.0
        return float((self.predict_logits(x).argmax(axis=1) == y).mean())

    # ------------------------------------------------------------ wake
    def _slow_sgd_steps(self, xb: mx.array, yb: mx.array, *, lr: float, n: int) -> None:
        opt = optim.SGD(learning_rate=lr)
        w_fast = self.fast.weights

        def loss_fn(model: _SlowNet, xx: mx.array, yy: mx.array) -> mx.array:
            return nn.losses.cross_entropy(model(xx, w_fast), yy, reduction="mean")

        lg = nn.value_and_grad(self.net, loss_fn)
        for _ in range(n):
            _loss, grads = lg(self.net, xb, yb)
            opt.update(self.net, grads)
            mx.eval(self.net.parameters(), opt.state)

    def wake_batches(
        self, task: dict[str, np.ndarray], *, epochs: int, batch_size: int
    ) -> list[np.ndarray]:
        """Deterministic list of index batches (G4 ordering)."""
        n = task["x_train"].shape[0]
        rng = np.random.default_rng(self.seed)
        out: list[np.ndarray] = []
        for _ in range(epochs):
            order = rng.permutation(n)
            for start in range(0, n, batch_size):
                idx = order[start : start + batch_size]
                if len(idx):
                    out.append(idx)
        return out

    def wake_step(self, task: dict[str, np.ndarray], idx: np.ndarray, *, lr: float) -> None:
        xb = mx.array(task["x_train"][idx])
        yb = mx.array(task["y_train"][idx])
        self._slow_sgd_steps(xb, yb, lr=lr, n=1)

    # ------------------------------------------------- consolidation
    def write_slow_replay(self, records: list[dict[str, Any]], *, lr: float, episode_id: str) -> None:
        """One G4 replay gradient step on the slow weights (1 write)."""
        if not records:
            return
        xb = mx.array(np.asarray([r["x"] for r in records], dtype=np.float32))
        yb = mx.array(np.asarray([r["y"] for r in records], dtype=np.int32))
        self._slow_sgd_steps(xb, yb, lr=lr, n=1)
        self._check_slow()
        self.ledger.record(
            episode_id=episode_id, op=OP_REPLAY,
            n_elements=self._n_slow, n_samples=len(records),
        )

    def write_slow_downscale(self, *, factor: float, episode_id: str) -> None:
        """G4 SHY downscale of every slow weight and bias (1 write)."""
        if not (0.0 < factor <= 1.0):
            raise ValueError(f"shrink_factor must be in (0, 1], got {factor}")
        for layer in (self.net.l1, self.net.l2, self.net.l3):
            layer.weight = layer.weight * factor
            layer.bias = layer.bias * factor
        mx.eval(self.net.parameters())
        self._check_slow()
        self.ledger.record(
            episode_id=episode_id, op=OP_DOWNSCALE,
            n_elements=self._n_slow, n_samples=0,
        )

    def write_sgd_control(self, xb: np.ndarray, yb: np.ndarray, *, lr: float, episode_id: str) -> None:
        """Matched-count SGD step on wake-stream data, no beta (1 write)."""
        self._slow_sgd_steps(mx.array(xb), mx.array(yb), lr=lr, n=1)
        self._check_slow()
        self.ledger.record(
            episode_id=episode_id, op=OP_SGD_CONTROL,
            n_elements=self._n_slow, n_samples=int(len(yb)),
        )

    def write_ttt_replay(self, records: list[dict[str, Any]], *, lr: float, episode_id: str) -> None:
        """TTT descent on W_f under the beta mini-batch (1 write)."""
        if not records:
            return
        x = mx.array(np.asarray([r["x"] for r in records], dtype=np.float32))
        y = mx.array(np.asarray([r["y"] for r in records], dtype=np.int32))
        h1 = mx.stop_gradient(self.net.hidden1(x))
        net = self.net

        def fwd(w: mx.array, hh: mx.array) -> mx.array:
            return net.head(apply_fast(w, hh))

        self.fast = self._ttt_replay(self.fast, h1, y, fwd, lr, episode_id)

    def write_ttt_decay(self, *, factor: float, episode_id: str) -> None:
        """Forgetting gate on W_f (1 write)."""
        self.fast = self._ttt_decay(self.fast, factor, episode_id)

    def _check_slow(self) -> None:
        check_finite(
            {k: np.asarray(v) for k, v in nn.utils.tree_flatten(self.net.parameters())}
        )
