"""Hierarchical MLX MLP classifier + dream-episode wrapper for G4-ter.

Architecture: input -> Linear(in_dim, hidden_1) -> ReLU ->
Linear(hidden_1, hidden_2) -> ReLU -> Linear(hidden_2, n_classes).

Compared to ``experiments.g4_split_fmnist.dream_wrap.G4Classifier``
the hierarchy exposes a *middle* hidden layer (hidden_2) that is
addressable as a RESTRUCTURE site (perturb its weight tensor without
touching input projection nor output classifier) and a latent
representation (hidden_2 activations) that is addressable as a
RECOMBINE site (Gaussian-MoG synthetic-latent injection).

DR-0 accountability is automatic: every call to ``dream_episode_hier``
appends one EpisodeLogEntry to ``profile.runtime.log`` regardless of
handler outcome.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md sec 3.1
    docs/osf-prereg-g4-ter-pilot.md sec 2-3
    docs/superpowers/plans/2026-05-03-g4-ter-hp-sweep-richer-substrate.md
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field
from typing import TypedDict

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
import numpy as np


class BetaRecordHier(TypedDict, total=False):
    """One curated episodic exemplar for the hierarchical head.

    Compared to ``BetaRecord`` (G4-bis), adds an optional
    ``latent`` field that holds the hidden_2 activation captured at
    push time, used as the support set for the RECOMBINE Gaussian-
    MoG sampler.
    """

    x: list[float]
    y: int
    latent: list[float] | None


@dataclass
class G4HierarchicalClassifier:
    """Hierarchical MLP classifier for Split-FMNIST 2-class tasks.

    Layers : Linear(in_dim, hidden_1) -> ReLU -> Linear(hidden_1,
    hidden_2) -> ReLU -> Linear(hidden_2, n_classes). Deterministic
    under a fixed ``seed`` via ``mx.random.seed`` at construction.
    """

    in_dim: int
    hidden_1: int
    hidden_2: int
    n_classes: int
    seed: int
    _l1: nn.Linear = field(init=False, repr=False)
    _l2: nn.Linear = field(init=False, repr=False)
    _l3: nn.Linear = field(init=False, repr=False)
    _model: nn.Module = field(init=False, repr=False)

    def __post_init__(self) -> None:
        mx.random.seed(self.seed)
        np.random.seed(self.seed)
        self._l1 = nn.Linear(self.in_dim, self.hidden_1)
        self._l2 = nn.Linear(self.hidden_1, self.hidden_2)
        self._l3 = nn.Linear(self.hidden_2, self.n_classes)
        self._model = nn.Sequential(
            self._l1, nn.ReLU(), self._l2, nn.ReLU(), self._l3
        )
        mx.eval(self._model.parameters())

    def predict_logits(self, x: np.ndarray) -> np.ndarray:
        """Return raw logits as a numpy array shape ``(N, n_classes)``."""
        out = self._model(mx.array(x))
        mx.eval(out)
        return np.asarray(out)

    def latent(self, x: np.ndarray) -> np.ndarray:
        """Return hidden_2 activations shape ``(N, hidden_2)``.

        Used by the beta buffer to capture per-record latents at push
        time for the RECOMBINE Gaussian-MoG sampler.
        """
        h1 = nn.relu(self._l1(mx.array(x)))
        h2 = nn.relu(self._l2(h1))
        mx.eval(h2)
        return np.asarray(h2)

    def eval_accuracy(self, x: np.ndarray, y: np.ndarray) -> float:
        if len(x) == 0:
            return 0.0
        logits = self.predict_logits(x)
        pred = logits.argmax(axis=1)
        return float((pred == y).mean())

    def train_task(
        self,
        task: dict[str, np.ndarray],
        *,
        epochs: int,
        batch_size: int,
        lr: float,
    ) -> None:
        x = mx.array(task["x_train"])
        y = mx.array(task["y_train"])
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

    def _restructure_step(self, *, factor: float, seed: int) -> None:
        """Add ``factor * sigma * N(0, 1)`` to hidden_2 weights only.

        ``sigma`` is the per-tensor standard deviation of
        ``self._l2.weight`` at call time. ``factor=0`` is a no-op
        (early exit). ``factor < 0`` raises ValueError. Determinism
        is provided by ``np.random.default_rng(seed)``.

        Hierarchy / RESTRUCTURE channel rationale: only the *middle*
        hidden layer is mutated, so the input projection and output
        classifier are preserved. This matches the framework-C sec
        3.1 H_DR4 monotonicity intuition that RESTRUCTURE escalates
        framework C's representational rearrangement without
        wholesale weight collapse.
        """
        if factor < 0.0:
            raise ValueError(
                f"factor must be non-negative, got {factor}"
            )
        if factor == 0.0:
            return
        w = np.asarray(self._l2.weight)
        sigma = float(w.std()) if w.size > 0 else 0.0
        if sigma == 0.0:
            return
        rng = np.random.default_rng(seed)
        noise = rng.standard_normal(size=w.shape).astype(np.float32)
        new_w = w + factor * sigma * noise
        self._l2.weight = mx.array(new_w)
        mx.eval(self._l2.weight)

    def _recombine_step(
        self,
        *,
        latents: list[tuple[list[float], int]],
        n_synthetic: int,
        lr: float,
        seed: int,
    ) -> None:
        """Sample ``n_synthetic`` synthetic latents from a per-class
        Gaussian-MoG and run one CE-loss SGD pass through ``_l3`` only.

        ``latents`` is a list of ``(latent_vector, class_label)``
        pairs accumulated from past tasks via ``BetaBufferHierFIFO``.
        Per-class component means/std are estimated empirically;
        synthetic samples are drawn from N(mean_c, std_c) and labeled
        by ``c``. The forward pass uses ``self._l3`` only, so the
        gradient flows into the output classifier weights - leaving
        ``_l1`` / ``_l2`` untouched.

        Empty ``latents`` -> no-op (S1-trivial branch). Single-class
        ``latents`` -> no-op (degenerate MoG, no recombination signal).
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

        x = mx.array(np.stack(synth_x).astype(np.float32))
        y = mx.array(np.asarray(synth_y, dtype=np.int32))

        opt = optim.SGD(learning_rate=lr)

        def loss_fn(layer: nn.Linear, xb: mx.array, yb: mx.array) -> mx.array:
            return nn.losses.cross_entropy(layer(xb), yb, reduction="mean")

        loss_and_grad = nn.value_and_grad(self._l3, loss_fn)
        _loss, grads = loss_and_grad(self._l3, x, y)
        opt.update(self._l3, grads)
        mx.eval(self._l3.parameters(), opt.state)

    def _replay_optimizer_step(
        self,
        records: list[BetaRecordHier],
        *,
        lr: float,
        n_steps: int,
    ) -> None:
        if not records:
            return
        x = mx.array([r["x"] for r in records])
        y = mx.array([r["y"] for r in records])
        opt = optim.SGD(learning_rate=lr)

        def loss_fn(model: nn.Module, xb: mx.array, yb: mx.array) -> mx.array:
            return nn.losses.cross_entropy(model(xb), yb, reduction="mean")

        loss_and_grad = nn.value_and_grad(self._model, loss_fn)
        for _ in range(n_steps):
            _loss, grads = loss_and_grad(self._model, x, y)
            opt.update(self._model, grads)
            mx.eval(self._model.parameters(), opt.state)

    def _downscale_step(self, *, factor: float) -> None:
        """Multiply every weight + bias in ``self._model`` by ``factor``.

        Bounds : ``factor`` must lie in ``(0, 1]`` - same constraint
        as ``downscale_real_handler``.
        """
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )
        for layer in (self._l1, self._l2, self._l3):
            layer.weight = layer.weight * factor
            if getattr(layer, "bias", None) is not None:
                layer.bias = layer.bias * factor
        mx.eval(self._model.parameters())

    def dream_episode_hier(
        self,
        profile: object,
        seed: int,
        *,
        beta_buffer: "BetaBufferHierFIFO",
        replay_n_records: int,
        replay_n_steps: int,
        replay_lr: float,
        downscale_factor: float,
        restructure_factor: float,
        recombine_n_synthetic: int,
        recombine_lr: float,
    ) -> None:
        """Drive one DreamEpisode and mutate classifier weights for G4-ter.

        Coupling map (richer substrate):
        - ``Operation.REPLAY`` -> ``_replay_optimizer_step`` over
          buffer sample of ``replay_n_records``.
        - ``Operation.DOWNSCALE`` -> ``_downscale_step`` with
          ``downscale_factor``.
        - ``Operation.RESTRUCTURE`` -> ``_restructure_step`` on
          ``_l2.weight`` with ``restructure_factor``.
        - ``Operation.RECOMBINE`` -> ``_recombine_step`` over Gaussian-
          MoG synthetic latents drawn from ``beta_buffer``.

        DR-0 spectator runtime path is preserved (synthetic
        input_slice values), so every episode appends one
        EpisodeLogEntry to ``profile.runtime.log``.
        """
        # Local imports to keep module-level deps minimal.
        from kiki_oniric.dream.episode import (
            BudgetCap,
            DreamEpisode,
            EpisodeTrigger,
            Operation,
            OutputChannel,
        )
        from kiki_oniric.profiles.p_min import PMinProfile

        if isinstance(profile, PMinProfile):
            ops: tuple[Operation, ...] = (
                Operation.REPLAY,
                Operation.DOWNSCALE,
            )
            channels: tuple[OutputChannel, ...] = (
                OutputChannel.WEIGHT_DELTA,
            )
        else:
            ops = (
                Operation.REPLAY,
                Operation.DOWNSCALE,
                Operation.RESTRUCTURE,
                Operation.RECOMBINE,
            )
            channels = (
                OutputChannel.WEIGHT_DELTA,
                OutputChannel.HIERARCHY_CHG,
                OutputChannel.LATENT_SAMPLE,
            )

        # Spectator runtime path (DR-0 logging).
        rng = np.random.default_rng(seed + 10_000)
        synthetic_records = [
            {
                "x": rng.standard_normal(4).astype(np.float32).tolist(),
                "y": rng.standard_normal(4).astype(np.float32).tolist(),
            }
            for _ in range(4)
        ]
        delta_latents = [
            rng.standard_normal(4).astype(np.float32).tolist()
            for _ in range(2)
        ]
        episode = DreamEpisode(
            trigger=EpisodeTrigger.SCHEDULED,
            input_slice={
                "beta_records": synthetic_records,
                "shrink_factor": 0.99,
                "topo_op": "reroute",
                "swap_indices": [0, 1],
                "delta_latents": delta_latents,
            },
            operation_set=ops,
            output_channels=channels,
            budget=BudgetCap(
                flops=10_000_000, wall_time_s=10.0, energy_j=1.0
            ),
            episode_id=f"g4-ter-{type(profile).__name__}-seed{seed}",
        )
        profile.runtime.execute(episode)  # type: ignore[attr-defined]

        # ---- Plan G4-ter coupling on richer substrate ----
        if Operation.REPLAY in ops:
            sampled = beta_buffer.sample(n=replay_n_records, seed=seed)
            self._replay_optimizer_step(
                sampled, lr=replay_lr, n_steps=replay_n_steps
            )
        if Operation.DOWNSCALE in ops:
            self._downscale_step(factor=downscale_factor)
        if Operation.RESTRUCTURE in ops:
            self._restructure_step(
                factor=restructure_factor, seed=seed + 20_000
            )
        if Operation.RECOMBINE in ops:
            latents_records: list[tuple[list[float], int]] = []
            for r in beta_buffer.snapshot():
                lat = r["latent"]
                if lat is not None:
                    latents_records.append((list(lat), int(r["y"])))
            self._recombine_step(
                latents=latents_records,
                n_synthetic=recombine_n_synthetic,
                lr=recombine_lr,
                seed=seed + 30_000,
            )


class BetaBufferHierFIFO:
    """Bounded curated episodic buffer with optional latents (beta channel).

    FIFO eviction at capacity. Compared to ``BetaBufferFIFO`` (G4-bis),
    each record carries an optional ``latent`` field - used by the
    RECOMBINE Gaussian-MoG sampler. ``latent=None`` is allowed for
    legacy / pre-classifier-warmup pushes.
    """

    def __init__(self, capacity: int) -> None:
        if capacity <= 0:
            raise ValueError(
                f"capacity must be positive, got {capacity}"
            )
        self._capacity = capacity
        self._records: deque[BetaRecordHier] = deque(maxlen=capacity)

    def __len__(self) -> int:
        return len(self._records)

    def push(
        self, *, x: np.ndarray, y: int, latent: np.ndarray | None
    ) -> None:
        record: BetaRecordHier = {
            "x": x.astype(np.float32).tolist(),
            "y": int(y),
            "latent": (
                latent.astype(np.float32).tolist()
                if latent is not None
                else None
            ),
        }
        self._records.append(record)

    def snapshot(self) -> list[BetaRecordHier]:
        return [
            {
                "x": list(r["x"]),
                "y": int(r["y"]),
                "latent": (
                    list(r["latent"]) if r["latent"] is not None else None
                ),
            }
            for r in self._records
        ]

    def sample(self, n: int, seed: int) -> list[BetaRecordHier]:
        n_avail = len(self._records)
        if n_avail == 0:
            return []
        rng = np.random.default_rng(seed)
        n_take = min(n, n_avail)
        indices = rng.choice(n_avail, size=n_take, replace=False)
        snapshot = list(self._records)
        return [
            {
                "x": list(snapshot[i]["x"]),
                "y": int(snapshot[i]["y"]),
                "latent": (
                    list(snapshot[i]["latent"])
                    if snapshot[i]["latent"] is not None
                    else None
                ),
            }
            for i in sorted(indices.tolist())
        ]

    def latents(self) -> list[list[float]]:
        """Return the list of populated latents (skips None)."""
        return [
            list(r["latent"])
            for r in self._records
            if r["latent"] is not None
        ]
