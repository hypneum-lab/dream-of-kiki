"""Real-weight replay op — gradient update over MLX model params.

Cycle-3 C3.3 counterpart to :mod:`kiki_oniric.dream.operations.replay`
that actually drives an MLX gradient step and tags K1 compute_flops
on the state, so the dream-episode scheduler can enforce budget
caps when real-weight updates land.

Contract (mirrors synthetic op style) :

- State is a mutable :class:`ReplayRealState` dataclass ; the factory
  :func:`replay_real_handler` closes over ``model`` + ``lr`` and
  returns the episode handler.
- Empty ``beta_records`` → no-op : no loss, no gradient step, no
  FLOP tag (test 5, preserves S1 trivially).
- Non-empty ``beta_records`` → MSE loss against ``y`` targets, one
  SGD step via :class:`mlx.optimizers.SGD`, tag last/total FLOPs,
  bump ``total_records_consumed``.
- Weight tensor shapes are preserved by construction — SGD only
  mutates values, never shapes (test 4).

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kiki_oniric.dream.episode import DreamEpisode


@dataclass
class ReplayRealState:
    """K1-tagged replay state across multiple episodes."""

    total_records_consumed: int = 0
    last_loss: float | None = None
    last_compute_flops: int = 0
    total_compute_flops: int = 0


def _flop_estimate(n_records: int, x_dim: int, y_dim: int) -> int:
    """Rough FLOP count : forward + backward over ``n_records``.

    Heuristic matches the K1 accounting style in ``qwen_mlx.py``
    (``2 * params * tokens``). Tiny MLP + 1 record ≈ a few thousand
    FLOPs — enough for the scheduler to see a non-zero tag.
    """
    # 4-8-2 MLP : ~48 mul-adds in fwd × 2 (bwd) × n_records
    per_record = max(2 * (x_dim * y_dim + x_dim + y_dim), 1)
    return max(per_record * n_records, 1)


def replay_real_handler(
    state: ReplayRealState,
    *,
    model,  # mlx.nn.Module — typed loosely for lazy import
    lr: float = 0.01,
) -> Callable[[DreamEpisode], None]:
    """Build a real-weight replay handler bound to ``state``.

    Imports MLX lazily so modules that never invoke the handler
    (e.g. synthetic-only tests) don't pay the import cost.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    optimizer = optim.SGD(learning_rate=lr)

    def loss_fn(model_inner, x, y):
        pred = model_inner(x)
        return mx.mean((pred - y) ** 2)

    grad_fn = nn.value_and_grad(model, loss_fn)

    def handler(episode: DreamEpisode) -> None:
        records = episode.input_slice.get("beta_records", [])
        if not records:
            # S1 no-op branch : no signal, no compute, no tag.
            state.last_loss = None
            state.last_compute_flops = 0
            return

        for idx, r in enumerate(records):
            if "x" not in r or "y" not in r:
                raise ValueError(
                    f"record {idx} missing 'x' or 'y' key: {r!r}"
                )

        xs = mx.array([r["x"] for r in records])
        ys = mx.array([r["y"] for r in records])
        loss, grads = grad_fn(model, xs, ys)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        # ``x`` / ``y`` may be scalars (int/float) in degenerate
        # fixtures — ``len()`` would raise TypeError. Treat scalars
        # as 1-D (one feature) so the FLOP estimate stays valid.
        def _dim(v) -> int:
            try:
                return len(v)
            except TypeError:
                return 1

        x_dim = _dim(records[0]["x"])
        y_dim = _dim(records[0]["y"])
        flops = _flop_estimate(len(records), x_dim, y_dim)

        state.total_records_consumed += len(records)
        state.last_loss = float(loss.item())
        state.last_compute_flops = flops
        state.total_compute_flops += flops

    return handler


__all__ = [
    "ReplayRealState",
    "replay_real_handler",
]
