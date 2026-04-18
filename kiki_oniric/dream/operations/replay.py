"""Replay operation — A-Walker/Stickgold consolidation source.

Skeleton version (S5.4): counts consumed records, logs episode.
Real gradient-based replay (sample β → forward through W → update
via retention-objective gradient) lands alongside MLX integration
S7+ with swap protocol.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kiki_oniric.dream.episode import DreamEpisode


@dataclass
class ReplayOpState:
    """Mutable counter state for replay op across multiple episodes."""

    total_records_consumed: int = 0
    total_episodes_handled: int = 0
    last_loss: float | None = None


def replay_handler(state: ReplayOpState) -> Callable[[DreamEpisode], None]:
    """Build a replay handler bound to a state instance.

    Handler consumes all `beta_records` in the DE's input_slice,
    updates the state counters. No-op on weights for now
    (skeleton) — gradient integration S7+ with MLX.
    """

    def handler(episode: DreamEpisode) -> None:
        records = episode.input_slice.get("beta_records", [])
        state.total_records_consumed += len(records)
        state.total_episodes_handled += 1

    return handler


def replay_handler_mlx(
    state: ReplayOpState,
    model,  # mlx.nn.Module — typed loosely for lazy import
    lr: float = 0.01,
) -> Callable[[DreamEpisode], None]:
    """Build a replay handler with real MLX gradient updates.

    Records expected as `{"x": list[float], "y": list[float]}`.
    Forward pass + MSE loss + SGD step on model parameters.

    The skeleton `replay_handler` remains for tests / contexts not
    requiring MLX. This MLX variant is the production path for the
    G2 GO-FULL gate (real weight updates).

    Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
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
        state.total_episodes_handled += 1
        if not records:
            state.last_loss = None
            return

        # Validate record schema before MLX conversion
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
        state.total_records_consumed += len(records)
        state.last_loss = float(loss.item())

    return handler
