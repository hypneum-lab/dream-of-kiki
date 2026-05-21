"""Diffusion-substrate emitting wrappers for replay and downscale.

These wrappers mirror ``kiki_oniric.dream.operations.{replay,downscale}_real``
in behaviour but additionally capture the dense per-layer delta on
the bound ``MLPDenoiser`` and emit a channel-1 ``WeightUpdate`` so
the B5 awake/dream loop can apply the consolidation to the denoiser
via ``DenoiserWeightDeltaChannel``.

Substrate-local per ``kiki_oniric/dream/operations/CLAUDE.md`` (no
4th variant under ``dream/operations/``; new substrates dispatch
via ``kiki_oniric/substrates/``).

See ``docs/superpowers/specs/2026-05-21-m7-delta-acc-consolidation-eval-design.md``
§ Components 2 + § Components 3.
"""
from __future__ import annotations

from typing import Any, Callable

import mlx.core as mx
import mlx.nn as _nn
import mlx.optimizers as optim
import numpy as np

from kiki_oniric.dream.channels import WeightUpdate
from kiki_oniric.dream.episode import DreamEpisode
from kiki_oniric.dream.operations.downscale_real import DownscaleRealState
from kiki_oniric.dream.operations.replay_real import ReplayRealState

nn: Any = _nn


def _snapshot_layers(layers: list[Any]) -> dict[str, np.ndarray]:
    """Capture a numpy snapshot of every layer.weight / layer.bias."""
    snap: dict[str, np.ndarray] = {}
    for i, layer in enumerate(layers):
        for attr in ("weight", "bias"):
            t = getattr(layer, attr, None)
            if t is not None:
                snap[f"layer_{i}_{attr}"] = np.asarray(t).copy()
    return snap


def _diff(
    pre: dict[str, np.ndarray], post: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    """Return per-key (post − pre) deltas as float32 arrays."""
    return {
        k: (post[k] - pre[k]).astype(np.float32, copy=False)
        for k in pre
    }


def _restore_layers(layers: list[Any], snap: dict[str, np.ndarray]) -> None:
    """Restore each layer.weight / layer.bias from a snapshot.

    Used by ``replay_diffusion_handler`` to roll back the in-place
    SGD step after capturing the delta — the handler is then
    purely compute-only, and ``apply_channel_outputs`` is the sole
    applier (B5 contract).
    """
    for i, layer in enumerate(layers):
        for attr in ("weight", "bias"):
            key = f"layer_{i}_{attr}"
            if key in snap:
                setattr(layer, attr, mx.array(snap[key]))
    tensors: list[Any] = []
    for layer in layers:
        for attr in ("weight", "bias"):
            t = getattr(layer, attr, None)
            if t is not None:
                tensors.append(t)
    if tensors:
        mx.eval(*tensors)


def replay_diffusion_handler(
    state: ReplayRealState,
    *,
    model: Any,
    lr: float = 1e-2,
) -> Callable[[DreamEpisode], "WeightUpdate | None"]:
    """Build a diffusion-substrate replay handler that emits.

    Mirrors ``replay_real_handler``'s SGD step on the denoiser but
    snapshots all ``layer_i.{weight,bias}`` before / after and emits
    a ``WeightUpdate`` carrying the dense delta. Empty
    ``beta_records`` → no-op, returns ``None`` (S1 contract).
    """
    optimizer = optim.SGD(learning_rate=lr)

    def loss_fn(model_inner: Any, x: mx.array, y: mx.array) -> mx.array:
        pred = model_inner(x)
        return mx.mean((pred - y) ** 2)

    grad_fn = nn.value_and_grad(model, loss_fn)

    def handler(episode: DreamEpisode) -> "WeightUpdate | None":
        records = episode.input_slice.get("beta_records", [])
        if not records:
            state.last_loss = None
            state.last_compute_flops = 0
            return None

        for idx, r in enumerate(records):
            if "x" not in r or "y" not in r:
                raise ValueError(
                    f"record {idx} missing 'x' or 'y' key: {r!r}"
                )

        pre = _snapshot_layers(model.layers)

        xs = mx.array([r["x"] for r in records])
        ys = mx.array([r["y"] for r in records])
        loss, grads = grad_fn(model, xs, ys)
        optimizer.update(model, grads)
        mx.eval(model.parameters())

        post = _snapshot_layers(model.layers)
        delta = _diff(pre, post)

        # ROLLBACK: restore the pre-update params so the handler is
        # compute-only. apply_channel_outputs(weight_channel=...) is
        # the sole applier (B5 awake/dream loop).
        _restore_layers(model.layers, pre)

        state.total_records_consumed += len(records)
        state.last_loss = float(loss.item())
        # K1 tag: a crude record-count proxy is enough for non-LoRA.
        state.last_compute_flops = max(len(records), 1)
        state.total_compute_flops += state.last_compute_flops

        return WeightUpdate(lora_delta=delta, fisher_bump=None)

    return handler


def downscale_diffusion_handler(
    state: DownscaleRealState,
    *,
    model: Any,
) -> Callable[[DreamEpisode], "WeightUpdate | None"]:
    """Build a diffusion-substrate downscale handler that emits.

    Mirrors ``downscale_real_handler`` (multiply weight + bias by
    ``shrink_factor`` per layer) but snapshots all
    ``layer_i.{weight,bias}`` before / after and emits a
    ``WeightUpdate`` carrying the dense (factor − 1) · W delta.
    """

    def handler(episode: DreamEpisode) -> "WeightUpdate | None":
        factor = episode.input_slice.get("shrink_factor", 1.0)
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )

        # Delta = (factor - 1) * W for each layer's weight + bias.
        # Pure arithmetic from the current params — no in-place mutation.
        delta: dict[str, np.ndarray] = {}
        param_count = 0
        for i, layer in enumerate(model.layers):
            for attr in ("weight", "bias"):
                t = getattr(layer, attr, None)
                if t is not None:
                    arr = np.asarray(t)
                    delta[f"layer_{i}_{attr}"] = (
                        ((factor - 1.0) * arr).astype(np.float32, copy=False)
                    )
                    param_count += arr.size

        state.compound_factor *= factor
        state.last_compute_flops = max(param_count, 1)

        return WeightUpdate(lora_delta=delta, fisher_bump=None)

    return handler


__all__ = [
    "replay_diffusion_handler",
    "downscale_diffusion_handler",
]
