"""SNN-substrate replay op — spike-rate-proxy (cycle-3 C3.12).

Norse-substrate counterpart to
:mod:`kiki_oniric.dream.operations.replay_real`. Instead of driving
an MLX gradient step, this variant treats model weights as spike
rates, applies the dream op in spike-rate domain, and projects back
via an inverse-sigmoid (logit) map.

The implementation is deliberately a **pure-numpy proxy** : no
PyTorch / Norse runtime is required for the op to execute. The
Norse substrate in :mod:`kiki_oniric.substrates.esnn_norse` owns the
actual LIF dynamics ; this module exposes the 8-primitive-Protocol
surface (same signatures as ``*_real.py``) so the DR-3 Conformance
Criterion condition (1) applies to the Norse substrate end-to-end.

Contract (mirrors :mod:`replay_real`) :

- ``weights`` is passed once at factory construction — a numpy
  array whose entries are treated as logits of per-synapse spike
  rates in ``[0, max_rate]`` via
  :func:`weights_to_spike_rates`. The op updates the array in-place
  (caller keeps the reference), mirroring the MLX variant's
  in-place ``model.parameters()`` mutation.
- ``target_rates`` read from ``episode.input_slice`` ; must be a
  numpy array shape-compatible with ``weights``.
- Rate-domain SGD step : ``rates ← rates + lr * (target - rates)``.
- Clip rates to ``[eps, max_rate - eps]`` before the inverse-sigmoid
  map so ``log(r / (max_rate - r))`` stays finite (S2 guard).
- Updates ``state.last_spike_delta_norm`` with
  ``np.linalg.norm(target_rates - rates_before)``.
- K1 tag : ~1 multiply + 1 add per weight scalar (rate update) +
  the sigmoid / inverse-sigmoid round-trip.

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2, §6.2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np

from kiki_oniric.dream.episode import DreamEpisode
from kiki_oniric.dream.guards.finite import check_finite


# ===== Shared spike-rate proxy helpers =====
#
# These two functions form the invertible weights ↔ rates bridge
# used by every SNN-proxy op. The mapping is sigmoid-based :
#
#     rates = max_rate * sigmoid(weights)
#     weights = logit(rates / max_rate)
#
# which is exact up to ``eps`` saturation near the extremes
# (``|w| >~ 10`` pushes rates to the clipped boundary and the
# inverse map loses precision — documented concern, not a bug).


def weights_to_spike_rates(
    w: np.ndarray, max_rate: float = 100.0
) -> np.ndarray:
    """Map real-valued weights to spike rates in ``[0, max_rate]``.

    Uses the numerically-stable sigmoid ``1 / (1 + exp(-x))`` scaled
    by ``max_rate``. Positive weights map to high firing rates,
    negative weights to low firing rates, and ``w = 0`` to
    ``max_rate / 2`` (the neutral baseline for a Poisson-rate code).
    """
    # No explicit clip on ``-w_arr`` is needed : numpy's ``exp``
    # handles overflow gracefully (``exp(large) → +inf`` yields
    # ``max_rate / (1 + inf) → 0``, ``exp(-large) → 0`` yields
    # ``max_rate / 1 → max_rate``), so the result stays inside
    # ``[0, max_rate]`` without an extra guard. The inverse map
    # below still needs its eps-guard because ``log(0)`` is ``-inf``.
    w_arr = np.asarray(w, dtype=float)
    return max_rate / (1.0 + np.exp(-w_arr))


def spike_rates_to_weights(
    r: np.ndarray, max_rate: float = 100.0, eps: float = 1e-6
) -> np.ndarray:
    """Inverse of :func:`weights_to_spike_rates` (logit map).

    Clips rates to ``[eps, max_rate - eps]`` so the argument of
    ``log`` stays strictly positive. Round-trip
    ``spike_rates_to_weights(weights_to_spike_rates(w))`` is an
    identity to ~6 decimal digits for ``|w| < 10``.
    """
    r_arr = np.asarray(r, dtype=float)
    r_clipped = np.clip(r_arr, eps, max_rate - eps)
    return np.log(r_clipped / (max_rate - r_clipped))


@dataclass
class ReplaySNNState:
    """K1-tagged SNN-proxy replay state across multiple episodes.

    Mirrors :class:`ReplayRealState` but tracks the spike-rate
    domain delta norm rather than the MSE loss (the notion of
    "loss" is not well-defined in a pure rate-proxy update).
    """

    total_records_consumed: int = 0
    last_compute_flops: int = 0
    total_compute_flops: int = 0
    last_spike_delta_norm: float | None = None


def _flop_estimate(n_scalar: int) -> int:
    """Rough FLOP count for a rate-domain SGD step over ``n_scalar``
    weights (1 sub + 1 mul + 1 add per scalar, plus sigmoid /
    inverse-sigmoid ~5 FLOPs each).
    """
    return max(8 * max(n_scalar, 1), 1)


def replay_snn_handler(
    state: ReplaySNNState,
    *,
    weights: np.ndarray,
    lr: float = 0.01,
    max_rate: float = 100.0,
) -> Callable[[DreamEpisode], None]:
    """Build an SNN-proxy replay handler bound to ``state``.

    The caller passes a **mutable** numpy array ``weights`` that
    the handler updates in place — same contract as the MLX
    variant, which mutates the model parameter tree.
    """

    def handler(episode: DreamEpisode) -> None:
        target_rates = episode.input_slice.get("target_rates", None)
        if target_rates is None:
            # S1 no-op branch : mirror ``replay_real`` on empty
            # records — no rates, no compute, no tag.
            state.last_spike_delta_norm = None
            state.last_compute_flops = 0
            return

        target = np.asarray(target_rates, dtype=float)
        if target.shape != weights.shape:
            raise ValueError(
                f"target_rates shape {target.shape} "
                f"!= weights shape {weights.shape}"
            )

        rates = weights_to_spike_rates(weights, max_rate=max_rate)
        delta = target - rates
        new_rates = rates + lr * delta
        # S2 guard : clip + finite-check before the inverse map
        new_rates = np.clip(new_rates, 1e-6, max_rate - 1e-6)
        check_finite(new_rates)

        new_weights = spike_rates_to_weights(
            new_rates, max_rate=max_rate
        )
        # In-place update preserves caller's reference (mirrors
        # MLX `model.update(new_params)`).
        weights[...] = new_weights

        state.total_records_consumed += int(target.size)
        state.last_spike_delta_norm = float(
            np.linalg.norm(delta)
        )
        flops = _flop_estimate(int(weights.size))
        state.last_compute_flops = flops
        state.total_compute_flops += flops

    return handler


__all__ = [
    "ReplaySNNState",
    "replay_snn_handler",
    "weights_to_spike_rates",
    "spike_rates_to_weights",
]
