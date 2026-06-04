"""Continual-learning metric adapters for the diffusion bench.

Three minimal CL metrics, adapted to the M4 smoke run + M5 full
bench :

- :func:`accuracy_per_task` — per-task held-out accuracy of the
  awake classifier (proxied at M4 smoke as the negative final
  denoiser loss for shape parity ; M5 substitutes the real metric).
- :func:`average_accuracy` — mean of per-task accuracy across the
  5 Split-CIFAR-100 tasks (the "ACC" CL metric).
- :func:`forgetting` — mean drop from the per-task peak to the
  end-of-training value (the "BWT" CL metric, sign-flipped to
  positive = bad).

The functions are deliberately self-contained Python (no MLX in
the signature) so they can run on Linux CI runners that skip the
MLX tests. They accept either Python floats or any iterable
materialised to floats via ``float(...)``.

References:
- ``docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md``
  §3.1 (diffusion_metrics.py contract).
- Lopez-Paz & Ranzato, "Gradient Episodic Memory" (NeurIPS 2017)
  for the ACC + BWT conventions.
"""
from __future__ import annotations

from typing import Iterable, Sequence

# Public re-export of the metric names so the M5 driver can build
# a typed metric dict.
ACCURACY_PER_TASK = "accuracy_per_task"
AVERAGE_ACCURACY = "average_accuracy"
FORGETTING = "forgetting"

ALL_METRICS = (ACCURACY_PER_TASK, AVERAGE_ACCURACY, FORGETTING)


def accuracy_per_task(per_task_scores: Iterable[float]) -> list[float]:
    """Return per-task accuracy as a list of plain floats.

    M4 smoke contract : ``per_task_scores`` is the per-task end-of-
    training value the substrate returns ; the adapter just casts
    + materialises so the downstream dump is JSON-serialisable.

    Raises:
        ValueError: any score is NaN or outside ``[0.0, 1.0]`` —
            both indicate the upstream metric is mis-wired (the
            smoke path uses ``1.0 - normalised_loss`` to keep
            this contract intact).
    """
    out: list[float] = []
    for raw in per_task_scores:
        value = float(raw)
        if value != value:  # NaN check
            raise ValueError("accuracy_per_task : NaN score")
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                f"accuracy_per_task : score {value!r} outside [0, 1]"
            )
        out.append(value)
    return out


def average_accuracy(per_task_scores: Sequence[float]) -> float:
    """Mean of per-task accuracy (the CL "ACC" metric).

    Raises:
        ValueError: empty input (cannot mean over zero tasks).
    """
    values = accuracy_per_task(per_task_scores)
    if not values:
        raise ValueError("average_accuracy : empty per_task_scores")
    return sum(values) / len(values)


def forgetting(
    per_task_peak: Sequence[float],
    per_task_final: Sequence[float],
) -> float:
    """Mean drop from per-task peak accuracy to per-task final.

    Sign convention : positive = catastrophic forgetting (the model
    lost accuracy on earlier tasks while learning later ones), zero
    = no forgetting, negative = backward transfer (gained on
    earlier tasks). M4 smoke does not exercise the forgetting
    regime ; this lives here so M5 can wire it without a new
    module.

    Raises:
        ValueError: the two sequences have different lengths or
            either is empty.
    """
    peak = accuracy_per_task(per_task_peak)
    final = accuracy_per_task(per_task_final)
    if len(peak) != len(final):
        raise ValueError(
            f"forgetting : len mismatch peak={len(peak)} "
            f"final={len(final)}"
        )
    if not peak:
        raise ValueError("forgetting : empty per-task sequences")
    drops = [p - f for p, f in zip(peak, final, strict=True)]
    return sum(drops) / len(drops)
