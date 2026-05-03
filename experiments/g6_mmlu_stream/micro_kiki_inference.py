"""Path B inference-only adaptation shim for the G6 pilot.

When the run host lacks ``KIKI-Mac_tunner`` (the training pipeline +
``mlx_lm.lora`` fork), the G6 driver falls back to this Path B shim.
Per-subdomain "training" is replaced by a deterministic perturbation
of a synthetic LoRA delta tensor seeded by ``(subdomain, seed)``. The
shim does NOT actually fine-tune Qwen; it provides a state surface
on which the four MicroKikiSubstrate handlers can operate so the
dream-episode signature is observable end-to-end.

Any forgetting effect reported under Path B is **exploratory only** —
it reflects the dream-episode handlers' impact on a synthetic adapter,
not a true CL signal. The pre-reg explicitly forbids STABLE/UNSTABLE
promotion under Path B (see ``docs/osf-prereg-g6-pilot.md`` §6).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from numpy.typing import NDArray

from harness.real_benchmarks.mmlu import MMLURecord


def _stable_seed(*tokens: object) -> int:
    """Deterministic positive int seed from a tuple of hashable tokens.

    Cross-process stable (does not rely on builtins.hash, which is
    PYTHONHASHSEED-randomised by default).
    """
    raw = "|".join(repr(t) for t in tokens).encode("utf-8")
    return int(hashlib.sha256(raw).hexdigest()[:8], 16)


@dataclass
class InferenceOnlyAdapter:
    """Mock LoRA adapter buffer keyed by tensor name.

    Attributes
    ----------
    out_dim
        LoRA B-matrix output dimension. 4096 in production (Qwen
        hidden size); smaller in tests.
    rank
        LoRA rank.
    seed
        Pins the initial delta state.
    """

    out_dim: int
    rank: int
    seed: int
    _deltas: dict[str, NDArray[np.float32]] = field(
        default_factory=dict, init=False,
    )

    def current_delta(self, key: str) -> NDArray[np.float32]:
        """Return the current delta for ``key`` (zero-init on first read)."""
        if key not in self._deltas:
            self._deltas[key] = np.zeros(
                (self.out_dim, self.rank), dtype=np.float32,
            )
        return self._deltas[key]

    def set_delta(self, key: str, value: NDArray[np.float32]) -> None:
        if value.shape != (self.out_dim, self.rank):
            raise ValueError(
                f"delta shape {value.shape} != expected "
                f"({self.out_dim}, {self.rank})"
            )
        self._deltas[key] = value.astype(np.float32, copy=False)


def adapt_subdomain(
    *,
    adapter: InferenceOnlyAdapter,
    subdomain: str,
    train: Sequence[MMLURecord],
    seed: int,
    key: str = "layer_0_lora_B",
    step_magnitude: float = 0.05,
) -> None:
    """Apply a deterministic perturbation modelling per-subdomain LoRA training.

    The perturbation magnitude scales with the number of training
    records (linear) and is reproducible from ``(subdomain, seed,
    len(train))``. This is a stand-in for real LoRA training; the
    point is to give the dream-episode handlers a non-trivial state
    surface to operate on under Path B.

    Parameters
    ----------
    adapter
        InferenceOnlyAdapter to mutate.
    subdomain
        Subject name; folded into the seed so different subjects
        yield distinct perturbations.
    train
        Training records. Only ``len(train)`` is used (perturbation
        magnitude scales linearly).
    seed
        Cell-level seed.
    key
        LoRA tensor key to perturb.
    step_magnitude
        Per-record perturbation scale. Default 0.05 — small enough
        that several subdomains accumulate without saturating the
        bf16 numeric range, large enough that dream-handler
        downscale (factor 0.99) produces a measurable post-handler
        delta.
    """
    rng_seed = _stable_seed("g6-adapt", subdomain, seed, len(train))
    rng = np.random.default_rng(rng_seed)
    delta_step = (
        step_magnitude
        * len(train)
        * rng.standard_normal(
            (adapter.out_dim, adapter.rank),
        ).astype(np.float32)
    )
    cur = adapter.current_delta(key)
    adapter.set_delta(key, cur + delta_step)


__all__ = ["InferenceOnlyAdapter", "adapt_subdomain"]
