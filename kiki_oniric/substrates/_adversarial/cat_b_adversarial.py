"""Cat B — adversarial-construction substrates.

7 substrates engineered to trick specific axioms. Conformance SHOULD
reject all 7.

Q2 Task 7 from HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from kiki_oniric.substrates.factory import CellRequest


class _BaseAdversarial:
    """Common no-op teardown for Cat B substrates."""

    def teardown(self) -> None:
        return None


class ShapePreservingNoise(_BaseAdversarial):
    """Returns Gaussian noise with the right tensor shape but no
    semantic content. Defeats axioms that only check output shape, not
    value distribution."""

    def execute_profile(self, request: CellRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        rng = np.random.default_rng(request.seed)
        return {
            "replay_rate": float(np.clip(rng.standard_normal() * 0.5 + 0.5, 0, 1)),
            "downscale_norm": float(abs(rng.standard_normal()) + 0.1),
            "restructure_sum": float(rng.standard_normal() * 0.01),
            "recombine_rate": float(np.clip(rng.uniform(), 0.1, 0.9)),
            "delta_acc": float(rng.standard_normal() * 0.001),
            "wall_time_s": time.perf_counter() - t0,
        }


class BudgetGamingSubstrate(_BaseAdversarial):
    """Satisfies DR-2 budget-additivity by construction: deltas always
    sum to a fixed budget regardless of semantic content."""

    def execute_profile(self, request: CellRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        rng = np.random.default_rng(request.seed)
        # Dirichlet draw guarantees sum == 1.0 by construction
        deltas = rng.dirichlet([1.0, 1.0, 1.0, 1.0])
        return {
            "replay_rate": float(deltas[0]),
            "downscale_norm": float(deltas[1]),
            "restructure_sum": float(deltas[2]),
            "recombine_rate": float(deltas[3]),
            "delta_acc": 0.001,
            "wall_time_s": time.perf_counter() - t0,
        }


class PermutedReplay(_BaseAdversarial):
    """Replays the canonical positive substrate's outputs but with
    op-order permuted. Same per-op statistics, wrong dependency graph."""

    _CANNED = (0.5, 0.7, 0.05, 0.6)

    def execute_profile(self, request: CellRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        rng = np.random.default_rng(request.seed)
        permuted = rng.permutation(self._CANNED)
        return {
            "replay_rate": float(permuted[0]),
            "downscale_norm": float(permuted[1]),
            "restructure_sum": float(permuted[2]),
            "recombine_rate": float(permuted[3]),
            "delta_acc": 0.001,
            "wall_time_s": time.perf_counter() - t0,
        }


class OverloadRecombine(_BaseAdversarial):
    """Inflates recombine_rate to dominate the metric vector; passes
    any axiom that rewards 'high recombination' without checking source."""

    def execute_profile(self, request: CellRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        return {
            "replay_rate": 0.05,
            "downscale_norm": 0.1,
            "restructure_sum": 0.0,
            "recombine_rate": 0.99,  # overloaded
            "delta_acc": 0.001,
            "wall_time_s": time.perf_counter() - t0,
        }


class StatelessAccident(_BaseAdversarial):
    """No persistent state across calls; each invocation is independent.
    Defeats axioms that probe temporal coherence implicitly."""

    def execute_profile(self, request: CellRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        # Ignore request.seed entirely — pure stateless lookup
        return {
            "replay_rate": 0.5,
            "downscale_norm": 0.5,
            "restructure_sum": 0.0,
            "recombine_rate": 0.5,
            "delta_acc": 0.001,
            "wall_time_s": time.perf_counter() - t0,
        }


class CommutativityViolator(_BaseAdversarial):
    """Returns op outputs that depend on call order: replay-then-restructure
    differs from restructure-then-replay. Targets DR-2 compositionality."""

    def __init__(self) -> None:
        self._call_count = 0

    def execute_profile(self, request: CellRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        self._call_count += 1
        # Output depends on call ordinal — non-commutative under repeat
        phase = self._call_count % 2
        return {
            "replay_rate": 0.8 if phase == 0 else 0.2,
            "downscale_norm": 0.3 if phase == 0 else 0.7,
            "restructure_sum": float(phase * 0.05),
            "recombine_rate": 0.6 if phase == 0 else 0.4,
            "delta_acc": 0.001,
            "wall_time_s": time.perf_counter() - t0,
        }


class BoundaryCheater(_BaseAdversarial):
    """Returns values exactly at the axiom boundary thresholds — passes
    inclusive checks but the values carry no information."""

    def execute_profile(self, request: CellRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        return {
            "replay_rate": 0.0 + 1e-9,        # boundary of [0, 1]
            "downscale_norm": 1.0 - 1e-9,     # boundary
            "restructure_sum": 0.0,           # boundary of nonneg
            "recombine_rate": 0.5 + 1e-9,     # midpoint + epsilon
            "delta_acc": 0.0 + 1e-9,
            "wall_time_s": time.perf_counter() - t0,
        }
