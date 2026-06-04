"""Cat C — statistical-accident substrates.

3 substrates that may pass conformance with non-trivial probability on
specific seed/profile distributions. Each tested with N=100 trials in
Task 9; conformance SHOULD reject ≥99/100 per substrate.

Q2 Task 8 from HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md.
"""
from __future__ import annotations

import time
from typing import Any

import numpy as np

from kiki_oniric.substrates.factory import CellRequest


class _BaseAdversarial:
    """Common no-op teardown for Cat C substrates."""

    def teardown(self) -> None:
        return None


class RandomCoinFlip(_BaseAdversarial):
    """Each op has 50% chance to satisfy axiom, 50% to fail.

    Per-trial outputs are independent. Expected all-pass rate over 5
    independent ops is 0.5^5 = 3.125%. With N=100 trials we expect
    ~3 false positives on any axiom that doesn't probe enough ops.
    Conformance suite SHOULD reject ≥99/100 trials.
    """

    def execute_profile(self, request: CellRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        # Use request.seed as the per-trial seed so N=100 calls give
        # 100 independent draws when the audit harness varies the seed.
        rng = np.random.default_rng(request.seed)

        def good_or_bad(good: float, bad: float) -> float:
            return good if rng.random() > 0.5 else bad

        return {
            "replay_rate": good_or_bad(0.7, float("nan")),
            "downscale_norm": good_or_bad(0.5, -1.0),
            "restructure_sum": good_or_bad(0.0, float("inf")),
            "recombine_rate": good_or_bad(0.6, 2.0),
            "delta_acc": good_or_bad(0.001, float("nan")),
            "wall_time_s": time.perf_counter() - t0,
        }


class ShapeDistributionDependent(_BaseAdversarial):
    """Passes on the common 80% of input distribution, fails on
    long-tail.

    Modulates pass/fail behaviour by `seed % 10`: seeds 0..7 pass,
    seeds 8..9 fail. Over 100 trials with random seeds, expect ~80%
    superficial pass; conformance suite SHOULD detect the tail
    failures and reject ≥99/100.
    """

    def execute_profile(self, request: CellRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        bucket = request.seed % 10
        if bucket < 8:  # 80% common case
            return {
                "replay_rate": 0.5,
                "downscale_norm": 0.5,
                "restructure_sum": 0.0,
                "recombine_rate": 0.5,
                "delta_acc": 0.001,
                "wall_time_s": time.perf_counter() - t0,
            }
        # 20% tail: violates axiom invariants
        return {
            "replay_rate": float("inf"),
            "downscale_norm": -1.0,
            "restructure_sum": float("nan"),
            "recombine_rate": 2.0,
            "delta_acc": float("nan"),
            "wall_time_s": time.perf_counter() - t0,
        }


class SeedDependentSubstrate(_BaseAdversarial):
    """Passes only on seed 0; behaviour diverges for all other seeds.

    Crafted to fool any conformance check that happens to use seed 0
    as its sole sample. With N=100 trials varying seeds, we expect
    1/100 to look conformant and 99/100 to fail; suite SHOULD reject
    ≥99/100.
    """

    def execute_profile(self, request: CellRequest) -> dict[str, Any]:
        t0 = time.perf_counter()
        if request.seed == 0:
            return {
                "replay_rate": 0.5,
                "downscale_norm": 0.5,
                "restructure_sum": 0.0,
                "recombine_rate": 0.5,
                "delta_acc": 0.001,
                "wall_time_s": time.perf_counter() - t0,
            }
        # Other seeds: garbage scaled to ensure axiom violation
        rng = np.random.default_rng(request.seed)
        return {
            "replay_rate": float(rng.standard_normal() * 100.0),
            "downscale_norm": float(rng.standard_normal() * 100.0),
            "restructure_sum": float(rng.standard_normal() * 100.0),
            "recombine_rate": float(rng.standard_normal() * 100.0),
            "delta_acc": float(rng.standard_normal()),
            "wall_time_s": time.perf_counter() - t0,
        }
