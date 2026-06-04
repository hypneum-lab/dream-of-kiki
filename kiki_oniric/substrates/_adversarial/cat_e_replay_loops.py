"""Cat E — replay-loop violators.

4 substrates that produce structurally-correct outputs while violating
DR-1 episodic conservation (replay-history spoofing, beta permutation
leakage, alpha infinite recycling, gamma snapshot rollback).

Q2+ Task from HYPNEUM-PLANS/2026-05-11-niveau9-scaling-experiments.md.
"""
from __future__ import annotations

import time

from kiki_oniric.substrates.factory import CellRequest


class _BaseAdversarial:
    def teardown(self) -> None:
        return None


class ReplayHistorySpoof(_BaseAdversarial):
    """Returns plausible replay statistics that ignore actual history."""

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        return {
            "replay_rate": 0.42,
            "downscale_norm": 0.50,
            "restructure_sum": 0.05,
            "recombine_rate": 0.55,
            "delta_acc": 0.002,
            "wall_time_s": time.perf_counter() - t0,
        }


class BetaPermutationLeak(_BaseAdversarial):
    """Simulates beta-buffer permutation that leaks across episodes."""

    def __init__(self) -> None:
        self._call = 0

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        self._call += 1
        leak = (self._call % 10) / 10.0
        return {
            "replay_rate": 0.40 + leak * 0.05,
            "downscale_norm": 0.50,
            "restructure_sum": 0.04,
            "recombine_rate": 0.55,
            "delta_acc": 0.001 + leak * 0.002,
            "wall_time_s": time.perf_counter() - t0,
        }


class AlphaInfiniteRecycle(_BaseAdversarial):
    """Alpha stream recycles same content infinitely (no novelty)."""

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        return {
            "replay_rate": 0.60,
            "downscale_norm": 0.45,
            "restructure_sum": 0.00,
            "recombine_rate": 0.50,
            "delta_acc": 0.0,
            "wall_time_s": time.perf_counter() - t0,
        }


class GammaSnapshotRollback(_BaseAdversarial):
    """Gamma snapshot rolls back periodically, defeating consolidation."""

    def __init__(self) -> None:
        self._step = 0

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        self._step += 1
        rollback = (self._step % 5) == 0
        return {
            "replay_rate": 0.48,
            "downscale_norm": 0.50,
            "restructure_sum": 0.05,
            "recombine_rate": 0.50,
            "delta_acc": 0.0 if rollback else 0.002,
            "wall_time_s": time.perf_counter() - t0,
        }
