"""Cat F — asymmetric-channel violators.

2 substrates that violate the spirit of DR-3 substrate-agnosticism by
loading work asymmetrically across the 4 ops while structurally
satisfying the dict-shape contract.

Q2+ Task from HYPNEUM-PLANS/2026-05-11-niveau9-scaling-experiments.md.
"""
from __future__ import annotations

import time

from kiki_oniric.substrates.factory import CellRequest


class _BaseAdversarial:
    def teardown(self) -> None:
        return None


class AsymReplayHeavy(_BaseAdversarial):
    """All semantic load goes to replay ; other ops are dummies."""

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        return {
            "replay_rate": 0.95,
            "downscale_norm": 0.05,
            "restructure_sum": 0.001,
            "recombine_rate": 0.05,
            "delta_acc": 0.001,
            "wall_time_s": time.perf_counter() - t0,
        }


class AsymRecombineEmpty(_BaseAdversarial):
    """Recombine op never produces output ; other ops normal."""

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        return {
            "replay_rate": 0.50,
            "downscale_norm": 0.50,
            "restructure_sum": 0.05,
            "recombine_rate": 0.0,
            "delta_acc": 0.001,
            "wall_time_s": time.perf_counter() - t0,
        }
