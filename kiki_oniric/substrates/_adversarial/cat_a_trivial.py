"""Cat A — trivial-accident substrates.

5 substrates that pass axioms for degenerate reasons. Conformance
SHOULD reject all 5.

Q2 Task 6 from HYPNEUM-PLANS/2026-05-10-niveau8-three-experiments.md.
"""
from __future__ import annotations

import time

import numpy as np

from kiki_oniric.substrates.factory import CellRequest


class _BaseAdversarial:
    """Common no-op teardown for all Cat A substrates."""

    def teardown(self) -> None:
        return None


class IdentitySubstrate(_BaseAdversarial):
    """All ops = identity transform; no learning, no information flow."""

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        return {
            "replay_rate": 1.0,
            "downscale_norm": 1.0,
            "restructure_sum": 0.0,
            "recombine_rate": 1.0,
            "delta_acc": 0.0,
            "wall_time_s": time.perf_counter() - t0,
        }


class RandomNoiseSubstrate(_BaseAdversarial):
    """All ops = pure Gaussian noise tied to the request seed."""

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        rng = np.random.default_rng(request.seed)
        return {
            "replay_rate": float(rng.standard_normal()),
            "downscale_norm": float(abs(rng.standard_normal())),
            "restructure_sum": float(rng.standard_normal()),
            "recombine_rate": float(rng.uniform()),
            "delta_acc": float(rng.standard_normal() * 0.01),
            "wall_time_s": time.perf_counter() - t0,
        }


class LookupSubstrate(_BaseAdversarial):
    """Memoized lookup table; no computation, returns canned values."""

    _DEFAULT = {
        "replay_rate": 0.5,
        "downscale_norm": 0.5,
        "restructure_sum": 0.0,
        "recombine_rate": 0.5,
        "delta_acc": 0.0,
    }
    _RESPONSES = {
        ("p_min", 0): {
            "replay_rate": 0.5,
            "downscale_norm": 0.7,
            "restructure_sum": 0.0,
            "recombine_rate": 0.6,
            "delta_acc": 0.001,
        },
    }

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        profile = getattr(request, "profile", None)
        key = (profile, request.seed % 5)
        canned = self._RESPONSES.get(key, self._DEFAULT)
        return {**canned, "wall_time_s": time.perf_counter() - t0}


class FrozenZerosSubstrate(_BaseAdversarial):
    """All ops return zero; passes any axiom that tolerates trivial state."""

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        return {
            "replay_rate": 0.0,
            "downscale_norm": 0.0,
            "restructure_sum": 0.0,
            "recombine_rate": 0.0,
            "delta_acc": 0.0,
            "wall_time_s": time.perf_counter() - t0,
        }


class ConstantSubstrate(_BaseAdversarial):
    """All ops return the same non-zero constant regardless of input."""

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        return {
            "replay_rate": 0.42,
            "downscale_norm": 0.42,
            "restructure_sum": 0.42,
            "recombine_rate": 0.42,
            "delta_acc": 0.0042,
            "wall_time_s": time.perf_counter() - t0,
        }
