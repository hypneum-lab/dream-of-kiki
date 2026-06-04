"""Cat D — modular-arithmetic obfuscation substrates.

4 substrates that encode outputs via modular cycles (mod 7, mod 13,
gray-binary, affine mod N). Outputs are deterministic functions of
seed but encode no axiom-relevant computation.

Q2+ Task from HYPNEUM-PLANS/2026-05-11-niveau9-scaling-experiments.md.
"""
from __future__ import annotations

import time

from kiki_oniric.substrates.factory import CellRequest


class _BaseAdversarial:
    def teardown(self) -> None:
        return None


class Mod7Cycler(_BaseAdversarial):
    """Outputs cycle through 7 fixed value-tuples by seed mod 7."""

    _CYCLE = [
        (0.30, 0.50, 0.05, 0.55),
        (0.40, 0.45, 0.10, 0.60),
        (0.50, 0.60, 0.00, 0.40),
        (0.45, 0.55, 0.08, 0.50),
        (0.55, 0.50, 0.06, 0.45),
        (0.35, 0.65, 0.04, 0.65),
        (0.60, 0.40, 0.07, 0.35),
    ]

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        idx = request.seed % 7
        rr, dn, rs, rc = self._CYCLE[idx]
        return {
            "replay_rate": rr, "downscale_norm": dn,
            "restructure_sum": rs, "recombine_rate": rc,
            "delta_acc": 0.001 * (idx + 1),
            "wall_time_s": time.perf_counter() - t0,
        }


class Mod13Hasher(_BaseAdversarial):
    """Hashes seed mod 13 to deterministic bounded floats."""

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        h = request.seed % 13
        return {
            "replay_rate": (h * 7 % 100) / 100.0,
            "downscale_norm": (h * 11 % 100) / 100.0,
            "restructure_sum": (h * 3 % 10) / 100.0,
            "recombine_rate": (h * 5 % 100) / 100.0,
            "delta_acc": (h % 5) * 0.001,
            "wall_time_s": time.perf_counter() - t0,
        }


class BinaryGray(_BaseAdversarial):
    """Encodes seed via Gray-code-like reflective binary into outputs."""

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        g = request.seed ^ (request.seed >> 1)
        return {
            "replay_rate": (g & 0x7F) / 128.0,
            "downscale_norm": ((g >> 7) & 0x7F) / 128.0,
            "restructure_sum": ((g >> 14) & 0x07) / 100.0,
            "recombine_rate": ((g >> 3) & 0x7F) / 128.0,
            "delta_acc": ((g & 0xF) - 8) * 0.0005,
            "wall_time_s": time.perf_counter() - t0,
        }


class AffineModN(_BaseAdversarial):
    """Affine transform a*seed+b mod N for each metric."""

    _COEFFS = [(13, 7, 100), (17, 3, 100), (5, 1, 50), (11, 9, 100), (19, 5, 200)]

    def execute_profile(self, request: CellRequest) -> dict:
        t0 = time.perf_counter()
        a1, b1, m1 = self._COEFFS[0]
        a2, b2, m2 = self._COEFFS[1]
        a3, b3, m3 = self._COEFFS[2]
        a4, b4, m4 = self._COEFFS[3]
        a5, b5, m5 = self._COEFFS[4]
        s = request.seed
        return {
            "replay_rate": ((a1 * s + b1) % m1) / m1,
            "downscale_norm": ((a2 * s + b2) % m2) / m2,
            "restructure_sum": ((a3 * s + b3) % m3) / m3,
            "recombine_rate": ((a4 * s + b4) % m4) / m4,
            "delta_acc": ((a5 * s + b5) % m5) / m5 * 0.01,
            "wall_time_s": time.perf_counter() - t0,
        }
