"""Diffusion-eval harness package — Wave 3b M4.

Sibling of :mod:`harness.real_benchmarks`. Houses the Split-CIFAR-100
5-task loader (D1) consumed by :mod:`scripts.ablation_cycle3_diffusion`
and the M5 full bench, plus the diffusion-specific metric adapters.

References:
- ``docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md`` §3.1
  (module layout) + §2.2 D2 (corpus size).
"""
from __future__ import annotations

__version__ = "0.1.0"

__all__ = [
    "__version__",
]
