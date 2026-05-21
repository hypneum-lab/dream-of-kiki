"""Internal package for the MLX latent-diffusion substrate skeleton.

Wave 3b M2 — skeleton only. Houses the three building blocks that
the public ``MLXLatentDiffusionSubstrate`` adapter wires together:

- ``Encoder``       (E) — awake activations → latent z
- ``MLPDenoiser``   (U) — reverse-process denoiser network
- ``NoiseSchedule`` (σ) — forward-process noise schedule

Training, sampling, and full DR-3 conformance wiring land in
Wave 3b M3+. See ``docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md``
§3.1 and §3.2 for the typed-primitives mapping.
"""
from __future__ import annotations

from kiki_oniric.substrates._diffusion.model import (
    Encoder,
    MLPDenoiser,
    NoiseSchedule,
)

__all__ = ["Encoder", "MLPDenoiser", "NoiseSchedule"]
