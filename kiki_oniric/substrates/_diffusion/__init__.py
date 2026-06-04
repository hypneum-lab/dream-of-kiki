"""Internal package for the MLX latent-diffusion substrate.

Wave 3b M3 — building blocks + training loop + sampler. Houses
the components that the public ``MLXLatentDiffusionSubstrate``
adapter wires together :

- ``Encoder``       (E) — awake activations → latent z
- ``MLPDenoiser``   (U) — reverse-process denoiser network
- ``NoiseSchedule`` (σ) — forward-process noise schedule
- ``Sampler``           — DDPM reverse sampler (R1-clean keys)
- ``Trainer``           — noise-prediction SGD loop (R1-clean keys)

Full DR-3 conformance wiring lands via the public
``MLXLatentDiffusionSubstrate`` adapter ; see
``docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md``
§3.1 / §3.2 for the typed-primitives mapping.
"""
from __future__ import annotations

from kiki_oniric.substrates._diffusion.cl_eval_head import (
    ClEvalHead,
    eval_head_accuracy,
    train_head_inplace,
)
from kiki_oniric.substrates._diffusion.decoder import Decoder
from kiki_oniric.substrates._diffusion.dream_ops_adapter import (
    bind_real_handlers,
)
from kiki_oniric.substrates._diffusion.model import (
    Encoder,
    MLPDenoiser,
    NoiseSchedule,
)
from kiki_oniric.substrates._diffusion.sampler import Sampler
from kiki_oniric.substrates._diffusion.trainer import Trainer
from kiki_oniric.substrates._diffusion.denoiser_weight_channel import (
    DenoiserWeightDeltaChannel,
)
from kiki_oniric.substrates._diffusion.handlers_emit import (
    downscale_diffusion_handler,
    replay_diffusion_handler,
)
from kiki_oniric.substrates._diffusion.cl_eval_head import (
    denoiser_feature,
)

__all__ = [
    "ClEvalHead",
    "Decoder",
    "DenoiserWeightDeltaChannel",
    "Encoder",
    "MLPDenoiser",
    "NoiseSchedule",
    "Sampler",
    "Trainer",
    "bind_real_handlers",
    "denoiser_feature",
    "downscale_diffusion_handler",
    "eval_head_accuracy",
    "replay_diffusion_handler",
    "train_head_inplace",
]
