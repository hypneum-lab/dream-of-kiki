"""MLX latent-diffusion substrate adapter — Wave 3b M2 skeleton.

Fourth member of the substrate family, alongside
``mlx_kiki_oniric``, ``esnn_thalamocortical``, and ``micro_kiki``.
This file is the **public adapter** that wires the three building
blocks (``Encoder``, ``MLPDenoiser``, ``NoiseSchedule``) into the
substrate ABI consumed by ``DreamRuntime.execute()`` via the
``SubstrateAdapter`` Protocol defined in
``kiki_oniric/substrates/factory.py``.

Wave 3b M2 scope (this file):
- Public class ``MLXLatentDiffusionSubstrate`` instantiable with
  the D2 defaults (``d_latent=64``, ``n_layers=3``) per the plan.
- ``SubstrateAdapter`` Protocol surface present: ``execute_profile``
  and ``teardown`` typed at the skeleton level.
- ``NotImplementedError`` raised on any operation that requires
  training, sampling, or registry I/O — those land in Wave 3b M3+.

The Track S vs Track B decision (full DR-3 conformance vs
baseline-only adapter, per plan §1.2) is **deferred to PR review**.
This file is scope-neutral: it does not add conformance test
artefacts, and it does not commit to a baseline rebrand.

References:
- ``docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md``
  §3.1 (module layout) + §3.3 (factory integration) + §4 M2
- ``kiki_oniric/substrates/factory.py`` (``SubstrateAdapter`` Protocol)
- ``kiki_oniric/substrates/mlx_kiki_oniric.py`` (sibling pattern)
"""
from __future__ import annotations

from dataclasses import dataclass

from kiki_oniric.substrates._diffusion import (
    Encoder,
    MLPDenoiser,
    NoiseSchedule,
)


# Substrate identity (mirrors the cycle-1/2/3 substrate identity blocks).
MLX_LATENT_DIFFUSION_SUBSTRATE_NAME = "mlx_latent_diffusion"
# DualVer empirical axis : Wave 3b skeleton ships at PARTIAL until the
# M3 trainer + M5 full bench land. Bump rules per framework-C §12.
MLX_LATENT_DIFFUSION_SUBSTRATE_VERSION = "C-v0.13.0+PARTIAL"

# D2 defaults from the plan (§2.2, blocker D2 resolved at M1).
_DEFAULT_D_LATENT = 64
_DEFAULT_N_LAYERS = 3
_DEFAULT_D_HIDDEN = 256
_DEFAULT_D_IN = 512
_DEFAULT_T_STEPS = 1000
_DEFAULT_BETA_MIN = 1e-4
_DEFAULT_BETA_MAX = 2e-2


@dataclass(frozen=True)
class MLXLatentDiffusionConfig:
    """Frozen configuration for the latent-diffusion substrate.

    Captures the D2 hyper-parameters at adapter construction so
    the M3 trainer + M5 bench can pin them in the run registry.
    """

    d_in: int = _DEFAULT_D_IN
    d_latent: int = _DEFAULT_D_LATENT
    d_hidden: int = _DEFAULT_D_HIDDEN
    n_layers: int = _DEFAULT_N_LAYERS
    t_steps: int = _DEFAULT_T_STEPS
    beta_min: float = _DEFAULT_BETA_MIN
    beta_max: float = _DEFAULT_BETA_MAX


class MLXLatentDiffusionSubstrate:
    """Public substrate adapter — Wave 3b M2 skeleton.

    Wires ``Encoder`` (E), ``MLPDenoiser`` (U), and
    ``NoiseSchedule`` (σ) into one object that satisfies the
    ``SubstrateAdapter`` Protocol from
    ``kiki_oniric/substrates/factory.py``.

    The adapter is intentionally minimal at M2: the three
    components are instantiated eagerly so import-time errors
    surface (Protocol conformance, shape validation), but every
    Protocol method that requires training, sampling, or
    benchmark I/O raises ``NotImplementedError`` with a pointer
    to the milestone where it lands.

    Args:
        d_latent: Latent dimensionality. Plan D2 default = 64.
        n_layers: Number of hidden layers in ``MLPDenoiser``.
            Plan D2 default = 3.
        config: Optional full ``MLXLatentDiffusionConfig`` override.
            When provided, ``d_latent`` and ``n_layers`` are
            ignored (the config wins) — this lets M3 pin the full
            hyper-parameter set in one shot.
    """

    def __init__(
        self,
        d_latent: int = _DEFAULT_D_LATENT,
        n_layers: int = _DEFAULT_N_LAYERS,
        *,
        config: MLXLatentDiffusionConfig | None = None,
    ) -> None:
        if config is None:
            config = MLXLatentDiffusionConfig(
                d_latent=d_latent,
                n_layers=n_layers,
            )
        self.config: MLXLatentDiffusionConfig = config
        self.encoder: Encoder = Encoder(
            d_in=config.d_in,
            d_latent=config.d_latent,
        )
        self.denoiser: MLPDenoiser = MLPDenoiser(
            d_latent=config.d_latent,
            d_hidden=config.d_hidden,
            n_layers=config.n_layers,
        )
        self.schedule: NoiseSchedule = NoiseSchedule(
            t_steps=config.t_steps,
            beta_min=config.beta_min,
            beta_max=config.beta_max,
        )

    # ------------------------------------------------------------------
    # SubstrateAdapter Protocol surface (factory.py).
    # ------------------------------------------------------------------

    def execute_profile(self, request: object) -> dict[str, object]:
        """Run pre-eval → dream ops → post-eval for one cell.

        Wave 3b M2 skeleton: not implemented. The dream-ops
        wiring (replay / downscale / restructure / recombine
        translated into diffusion train + sample steps) lands
        in Wave 3b M3 with the trainer + sampler. See plan §4 M3.
        """
        raise NotImplementedError(
            "MLXLatentDiffusionSubstrate.execute_profile is deferred "
            "to Wave 3b M3 (trainer + sampler + ablation_cycle3 "
            "integration). M2 ships skeleton + Protocol conformance only."
        )

    def teardown(self) -> None:
        """Release MLX Metal buffers / close files.

        M2 skeleton: no resource is acquired beyond Python-level
        config dataclasses and the pre-computed ``NoiseSchedule``
        tables (small arrays in MLX). Nothing to release explicitly.
        M3 will tear down the trainer state and any open run-registry
        handles here.
        """
        return None

    # ------------------------------------------------------------------
    # Convenience accessors used by future M3 wiring and by the
    # M2 unit tests. Kept thin: no logic, just structured access.
    # ------------------------------------------------------------------

    def components(self) -> dict[str, str]:
        """Return the canonical map of substrate components.

        Mirrors ``mlx_substrate_components`` / ``esnn_substrate_components``
        for parity at the substrate-package level.
        """
        return {
            "encoder": "kiki_oniric.substrates._diffusion.model.Encoder",
            "denoiser": "kiki_oniric.substrates._diffusion.model.MLPDenoiser",
            "schedule": "kiki_oniric.substrates._diffusion.model.NoiseSchedule",
            "adapter": (
                "kiki_oniric.substrates.mlx_latent_diffusion."
                "MLXLatentDiffusionSubstrate"
            ),
        }


def mlx_latent_diffusion_substrate_components() -> dict[str, str]:
    """Return the canonical map of MLX latent-diffusion components.

    Free function variant mirroring the
    ``{mlx,esnn,micro_kiki,wake_sleep}_substrate_components``
    sibling functions, used by the substrates package
    ``__init__`` re-export surface.
    """
    return MLXLatentDiffusionSubstrate().components()
