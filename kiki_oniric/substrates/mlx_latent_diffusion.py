"""MLX latent-diffusion substrate adapter — Wave 3b M3.

Fourth member of the substrate family, alongside
``mlx_kiki_oniric``, ``esnn_thalamocortical``, and ``micro_kiki``.
This file is the **public adapter** that wires the building
blocks (``Encoder``, ``MLPDenoiser``, ``NoiseSchedule``,
``Trainer``, ``Sampler``) into the substrate ABI consumed by
``DreamRuntime.execute()`` via the ``SubstrateAdapter`` Protocol
defined in ``kiki_oniric/substrates/factory.py``.

Wave 3b M3 scope (this file):
- Public class ``MLXLatentDiffusionSubstrate`` instantiable with
  the D2 defaults (``d_latent=64``, ``n_layers=3``) per the plan.
- ``SubstrateAdapter`` Protocol surface ``execute_profile`` /
  ``teardown`` now wires a real (short) training + sampling pass
  for ablation_cycle3 smoke runs (M4 will plug the CIFAR-100
  loader ; M3 ships a deterministic synthetic-latent driver so
  the substrate is exercised end-to-end and R1-hashable today).
- ``Track S`` chosen at the M2 review : DR-3 conformance tests
  land under ``tests/conformance/axioms/`` in this PR.

References:
- ``docs/plans/2026-05-20-wave3b-mlx-diffusion-substrate-plan.md``
  §3.1 (module layout) + §3.3 (factory integration) + §4 M3
- ``kiki_oniric/substrates/factory.py`` (``SubstrateAdapter`` Protocol)
- ``kiki_oniric/substrates/mlx_kiki_oniric.py`` (sibling pattern)
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from typing import TYPE_CHECKING

from kiki_oniric.substrates._diffusion import (
    Encoder,
    MLPDenoiser,
    NoiseSchedule,
    Sampler,
    Trainer,
)

if TYPE_CHECKING:
    from kiki_oniric.substrates.factory import CellRequest


# Substrate identity (mirrors the cycle-1/2/3 substrate identity blocks).
MLX_LATENT_DIFFUSION_SUBSTRATE_NAME = "mlx_latent_diffusion"
# DualVer empirical axis : Wave 3b PARTIAL until the M5 full bench
# closes. Bump rules per framework-C §12.
# M2 → M3 substrate-internal MINOR : C-v0.13.0 → C-v0.14.0 (trainer +
# sampler + R1 entries shipped). EC stays PARTIAL ; M6 will flip to
# STABLE iff §12.3 conditions all hold.
MLX_LATENT_DIFFUSION_SUBSTRATE_VERSION = "C-v0.14.0+PARTIAL"

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

    def execute_profile(self, request: "CellRequest | object") -> dict[str, object]:
        """Run a minimal train + sample cycle for one ablation cell.

        Wave 3b M3 wiring (synthetic-latent driver, M4 will swap in
        the CIFAR-100 loader). The cycle :

        1. Seed MLX RNG from ``request.seed`` so the synthetic
           training latents and the per-step subkey trees are
           deterministic.
        2. Build a small synthetic dataset of normal latents.
        3. Run a short :class:`Trainer.fit` on the denoiser
           (noise-prediction MSE).
        4. Draw one reverse-process sample via :class:`Sampler`.
        5. Return a metrics dict shaped like the sibling
           substrates (``replay_rate`` / ``downscale_norm`` etc.
           proxied from the trainer + sampler state) so the
           ablation_cycle3 row reducer accepts it without a
           schema branch.

        The output is intentionally a self-contained smoke trace
        — the bench-grade objective lands in M5. The
        ``synthetic`` marker in the returned dict makes that
        explicit per CLAUDE.md §Working rules item 3.
        """
        import mlx.core as mx

        seed = int(getattr(request, "seed", 0))
        # Per the R1 contract : derive the training and sampling
        # root keys from one split of the seed key. Never consume
        # a raw mx.random.key(seed) directly with mx.random.normal.
        root = mx.random.key(seed)
        train_root, sample_root, data_root = mx.random.split(root, num=3)

        # Tiny synthetic latent dataset : 4 batches × batch_size 8.
        # Held small so the M3 smoke fits in seconds on M5.
        d_latent = self.config.d_latent
        n_batches = 4
        batch_size = 8
        data_keys = mx.random.split(data_root, num=n_batches)
        dataset = [
            mx.random.normal(shape=(batch_size, d_latent), key=k)
            for k in data_keys
        ]

        t0 = time.perf_counter()

        trainer = Trainer(
            model=self.denoiser,
            schedule=self.schedule,
            optimizer_kwargs={"lr": 1e-3},
        )
        history = trainer.fit(
            dataset=dataset, n_epochs=1, seed_key=train_root
        )
        losses = history["loss"]

        sampler = Sampler(model=self.denoiser, schedule=self.schedule)
        sample = sampler.sample(
            key=sample_root,
            n_steps=min(8, self.schedule.t_steps),
            shape=(1, d_latent),
        )

        wall = time.perf_counter() - t0
        # Convert to plain Python floats so the row reducer can
        # serialize without an MLX dependency.
        loss_first = float(losses[0]) if losses else 0.0
        loss_last = float(losses[-1]) if losses else 0.0
        sample_norm = float(mx.sqrt(mx.sum(sample * sample)).item())

        return {
            "replay_rate": loss_last,       # proxy : trained denoiser loss
            "downscale_norm": sample_norm,  # proxy : sample magnitude
            "restructure_sum": 0.0,         # canal 3 inactive in smoke
            "recombine_rate": float(len(losses)),
            "delta_acc": loss_first - loss_last,
            "wall_time_s": wall,
            "synthetic": True,
            "profile": getattr(request, "profile", "unknown"),
            "seed": seed,
            "substrate": MLX_LATENT_DIFFUSION_SUBSTRATE_NAME,
            "substrate_version": MLX_LATENT_DIFFUSION_SUBSTRATE_VERSION,
        }

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
