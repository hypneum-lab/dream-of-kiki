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
)

if TYPE_CHECKING:
    import mlx.core as mx  # noqa: F401 — used in string annotations below

    from kiki_oniric.substrates.factory import CellRequest


# Substrate identity (mirrors the cycle-1/2/3 substrate identity blocks).
MLX_LATENT_DIFFUSION_SUBSTRATE_NAME = "mlx_latent_diffusion"
# DualVer empirical axis : Wave 3b PARTIAL until the M5 full bench
# closes. Bump rules per framework-C §12.
# M2 → M3 substrate-internal MINOR : C-v0.13.0 → C-v0.14.0 (trainer +
# sampler + R1 entries shipped). EC stays PARTIAL ; M6 will flip to
# STABLE iff §12.3 conditions all hold.
MLX_LATENT_DIFFUSION_SUBSTRATE_VERSION = "C-v0.15.0+PARTIAL"

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

    # Raw CIFAR-100 input dimensionality: 32 × 32 × 3 = 3072.
    _CIFAR_D_IN: int = 3072

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
        # Separate encoder for raw CIFAR-100 features (3072 → d_latent).
        # Used by _encode_features when loader_batches are present.
        self._cifar_encoder: Encoder = Encoder(
            d_in=self._CIFAR_D_IN,
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
        from kiki_oniric.substrates._diffusion.decoder import Decoder
        from kiki_oniric.substrates._diffusion.cl_eval_head import (
            ClEvalHead,
        )
        self._diffusion_decoder: Decoder = Decoder(
            d_latent=config.d_latent, d_out=self._CIFAR_D_IN,
        )
        self._cl_head: ClEvalHead = ClEvalHead(
            d_latent=config.d_latent,
            n_classes=20,  # CIFAR-100 task-window size
        )

    # ------------------------------------------------------------------
    # SubstrateAdapter Protocol surface (factory.py).
    # ------------------------------------------------------------------

    def execute_profile(self, request: "CellRequest | object") -> dict[str, object]:
        """Run one ablation cell honouring framework-C §3.1.

        M7 wiring: instantiate the profile dataclass, override its
        skeleton handlers with the diffusion-bound _real.py handlers
        via dream_ops_adapter.bind_real_handlers, drive one
        DreamEpisode per loader batch through the profile's runtime,
        and snapshot the per-op metrics off the profile's *RealState
        fields.

        Synthetic fallback: when no loader_batches are provided, the
        method generates deterministic normal latents (same posture as
        M3) and drives the dream cycle over them — ensuring the
        existing R1 hashes for the synthetic path remain stable
        post-regeneration (Task 6).

        See docs/superpowers/specs/2026-05-21-m7-substrate-dr3-design.md
        §3 D3 + D7.
        """
        import mlx.core as mx
        from kiki_oniric.dream.episode import (
            BudgetCap, DreamEpisode, EpisodeTrigger, Operation,
        )
        from kiki_oniric.profiles.p_min import PMinProfile
        from kiki_oniric.profiles.p_equ import PEquProfile
        from kiki_oniric.profiles.p_max import PMaxProfile
        from kiki_oniric.substrates._diffusion.dream_ops_adapter import (
            bind_real_handlers,
        )
        from kiki_oniric.substrates._diffusion.cl_eval_head import (
            train_head_inplace, eval_head_accuracy,
        )

        seed = int(getattr(request, "seed", 0))
        profile_tag = str(getattr(request, "profile", "p_equ"))
        loader_batches = getattr(request, "loader_batches", ())

        root = mx.random.key(seed)
        train_root, sample_root, data_root, head_root = mx.random.split(
            root, num=4,
        )

        # Build the dataset: encoded latents (one per loader batch),
        # or a synthetic fallback identical to the M3 skeleton so
        # existing R1 hashes for the synthetic path stay aligned
        # (post-regeneration; see plan Task 6).
        if loader_batches:
            dataset = [
                self._encode_features(batch.features)
                for batch in loader_batches
            ]
            labels_per_batch = [batch.labels for batch in loader_batches]
            synthetic = False
        else:
            d_latent = self.config.d_latent
            n_batches = 4
            batch_size = 8
            data_keys = mx.random.split(data_root, num=n_batches)
            dataset = [
                mx.random.normal(shape=(batch_size, d_latent), key=k)
                for k in data_keys
            ]
            labels_per_batch = [
                mx.zeros((batch_size,), dtype=mx.int32)
                for _ in range(n_batches)
            ]
            synthetic = True

        # delta_acc baseline: train + eval the head BEFORE dream cycle.
        # All cells use the first batch as the eval slice for simplicity.
        baseline_acc = 0.0
        if dataset:
            train_head_inplace(
                self._cl_head, dataset[0], labels_per_batch[0],
            )
            baseline_acc = eval_head_accuracy(
                self._cl_head, dataset[0], labels_per_batch[0],
            )

        # Instantiate the profile and bind real handlers.
        # The replay_real_handler calls model(x) with a single arg;
        # MLPDenoiser requires (z, t). Wrap the denoiser in a thin
        # nn.Module subclass that injects a zero timestep so the
        # handler's loss_fn can call model(x) without modification.
        # We subclass nn.Module (via cast-to-Any alias to avoid mypy
        # issues with MLX's star-import resolution — mirrors the
        # pattern used in model.py and decoder.py).
        from typing import Any as _Any, cast as _cast
        import mlx.nn as _nn_raw
        _nn_any: _Any = _cast(_Any, _nn_raw)

        class _DenoiserSingleArgAdapter(_nn_any.Module):  # type: ignore[misc]
            """Thin nn.Module wrapper: model(x) → denoiser(x, t=0).

            Exposes model.layers so downscale_real / restructure_real
            can iterate and mutate weights in-place. The underlying
            denoiser's parameters are addressable via the standard
            nn.Module API (trainable_parameters, parameters, etc.)
            because they are stored as a sub-attribute.
            """

            def __init__(self, denoiser: _Any) -> None:
                super().__init__()
                self._denoiser = denoiser
                # Expose layers as a direct attribute so
                # downscale_real_handler / restructure_real_handler
                # can iterate ``model.layers``.
                self.layers = denoiser.layers

            def __call__(self, x: "mx.array") -> "mx.array":
                batch = x.shape[0] if x.ndim > 1 else 1
                t = mx.zeros((batch,), dtype=mx.int32)
                out: "mx.array" = self._denoiser(x, t)
                return out

        model_adapter = _DenoiserSingleArgAdapter(self.denoiser)

        # The recombine_real_handler expects a VAEEncoder that returns
        # (mu, log_var). The delta_latents are already encoded latents
        # (d_latent-dimensional vectors from _encode_features), not
        # raw features. Treat the latent directly as mu with zero
        # log_var so reparameterisation has zero variance — the latent
        # is deterministic and the VAEEncoder Protocol contract is
        # satisfied without a re-encoding pass.
        class _LatentPassthroughVAEEncoder:
            """Returns (latent_as_mu, zeros_log_var): no re-encoding."""

            def __call__(
                self, x: "mx.array"
            ) -> "tuple[mx.array, mx.array]":
                if x.ndim == 1:
                    x = x[None, :]
                log_var: "mx.array" = mx.zeros_like(x)
                return x, log_var

        vae_encoder = _LatentPassthroughVAEEncoder()

        profile_ctor = {
            "p_min": PMinProfile,
            "p_equ": PEquProfile,
            "p_max": PMaxProfile,
        }[profile_tag]
        profile = profile_ctor()
        activated = bind_real_handlers(
            profile, model=model_adapter,
            encoder=vae_encoder,
            decoder=self._diffusion_decoder, seed=seed,
        )

        t0 = time.perf_counter()

        # Drive one DreamEpisode per loader batch through the
        # profile's runtime. Episode input_slice carries every key
        # any handler might need; inactive ops are simply not in
        # operation_set.
        op_order = tuple(
            op for op in (
                Operation.REPLAY, Operation.DOWNSCALE,
                Operation.RESTRUCTURE, Operation.RECOMBINE,
            ) if op in activated
        )
        for batch_idx, latents in enumerate(dataset):
            # replay_real_handler computes MSE(model(x), y) so x and
            # y must be compatible with the model output shape. Since
            # the adapter outputs (batch, d_latent), y must also be
            # (d_latent). We use the latent itself as the reconstruction
            # target (self-supervised replay), consistent with the
            # spec's replay semantics (β-buffer gradient step on the
            # latent representation). Labels are available in
            # labels_per_batch but do not match the denoiser's output
            # shape — they are used for the CL eval head only.
            records = [
                {"x": latents[i], "y": latents[i]}
                for i in range(latents.shape[0])
            ]
            episode = DreamEpisode(
                trigger=EpisodeTrigger.SCHEDULED,
                input_slice={
                    "beta_records": records,
                    "shrink_factor": 0.95,
                    "topo_op": "reroute",
                    "swap_indices": (0, min(1, self.config.n_layers - 1)),
                    "delta_latents": [latents[i] for i in range(latents.shape[0])],
                    "species": "diffusion",
                },
                operation_set=op_order,
                output_channels=(),
                budget=BudgetCap(
                    flops=10 ** 9,
                    wall_time_s=60.0,
                    energy_j=1.0,
                ),
                episode_id=f"diff/{profile_tag}/seed={seed}/b={batch_idx}",
            )
            profile.runtime.execute(episode)

        # delta_acc post-cycle: same head, eval again after the dream.
        post_acc = 0.0
        if dataset:
            post_acc = eval_head_accuracy(
                self._cl_head, dataset[0], labels_per_batch[0],
            )

        wall = time.perf_counter() - t0

        # Read metrics off the profile's state fields. Inactive ops
        # return field defaults (legitimate zeros — that is the no-op
        # semantic per framework-C §3.1).
        replay_rate = float(profile.replay_state.last_loss or 0.0) \
            if hasattr(profile, "replay_state") else 0.0
        downscale_norm = float(profile.downscale_state.compound_factor) \
            if hasattr(profile, "downscale_state") else 1.0
        restructure_sum = int(profile.restructure_state.total_reroutes) \
            if hasattr(profile, "restructure_state") else 0
        recombine_rate = int(profile.recombine_state._episode_count) \
            if hasattr(profile, "recombine_state") else 0
        op_flops_total = sum(
            getattr(getattr(profile, f"{name}_state"), "last_compute_flops", 0)
            for name in ("replay", "downscale", "restructure", "recombine")
            if hasattr(profile, f"{name}_state")
        )

        return {
            "replay_rate": replay_rate,
            "downscale_norm": downscale_norm,
            "restructure_sum": restructure_sum,
            "recombine_rate": recombine_rate,
            "delta_acc": post_acc - baseline_acc,
            "op_flops_total": int(op_flops_total),
            "wall_time_s": wall,
            "synthetic": synthetic,
            "profile": profile_tag,
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
    # Internal helpers.
    # ------------------------------------------------------------------

    def _encode_features(self, features: "mx.array") -> "mx.array":
        """Project raw (B, 3072) CIFAR features to (B, d_latent).

        Routes through ``_cifar_encoder`` (Encoder sized
        3072 → config.d_latent) — distinct from ``self.encoder``
        which handles the default d_in=512 awake activations.
        """
        return self._cifar_encoder(features)

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
