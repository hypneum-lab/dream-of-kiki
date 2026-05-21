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


# ---------------------------------------------------------------------------
# Module-level adapter classes (FIX 5: lifted from execute_profile body).
# ---------------------------------------------------------------------------

_DENOISER_ADAPTER_CLS: "type | None" = None


def _make_denoiser_adapter_class() -> type:
    """Return the nn.Module subclass for _DenoiserSingleArgAdapter.

    Deferred to a factory function so the ``mlx.nn`` import happens
    at call time rather than at module import time (preserves Linux
    CI importability where mlx is absent). The class object is built
    once and cached at module level; repeated ``execute_profile``
    calls reuse it (only the per-cell instance differs).
    """
    global _DENOISER_ADAPTER_CLS
    if _DENOISER_ADAPTER_CLS is not None:
        return _DENOISER_ADAPTER_CLS
    import mlx.core as mx
    import mlx.nn as _nn_raw
    from typing import Any as _Any, cast as _cast

    _nn: _Any = _cast(_Any, _nn_raw)

    class _Cls(_nn.Module):  # type: ignore[misc]
        """Thin nn.Module wrapper: model(x) → denoiser(x, t~Uniform[0,T)).

        Exposes model.layers so downscale_real / restructure_real
        can iterate and mutate weights in-place. The underlying
        denoiser's parameters are addressable via the standard
        nn.Module API (trainable_parameters, parameters, etc.)
        because they are stored as a sub-attribute.

        The diffusion timestep ``t`` is sampled uniformly from
        ``[0, schedule.t_steps)`` using a deterministic RNG key
        derived from the cell seed (R1 byte-stability).

        Because ``mx.random.randint`` cannot be called inside an
        MLX gradient trace, ``t`` is sampled OUTSIDE the
        differentiable forward pass via ``advance_timestep()``
        and cached as a plain Python-side ``mx.array`` in
        ``self._t_cache``.  The default ``_t_cache`` is zeros
        (matching the original M7 behaviour before this fix) and
        is overwritten by ``advance_timestep()`` before each
        replay episode loop.

        Args:
            denoiser: The underlying ``MLPDenoiser`` module.
            t_steps: Upper bound for timestep sampling (exclusive).
            rng_root: Seeded ``mx.random.key`` from
                ``execute_profile`` (``train_root``).
        """

        def __init__(
            self, denoiser: _Any, t_steps: int, rng_root: _Any
        ) -> None:
            super().__init__()
            self._denoiser = denoiser
            self._t_steps = t_steps
            self._rng_root = rng_root
            self._step_counter: int = 0
            # Default t cache: zeros (1,) — replaced by
            # advance_timestep() before each episode.
            self._t_cache: _Any = mx.zeros((1,), dtype=mx.int32)
            # Expose a shallow copy of the denoiser's layer list so
            # downscale_real_handler / restructure_real_handler can
            # iterate and swap ``model.layers`` without reordering the
            # denoiser's own internal layer list (which would break its
            # forward pass after a restructure swap).
            self.layers = list(denoiser.layers)

        def advance_timestep(self, batch: int) -> None:
            """Sample a fresh t for the next forward pass.

            Must be called OUTSIDE any gradient trace, once per
            episode, before the episode's replay loop.  The key
            is derived from ``_rng_root`` and ``_step_counter``
            so R1 byte-stability holds across re-runs.
            """
            # NOTE: this split allocates step_counter+2 keys and
            # indexes the last — O(n) in the episode count. Negligible
            # for the M7 4-batch workload; a fold-based O(1) derivation
            # would change the R1 key sequence and require regenerating
            # the diffusion golden hashes, so it is deferred.
            subkey = mx.random.split(
                self._rng_root, num=self._step_counter + 2
            )[self._step_counter]
            self._step_counter += 1
            self._t_cache = mx.random.randint(
                0, self._t_steps, (batch,), key=subkey,
            )

        def __call__(self, x: mx.array) -> mx.array:
            # Use the pre-sampled timestep; no random ops inside
            # the differentiable trace (MLX grad-trace safety).
            batch = x.shape[0] if x.ndim > 1 else 1
            t = self._t_cache
            if t.shape[0] == 1 and batch != 1:
                t = mx.broadcast_to(t, (batch,))
            out: mx.array = self._denoiser(x, t)
            return out

    _DENOISER_ADAPTER_CLS = _Cls
    return _Cls


class _LatentPassthroughVAEEncoder:
    """Returns ``(latent_as_mu, zeros_log_var)``: no re-encoding.

    log_var = zeros gives sigma = exp(0) = 1.0, so the
    reparameterisation in the recombine handler applies
    unit-variance Gaussian perturbation.  The noise is seeded
    deterministically by the recombine handler's ``seed``.
    The stochastic recombination is intentional and matches the
    framework-C recombine semantics.
    """

    def __call__(
        self, x: "mx.array"
    ) -> "tuple[mx.array, mx.array]":
        import mlx.core as mx

        if x.ndim == 1:
            x = x[None, :]
        log_var: "mx.array" = mx.zeros_like(x)
        return x, log_var


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

        self._diffusion_decoder: Decoder = Decoder(
            d_latent=config.d_latent, d_out=self._CIFAR_D_IN,
        )
        # NOTE: ClEvalHead is NOT stored on self.  A fresh head is
        # constructed at the start of each execute_profile call so
        # that delta_acc depends only on (seed, profile, task_idx)
        # and never on cell execution order (R1 order-independence,
        # FIX 1 — same class of bug as the M5 shared-substrate
        # weight leak).

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
            ClEvalHead,
            train_head_inplace,
            eval_head_accuracy,
        )

        seed = int(getattr(request, "seed", 0))
        profile_tag = str(getattr(request, "profile", "p_equ"))
        loader_batches = getattr(request, "loader_batches", ())

        root = mx.random.key(seed)
        # train_root: feeds the denoiser adapter's RNG (FIX 2).
        # data_root: used for synthetic dataset generation.
        # sample_root and head_root are not consumed in this milestone;
        # they are reserved for M5 (real sampler) and future per-cell
        # head seeding respectively.
        train_root, _sample_root, data_root, _head_root = (
            mx.random.split(root, num=4)
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

        # Fresh ClEvalHead per cell — do NOT reuse across calls.
        # This ensures delta_acc depends only on (seed, profile,
        # task_idx) and never on the order in which cells are run
        # (R1 order-independence; see __init__ NOTE).
        cl_head = ClEvalHead(
            d_latent=self.config.d_latent,
            n_classes=20,  # CIFAR-100 task-window size
        )

        # delta_acc baseline: train + eval the head BEFORE dream cycle.
        # All cells use the first batch as the eval slice for simplicity.
        baseline_acc = 0.0
        if dataset:
            train_head_inplace(
                cl_head, dataset[0], labels_per_batch[0],
            )
            baseline_acc = eval_head_accuracy(
                cl_head, dataset[0], labels_per_batch[0],
            )

        # Wrap the denoiser: model(x) → denoiser(x, t) where t is
        # sampled uniformly from [0, T) using train_root (FIX 2).
        # The adapter's RNG state is per-call (call counter resets
        # each execute_profile invocation) so R1 holds.
        _DenoiserSingleArgAdapter = _make_denoiser_adapter_class()
        model_adapter = _DenoiserSingleArgAdapter(
            self.denoiser,
            t_steps=self.config.t_steps,
            rng_root=train_root,
        )

        # The recombine_real_handler expects a VAEEncoder returning
        # (mu, log_var).  The delta_latents are already d_latent-
        # dimensional vectors from _encode_features.  Treat the latent
        # as mu; log_var = zeros gives unit-variance reparameterisation
        # seeded deterministically by the recombine handler's seed.
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
        n_layers = self.config.n_layers
        for batch_idx, latents in enumerate(dataset):
            # Sample a fresh diffusion timestep for this episode
            # outside the gradient trace (MLX grad-trace safety).
            # advance_timestep() uses train_root + a step counter so
            # each batch gets a different but deterministic t.
            model_adapter.advance_timestep(latents.shape[0])

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
            # Vary the restructure swap pair deterministically per
            # batch so restructure does non-degenerate work across
            # batches (FIX 6).  Indices are functions of batch_idx
            # only → deterministic, no RNG required.
            swap_a = batch_idx % n_layers
            swap_b = (batch_idx + 1) % n_layers
            episode = DreamEpisode(
                trigger=EpisodeTrigger.SCHEDULED,
                input_slice={
                    "beta_records": records,
                    "shrink_factor": 0.95,
                    "topo_op": "reroute",
                    "swap_indices": (swap_a, swap_b),
                    "delta_latents": [
                        latents[i] for i in range(latents.shape[0])
                    ],
                    "species": "diffusion",
                },
                operation_set=op_order,
                output_channels=(),
                budget=BudgetCap(
                    flops=10 ** 9,
                    wall_time_s=60.0,
                    energy_j=1.0,
                ),
                episode_id=(
                    f"diff/{profile_tag}/seed={seed}/b={batch_idx}"
                ),
            )
            profile.runtime.execute(episode)

        # delta_acc post-cycle: same (per-cell) head, eval again after
        # the dream.
        post_acc = 0.0
        if dataset:
            post_acc = eval_head_accuracy(
                cl_head, dataset[0], labels_per_batch[0],
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
        # _episode_count is a plain dataclass field on the recombine
        # handler's RealState, read intentionally as the recombine-
        # event count metric.  The leading underscore is the handler's
        # naming convention; the field is not a property (FIX 4).
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
