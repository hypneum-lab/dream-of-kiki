"""DR-3 Conformance Criterion — MLX latent-diffusion substrate (Wave 3b).

Validates that the latent-diffusion substrate satisfies the 3
conditions of the DR-3 Conformance Criterion (framework-C spec
§6.2) :

1. **Signature typing** — substrate exports the identity constants
   and a callable adapter ; the 7/8 typed primitives mapping (plan
   §3.2) is exercised by walking the public surface
   ``MLXLatentDiffusionSubstrate.execute_profile`` returns. Canal 4
   is the documented no-op (flat latent space) and is asserted as
   such — the criterion permits a *typed-but-empty* return for the
   no-op primitive (framework-C §6.2).

2. **Axiom property tests** — DR-0 budget accountability is tested
   via the smoke cycle's deterministic step-count ; DR-1 finite-loss
   propagation is tested via the loss-history return ; DR-2'
   canonical-order determinism is tested via repeated identical
   execution.

3. **BLOCKING invariants enforceable** — S2 (finite values) on the
   trainer loss and the sampler output ; S3 (topology) is N/A for
   this substrate (no graph mutation), explicitly documented.

Reference : docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import pytest

mx = pytest.importorskip("mlx.core")

from kiki_oniric.substrates._diffusion import (  # noqa: E402
    Encoder,
    MLPDenoiser,
    NoiseSchedule,
    Sampler,
    Trainer,
)
from kiki_oniric.substrates.factory import (  # noqa: E402
    SUBSTRATE_NAMES,
    build_substrate_adapter,
)
from kiki_oniric.substrates.mlx_latent_diffusion import (  # noqa: E402
    MLX_LATENT_DIFFUSION_SUBSTRATE_NAME,
    MLX_LATENT_DIFFUSION_SUBSTRATE_VERSION,
    MLXLatentDiffusionSubstrate,
)


@dataclass
class _StubRequest:
    seed: int = 7
    profile: str = "p_min"


# ===== Condition 1 : signature typing =====


def test_c1_substrate_exports_required_identity_constants() -> None:
    """Identity constants follow the substrate naming convention."""
    assert MLX_LATENT_DIFFUSION_SUBSTRATE_NAME == "mlx_latent_diffusion"
    assert MLX_LATENT_DIFFUSION_SUBSTRATE_VERSION.startswith("C-v")
    assert MLX_LATENT_DIFFUSION_SUBSTRATE_VERSION.endswith("+PARTIAL")
    assert MLX_LATENT_DIFFUSION_SUBSTRATE_NAME in SUBSTRATE_NAMES


def test_c1_factory_returns_substrate_for_canonical_name() -> None:
    """Factory dispatch wires the substrate by its canonical name."""
    adapter = build_substrate_adapter("mlx_latent_diffusion")
    assert isinstance(adapter, MLXLatentDiffusionSubstrate)


def test_c1_typed_primitives_7_of_8() -> None:
    """The 7 typed primitives (plan §3.2) each have a callable
    correspondent in the substrate surface. The 8th (Canal 4,
    ``AttentionPriorChannel``) is the documented no-op.

    We exercise each typed primitive by asserting the relevant
    M3 building block exposes the expected callable :

    * α (AlphaStreamProtocol)    → ``Encoder.__call__``
    * β (BetaBufferProtocol)     → adapter-level dataset feed (the
                                    Trainer accepts batches of latents
                                    consumed in FIFO order)
    * γ (GammaSnapshotProtocol)  → ``MLPDenoiser`` parameter readout
                                    (snapshotable via .parameters())
    * δ (DeltaLatentsProtocol)   → ``MLPDenoiser`` internal activations
                                    available via the forward call
    * Canal 1 (WeightDelta)      → ``MLPDenoiser.train_step``
    * Canal 2 (LatentSample)     → ``Sampler.sample``
    * Canal 3 (HierarchyChange)  → adapter-level restructure no-op
                                    (typed at the substrate Protocol
                                    surface ; the latent-MLP has no
                                    explicit graph but the contract
                                    accepts an empty diff list)
    """
    adapter = MLXLatentDiffusionSubstrate()
    # α — encoder forward call.
    assert callable(adapter.encoder.__call__)
    # γ — denoiser parameters are readable for snapshotting.
    params = adapter.denoiser.parameters()
    assert params is not None
    # δ — denoiser forward call.
    assert callable(adapter.denoiser.__call__)
    # Canal 1 — denoiser training step.
    assert callable(adapter.denoiser.train_step)
    # Canal 2 — sampler is constructible from the adapter's
    # building blocks.
    sampler = Sampler(model=adapter.denoiser, schedule=adapter.schedule)
    assert callable(sampler.sample)
    # β — trainer accepts a batch dataset (FIFO buffer surrogate).
    trainer = Trainer(
        model=adapter.denoiser,
        schedule=adapter.schedule,
    )
    assert callable(trainer.fit)
    assert callable(trainer.train_step)
    # Canal 3 — empty topology-diff acceptance : the adapter's
    # public surface returns a "restructure_sum" key set to 0.0
    # in the M3 smoke cycle, mirroring the typed-but-empty
    # contract for a flat latent space.
    out = adapter.execute_profile(_StubRequest())
    assert "restructure_sum" in out
    assert out["restructure_sum"] == 0.0


def test_c1_canal_4_attention_prior_is_documented_noop() -> None:
    """Canal 4 (AttentionPriorChannel) is the documented no-op.

    Per plan §3.2 the substrate scores 7/8 typed + 1/8 no-op. The
    Canal 4 absence is allowed by framework-C §6.2 provided the
    no-op is explicitly documented. We assert here that the
    adapter does NOT advertise an attention prior in its
    ``execute_profile`` return — this is the empirical witness of
    the no-op contract.
    """
    adapter = MLXLatentDiffusionSubstrate()
    out = adapter.execute_profile(_StubRequest())
    assert "attention_prior" not in out
    # And that the components map does not list one either.
    comps = adapter.components()
    assert "attention_prior" not in comps


# ===== Condition 2 : axiom property tests =====


def test_c2_dr0_budget_accountability_via_loss_history() -> None:
    """DR-0 : every training step is accountable in the loss history.

    The Trainer's ``fit`` MUST return a loss list of length
    ``n_epochs * steps_per_epoch`` — one entry per executed DE
    (interpreted as one training step). This is the substrate-side
    translation of DR-0 (every DE logged).
    """
    mx.random.seed(0)
    denoiser = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    schedule = NoiseSchedule(t_steps=20, beta_min=1e-4, beta_max=2e-2)
    trainer = Trainer(model=denoiser, schedule=schedule)
    root = mx.random.key(7)
    data_keys = mx.random.split(root, num=3)
    dataset = [
        mx.random.normal(shape=(2, 4), key=k) for k in data_keys
    ]
    history = trainer.fit(dataset=dataset, n_epochs=2, seed_key=root)
    losses = history["loss"]
    assert len(losses) == 2 * 3  # n_epochs * steps_per_epoch
    for loss in losses:
        assert math.isfinite(loss)


def test_c2_dr2_prime_deterministic_canonical_order() -> None:
    """DR-2' : repeated execution under the same canonical order
    yields the same trace (loss list).

    This is the substrate-side translation of DR-2' (strict
    canonical-order determinism) : two independent runs with
    identical seeds + identical dataset produce the same
    loss-list-shaped trace.
    """
    def _run() -> list[float]:
        mx.random.seed(0)
        denoiser = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
        schedule = NoiseSchedule(t_steps=20, beta_min=1e-4, beta_max=2e-2)
        trainer = Trainer(model=denoiser, schedule=schedule)
        root = mx.random.key(7)
        data_keys = mx.random.split(root, num=2)
        dataset = [
            mx.random.normal(shape=(2, 4), key=k) for k in data_keys
        ]
        return trainer.fit(dataset=dataset, n_epochs=1, seed_key=root)["loss"]

    a = _run()
    b = _run()
    assert a == b


# ===== Condition 3 : BLOCKING invariants enforceable =====


def test_c3_s2_finite_guard_on_sampler_output() -> None:
    """S2 invariant : the sampler output must be finite.

    Mirrors the S2 finite check used across the sibling substrates
    (cf. ``test_dr3_micro_kiki_substrate.py::test_c3_s2_finite_*``).
    """
    import numpy as np

    mx.random.seed(0)
    denoiser = MLPDenoiser(d_latent=4, d_hidden=8, n_layers=2)
    schedule = NoiseSchedule(t_steps=10, beta_min=1e-4, beta_max=2e-2)
    sampler = Sampler(model=denoiser, schedule=schedule)
    out = sampler.sample(key=mx.random.key(0), n_steps=4, shape=(2, 4))
    arr = np.asarray(out)
    assert np.all(np.isfinite(arr)), "S2 violated : sampler output non-finite"


def test_c3_encoder_train_step_returns_finite_loss() -> None:
    """S2 invariant : encoder train_step loss must be finite."""
    mx.random.seed(0)
    enc = Encoder(d_in=8, d_latent=4)
    x = mx.random.normal(shape=(3, 8), key=mx.random.key(0))
    loss = enc.train_step(x, lr=1e-3)
    assert math.isfinite(loss), "S2 violated : encoder train_step non-finite"
    assert loss >= 0.0
