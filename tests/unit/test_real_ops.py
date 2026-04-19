"""Unit tests for cycle-3 C3.3 real-weight dream operations.

Covers four modules in :mod:`kiki_oniric.dream.operations` :

- ``replay_real`` — gradient replay with real-weight K1 tagging
- ``downscale_real`` — SHY homeostasis with real-weight K1 tagging
- ``restructure_real`` — topology mutation with S3 guard reuse
- ``recombine_real`` — VAE-light recombine with S2 finite check

Each op is tested against the 4 invariants mandated by the cycle-3
plan (collapsed here into 2 tests per op for clarity) :

1. DR-2 canonical order preserved when chaining the 4 ops in a
   single runtime.
2. S1 budget respected — compute_flops tag on every DE log entry.
3. Channel 1-4 emission typed — input_slice carries the right
   schema per §2.1 (α/β/γ/δ).
4. Weight-tensor shape invariant — the op does not corrupt the
   wrapper's model weights shape.

Tests are MLX-backed but tiny-scale (4-dim, 8 params) so the suite
stays deterministic and under 1 s.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

# Deterministic seed for MLX random init across all tests.
mx.random.seed(7)

from kiki_oniric.dream.episode import (  # noqa: E402
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.guards.finite import FiniteGuardError  # noqa: E402
from kiki_oniric.dream.operations.downscale_real import (  # noqa: E402
    DownscaleRealState,
    downscale_real_handler,
)
from kiki_oniric.dream.operations.recombine_real import (  # noqa: E402
    RecombineRealState,
    recombine_real_handler,
)
from kiki_oniric.dream.operations.replay_real import (  # noqa: E402
    ReplayRealState,
    replay_real_handler,
)
from kiki_oniric.dream.operations.restructure_real import (  # noqa: E402
    RestructureRealState,
    restructure_real_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime  # noqa: E402


# --------------------------------------------------------------------------
# Tiny MLP fixture for the real-weight ops — 4-input / 8-hidden / 2-output.
# --------------------------------------------------------------------------


class _TinyMLP(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = [nn.Linear(4, 8), nn.Linear(8, 2)]
        self.input_dim = 4

    def __call__(self, x):
        h = nn.relu(self.layers[0](x))
        return self.layers[1](h)


class _TinyEncoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def __call__(self, x):
        h = self.fc(x)
        mu = h
        log_var = h * 0.0  # zero log-var ⇒ deterministic sigma = 1
        return mu, log_var


class _TinyDecoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def __call__(self, z):
        return self.fc(z)


def _make_episode(
    ep_id: str,
    input_slice: dict,
    operations: tuple[Operation, ...],
    channels: tuple[OutputChannel, ...],
    flops: int = 1_000_000,
) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice=input_slice,
        operation_set=operations,
        output_channels=channels,
        budget=BudgetCap(flops=flops, wall_time_s=1.0, energy_j=0.1),
        episode_id=ep_id,
    )


# --------------------------------------------------------------------------
# Test 1 — DR-2 canonical order replay → downscale → restructure → recombine
# (invariant 1 : order preserved when chained in one DreamRuntime run)
# --------------------------------------------------------------------------


def test_dr2_canonical_order_preserved_across_real_ops() -> None:
    """The four real-weight ops respect DR-2 canonical order.

    canonical order = replay → downscale → restructure → recombine
    (cf. docs/proofs/op-pair-analysis.md). The DreamRuntime log
    records them in that order when the DE lists them accordingly.
    """
    model = _TinyMLP()
    encoder, decoder = _TinyEncoder(), _TinyDecoder()

    replay_state = ReplayRealState()
    downscale_state = DownscaleRealState()
    restructure_state = RestructureRealState()
    recombine_state = RecombineRealState()

    rt = DreamRuntime()
    rt.register_handler(
        Operation.REPLAY,
        replay_real_handler(replay_state, model=model, lr=0.01),
    )
    rt.register_handler(
        Operation.DOWNSCALE,
        downscale_real_handler(downscale_state, model=model),
    )
    rt.register_handler(
        Operation.RESTRUCTURE,
        restructure_real_handler(restructure_state, model=model),
    )
    rt.register_handler(
        Operation.RECOMBINE,
        recombine_real_handler(
            recombine_state, encoder=encoder, decoder=decoder, seed=0
        ),
    )

    ep = _make_episode(
        "de-order",
        input_slice={
            "beta_records": [
                {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
            ],
            "shrink_factor": 0.99,
            "topo_op": "reroute",
            "swap_indices": [0, 1],
            "delta_latents": [[0.1, 0.2, 0.3, 0.4], [0.2, 0.3, 0.4, 0.5]],
        },
        operations=(
            Operation.REPLAY,
            Operation.DOWNSCALE,
            Operation.RESTRUCTURE,
            Operation.RECOMBINE,
        ),
        channels=(
            OutputChannel.WEIGHT_DELTA,
            OutputChannel.HIERARCHY_CHG,
            OutputChannel.LATENT_SAMPLE,
        ),
    )
    rt.execute(ep)
    assert len(rt.log) == 1
    entry = rt.log[0]
    assert entry.completed is True
    assert entry.operations_executed == (
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    )


# --------------------------------------------------------------------------
# Test 2 — K1 compute_flops tag on every real op (invariant 2 : budget)
# --------------------------------------------------------------------------


def test_k1_compute_flops_tagged_on_every_real_op() -> None:
    """Every real-weight op tags compute_flops on its state.

    K1 : dream-episodes are budget-bounded. A real op that does not
    tag a FLOP estimate cannot be scheduled safely.
    """
    model = _TinyMLP()
    encoder, decoder = _TinyEncoder(), _TinyDecoder()

    rs = ReplayRealState()
    ds = DownscaleRealState()
    ts = RestructureRealState()
    cs = RecombineRealState()

    replay = replay_real_handler(rs, model=model, lr=0.01)
    downscale = downscale_real_handler(ds, model=model)
    restructure = restructure_real_handler(ts, model=model)
    recombine = recombine_real_handler(
        cs, encoder=encoder, decoder=decoder, seed=0
    )

    replay(
        _make_episode(
            "de-r",
            {
                "beta_records": [
                    {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
                ]
            },
            (Operation.REPLAY,),
            (OutputChannel.WEIGHT_DELTA,),
        )
    )
    downscale(
        _make_episode(
            "de-d",
            {"shrink_factor": 0.95},
            (Operation.DOWNSCALE,),
            (OutputChannel.WEIGHT_DELTA,),
        )
    )
    restructure(
        _make_episode(
            "de-s",
            {"topo_op": "reroute", "swap_indices": [0, 1]},
            (Operation.RESTRUCTURE,),
            (OutputChannel.HIERARCHY_CHG,),
        )
    )
    recombine(
        _make_episode(
            "de-c",
            {
                "delta_latents": [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.2, 0.3, 0.4, 0.5],
                ]
            },
            (Operation.RECOMBINE,),
            (OutputChannel.LATENT_SAMPLE,),
        )
    )

    assert rs.last_compute_flops > 0
    assert ds.last_compute_flops > 0
    assert ts.last_compute_flops > 0
    assert cs.last_compute_flops > 0
    # Totals accumulate across calls.
    assert rs.total_compute_flops == rs.last_compute_flops


# --------------------------------------------------------------------------
# Test 3 — Channel emission typed per §2.1 (invariant 3 : channel typing)
# --------------------------------------------------------------------------


def test_channel_emission_typed_per_spec() -> None:
    """Channels 1-4 : each real op populates the right state field.

    - replay_real → canal 1 WEIGHT_DELTA : state tracks last_loss
    - downscale_real → canal 1 WEIGHT_DELTA : state tracks compound_factor
    - restructure_real → canal 3 HIERARCHY_CHG : state appends to diff_history
    - recombine_real → canal 2 LATENT_SAMPLE : state populates last_sample
    """
    model = _TinyMLP()
    encoder, decoder = _TinyEncoder(), _TinyDecoder()

    rs = ReplayRealState()
    ds = DownscaleRealState()
    ts = RestructureRealState()
    cs = RecombineRealState()

    replay_real_handler(rs, model=model, lr=0.01)(
        _make_episode(
            "de-r",
            {"beta_records": [{"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]}]},
            (Operation.REPLAY,),
            (OutputChannel.WEIGHT_DELTA,),
        )
    )
    downscale_real_handler(ds, model=model)(
        _make_episode(
            "de-d",
            {"shrink_factor": 0.95},
            (Operation.DOWNSCALE,),
            (OutputChannel.WEIGHT_DELTA,),
        )
    )
    restructure_real_handler(ts, model=model)(
        _make_episode(
            "de-s",
            {"topo_op": "reroute", "swap_indices": [0, 1]},
            (Operation.RESTRUCTURE,),
            (OutputChannel.HIERARCHY_CHG,),
        )
    )
    recombine_real_handler(cs, encoder=encoder, decoder=decoder, seed=0)(
        _make_episode(
            "de-c",
            {
                "delta_latents": [
                    [0.1, 0.2, 0.3, 0.4],
                    [0.2, 0.3, 0.4, 0.5],
                ]
            },
            (Operation.RECOMBINE,),
            (OutputChannel.LATENT_SAMPLE,),
        )
    )

    assert rs.last_loss is not None and rs.last_loss >= 0.0
    assert 0.0 < ds.compound_factor < 1.0
    assert ts.diff_history == ["reroute"]
    assert cs.last_sample is not None
    assert len(cs.last_sample) == 4


# --------------------------------------------------------------------------
# Test 4 — Weight-tensor shape invariant (invariant 4 : shape preserved)
# --------------------------------------------------------------------------


def test_weight_tensor_shape_invariant_preserved() -> None:
    """replay + downscale must not alter model weight shapes.

    The real ops operate on weight tensors via MLX primitives ;
    if they accidentally replace a tensor with one of a different
    shape, downstream forward passes would break.
    restructure is allowed to change shapes (it is the topology op)
    but reroute specifically swaps without resizing.
    """
    model = _TinyMLP()

    def _shapes():
        return [
            (layer.weight.shape, layer.bias.shape)
            for layer in model.layers
        ]

    before = _shapes()

    rs = ReplayRealState()
    ds = DownscaleRealState()
    ts = RestructureRealState()

    replay_real_handler(rs, model=model, lr=0.01)(
        _make_episode(
            "de-r",
            {"beta_records": [{"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]}]},
            (Operation.REPLAY,),
            (OutputChannel.WEIGHT_DELTA,),
        )
    )
    downscale_real_handler(ds, model=model)(
        _make_episode(
            "de-d",
            {"shrink_factor": 0.99},
            (Operation.DOWNSCALE,),
            (OutputChannel.WEIGHT_DELTA,),
        )
    )
    restructure_real_handler(ts, model=model)(
        _make_episode(
            "de-s",
            {"topo_op": "reroute", "swap_indices": [0, 1]},
            (Operation.RESTRUCTURE,),
            (OutputChannel.HIERARCHY_CHG,),
        )
    )

    after = _shapes()
    # replay + downscale leave shapes intact ; reroute swaps the
    # 2 layers so shapes swap positions (but the multiset of shapes
    # is preserved).
    assert sorted(map(str, before)) == sorted(map(str, after))


# --------------------------------------------------------------------------
# Test 5 — S1 guard : replay with empty records is a no-op (budget=0 tag)
# --------------------------------------------------------------------------


def test_replay_real_empty_records_no_op_with_zero_flops() -> None:
    """Empty beta_records : no gradient step, no FLOP tag.

    Preserves S1 (retained non-regression) trivially : without
    training signal the op is a budget-accounting no-op.
    """
    model = _TinyMLP()
    state = ReplayRealState()
    handler = replay_real_handler(state, model=model, lr=0.01)
    handler(
        _make_episode(
            "de-empty",
            {"beta_records": []},
            (Operation.REPLAY,),
            (OutputChannel.WEIGHT_DELTA,),
        )
    )
    assert state.total_records_consumed == 0
    assert state.last_loss is None
    assert state.last_compute_flops == 0


# --------------------------------------------------------------------------
# Test 6 — S2 finite guard wiring (downscale with NaN weights raises)
# --------------------------------------------------------------------------


def test_s2_finite_guard_wired_on_downscale() -> None:
    """After downscale, S2 guard runs — NaN weights raise.

    We manually inject a NaN into a weight tensor and verify the
    downscale_real op surfaces the S2 violation via FiniteGuardError.
    """
    model = _TinyMLP()
    # Taint the first layer weight with NaN.
    bad = model.layers[0].weight
    arr = np.asarray(bad).copy()
    arr[0, 0] = float("nan")
    model.layers[0].weight = mx.array(arr)

    state = DownscaleRealState()
    handler = downscale_real_handler(state, model=model)
    with pytest.raises(FiniteGuardError):
        handler(
            _make_episode(
                "de-nan",
                {"shrink_factor": 0.5},
                (Operation.DOWNSCALE,),
                (OutputChannel.WEIGHT_DELTA,),
            )
        )


# --------------------------------------------------------------------------
# Test 7 — restructure_real rejects invalid topo_op (S3 vocabulary guard)
# --------------------------------------------------------------------------


def test_restructure_real_rejects_unknown_topo_op() -> None:
    """Unknown topo_op → ValueError with S3 tag."""
    model = _TinyMLP()
    state = RestructureRealState()
    handler = restructure_real_handler(state, model=model)
    with pytest.raises(ValueError, match="S3"):
        handler(
            _make_episode(
                "de-bad",
                {"topo_op": "teleport"},
                (Operation.RESTRUCTURE,),
                (OutputChannel.HIERARCHY_CHG,),
            )
        )


# --------------------------------------------------------------------------
# Test 8 — recombine_real deterministic under (seed, episode_count)
# --------------------------------------------------------------------------


def test_recombine_real_deterministic_under_seed() -> None:
    """Same seed + same episode index → identical sample."""
    encoder1, decoder1 = _TinyEncoder(), _TinyDecoder()
    # Use same parameters for second run so samples match.
    encoder2 = _TinyEncoder()
    encoder2.fc.weight = encoder1.fc.weight
    encoder2.fc.bias = encoder1.fc.bias
    decoder2 = _TinyDecoder()
    decoder2.fc.weight = decoder1.fc.weight
    decoder2.fc.bias = decoder1.fc.bias

    s1 = RecombineRealState()
    s2 = RecombineRealState()
    h1 = recombine_real_handler(
        s1, encoder=encoder1, decoder=decoder1, seed=42
    )
    h2 = recombine_real_handler(
        s2, encoder=encoder2, decoder=decoder2, seed=42
    )
    latents = [[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]]
    ep = _make_episode(
        "de-det",
        {"delta_latents": latents},
        (Operation.RECOMBINE,),
        (OutputChannel.LATENT_SAMPLE,),
    )
    h1(ep)
    h2(ep)
    assert s1.last_sample == s2.last_sample


import numpy as np  # placed at bottom because pytest.importorskip gates file
