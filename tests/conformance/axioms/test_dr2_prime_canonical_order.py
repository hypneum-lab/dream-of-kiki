"""DR-2' Canonical-order compositionality — determinism conformance test.

DR-2' (fallback, DR-2 unproven — see
docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §5.1) states
that operations applied in the canonical order are compositional
within the operation set :

    canonical order = replay < downscale < restructure < recombine

Operationally, this test verifies that chaining the four canonical
operations through a single :class:`DreamRuntime` run under an
identical seed produces a **byte-identical** final state across
two independent runs. This is the empirical contract retained by
the G2/G4 pilots until a strict DR-2 proof is available.

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §5.1
  (DR-2 unproven working axiom) and the DR-2' fallback definition
  in the same section.
"""
from __future__ import annotations

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

import numpy as np  # noqa: E402

from kiki_oniric.dream.episode import (  # noqa: E402
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
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
# Minimal MLX fixtures — mirror tests/unit/test_real_ops.py so both
# suites share the same tiny-scale deterministic shape.
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


def _make_canonical_episode(ep_id: str) -> DreamEpisode:
    """Build a DE whose operation_set is the canonical order.

    canonical order (cf. ``Operation`` enum in
    kiki_oniric/dream/episode.py) = REPLAY, DOWNSCALE, RESTRUCTURE,
    RECOMBINE. DR-2' is formulated against this exact ordering.
    """
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={
            "beta_records": [
                {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
                {"x": [0.5, 0.6, 0.7, 0.8], "y": [0.0, 1.0]},
            ],
            "shrink_factor": 0.97,
            "topo_op": "reroute",
            "swap_indices": [0, 1],
            "delta_latents": [
                [0.1, 0.2, 0.3, 0.4],
                [0.5, 0.6, 0.7, 0.8],
            ],
        },
        operation_set=(
            Operation.REPLAY,
            Operation.DOWNSCALE,
            Operation.RESTRUCTURE,
            Operation.RECOMBINE,
        ),
        output_channels=(
            OutputChannel.WEIGHT_DELTA,
            OutputChannel.HIERARCHY_CHG,
            OutputChannel.LATENT_SAMPLE,
        ),
        budget=BudgetCap(flops=1_000_000, wall_time_s=1.0, energy_j=0.1),
        episode_id=ep_id,
    )


def _build_runtime(
    model: _TinyMLP,
    encoder: _TinyEncoder,
    decoder: _TinyDecoder,
    seed: int,
) -> tuple[
    DreamRuntime,
    ReplayRealState,
    DownscaleRealState,
    RestructureRealState,
    RecombineRealState,
]:
    """Assemble a DreamRuntime wired with all 4 canonical handlers."""
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
            recombine_state, encoder=encoder, decoder=decoder, seed=seed
        ),
    )
    return rt, replay_state, downscale_state, restructure_state, recombine_state


def _weight_snapshot(model: _TinyMLP) -> list[np.ndarray]:
    """Byte-for-byte snapshot of every (weight, bias) in ``model.layers``."""
    snap: list[np.ndarray] = []
    for layer in model.layers:
        snap.append(np.asarray(layer.weight).copy())
        snap.append(np.asarray(layer.bias).copy())
    return snap


def _assert_snapshots_equal(
    left: list[np.ndarray], right: list[np.ndarray]
) -> None:
    assert len(left) == len(right)
    for a, b in zip(left, right):
        assert a.shape == b.shape
        assert a.dtype == b.dtype
        assert np.array_equal(a, b), (
            "DR-2' violated — canonical-order composition is "
            "not deterministic under identical seed"
        )


def test_dr2_prime_canonical_order_is_deterministic() -> None:
    """DR-2' (fallback, DR-2 unproven — see spec §5.1).

    Applying REPLAY → DOWNSCALE → RESTRUCTURE → RECOMBINE in a single
    DE under identical initial conditions (same seed, same input
    slice) must yield byte-identical final model weights and an
    identical recombine latent sample across two independent runs.
    """
    # Run 1 — seed, build fresh substrate, execute canonical DE.
    mx.random.seed(7)
    model1 = _TinyMLP()
    encoder1 = _TinyEncoder()
    decoder1 = _TinyDecoder()
    rt1, _, _, rs1, cs1 = _build_runtime(
        model1, encoder1, decoder1, seed=42
    )
    rt1.execute(_make_canonical_episode("de-dr2p-1"))

    # Run 2 — re-seed, rebuild identical substrate, execute same DE.
    mx.random.seed(7)
    model2 = _TinyMLP()
    encoder2 = _TinyEncoder()
    decoder2 = _TinyDecoder()
    rt2, _, _, rs2, cs2 = _build_runtime(
        model2, encoder2, decoder2, seed=42
    )
    rt2.execute(_make_canonical_episode("de-dr2p-2"))

    # Final MLP weights must match byte-for-byte (after reroute both
    # runs swap layers identically, so positional order also matches).
    _assert_snapshots_equal(
        _weight_snapshot(model1), _weight_snapshot(model2)
    )

    # Runtime log records the canonical-order execution identically.
    assert len(rt1.log) == len(rt2.log) == 1
    entry1, entry2 = rt1.log[0], rt2.log[0]
    assert entry1.completed is True
    assert entry2.completed is True
    assert entry1.operations_executed == entry2.operations_executed == (
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    )

    # Restructure diff trace is identical.
    assert rs1.diff_history == rs2.diff_history == ["reroute"]

    # Recombine latent sample is byte-identical (the op that carries
    # the only RNG-driven step — its determinism anchors DR-2').
    assert cs1.last_sample is not None
    assert cs2.last_sample is not None
    assert cs1.last_sample == cs2.last_sample
