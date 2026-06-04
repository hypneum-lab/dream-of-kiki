"""R1 bit-exact reproducibility tests for the 4 real-weight ops.

Each test builds the same canonical minimal scenario :

* fixed seed (``CANONICAL_SEED = 7``)
* fixed number of episodes (``CANONICAL_N_EPISODES = 3``)
* same initial model / encoder / decoder weights (seeded before
  construction)
* same episode input_slice payload

It then runs the op handler(s), serializes the resulting state
(weight tensors, latents, or topology history — whichever the op
produces), hashes with SHA-256, and defers to
:func:`_r1_helpers.compare_or_bootstrap` to enforce the golden.

The full-pipeline test chains the 4 ops in canonical order
(REPLAY → DOWNSCALE → RESTRUCTURE → RECOMBINE — see
``kiki_oniric.dream.episode.Operation``) and hashes the final
state of the model + recombine sample.

These tests are MLX-backed ; they are skipped when MLX is not
installed (``pytest.importorskip``). CI runs on Apple Silicon
(``macos-14``) — see ``.github/workflows/r1-nightly.yml``.
"""
from __future__ import annotations

from typing import Any

import pytest

mx = pytest.importorskip("mlx.core")
nn = pytest.importorskip("mlx.nn")

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

from tests.reproducibility._r1_helpers import (  # noqa: E402
    CANONICAL_N_EPISODES,
    CANONICAL_SEED,
    canonical_hash,
    compare_or_bootstrap,
    tensor_to_list,
)


# --------------------------------------------------------------------------
# Canonical fixtures — every test rebuilds these from scratch so the suite
# is order-independent and no fixture state leaks between tests.
# --------------------------------------------------------------------------


class _TinyMLP(nn.Module):  # type: ignore[misc,name-defined]  # mlx.nn star re-export
    """4-in / 8-hidden / 2-out MLP — matches ``test_real_ops.py``."""

    def __init__(self) -> None:
        super().__init__()
        self.layers = [nn.Linear(4, 8), nn.Linear(8, 2)]
        self.input_dim = 4

    def __call__(self, x: Any) -> Any:
        h = nn.relu(self.layers[0](x))
        return self.layers[1](h)


class _TinyEncoder(nn.Module):  # type: ignore[misc,name-defined]  # mlx.nn star re-export
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def __call__(self, x: Any) -> tuple[Any, Any]:
        h = self.fc(x)
        log_var = h * 0.0  # deterministic sigma = 1
        return h, log_var


class _TinyDecoder(nn.Module):  # type: ignore[misc,name-defined]  # mlx.nn star re-export
    def __init__(self) -> None:
        super().__init__()
        self.fc = nn.Linear(4, 4)

    def __call__(self, z: Any) -> Any:
        return self.fc(z)


def _seed_all() -> None:
    """Seed the process-wide MLX RNG so model init is deterministic.

    ``nn.Linear`` pulls from ``mx.random`` at construction time, so
    we MUST reseed before every model / encoder / decoder instance
    that we intend to hash.
    """
    mx.random.seed(CANONICAL_SEED)


def _make_mlp() -> _TinyMLP:
    _seed_all()
    return _TinyMLP()


def _make_encoder_decoder() -> tuple[_TinyEncoder, _TinyDecoder]:
    _seed_all()
    enc = _TinyEncoder()
    dec = _TinyDecoder()
    return enc, dec


def _make_episode(
    ep_id: str,
    input_slice: dict[str, Any],
    operations: tuple[Operation, ...],
    channels: tuple[OutputChannel, ...],
) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice=input_slice,
        operation_set=operations,
        output_channels=channels,
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=0.1),
        episode_id=ep_id,
    )


def _canonical_replay_records() -> list[dict[str, list[float]]]:
    """Three canonical (x, y) records — fixed across all replay tests."""
    return [
        {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
        {"x": [0.2, 0.3, 0.4, 0.5], "y": [0.0, 1.0]},
        {"x": [0.3, 0.4, 0.5, 0.6], "y": [1.0, 1.0]},
    ]


def _canonical_delta_latents() -> list[list[float]]:
    """Three canonical latents — fixed across recombine tests."""
    return [
        [0.1, 0.2, 0.3, 0.4],
        [0.2, 0.3, 0.4, 0.5],
        [0.3, 0.4, 0.5, 0.6],
    ]


def _model_weight_payload(model: _TinyMLP) -> list[list[list[float]] | list[float]]:
    """Serialize every (weight, bias) of every layer to nested floats."""
    payload: list[list[list[float]] | list[float]] = []
    for layer in model.layers:
        payload.append(tensor_to_list(layer.weight))
        payload.append(tensor_to_list(layer.bias))
    return payload


# --------------------------------------------------------------------------
# B1.1 — replay_real : hash the updated weight tensor after N episodes
# --------------------------------------------------------------------------


def test_r1_replay() -> None:
    model = _make_mlp()
    state = ReplayRealState()
    handler = replay_real_handler(state, model=model, lr=0.01)

    records = _canonical_replay_records()
    for i in range(CANONICAL_N_EPISODES):
        handler(
            _make_episode(
                f"de-replay-{i}",
                {"beta_records": records},
                (Operation.REPLAY,),
                (OutputChannel.WEIGHT_DELTA,),
            )
        )

    payload = {
        "op": "replay_real",
        "weights": _model_weight_payload(model),
        "total_records_consumed": state.total_records_consumed,
        "n_episodes": CANONICAL_N_EPISODES,
        "seed": CANONICAL_SEED,
    }
    digest = canonical_hash(payload)
    compare_or_bootstrap("test_r1_replay", digest)


# --------------------------------------------------------------------------
# B1.2 — downscale_real : hash the shrunk weight tensor after N episodes
# --------------------------------------------------------------------------


def test_r1_downscale() -> None:
    model = _make_mlp()
    state = DownscaleRealState()
    handler = downscale_real_handler(state, model=model)

    for i in range(CANONICAL_N_EPISODES):
        handler(
            _make_episode(
                f"de-downscale-{i}",
                {"shrink_factor": 0.95},
                (Operation.DOWNSCALE,),
                (OutputChannel.WEIGHT_DELTA,),
            )
        )

    payload = {
        "op": "downscale_real",
        "weights": _model_weight_payload(model),
        "compound_factor": state.compound_factor,
        "n_episodes": CANONICAL_N_EPISODES,
        "seed": CANONICAL_SEED,
    }
    digest = canonical_hash(payload)
    compare_or_bootstrap("test_r1_downscale", digest)


# --------------------------------------------------------------------------
# B1.3 — restructure_real : hash the topology after N episodes
# --------------------------------------------------------------------------


def test_r1_restructure() -> None:
    model = _make_mlp()
    state = RestructureRealState()
    handler = restructure_real_handler(state, model=model)

    for i in range(CANONICAL_N_EPISODES):
        handler(
            _make_episode(
                f"de-restructure-{i}",
                {"topo_op": "reroute", "swap_indices": [0, 1]},
                (Operation.RESTRUCTURE,),
                (OutputChannel.HIERARCHY_CHG,),
            )
        )

    payload = {
        "op": "restructure_real",
        "weights": _model_weight_payload(model),
        "diff_history": list(state.diff_history),
        "n_episodes": CANONICAL_N_EPISODES,
        "seed": CANONICAL_SEED,
    }
    digest = canonical_hash(payload)
    compare_or_bootstrap("test_r1_restructure", digest)


# --------------------------------------------------------------------------
# B1.4 — recombine_real : hash the decoded latent sample after N episodes
# --------------------------------------------------------------------------


def test_r1_recombine() -> None:
    encoder, decoder = _make_encoder_decoder()
    state = RecombineRealState()
    handler = recombine_real_handler(
        state, encoder=encoder, decoder=decoder, seed=CANONICAL_SEED
    )

    latents = _canonical_delta_latents()
    samples: list[list[float]] = []
    for i in range(CANONICAL_N_EPISODES):
        handler(
            _make_episode(
                f"de-recombine-{i}",
                {"delta_latents": latents},
                (Operation.RECOMBINE,),
                (OutputChannel.LATENT_SAMPLE,),
            )
        )
        assert state.last_sample is not None
        samples.append(list(state.last_sample))

    payload = {
        "op": "recombine_real",
        "samples": samples,
        "n_episodes": CANONICAL_N_EPISODES,
        "seed": CANONICAL_SEED,
    }
    digest = canonical_hash(payload)
    compare_or_bootstrap("test_r1_recombine", digest)


# --------------------------------------------------------------------------
# B1.5 — full pipeline REPLAY → DOWNSCALE → RESTRUCTURE → RECOMBINE
# --------------------------------------------------------------------------


def test_r1_full_pipeline() -> None:
    """Chain all 4 ops in canonical DR-2 order and hash the final state.

    The order matches ``kiki_oniric.dream.episode.Operation`` enum
    declaration (REPLAY, DOWNSCALE, RESTRUCTURE, RECOMBINE) — this
    is the DR-2 canonical order (see
    ``docs/proofs/op-pair-analysis.md``).

    Single-episode pipeline : reroute would invert ``model.layers``
    so a second episode's REPLAY could not forward-pass through the
    now-swapped Linear(8,2) layer. The canonical R1 scenario runs
    one DE that exercises all 4 ops in order, hashing the terminal
    state.
    """
    model = _make_mlp()
    encoder, decoder = _make_encoder_decoder()

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
            recombine_state,
            encoder=encoder,
            decoder=decoder,
            seed=CANONICAL_SEED,
        ),
    )

    records = _canonical_replay_records()
    latents = _canonical_delta_latents()

    ep = _make_episode(
        "de-pipeline-canonical",
        {
            "beta_records": records,
            "shrink_factor": 0.95,
            "topo_op": "reroute",
            "swap_indices": [0, 1],
            "delta_latents": latents,
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

    assert recombine_state.last_sample is not None
    # DR-0 : single DE produced a completed log entry.
    assert len(rt.log) == 1
    assert rt.log[0].completed is True

    payload = {
        "op": "full_pipeline",
        "canonical_order": [
            Operation.REPLAY.value,
            Operation.DOWNSCALE.value,
            Operation.RESTRUCTURE.value,
            Operation.RECOMBINE.value,
        ],
        "final_weights": _model_weight_payload(model),
        "final_sample": list(recombine_state.last_sample),
        "compound_factor": downscale_state.compound_factor,
        "diff_history": list(restructure_state.diff_history),
        "total_records_consumed": replay_state.total_records_consumed,
        "seed": CANONICAL_SEED,
    }
    digest = canonical_hash(payload)
    compare_or_bootstrap("test_r1_full_pipeline", digest)
