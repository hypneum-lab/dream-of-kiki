"""Unit tests for the LoRA restructure handler (B3)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pytest

if TYPE_CHECKING:
    import mlx.core as mx
else:
    mx = pytest.importorskip("mlx.core")

from kiki_oniric.dream.channels import TopologyDiff
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.restructure_real import (
    RestructureRealState,
    restructure_lora_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime, EpisodeLogEntry
from kiki_oniric.substrates.micro_kiki.lora_model import LoRALinear, LoRAModel


def _episode(
    topo_ops: list[dict[str, object]],
    episode_id: str = "de-restr",
) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"topo_ops": topo_ops},
        operation_set=(Operation.RESTRUCTURE,),
        output_channels=(OutputChannel.HIERARCHY_CHG,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id=episode_id,
    )


def _run(
    model: LoRAModel,
    topo_ops: list[dict[str, object]],
    *,
    max_adds: int = 1,
    seed: int = 0,
    episode_id: str = "de-restr",
) -> tuple[RestructureRealState, EpisodeLogEntry]:
    state = RestructureRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.RESTRUCTURE,
        restructure_lora_handler(
            state,
            model=model,
            max_adds_per_episode=max_adds,
            seed=seed,
        ),
    )
    runtime.execute(_episode(topo_ops, episode_id=episode_id))
    return state, runtime.log[-1]


def _reroute(i: int, j: int) -> dict[str, object]:
    return {"op": "reroute", "swap_indices": [i, j]}


def _add(
    index: int,
    in_features: int = 4,
    out_features: int = 8,
    rank: int = 2,
    alpha: float = 4.0,
) -> dict[str, object]:
    return {
        "op": "add",
        "index": index,
        "in_features": in_features,
        "out_features": out_features,
        "rank": rank,
        "alpha": alpha,
    }


def _remove(index: int) -> dict[str, object]:
    return {"op": "remove", "index": index}


def test_restructure_emits_topology_diff() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _, entry = _run(model, [_reroute(0, 1)])
    assert isinstance(entry.channel_outputs[0], TopologyDiff)


def test_restructure_s3_vocab_rejects_unknown_op() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    state = RestructureRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.RESTRUCTURE,
        restructure_lora_handler(state, model=model),
    )
    with pytest.raises(ValueError, match=r"^S3"):
        runtime.execute(_episode([{"op": "bogus"}]))
    assert len(model.layers) == pre_len  # untouched


def test_restructure_s3_add_missing_dims_rejected() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    state = RestructureRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.RESTRUCTURE,
        restructure_lora_handler(state, model=model),
    )
    with pytest.raises(ValueError, match=r"^S3"):
        runtime.execute(_episode([{"op": "add", "index": 0}]))
    assert len(model.layers) == pre_len


def test_restructure_add_grows_model() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    state, entry = _run(
        model, [_add(index=pre_len, in_features=2, out_features=2, rank=2)],
        max_adds=1,
    )
    assert len(model.layers) == pre_len + 1
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    assert td.diff[0][0] == "add"
    payload = td.diff[0][1]
    assert payload["index"] == pre_len
    assert payload["in_features"] == 2
    assert payload["out_features"] == 2
    assert payload["rank"] == 2
    assert state.total_adds == 1


def test_restructure_add_reconstruction_round_trip() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    _, entry = _run(
        model,
        [_add(index=pre_len, in_features=3, out_features=5, rank=2, alpha=4.0)],
        seed=42,
        episode_id="de-rt",
    )
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    payload = cast(dict[str, Any], td.diff[0][1])
    inserted = model.layers[pre_len]
    # Reconstruct from the payload seed: same key seed → same init.
    rebuilt = LoRALinear(
        in_features=int(payload["in_features"]),
        out_features=int(payload["out_features"]),
        rank=int(payload["rank"]),
        alpha=float(payload["alpha"]),
        key=mx.random.key(int(payload["seed"])),
    )
    np.testing.assert_array_equal(
        np.asarray(inserted.base_weight), np.asarray(rebuilt.base_weight),
    )
    np.testing.assert_array_equal(
        np.asarray(inserted.lora_a), np.asarray(rebuilt.lora_a),
    )
    np.testing.assert_array_equal(
        np.asarray(inserted.lora_b), np.asarray(rebuilt.lora_b),
    )


def test_restructure_remove_shrinks_model() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    state, entry = _run(model, [_remove(0)])
    assert len(model.layers) == pre_len - 1
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    snap = cast(dict[str, Any], td.diff[0][1]["snapshot"])
    for k in ("base_weight", "lora_a", "lora_b"):
        assert snap[k].dtype == np.float32
        assert bool(np.isfinite(snap[k]).all())
    assert state.total_removes == 1


def test_restructure_remove_undo_round_trip() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_base = np.asarray(model.layers[0].base_weight)
    pre_a = np.asarray(model.layers[0].lora_a)
    pre_b = np.asarray(model.layers[0].lora_b)
    _, entry = _run(model, [_remove(0)])
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    snap = cast(dict[str, Any], td.diff[0][1]["snapshot"])
    rebuilt = LoRALinear(
        in_features=int(snap["in_features"]),
        out_features=int(snap["out_features"]),
        rank=int(snap["rank"]),
        alpha=float(snap["alpha"]),
    )
    rebuilt.base_weight = mx.array(snap["base_weight"])
    rebuilt.lora_a = mx.array(snap["lora_a"])
    rebuilt.lora_b = mx.array(snap["lora_b"])
    if snap["bias"] is not None:
        rebuilt.bias = mx.array(snap["bias"])
    np.testing.assert_array_equal(np.asarray(rebuilt.base_weight), pre_base)
    np.testing.assert_array_equal(np.asarray(rebuilt.lora_a), pre_a)
    np.testing.assert_array_equal(np.asarray(rebuilt.lora_b), pre_b)


def test_restructure_reroute_swaps_in_place() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    layer_a_id = id(model.layers[0])
    layer_b_id = id(model.layers[1])
    _, _ = _run(model, [_reroute(0, 1)])
    assert id(model.layers[0]) == layer_b_id
    assert id(model.layers[1]) == layer_a_id


def test_restructure_multi_op_diff_length() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    _, entry = _run(
        model,
        [
            _add(index=pre_len, in_features=2, out_features=2),
            _reroute(0, 1),
            _remove(0),
        ],
        max_adds=1,
    )
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    ops_in_order = tuple(e[0] for e in td.diff)
    assert ops_in_order == ("add", "reroute", "remove")


def test_restructure_inss_soft_cap_one_skip() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    state, entry = _run(
        model,
        [
            _add(index=pre_len, in_features=2, out_features=2),
            _add(index=pre_len + 1, in_features=2, out_features=2),
        ],
        max_adds=1,
    )
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    assert len(td.diff) == 1
    assert state.total_adds == 1
    # Only one add applied → model grew by exactly one.
    assert len(model.layers) == pre_len + 1


def test_restructure_inss_soft_cap_full_skip_returns_none() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    pre_len = len(model.layers)
    state, entry = _run(
        model,
        [_add(index=pre_len, in_features=2, out_features=2)],
        max_adds=0,
    )
    assert entry.channel_outputs[0] is None
    assert state.last_compute_flops == 0
    assert len(model.layers) == pre_len


def test_restructure_sha256_changes_under_mutation() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _, entry = _run(model, [_reroute(0, 1), _reroute(0, 1)])
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    sha_first = td.diff[0][1]["model_sha256_post"]
    sha_second = td.diff[1][1]["model_sha256_post"]
    # Two independent reroutes (0,1) → first swap differs from base,
    # second swap restores order, so the two shas must differ.
    assert sha_first != sha_second


def test_restructure_sha256_stable_under_identity_swap() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    # Capture the pre-handler hash via a dummy run that does nothing.
    from kiki_oniric.dream.operations.restructure_real import _model_sha256
    pre_sha = _model_sha256(model)
    _, entry = _run(model, [_reroute(0, 0)])
    td = entry.channel_outputs[0]
    assert isinstance(td, TopologyDiff)
    assert td.diff[0][1]["model_sha256_post"] == pre_sha


def test_restructure_is_deterministic() -> None:
    m1 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=3)
    m2 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=3)
    pre = len(m1.layers)
    _, e1 = _run(
        m1, [_add(index=pre, in_features=2, out_features=2)],
        max_adds=1, seed=7, episode_id="de-det",
    )
    _, e2 = _run(
        m2, [_add(index=pre, in_features=2, out_features=2)],
        max_adds=1, seed=7, episode_id="de-det",
    )
    td1 = e1.channel_outputs[0]
    td2 = e2.channel_outputs[0]
    assert isinstance(td1, TopologyDiff) and isinstance(td2, TopologyDiff)
    assert td1.diff[0][1]["model_sha256_post"] == td2.diff[0][1]["model_sha256_post"]
    assert td1.diff[0][1]["seed"] == td2.diff[0][1]["seed"]
