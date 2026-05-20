"""Unit tests for apply_channel_outputs() and concrete channels (B5)."""
from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
import pytest

if TYPE_CHECKING:
    import mlx.core as mx
else:
    mx = pytest.importorskip("mlx.core")

from kiki_oniric.substrates.micro_kiki.lora_model import LoRALinear, LoRAModel  # noqa: F401


def _clones(seed: int = 0) -> tuple[LoRAModel, LoRAModel]:
    """Return two bit-identical LoRAModels — dream-side and awake-side."""
    return (
        LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=seed),
        LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=seed),
    )


def _assert_lora_models_equal(a: LoRAModel, b: LoRAModel) -> None:
    """Assert bit-equality of every layer's base + adapters."""
    assert len(a.layers) == len(b.layers)
    for la, lb in zip(a.layers, b.layers):
        np.testing.assert_array_equal(
            np.asarray(la.base_weight), np.asarray(lb.base_weight),
        )
        np.testing.assert_array_equal(
            np.asarray(la.lora_a), np.asarray(lb.lora_a),
        )
        np.testing.assert_array_equal(
            np.asarray(la.lora_b), np.asarray(lb.lora_b),
        )
        if la.use_bias:
            np.testing.assert_array_equal(
                np.asarray(la.bias), np.asarray(lb.bias),
            )


def test_lora_weight_delta_channel_additive_apply() -> None:
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )

    _, target = _clones(seed=0)
    before_a = np.asarray(target.layers[0].lora_a, dtype=np.float32).copy()
    delta_a = np.ones_like(before_a) * 0.5
    channel = LoRAWeightDeltaChannel(target)
    channel.apply({"layer0.lora_a": delta_a})

    after_a = np.asarray(target.layers[0].lora_a, dtype=np.float32)
    np.testing.assert_allclose(after_a, before_a + delta_a, rtol=1e-6)


def test_lora_weight_delta_channel_rejects_non_finite() -> None:
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )

    _, target = _clones(seed=0)
    bad = np.full_like(
        np.asarray(target.layers[0].lora_a, dtype=np.float32),
        np.inf,
    )
    channel = LoRAWeightDeltaChannel(target)
    with pytest.raises(ValueError, match=r"^S2:"):
        channel.apply({"layer0.lora_a": bad})


def test_lora_weight_delta_channel_rejects_bad_key_format() -> None:
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )

    _, target = _clones(seed=0)
    channel = LoRAWeightDeltaChannel(target)
    with pytest.raises(ValueError, match=r"^S1:"):
        channel.apply({"oops": np.zeros((2, 4), dtype=np.float32)})


def test_lora_weight_delta_channel_rejects_out_of_range_layer() -> None:
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )

    _, target = _clones(seed=0)
    channel = LoRAWeightDeltaChannel(target)
    with pytest.raises(ValueError, match=r"^S1:"):
        channel.apply({"layer99.lora_a": np.zeros((2, 4), dtype=np.float32)})


# ---------------------------------------------------------------------------
# B5 T2 — LoRAHierarchyChangeChannel tests
# ---------------------------------------------------------------------------


def _make_topology_diff_add(
    seed: int,
    index: int = 1,
    in_f: int = 4,
    out_f: int = 8,
    rank: int = 2,
    alpha: float = 4.0,
    sha: str = "a" * 64,
) -> tuple[str, dict[str, object]]:
    return (
        "add",
        {
            "index": index,
            "in_features": in_f,
            "out_features": out_f,
            "rank": rank,
            "alpha": alpha,
            "seed": seed,
            "model_sha256_post": sha,
        },
    )


def test_hierarchy_change_channel_add_inserts_layer() -> None:
    """apply_diff with an add entry grows the layer stack by one (S3)."""
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )

    _, target = _clones(seed=0)
    before_len = len(target.layers)
    channel = LoRAHierarchyChangeChannel(target)
    entry = _make_topology_diff_add(seed=42, index=0)
    channel.apply_diff([entry])
    assert len(target.layers) == before_len + 1


def test_hierarchy_change_channel_add_seed_reproducible() -> None:
    """add with same seed reconstructs bit-identical layer (R1 linchpin)."""
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )

    op_seed = 99
    entry = _make_topology_diff_add(seed=op_seed, index=0)

    # Apply via channel onto first target.
    _, target_a = _clones(seed=0)
    LoRAHierarchyChangeChannel(target_a).apply_diff([entry])
    inserted_a = target_a.layers[0]

    # Reconstruct independently using the same seed.
    reference = LoRALinear(
        in_features=4,
        out_features=8,
        rank=2,
        alpha=4.0,
        key=mx.random.key(op_seed),
    )

    np.testing.assert_array_equal(
        np.asarray(inserted_a.lora_a), np.asarray(reference.lora_a),
    )
    np.testing.assert_array_equal(
        np.asarray(inserted_a.base_weight), np.asarray(reference.base_weight),
    )


def test_hierarchy_change_channel_remove_shrinks_layer() -> None:
    """apply_diff with a remove entry shrinks the layer stack by one (S3)."""
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )

    _, target = _clones(seed=0)
    before_len = len(target.layers)
    channel = LoRAHierarchyChangeChannel(target)
    remove_entry = (
        "remove",
        {
            "index": 0,
            "snapshot": {},
            "model_sha256_post": "b" * 64,
        },
    )
    channel.apply_diff([remove_entry])
    assert len(target.layers) == before_len - 1


def test_hierarchy_change_channel_reroute_swaps_layers() -> None:
    """apply_diff with reroute swaps two layers' lora_a arrays (S3)."""
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )

    _, target = _clones(seed=0)
    a_before = np.asarray(target.layers[0].lora_a, dtype=np.float32).copy()
    b_before = np.asarray(target.layers[1].lora_a, dtype=np.float32).copy()

    channel = LoRAHierarchyChangeChannel(target)
    reroute_entry = (
        "reroute",
        {
            "swap_indices": (0, 1),
            "model_sha256_post": "c" * 64,
        },
    )
    channel.apply_diff([reroute_entry])

    np.testing.assert_array_equal(
        np.asarray(target.layers[0].lora_a, dtype=np.float32), b_before,
    )
    np.testing.assert_array_equal(
        np.asarray(target.layers[1].lora_a, dtype=np.float32), a_before,
    )


# ---------------------------------------------------------------------------
# B5 T3 — LatentSampleQueue tests
# ---------------------------------------------------------------------------


def test_latent_sample_queue_enqueue_grows_len() -> None:
    """enqueue appends to the FIFO; len() tracks the depth."""
    from kiki_oniric.dream.channels.latent_sample import (
        LatentSampleQueue,
    )

    queue = LatentSampleQueue()
    assert len(queue) == 0
    vec = np.zeros(4, dtype=np.float32)
    queue.enqueue("default", vec, "recombine:de=ep0:ep=0:seed=0")
    assert len(queue) == 1
    queue.enqueue("replay-mix", vec, "recombine:de=ep1:ep=1:seed=1")
    assert len(queue) == 2


def test_latent_sample_queue_dequeue_returns_dict_fields() -> None:
    """dequeue returns a dict with species / latent_vector / provenance."""
    from kiki_oniric.dream.channels.latent_sample import (
        LatentSampleQueue,
    )

    queue = LatentSampleQueue()
    vec = np.array([0.1, 0.2, 0.3, 0.4], dtype=np.float32)
    prov = "recombine:de=test:ep=0:seed=7"
    queue.enqueue("myspecies", vec, prov)
    item = queue.dequeue()
    assert item is not None
    assert item["species"] == "myspecies"
    np.testing.assert_array_equal(item["latent_vector"], vec)
    assert item["provenance"] == prov
    assert len(queue) == 0


def test_latent_sample_queue_capacity_drops_oldest() -> None:
    """maxlen capacity: oldest item is dropped when the queue is full."""
    from kiki_oniric.dream.channels.latent_sample import (
        LatentSampleQueue,
    )

    queue = LatentSampleQueue(maxlen=2)
    v = np.zeros(2, dtype=np.float32)
    queue.enqueue("first", v, "prov:0")
    queue.enqueue("second", v, "prov:1")
    queue.enqueue("third", v, "prov:2")  # drops "first"
    assert len(queue) == 2
    oldest = queue.dequeue()
    assert oldest is not None
    assert oldest["species"] == "second"


def test_latent_sample_queue_dequeue_empty_returns_none() -> None:
    """dequeue on an empty queue returns None (I3 no-latent-buffer)."""
    from kiki_oniric.dream.channels.latent_sample import (
        LatentSampleQueue,
    )

    queue = LatentSampleQueue()
    assert queue.dequeue() is None


# ---------------------------------------------------------------------------
# B5 T4 — set_prior alias on AttentionPriorChannel
# ---------------------------------------------------------------------------


def test_attention_prior_set_prior_is_alias_of_emit() -> None:
    from kiki_oniric.dream.channels.attention_prior import (
        AttentionPriorChannel,
    )

    ch = AttentionPriorChannel()
    prior = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    ch.set_prior(prior)
    got = ch.get_prior()
    assert got is not None
    np.testing.assert_array_equal(got, prior)


# ---------------------------------------------------------------------------
# B5 T5 — apply_channel_outputs dispatch function
# ---------------------------------------------------------------------------


def _make_log_with_one_output(output: Any) -> "list[Any]":
    """Build a 1-entry EpisodeLogEntry log carrying ``output`` only."""
    from kiki_oniric.dream.episode import Operation
    from kiki_oniric.dream.runtime import EpisodeLogEntry

    return [
        EpisodeLogEntry(
            episode_id="de-test",
            operations_executed=(Operation.REPLAY,),
            completed=True,
            error=None,
            channel_outputs=(output,),
        ),
    ]


def test_apply_dispatches_weight_update() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    _, target = _clones(seed=0)
    delta_a = np.ones(
        np.asarray(target.layers[0].lora_a).shape, dtype=np.float32,
    ) * 0.25
    log = _make_log_with_one_output(WeightUpdate(lora_delta={"layer0.lora_a": delta_a}))
    count = apply_channel_outputs(
        log,
        weight_channel=LoRAWeightDeltaChannel(target),
        hierarchy_channel=LoRAHierarchyChangeChannel(target),
        latent_channel=LatentSampleQueue(),
    )
    assert count == 1


def test_apply_dispatches_topology_diff_reroute() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import TopologyDiff
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    _, target = _clones(seed=0)
    payload: dict[str, object] = {
        "swap_indices": (0, 1),
        "model_sha256_post": "0" * 64,
    }
    entry: tuple[str, dict[str, object]] = ("reroute", payload)
    log = _make_log_with_one_output(TopologyDiff(diff=(entry,)))
    apply_channel_outputs(
        log,
        weight_channel=LoRAWeightDeltaChannel(target),
        hierarchy_channel=LoRAHierarchyChangeChannel(target),
        latent_channel=LatentSampleQueue(),
    )
    # After one reroute, the model is mutated; here we just smoke-check
    # via length. End-to-end bit-exact behaviour is covered in Task 6.
    assert len(target.layers) == 2


def test_apply_dispatches_latent_sample() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import LatentSample
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    _, target = _clones(seed=0)
    q = LatentSampleQueue()
    log = _make_log_with_one_output(
        LatentSample(
            species="default",
            latent_vector=np.array([0.1, 0.2], dtype=np.float32),
            provenance="recombine:de=test:ep=0:seed=0",
        ),
    )
    apply_channel_outputs(
        log,
        weight_channel=LoRAWeightDeltaChannel(target),
        hierarchy_channel=LoRAHierarchyChangeChannel(target),
        latent_channel=q,
    )
    assert len(q) == 1


def test_apply_dispatches_attention_prior() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import AttentionPrior
    from kiki_oniric.dream.channels.attention_prior import (
        AttentionPriorChannel,
    )
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    _, target = _clones(seed=0)
    att = AttentionPriorChannel()
    prior = np.array([0.1, 0.2, 0.3], dtype=np.float32)
    log = _make_log_with_one_output(AttentionPrior(prior=prior))
    apply_channel_outputs(
        log,
        weight_channel=LoRAWeightDeltaChannel(target),
        hierarchy_channel=LoRAHierarchyChangeChannel(target),
        latent_channel=LatentSampleQueue(),
        attention_channel=att,
    )
    got = att.get_prior()
    assert got is not None
    np.testing.assert_array_equal(got, prior)


def test_apply_attention_required_when_emitted() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import AttentionPrior
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    _, target = _clones(seed=0)
    log = _make_log_with_one_output(
        AttentionPrior(prior=np.array([0.1], dtype=np.float32)),
    )
    with pytest.raises(ValueError, match="attention_channel"):
        apply_channel_outputs(
            log,
            weight_channel=LoRAWeightDeltaChannel(target),
            hierarchy_channel=LoRAHierarchyChangeChannel(target),
            latent_channel=LatentSampleQueue(),
        )


def test_apply_skips_none_entries() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue
    from kiki_oniric.dream.episode import Operation
    from kiki_oniric.dream.runtime import EpisodeLogEntry

    _, target = _clones(seed=0)
    delta_a = np.zeros(
        np.asarray(target.layers[0].lora_a).shape, dtype=np.float32,
    )
    log = [
        EpisodeLogEntry(
            episode_id="de-skip",
            operations_executed=(Operation.REPLAY, Operation.DOWNSCALE, Operation.RECOMBINE),
            completed=True,
            error=None,
            channel_outputs=(
                None,
                WeightUpdate(lora_delta={"layer0.lora_a": delta_a}),
                None,
            ),
        ),
    ]
    count = apply_channel_outputs(
        log,
        weight_channel=LoRAWeightDeltaChannel(target),
        hierarchy_channel=LoRAHierarchyChangeChannel(target),
        latent_channel=LatentSampleQueue(),
    )
    assert count == 1


def test_apply_empty_log_returns_zero() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    _, target = _clones(seed=0)
    count = apply_channel_outputs(
        [],
        weight_channel=LoRAWeightDeltaChannel(target),
        hierarchy_channel=LoRAHierarchyChangeChannel(target),
        latent_channel=LatentSampleQueue(),
    )
    assert count == 0


def test_apply_rejects_unknown_output_type() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels.weight_delta import LoRAWeightDeltaChannel
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue

    _, target = _clones(seed=0)
    # A bare object — not in the ChannelOutput union.
    bogus = object()
    log = _make_log_with_one_output(bogus)
    with pytest.raises(TypeError, match="unknown"):
        apply_channel_outputs(
            log,
            weight_channel=LoRAWeightDeltaChannel(target),
            hierarchy_channel=LoRAHierarchyChangeChannel(target),
            latent_channel=LatentSampleQueue(),
        )
