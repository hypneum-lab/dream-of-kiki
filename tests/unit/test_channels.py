"""Unit tests for the four dream-awake channel-output types."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kiki_oniric.dream.channels import (
    AttentionPrior,
    ChannelOutput,
    LatentSample,
    TopologyDiff,
    WeightUpdate,
)


def test_weight_update_construction() -> None:
    wu = WeightUpdate(
        lora_delta={"layer0": np.zeros(4, dtype=np.float32)},
    )
    assert wu.fisher_bump is None
    assert wu.lora_delta["layer0"].shape == (4,)


def test_weight_update_rejects_non_finite() -> None:
    bad = {"layer0": np.array([np.inf], dtype=np.float32)}
    with pytest.raises(ValueError, match="S2"):
        WeightUpdate(lora_delta=bad)


def test_latent_sample_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="S2"):
        LatentSample(
            species="x",
            latent_vector=np.array([np.nan], dtype=np.float32),
            provenance="test",
        )


def test_attention_prior_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="S2"):
        AttentionPrior(prior=np.array([np.inf], dtype=np.float32))


_VALID_SHA = "0" * 64


def _reroute_entry() -> tuple[str, dict[str, object]]:
    return (
        "reroute",
        {"swap_indices": (0, 1), "model_sha256_post": _VALID_SHA},
    )


def test_topology_diff_accepts_valid_reroute() -> None:
    td = TopologyDiff(diff=(_reroute_entry(),))
    assert td.diff[0][0] == "reroute"


def test_topology_diff_rejects_unknown_op() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(("add_node", {"model_sha256_post": _VALID_SHA}),))


def test_topology_diff_rejects_non_tuple_entry() -> None:
    bad = (["reroute", {"swap_indices": (0, 1), "model_sha256_post": _VALID_SHA}],)
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=bad)  # type: ignore[arg-type]


def test_topology_diff_rejects_non_dict_payload() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(("reroute", "not-a-dict"),))  # type: ignore[arg-type]


def test_topology_diff_rejects_missing_sha() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(("reroute", {"swap_indices": (0, 1)}),))


def test_topology_diff_rejects_short_sha() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(
            ("reroute", {"swap_indices": (0, 1), "model_sha256_post": "abc"}),
        ))


def test_topology_diff_rejects_add_missing_keys() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(
            ("add", {"index": 0, "model_sha256_post": _VALID_SHA}),
        ))


def test_topology_diff_rejects_add_non_positive_rank() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(
            ("add", {
                "index": 0,
                "in_features": 4,
                "out_features": 8,
                "rank": 0,
                "alpha": 4.0,
                "seed": 0,
                "model_sha256_post": _VALID_SHA,
            }),
        ))


def test_topology_diff_rejects_remove_missing_snapshot() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(
            ("remove", {"index": 0, "model_sha256_post": _VALID_SHA}),
        ))


def test_topology_diff_rejects_remove_non_finite_snapshot() -> None:
    bad_snap: dict[str, object] = {
        "base_weight": np.array([np.inf], dtype=np.float32),
        "lora_a": np.zeros(1, dtype=np.float32),
        "lora_b": np.zeros(1, dtype=np.float32),
        "bias": None,
        "in_features": 4,
        "out_features": 8,
        "rank": 2,
        "alpha": 4.0,
    }
    with pytest.raises(ValueError, match="S2"):
        TopologyDiff(diff=(
            ("remove", {
                "index": 0,
                "snapshot": bad_snap,
                "model_sha256_post": _VALID_SHA,
            }),
        ))


def test_topology_diff_rejects_reroute_bad_swap_indices() -> None:
    with pytest.raises(ValueError, match="S3"):
        TopologyDiff(diff=(
            ("reroute", {
                "swap_indices": (0, 1, 2),
                "model_sha256_post": _VALID_SHA,
            }),
        ))


def test_channel_types_are_frozen() -> None:
    ap = AttentionPrior(prior=np.zeros(3, dtype=np.float32))
    with pytest.raises(dataclasses.FrozenInstanceError):
        ap.prior = np.ones(3, dtype=np.float32)  # type: ignore[misc]


def test_channel_output_union_members() -> None:
    members = (WeightUpdate, LatentSample, TopologyDiff, AttentionPrior)
    for member in members:
        assert member in ChannelOutput.__args__


def test_weight_update_rejects_non_finite_fisher_bump() -> None:
    bad_fisher = {"layer0": np.array([np.inf], dtype=np.float32)}
    with pytest.raises(ValueError, match="S2"):
        WeightUpdate(
            lora_delta={"layer0": np.zeros(4, dtype=np.float32)},
            fisher_bump=bad_fisher,
        )
