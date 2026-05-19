"""Unit tests for the four dream-awake channel-output types."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kiki_oniric.dream.channels import (
    AttentionPrior,
    ChannelOutput,
    HierarchyDiff,
    LatentSample,
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


def test_hierarchy_diff_holds_tuple() -> None:
    hd = HierarchyDiff(diff=(("add_node", {"id": "n1"}),))
    assert hd.diff[0][0] == "add_node"


def test_channel_types_are_frozen() -> None:
    ap = AttentionPrior(prior=np.zeros(3, dtype=np.float32))
    with pytest.raises(dataclasses.FrozenInstanceError):
        ap.prior = np.ones(3, dtype=np.float32)  # type: ignore[misc]


def test_channel_output_union_members() -> None:
    members = (WeightUpdate, LatentSample, HierarchyDiff, AttentionPrior)
    for member in members:
        assert member in ChannelOutput.__args__  # type: ignore[attr-defined]
