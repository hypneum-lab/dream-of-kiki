"""Unit tests for apply_channel_outputs() and concrete channels (B5)."""
from __future__ import annotations

from typing import TYPE_CHECKING

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
