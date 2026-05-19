"""Unit tests for the LoRA-adapter model abstraction (B1a)."""
from __future__ import annotations

import pytest
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import mlx.core as mx
    from mlx.utils import tree_flatten
else:
    mx = pytest.importorskip("mlx.core")
    tree_flatten = pytest.importorskip("mlx.utils").tree_flatten

import numpy as np

from kiki_oniric.substrates.micro_kiki.lora_model import (
    LoRALinear,
    LoRAModel,
    adapter_delta,
)


def test_lora_linear_shapes() -> None:
    layer = LoRALinear(4, 8, rank=2, alpha=4.0, key=mx.random.key(0))
    assert layer.base_weight.shape == (8, 4)
    assert layer.lora_a.shape == (2, 4)
    assert layer.lora_b.shape == (8, 2)
    assert layer.bias.shape == (8,)


def test_lora_linear_b_init_is_zero() -> None:
    layer = LoRALinear(4, 8, rank=2, alpha=4.0, key=mx.random.key(0))
    assert bool(mx.all(mx.equal(layer.lora_b, 0.0)).item())


def test_lora_linear_initial_forward_equals_base() -> None:
    layer = LoRALinear(4, 8, rank=2, alpha=4.0, key=mx.random.key(1))
    x = mx.array([[0.1, 0.2, 0.3, 0.4]])
    got = layer(x)
    expected = x @ layer.base_weight.T + layer.bias
    assert bool(mx.allclose(got, expected).item())


def test_lora_linear_nonzero_b_changes_output() -> None:
    layer = LoRALinear(4, 8, rank=2, alpha=4.0, key=mx.random.key(2))
    x = mx.array([[0.1, 0.2, 0.3, 0.4]])
    base_out = layer(x)
    layer.lora_b = mx.ones((8, 2))
    assert not bool(mx.allclose(layer(x), base_out).item())


def test_lora_linear_scale_applied() -> None:
    layer = LoRALinear(3, 3, rank=1, alpha=6.0, key=mx.random.key(3))
    layer.lora_a = mx.ones((1, 3))
    layer.lora_b = mx.ones((3, 1))
    x = mx.array([[1.0, 0.0, 0.0]])
    delta = 6.0 * (layer.lora_b @ layer.lora_a)
    expected = x @ (layer.base_weight + delta).T + layer.bias
    assert bool(mx.allclose(layer(x), expected).item())


def test_lora_linear_base_weight_is_frozen() -> None:
    layer = LoRALinear(4, 8, rank=2, alpha=4.0, key=mx.random.key(4))
    trainable = dict(tree_flatten(layer.trainable_parameters()))
    assert "base_weight" not in trainable
    assert "bias" not in trainable
    assert "lora_a" in trainable
    assert "lora_b" in trainable


def test_lora_linear_rejects_nonpositive_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(4, 8, rank=0, alpha=4.0, key=mx.random.key(0))


def test_lora_model_forward_shape() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    x = mx.array([[0.1, 0.2, 0.3, 0.4]])
    assert model(x).shape == (1, 2)


def test_lora_model_adapter_parameters_keys() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    params = model.adapter_parameters()
    assert set(params) == {
        "layer0.lora_a",
        "layer0.lora_b",
        "layer1.lora_a",
        "layer1.lora_b",
    }
    assert params["layer0.lora_a"].shape == (2, 4)
    assert params["layer1.lora_b"].shape == (2, 2)
    assert params["layer0.lora_b"].shape == (8, 2)
    assert params["layer1.lora_a"].shape == (2, 8)


def test_lora_model_adapter_parameters_excludes_base() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    for name in model.adapter_parameters():
        assert "base_weight" not in name
        assert "bias" not in name


def test_lora_model_is_deterministic_under_seed() -> None:
    m1 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=7)
    m2 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=7)
    assert bool(
        mx.allclose(m1.layers[0].base_weight, m2.layers[0].base_weight).item()
    )
    assert bool(
        mx.allclose(m1.layers[1].lora_a, m2.layers[1].lora_a).item()
    )


def test_lora_model_rejects_too_few_sizes() -> None:
    with pytest.raises(ValueError, match="layer_sizes"):
        LoRAModel((4,), rank=2, alpha=4.0, seed=0)


def test_lora_model_different_seeds_differ() -> None:
    m1 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=7)
    m2 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=99)
    assert not bool(
        mx.allclose(m1.layers[0].base_weight, m2.layers[0].base_weight).item()
    )


def test_lora_linear_no_bias() -> None:
    layer = LoRALinear(4, 8, rank=2, alpha=4.0, bias=False, key=mx.random.key(5))
    assert not hasattr(layer, "bias")
    x = mx.array([[0.1, 0.2, 0.3, 0.4]])
    assert layer(x).shape == (1, 8)
    trainable = dict(tree_flatten(layer.trainable_parameters()))
    assert "bias" not in trainable
    assert "base_weight" not in trainable


def test_adapter_delta_computes_float32_difference() -> None:
    before = {"layer0.lora_a": mx.zeros((2, 4))}
    after = {"layer0.lora_a": mx.ones((2, 4))}
    delta = adapter_delta(before, after)
    assert delta["layer0.lora_a"].dtype == np.float32
    assert delta["layer0.lora_a"].shape == (2, 4)
    assert bool((delta["layer0.lora_a"] == 1.0).all())


def test_adapter_delta_rejects_key_mismatch() -> None:
    with pytest.raises(ValueError, match="keys"):
        adapter_delta({"a": mx.zeros((1,))}, {"b": mx.zeros((1,))})
