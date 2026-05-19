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

from kiki_oniric.substrates.micro_kiki.lora_model import LoRALinear


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
