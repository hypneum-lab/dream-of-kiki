"""Unit tests for the LoRA downscale handler (B2)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import mlx.core as mx
else:
    mx = pytest.importorskip("mlx.core")

from kiki_oniric.dream.channels import WeightUpdate
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.downscale_real import (
    DownscaleRealState,
    downscale_lora_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime, EpisodeLogEntry
from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


def _seed_nontrivial_b(model: LoRAModel) -> None:
    """Replace each ``lora_b`` with a known non-zero pattern.

    B1a inits ``B = 0`` so a shrinkage of an untouched model would
    leave ``lora_b`` deltas at zero. Seeding B makes the SHY
    shrinkage signal observable on both A and B.
    """
    for idx, layer in enumerate(model.layers):
        layer.lora_b = mx.ones(layer.lora_b.shape) * (0.1 * (idx + 1))


def _episode(factor: float) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"shrink_factor": factor},
        operation_set=(Operation.DOWNSCALE,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-lora-down",
    )


def _run(
    model: LoRAModel, factor: float,
) -> tuple[DownscaleRealState, EpisodeLogEntry]:
    state = DownscaleRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.DOWNSCALE,
        downscale_lora_handler(state, model=model),
    )
    runtime.execute(_episode(factor))
    return state, runtime.log[-1]


def test_downscale_lora_emits_weight_update() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    _, entry = _run(model, 0.5)
    assert isinstance(entry.channel_outputs[0], WeightUpdate)


def test_downscale_lora_delta_keys_match_adapters() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    _, entry = _run(model, 0.5)
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    assert set(out.lora_delta) == set(model.adapter_parameters())
    for arr in out.lora_delta.values():
        assert arr.dtype == np.float32
        assert bool(np.isfinite(arr).all())


def test_downscale_lora_deltas_are_non_positive() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    # capture before
    before = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    _, entry = _run(model, 0.5)
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    # Δ = before * (f - 1) — non-positive when before ≥ 0 and f ≤ 1.
    # Since A is seeded-random (signed) and B is positive here,
    # check the sign rule per-element: sign(Δ) == -sign(before).
    for k, delta in out.lora_delta.items():
        expected = before[k] * (0.5 - 1.0)
        np.testing.assert_allclose(delta, expected, rtol=1e-5, atol=1e-6)


def test_downscale_lora_magnitudes_scale_by_factor() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    before = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    _, entry = _run(model, 0.5)
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    for k, delta in out.lora_delta.items():
        np.testing.assert_allclose(
            np.abs(delta), 0.5 * np.abs(before[k]),
            rtol=1e-5, atol=1e-6,
        )


def test_downscale_lora_composed_dense_delta_property() -> None:
    """Recomposed ΔW = scale * (f² - 1) * (B_before @ A_before)."""
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    layer0 = model.layers[0]
    a_before = np.asarray(layer0.lora_a, dtype=np.float32)
    b_before = np.asarray(layer0.lora_b, dtype=np.float32)
    scale = layer0.scale
    factor = 0.5
    _, entry = _run(model, factor)
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    a_after = a_before + out.lora_delta["layer0.lora_a"]
    b_after = b_before + out.lora_delta["layer0.lora_b"]
    composed = scale * (b_after @ a_after - b_before @ a_before)
    expected = scale * (factor * factor - 1.0) * (b_before @ a_before)
    np.testing.assert_allclose(composed, expected, rtol=1e-5, atol=1e-6)


def test_downscale_lora_factor_one_is_noop() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    before = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    state, entry = _run(model, 1.0)
    assert entry.channel_outputs[0] is None
    assert state.last_compute_flops == 0
    after = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    for k in before:
        np.testing.assert_array_equal(before[k], after[k])


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2.0])
def test_downscale_lora_rejects_out_of_bounds_factor(bad: float) -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    before = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    state = DownscaleRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.DOWNSCALE,
        downscale_lora_handler(state, model=model),
    )
    with pytest.raises(ValueError, match="shrink_factor"):
        runtime.execute(_episode(bad))
    # validation-before-mutation: model untouched
    after = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    for k in before:
        np.testing.assert_array_equal(before[k], after[k])


def test_downscale_lora_compounds_multiplicatively() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    state = DownscaleRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.DOWNSCALE,
        downscale_lora_handler(state, model=model),
    )
    runtime.execute(_episode(0.9))
    runtime.execute(_episode(0.8))
    assert state.compound_factor == pytest.approx(0.72, rel=1e-6)


def test_downscale_lora_tags_k1_flops() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    state, _ = _run(model, 0.5)
    assert state.last_compute_flops > 0


def test_downscale_lora_is_deterministic() -> None:
    m1 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=3)
    m2 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=3)
    _seed_nontrivial_b(m1)
    _seed_nontrivial_b(m2)
    _, e1 = _run(m1, 0.5)
    _, e2 = _run(m2, 0.5)
    o1, o2 = e1.channel_outputs[0], e2.channel_outputs[0]
    assert isinstance(o1, WeightUpdate) and isinstance(o2, WeightUpdate)
    for k in o1.lora_delta:
        np.testing.assert_array_equal(o1.lora_delta[k], o2.lora_delta[k])
