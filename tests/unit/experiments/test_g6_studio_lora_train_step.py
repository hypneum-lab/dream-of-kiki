"""Unit tests for the G6-Studio Path A LoRA fine-tune shim.

Mock ``mlx_lm.tuner.lora`` to capture the ``args`` payload forwarded
by :func:`train_subdomain_lora`. Asserts that the locked
hyperparameters (Option-B decisions doc 2026-05-04) reach the
upstream trainer verbatim and that the post-step delta extraction
returns only the ``adapter_keys`` subset.

Reference :
- Plan : ``docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md`` Task 3 step 1
"""
from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def mock_tuner(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject a mock ``mlx_lm.tuner.lora.train`` capturing its args."""
    captured: dict[str, Any] = {}

    def fake_train(
        model: Any,
        tokenizer: Any,
        *,
        args: dict[str, Any],
        train_set: list[Any],
        **kwargs: Any,
    ) -> None:
        captured["args"] = args
        captured["len_train"] = len(train_set)

    fake_lora = types.ModuleType("mlx_lm.tuner.lora")
    fake_lora.train = fake_train  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "mlx_lm", types.ModuleType("mlx_lm"),
    )
    monkeypatch.setitem(
        sys.modules, "mlx_lm.tuner", types.ModuleType("mlx_lm.tuner"),
    )
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.lora", fake_lora)
    return captured


def test_train_step_forwards_locked_hyperparams(
    mock_tuner: dict[str, Any],
) -> None:
    """TDD-3.1 — lr=1e-4 / iters=50 / rank=8 / alpha=16 forwarded."""
    from experiments.g6_studio_path_a.lora_train_step import (
        TrainHyperparams,
        train_subdomain_lora,
    )

    class FakeModel:
        def parameters(self) -> dict[str, np.ndarray]:
            return {
                "layer_0_lora_B": np.zeros((8, 8), dtype=np.float32),
            }

    hp = TrainHyperparams(
        lr=1e-4, iters=50, rank=8, alpha=16, batch_size=1,
    )
    delta = train_subdomain_lora(
        model=FakeModel(),
        tokenizer=None,
        train_records=[{"text": "Q1"}, {"text": "Q2"}],
        hyperparams=hp,
        adapter_keys=("layer_0_lora_B",),
    )
    args = mock_tuner["args"]
    assert mock_tuner["len_train"] == 2
    assert args["learning_rate"] == 1e-4
    assert args["iters"] == 50
    assert args["batch_size"] == 1
    assert args["lora_layers"] == 8
    assert args["lora_alpha"] == 16
    assert "layer_0_lora_B" in delta
    assert delta["layer_0_lora_B"].dtype == np.float32


def test_train_step_returns_delta_subset(
    mock_tuner: dict[str, Any],
) -> None:
    """TDD-3.2 — only adapter_keys subset returned ; base weights excluded."""
    from experiments.g6_studio_path_a.lora_train_step import (
        TrainHyperparams,
        train_subdomain_lora,
    )

    class FakeModel:
        def parameters(self) -> dict[str, np.ndarray]:
            return {
                "layer_0_lora_B": np.ones((4, 8), dtype=np.float32),
                "layer_0_base_W": np.ones(
                    (4096, 4096), dtype=np.float32,
                ),
                "layer_1_lora_B": np.ones((4, 8), dtype=np.float32),
            }

    hp = TrainHyperparams(
        lr=1e-4, iters=1, rank=8, alpha=16, batch_size=1,
    )
    delta = train_subdomain_lora(
        model=FakeModel(),
        tokenizer=None,
        train_records=[],
        hyperparams=hp,
        adapter_keys=("layer_0_lora_B", "layer_1_lora_B"),
    )
    assert set(delta) == {"layer_0_lora_B", "layer_1_lora_B"}
    assert delta["layer_0_lora_B"].shape == (4, 8)
    assert "layer_0_base_W" not in delta


def test_train_step_handles_missing_adapter_key(
    mock_tuner: dict[str, Any],
) -> None:
    """TDD-3.3 — adapter_key absent from model.parameters() is dropped."""
    from experiments.g6_studio_path_a.lora_train_step import (
        TrainHyperparams,
        train_subdomain_lora,
    )

    class FakeModel:
        def parameters(self) -> dict[str, np.ndarray]:
            return {"layer_0_lora_B": np.zeros((4, 8), dtype=np.float32)}

    hp = TrainHyperparams(
        lr=1e-4, iters=1, rank=8, alpha=16, batch_size=1,
    )
    delta = train_subdomain_lora(
        model=FakeModel(),
        tokenizer=None,
        train_records=[],
        hyperparams=hp,
        adapter_keys=("layer_0_lora_B", "missing_key"),
    )
    assert set(delta) == {"layer_0_lora_B"}
