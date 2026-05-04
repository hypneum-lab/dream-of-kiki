"""Unit tests for the G6-Studio Path A LoRA fine-tune shim.

Mock the KIKI-Mac_tunner ``mlx_lm`` fork's
``mlx_lm.tuner.trainer.train`` to capture the
``(model, optimizer, train_dataset, args)`` payload forwarded by
:func:`train_subdomain_lora`. Asserts that the locked
hyperparameters (Option-B decisions doc 2026-05-04) reach the
fork trainer verbatim through ``TrainingArgs`` + the optimizer
constructor, and that the post-step delta extraction returns only
the ``adapter_keys`` subset.

Reference :
- Plan : ``docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md`` Task 3 step 1
- Fork CLI : ``mlx_lm_fork/lora.py`` lines 250-300.
"""
from __future__ import annotations

import sys
import types
from typing import Any

import numpy as np
import pytest


@pytest.fixture
def mock_tuner(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Inject mocks for the fork's ``mlx_lm`` training surface.

    Captures the kwargs ``train()`` is invoked with — including the
    constructed optimizer, ``TrainingArgs`` instance, and the
    dataset wrapper — so tests can assert hyperparameter forwarding
    without touching MLX.
    """
    captured: dict[str, Any] = {}

    # Fake mlx.optimizers (just need Adam constructor).
    fake_mlx = types.ModuleType("mlx")
    fake_mlx_core = types.ModuleType("mlx.core")
    fake_mlx_optim = types.ModuleType("mlx.optimizers")

    class FakeAdam:
        def __init__(self, learning_rate: float, **kwargs: Any) -> None:
            captured["optimizer_class"] = "Adam"
            captured["learning_rate"] = learning_rate
            captured["optimizer_kwargs"] = kwargs

    fake_mlx_optim.Adam = FakeAdam  # type: ignore[attr-defined]
    fake_mlx.optimizers = fake_mlx_optim  # type: ignore[attr-defined]
    fake_mlx.core = fake_mlx_core  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx", fake_mlx)
    monkeypatch.setitem(sys.modules, "mlx.core", fake_mlx_core)
    monkeypatch.setitem(sys.modules, "mlx.optimizers", fake_mlx_optim)

    # Fake mlx_lm.tuner.{trainer,datasets,utils}.
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_tuner = types.ModuleType("mlx_lm.tuner")

    fake_trainer = types.ModuleType("mlx_lm.tuner.trainer")

    class FakeTrainingArgs:
        def __init__(
            self,
            *,
            batch_size: int = 4,
            iters: int = 100,
            **kwargs: Any,
        ) -> None:
            self.batch_size = batch_size
            self.iters = iters
            self.extra = kwargs

    def fake_train(
        *,
        model: Any,
        optimizer: Any,
        train_dataset: Any,
        args: Any,
        **kwargs: Any,
    ) -> None:
        captured["model"] = model
        captured["optimizer"] = optimizer
        captured["train_dataset"] = train_dataset
        captured["args"] = args
        captured["len_train"] = len(train_dataset)

    fake_trainer.TrainingArgs = FakeTrainingArgs  # type: ignore[attr-defined]
    fake_trainer.train = fake_train  # type: ignore[attr-defined]

    fake_datasets = types.ModuleType("mlx_lm.tuner.datasets")

    class FakeCacheDataset:
        def __init__(self, data: Any) -> None:
            self._data = data

        def __len__(self) -> int:
            return len(self._data)

    fake_datasets.CacheDataset = FakeCacheDataset  # type: ignore[attr-defined]

    fake_utils = types.ModuleType("mlx_lm.tuner.utils")

    def fake_linear_to_lora_layers(
        model: Any, num_layers: int, config: dict, use_dora: bool = False,
    ) -> None:
        captured["lora_applied"] = True
        captured["lora_num_layers"] = num_layers
        captured["lora_config"] = dict(config)

    fake_utils.linear_to_lora_layers = fake_linear_to_lora_layers  # type: ignore[attr-defined]

    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner", fake_tuner)
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.trainer", fake_trainer)
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.datasets", fake_datasets)
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.utils", fake_utils)
    return captured


class _FakeTokenizer:
    eos_token_id = 99

    def encode(self, text: str) -> list[int]:
        return [1, 2, 3]


def test_train_step_forwards_locked_hyperparams(
    mock_tuner: dict[str, Any],
) -> None:
    """TDD-3.1 — lr=1e-4 / iters=50 / batch_size=1 forwarded via fork API."""
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
        tokenizer=_FakeTokenizer(),
        train_records=[{"text": "Q1"}, {"text": "Q2"}],
        hyperparams=hp,
        adapter_keys=("layer_0_lora_B",),
    )
    assert mock_tuner["len_train"] == 2
    # Optimizer hyperparameter (lr) must reach mlx.optimizers.Adam.
    assert mock_tuner["optimizer_class"] == "Adam"
    assert mock_tuner["learning_rate"] == 1e-4
    # iters / batch_size must reach TrainingArgs.
    args = mock_tuner["args"]
    assert args.iters == 50
    assert args.batch_size == 1
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
        tokenizer=_FakeTokenizer(),
        train_records=[{"text": "x"}],
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
        tokenizer=_FakeTokenizer(),
        train_records=[{"text": "x"}],
        hyperparams=hp,
        adapter_keys=("layer_0_lora_B", "missing_key"),
    )
    assert set(delta) == {"layer_0_lora_B"}


def test_train_step_applies_lora_layers_when_fresh_init(
    mock_tuner: dict[str, Any],
) -> None:
    """TDD-3.4 — apply_lora_layers=True triggers linear_to_lora_layers.

    Mirrors the fork's lora.py CLI sequence — when the wrapper
    carries ``fresh_init=True`` the driver must apply LoRA layers
    before training. Scale is mapped from ``alpha / rank``.
    """
    from experiments.g6_studio_path_a.lora_train_step import (
        TrainHyperparams,
        train_subdomain_lora,
    )

    class FakeModel:
        def __init__(self) -> None:
            self.frozen = False

        def freeze(self) -> None:
            self.frozen = True

        def parameters(self) -> dict[str, np.ndarray]:
            return {"layer_0_lora_B": np.zeros((4, 8), dtype=np.float32)}

    hp = TrainHyperparams(
        lr=1e-4, iters=1, rank=8, alpha=16, batch_size=1,
    )
    train_subdomain_lora(
        model=FakeModel(),
        tokenizer=_FakeTokenizer(),
        train_records=[{"text": "x"}],
        hyperparams=hp,
        adapter_keys=("layer_0_lora_B",),
        apply_lora_layers=True,
    )
    assert mock_tuner.get("lora_applied") is True
    assert mock_tuner["lora_num_layers"] == 8
    cfg = mock_tuner["lora_config"]
    assert cfg["rank"] == 8
    # alpha=16 / rank=8 → scale=2.0
    assert cfg["scale"] == pytest.approx(2.0)
    assert cfg["dropout"] == 0.0
