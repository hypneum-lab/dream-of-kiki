"""Unit tests for the G6-Studio Path A LoRA loader.

Tests mock ``mlx_lm`` so they exercise the wrapper logic on Linux
CI without touching MLX. The full 58 GB SpikingKiki-V4 load only
fires on Studio with ``DREAM_MICRO_KIKI_REAL=1`` (out of scope for
unit tests).

Reference :
- Plan : ``docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md`` Task 2 step 1
"""
from __future__ import annotations

import sys
import types
from pathlib import Path

import pytest


@pytest.fixture
def mock_mlx_lm(monkeypatch: pytest.MonkeyPatch) -> None:
    """Inject mocks for ``mlx_lm`` and ``mlx_lm.tuner.utils``."""
    fake_mlx_lm = types.ModuleType("mlx_lm")
    fake_mlx_lm.load = lambda path: ("MOCK_MODEL", "MOCK_TOK")  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm", fake_mlx_lm)

    fake_tuner = types.ModuleType("mlx_lm.tuner")
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner", fake_tuner)

    fake_utils = types.ModuleType("mlx_lm.tuner.utils")
    fake_utils.load_adapters = lambda model, path: f"WITH_ADAPTERS:{path}"  # type: ignore[attr-defined]
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.utils", fake_utils)


def test_load_with_adapters_present(
    tmp_path: Path, mock_mlx_lm: None,
) -> None:
    """TDD-2.1 — adapter dir present + safetensors → wrapper carries both."""
    from experiments.g6_studio_path_a.lora_loader import (
        load_qwen_with_adapters,
    )

    adapters = tmp_path / "SpikingKiki-V4-adapters"
    adapters.mkdir()
    (adapters / "adapters.safetensors").write_bytes(b"\x00" * 8)

    wrapper = load_qwen_with_adapters(
        base_path="/fake/qwen",
        adapter_path=adapters,
        rank=8,
    )
    assert wrapper.model == "WITH_ADAPTERS:" + str(adapters)
    assert wrapper.tokenizer == "MOCK_TOK"
    assert wrapper.adapter_path == adapters
    assert wrapper.fresh_init is False
    assert wrapper.rank == 8


def test_load_without_adapters_fresh_init(
    tmp_path: Path, mock_mlx_lm: None,
) -> None:
    """TDD-2.2 — adapter dir absent → fresh_init=True signalled."""
    from experiments.g6_studio_path_a.lora_loader import (
        load_qwen_with_adapters,
    )

    wrapper = load_qwen_with_adapters(
        base_path="/fake/qwen",
        adapter_path=tmp_path / "missing-dir",
        rank=8,
    )
    assert wrapper.model == "MOCK_MODEL"
    assert wrapper.tokenizer == "MOCK_TOK"
    assert wrapper.adapter_path is None
    assert wrapper.fresh_init is True
    assert wrapper.rank == 8


def test_load_with_dir_but_no_safetensors_fresh_init(
    tmp_path: Path, mock_mlx_lm: None,
) -> None:
    """TDD-2.3 — empty adapter dir (no .safetensors) → fresh_init=True."""
    from experiments.g6_studio_path_a.lora_loader import (
        load_qwen_with_adapters,
    )

    adapters = tmp_path / "empty-adapters"
    adapters.mkdir()

    wrapper = load_qwen_with_adapters(
        base_path="/fake/qwen",
        adapter_path=adapters,
        rank=16,
    )
    assert wrapper.fresh_init is True
    assert wrapper.adapter_path is None
    assert wrapper.rank == 16
