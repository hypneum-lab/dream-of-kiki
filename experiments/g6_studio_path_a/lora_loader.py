"""Qwen-35B-A3B-V4 base + LoRA adapter loader (Studio-only path).

Wraps ``mlx_lm.load`` + ``mlx_lm.tuner.utils.load_adapters`` into a
single :class:`QwenLoRAWrapper`. When the adapter directory is
absent or does not contain ``adapters.safetensors``, the wrapper
returns ``fresh_init=True`` so the caller can initialise a random
rank-r LoRA stack via :mod:`experiments.g6_studio_path_a.lora_train_step`.

``mlx_lm`` is imported lazily inside :func:`load_qwen_with_adapters`
so Linux CI hosts (no MLX wheel) can import the module and exercise
the wrapper logic via mocks. The full 58 GB SpikingKiki-V4 load
only happens on Studio with ``DREAM_MICRO_KIKI_REAL=1``.

Reference :
- Plan : ``docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md`` Task 2
- Pre-reg : ``docs/osf-prereg-g6-studio-path-a.md`` §5
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QwenLoRAWrapper:
    """Bundle of (model, tokenizer, adapter_path, rank, fresh_init).

    Attributes
    ----------
    model
        Model handle returned by ``mlx_lm.load`` (and optionally
        wrapped by ``mlx_lm.tuner.utils.load_adapters``).
    tokenizer
        Tokenizer handle returned by ``mlx_lm.load``.
    adapter_path
        Resolved path to the adapter directory iff loading succeeded ;
        ``None`` when ``fresh_init`` is True.
    rank
        LoRA rank carried for downstream :class:`TrainHyperparams`
        consistency.
    fresh_init
        ``True`` when no ``adapters.safetensors`` was found and the
        wrapper carries the bare base model — a per-cell random LoRA
        stack will be initialised by the first
        ``train_subdomain_lora`` call.
    """

    model: Any
    tokenizer: Any
    adapter_path: Path | None
    rank: int
    fresh_init: bool


def load_qwen_with_adapters(
    *,
    base_path: str,
    adapter_path: Path,
    rank: int,
) -> QwenLoRAWrapper:
    """Load Qwen base + (optional) LoRA stack via ``mlx_lm``.

    Parameters
    ----------
    base_path
        Filesystem path or HF repo id loadable by ``mlx_lm.load``.
        For G6-Studio Path A the production target is
        ``/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4``.
    adapter_path
        Filesystem path to a directory expected to contain
        ``adapters.safetensors``. When absent the wrapper signals
        ``fresh_init=True`` so the caller can initialise a fresh
        rank-r LoRA stack without raising.
    rank
        LoRA rank carried verbatim onto the returned wrapper.

    Returns
    -------
    QwenLoRAWrapper
        Wrapper bundling model + tokenizer + provenance flags.
    """
    from mlx_lm import load as mlx_load

    model, tokenizer = mlx_load(base_path)
    adapter_file = adapter_path / "adapters.safetensors"
    if adapter_path.is_dir() and adapter_file.is_file():
        from mlx_lm.tuner.utils import load_adapters

        model = load_adapters(model, str(adapter_path))
        return QwenLoRAWrapper(
            model=model,
            tokenizer=tokenizer,
            adapter_path=adapter_path,
            rank=rank,
            fresh_init=False,
        )
    return QwenLoRAWrapper(
        model=model,
        tokenizer=tokenizer,
        adapter_path=None,
        rank=rank,
        fresh_init=True,
    )


__all__ = ["QwenLoRAWrapper", "load_qwen_with_adapters"]
