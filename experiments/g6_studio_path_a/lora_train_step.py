"""Per-subdomain real LoRA fine-tune via ``mlx_lm.tuner.lora``.

Drives one fine-tune pass over a subdomain's training records, then
extracts the resulting LoRA delta — restricted to the
``adapter_keys`` subset — as numpy arrays so the dream runtime
(numpy-only) can mutate them in place via
:func:`experiments.g6_studio_path_a.dream_episode_real.dream_episode_real`.

The ``mlx_lm.tuner.lora.train`` signature drifts across versions ;
this shim binds to the 0.31.x flavour observed on Studio (Task 0
step 2 in the plan). API drift = log a deviation in
``docs/osf-deviations-g6-studio-path-a-*.md`` BEFORE re-running.

Reference :
- Plan : ``docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md`` Task 3
- Pre-reg : ``docs/osf-prereg-g6-studio-path-a.md`` §5
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrainHyperparams:
    """Locked LoRA fine-tune hyperparameters (decisions doc 2026-05-04).

    Default Option-B values : ``lr=1e-4, iters=50, rank=8, alpha=16,
    batch_size=1`` (5-min/cell budget on Studio M3 Ultra at 35B bf16).
    """

    lr: float
    iters: int
    rank: int
    alpha: int
    batch_size: int


def train_subdomain_lora(
    *,
    model: Any,
    tokenizer: Any,
    train_records: Sequence[Any],
    hyperparams: TrainHyperparams,
    adapter_keys: tuple[str, ...],
) -> dict[str, NDArray[np.float32]]:
    """Run one fine-tune pass and return the post-step LoRA delta subset.

    Parameters
    ----------
    model
        Model handle as carried on
        :class:`experiments.g6_studio_path_a.lora_loader.QwenLoRAWrapper`.
        Must expose a ``parameters()`` method returning a dict of
        named tensors (numpy-castable).
    tokenizer
        Tokenizer handle (forwarded to ``mlx_lm.tuner.lora.train``).
    train_records
        Subdomain training records (consumed verbatim by
        ``mlx_lm.tuner.lora.train``).
    hyperparams
        Locked :class:`TrainHyperparams` ; forwarded as the ``args``
        kwarg of ``mlx_lm.tuner.lora.train``.
    adapter_keys
        Names of LoRA tensors to extract from ``model.parameters()``
        after training. The production default is
        ``("layer_0_lora_B",)`` (single sparse tap, Option B).

    Returns
    -------
    dict[str, NDArray[np.float32]]
        Mapping ``{adapter_key -> ndarray}`` for keys that are
        present in ``model.parameters()`` after training. Missing
        keys are silently dropped — the caller decides whether the
        live-delta dict is allowed to be empty.
    """
    from mlx_lm.tuner.lora import train as lora_train

    args = {
        "learning_rate": hyperparams.lr,
        "iters": hyperparams.iters,
        "batch_size": hyperparams.batch_size,
        "lora_layers": hyperparams.rank,
        "lora_alpha": hyperparams.alpha,
    }
    lora_train(
        model,
        tokenizer,
        args=args,
        train_set=list(train_records),
    )
    params = model.parameters()
    return {
        k: np.asarray(params[k], dtype=np.float32)
        for k in adapter_keys
        if k in params
    }


__all__ = ["TrainHyperparams", "train_subdomain_lora"]
