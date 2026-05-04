"""Per-subdomain real LoRA fine-tune via the KIKI-Mac_tunner ``mlx_lm`` fork.

Drives one fine-tune pass over a subdomain's training records, then
extracts the resulting LoRA delta — restricted to the
``adapter_keys`` subset — as numpy arrays so the dream runtime
(numpy-only) can mutate them in place via
:func:`experiments.g6_studio_path_a.dream_episode_real.dream_episode_real`.

This shim binds to the FORK API surface
(``/Users/clems/KIKI-Mac_tunner/lib/mlx_lm_fork``) which is the
flavour that ships on Studio M3 Ultra. The fork's ``train()`` lives
under ``mlx_lm.tuner.trainer`` (not ``mlx_lm.tuner.lora``), takes
``(model, optimizer, train_dataset, ...)`` as positional / keyword
arguments, and consumes a :class:`mlx_lm.tuner.datasets.CacheDataset`
wrapping a tokenised dataset object (one that exposes
``__getitem__``, ``__len__``, and ``process()``). API drift = log a
deviation in ``docs/osf-deviations-g6-studio-path-a-*.md`` BEFORE
re-running.

Reference :
- Plan : ``docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md`` Task 3
- Pre-reg : ``docs/osf-prereg-g6-studio-path-a.md`` §5
- Fork CLI reference : ``mlx_lm_fork/lora.py`` lines 250-300.
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

    Note on ``alpha`` : the fork's ``linear_to_lora_layers`` does not
    accept a separate ``alpha`` argument ; instead, LoRA scale is
    passed as a single ``scale`` value in the layer config dict. We
    map ``scale = alpha / rank`` per the canonical LoRA paper
    convention (Hu et al. 2021 §3 ; matches HF PEFT default).
    """

    lr: float
    iters: int
    rank: int
    alpha: int
    batch_size: int


class _MMLURecordDataset:
    """Thin fork-compatible dataset wrapping MMLU-style records.

    Implements the trio expected by
    :class:`mlx_lm.tuner.datasets.CacheDataset` :
    ``__len__`` / ``__getitem__`` (raw record passthrough) and
    ``process(record) -> (tokens, offset)`` consumed by
    ``iterate_batches``.

    Records may be either :class:`harness.real_benchmarks.mmlu.MMLURecord`
    instances or ``{"text": ...}`` dicts (the latter is what the
    existing G6-Studio test fixtures inject). MMLU records are
    formatted into the canonical ``Question: … Answer: <letter>``
    text and tokenised end-to-end ; dict records use the ``text``
    key verbatim.
    """

    def __init__(self, data: Sequence[Any], tokenizer: Any) -> None:
        self._data = list(data)
        self._tokenizer = tokenizer

    def __len__(self) -> int:
        return len(self._data)

    def __getitem__(self, idx: int) -> Any:
        return self._data[idx]

    def process(self, record: Any) -> tuple[list[int], int]:
        """Tokenise one record and return ``(tokens, mask_offset=0)``."""
        if isinstance(record, dict) and "text" in record:
            text = str(record["text"])
        elif hasattr(record, "question") and hasattr(record, "choices"):
            letter = "ABCD"[int(record.answer)]
            text = (
                f"Question: {record.question}\n"
                f"A. {record.choices[0]}\n"
                f"B. {record.choices[1]}\n"
                f"C. {record.choices[2]}\n"
                f"D. {record.choices[3]}\n"
                f"Answer: {letter}"
            )
        else:
            text = str(record)
        tokens = self._tokenizer.encode(text)
        eos = getattr(self._tokenizer, "eos_token_id", None)
        if eos is not None and (not tokens or tokens[-1] != eos):
            tokens.append(eos)
        return (tokens, 0)


def train_subdomain_lora(
    *,
    model: Any,
    tokenizer: Any,
    train_records: Sequence[Any],
    hyperparams: TrainHyperparams,
    adapter_keys: tuple[str, ...],
    apply_lora_layers: bool = False,
) -> dict[str, NDArray[np.float32]]:
    """Run one fine-tune pass and return the post-step LoRA delta subset.

    Parameters
    ----------
    model
        Model handle as carried on
        :class:`experiments.g6_studio_path_a.lora_loader.QwenLoRAWrapper`.
        Must expose a ``parameters()`` method returning a dict of
        named tensors (numpy-castable). When ``apply_lora_layers`` is
        True, must also expose ``layers`` and ``freeze()`` per the
        fork's ``nn.Module`` contract.
    tokenizer
        Tokenizer handle (used to encode each record before training
        and forwarded to the trainer).
    train_records
        Subdomain training records — either ``MMLURecord`` instances
        or ``{"text": ...}`` dicts. Wrapped in
        :class:`_MMLURecordDataset` then in
        :class:`mlx_lm.tuner.datasets.CacheDataset`.
    hyperparams
        Locked :class:`TrainHyperparams` ; mapped onto
        ``mlx.optimizers.Adam`` + ``mlx_lm.tuner.trainer.TrainingArgs``.
    adapter_keys
        Names of LoRA tensors to extract from ``model.parameters()``
        after training. The production default is
        ``("layer_0_lora_B",)`` (single sparse tap, Option B).
    apply_lora_layers
        When True, invoke ``mlx_lm.tuner.utils.linear_to_lora_layers``
        before training to convert the bare base model's linear
        layers into LoRA layers. Set this iff the wrapper carries
        ``fresh_init=True`` (i.e. no ``adapters.safetensors`` was
        loaded). When False (default), assume the caller already ran
        ``load_adapters`` so LoRA layers are present and unfrozen.

    Returns
    -------
    dict[str, NDArray[np.float32]]
        Mapping ``{adapter_key -> ndarray}`` for keys that are
        present in ``model.parameters()`` after training. Missing
        keys are silently dropped — the caller decides whether the
        live-delta dict is allowed to be empty.
    """
    import mlx.optimizers as optim

    from mlx_lm.tuner.datasets import CacheDataset
    from mlx_lm.tuner.trainer import TrainingArgs, train as fork_train

    if apply_lora_layers:
        from mlx_lm.tuner.utils import linear_to_lora_layers

        model.freeze()
        # Fork's LoRA scale = alpha / rank (canonical LoRA convention).
        lora_cfg = {
            "rank": hyperparams.rank,
            "scale": float(hyperparams.alpha) / max(hyperparams.rank, 1),
            "dropout": 0.0,
        }
        # ``num_layers`` controls how many top blocks become LoRA ;
        # we map our ``rank`` onto the number of touched blocks for
        # parity with G6 Path B's "single sparse tap" semantics.
        linear_to_lora_layers(model, hyperparams.rank, lora_cfg)

    optimizer = optim.Adam(learning_rate=hyperparams.lr)
    training_args = TrainingArgs(
        batch_size=hyperparams.batch_size,
        iters=hyperparams.iters,
    )
    dataset = _MMLURecordDataset(train_records, tokenizer)
    fork_train(
        model=model,
        optimizer=optimizer,
        train_dataset=CacheDataset(dataset),
        args=training_args,
    )
    params = model.parameters()
    return {
        k: np.asarray(params[k], dtype=np.float32)
        for k in adapter_keys
        if k in params
    }


__all__ = ["TrainHyperparams", "train_subdomain_lora"]
