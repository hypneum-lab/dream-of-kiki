"""M5 prod CIFAR-100 split loader — happy path, offline, determinism."""

from __future__ import annotations

from typing import Any

import pytest

from harness.diffusion_eval.cifar100_split_loader import (
    N_CLASSES_PER_TASK,
    RAW_FEATURE_DIM,
    SplitCifar100Batch,
    load_split_cifar100,
)

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("datasets") is None,
    reason="datasets package not installed",
)


def _has_cifar100_cache() -> bool:
    try:
        from datasets import load_dataset

        load_dataset("uoft-cs/cifar100", split="test")
    except Exception:  # noqa: BLE001 - any cache miss disables the test
        return False
    return True


requires_cache = pytest.mark.skipif(
    not _has_cifar100_cache(),
    reason="uoft-cs/cifar100 not in the HuggingFace cache",
)


@requires_cache
def test_prod_loader_happy_path() -> None:
    batches = list(
        load_split_cifar100(task_idx=0, batch_size=32, seed=0,
                             smoke=False, split="val")
    )
    assert batches, "prod loader yielded no batches"
    first = batches[0]
    assert isinstance(first, SplitCifar100Batch)
    assert first.task_idx == 0
    assert first.features.shape[1] == RAW_FEATURE_DIM
    from typing import cast
    labels = [
        int(lbl)
        for b in batches
        for lbl in cast(list[int], b.labels.tolist())
    ]
    assert min(labels) >= 0
    assert max(labels) < N_CLASSES_PER_TASK


@requires_cache
def test_prod_loader_is_seed_deterministic() -> None:
    def _hashes(seed: int) -> list[tuple[Any, ...]]:
        return [
            (tuple(b.features.reshape(-1)[:8].tolist()),  # type: ignore[arg-type]
             tuple(b.labels.tolist()))  # type: ignore[arg-type]
            for b in load_split_cifar100(
                task_idx=1, batch_size=32, seed=seed,
                smoke=False, split="val")
        ]

    assert _hashes(0) == _hashes(0)
    assert _hashes(0) != _hashes(1)


def test_prod_loader_offline_raises_with_hint(monkeypatch: pytest.MonkeyPatch) -> None:
    import datasets

    def _raise(*_args: Any, **_kwargs: Any) -> None:
        raise ConnectionError("simulated offline cache miss")

    monkeypatch.setattr(datasets, "load_dataset", _raise)
    with pytest.raises(FileNotFoundError, match="huggingface-cli download"):
        list(
            load_split_cifar100(task_idx=0, batch_size=32, seed=0,
                                smoke=False, split="val")
        )
