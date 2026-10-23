"""G4-nono domain stream composition (spec section 5)."""
from __future__ import annotations

import numpy as np

from experiments.g4_nono_test.stream import (
    DOMAIN_A,
    DOMAIN_B,
    TASK_CLASS_PAIRS,
    build_domain_stream,
    domain_of,
)


def test_domains_partition_20_coarse_ids() -> None:
    assert DOMAIN_A == (0, 1, 7, 8, 11, 12, 13, 14, 15, 16)
    assert DOMAIN_B == (2, 3, 4, 5, 6, 9, 10, 17, 18, 19)
    assert sorted(DOMAIN_A + DOMAIN_B) == list(range(20))


def test_task_pairs_exact_order() -> None:
    assert TASK_CLASS_PAIRS == (
        (0, 1), (7, 8), (11, 12), (13, 14), (15, 16),
        (2, 3), (4, 5), (6, 9), (10, 17), (18, 19),
    )
    assert all(domain_of(c) == "A" for p in TASK_CLASS_PAIRS[:5] for c in p)
    assert all(domain_of(c) == "B" for p in TASK_CLASS_PAIRS[5:] for c in p)


def _fake(n_per: int, seed: int):
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(20, dtype=np.uint8), n_per)
    x = rng.integers(0, 256, (len(y), 32, 32, 3), dtype=np.uint8)
    return x, y


def test_build_stream_uses_coarse_ids_and_all_images() -> None:
    xtr, ytr = _fake(5, 0)
    xte, yte = _fake(2, 1)
    stream = build_domain_stream(xtr, ytr, xte, yte)
    assert len(stream) == 10
    for task, pair in zip(stream, TASK_CLASS_PAIRS):
        assert sorted(set(task["y_train"].tolist())) == sorted(pair)
        assert sorted(set(task["y_test"].tolist())) == sorted(pair)
        assert task["x_train_nhwc"].shape == (10, 32, 32, 3)
        assert task["x_train_nhwc"].dtype == np.float32
        assert task["x_train_nhwc"].max() <= 1.0
        assert task["y_train"].dtype == np.int64


def test_build_stream_is_deterministic_no_reshuffle() -> None:
    xtr, ytr = _fake(5, 0)
    xte, yte = _fake(2, 1)
    a = build_domain_stream(xtr, ytr, xte, yte)
    b = build_domain_stream(xtr, ytr, xte, yte)
    for ta, tb in zip(a, b):
        assert np.array_equal(ta["x_train_nhwc"], tb["x_train_nhwc"])
