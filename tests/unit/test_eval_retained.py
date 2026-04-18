"""Unit tests for retained benchmark eval bridge (S9.3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.benchmarks.retained.retained import (
    RetainedBenchmark,
    load_retained,
)
from kiki_oniric.dream.eval_retained import evaluate_retained


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "harness" / "benchmarks" / "retained"


def _always_correct_model(item: dict) -> str:
    """Mock model: returns the expected output verbatim."""
    return item["expected"]


def _always_wrong_model(item: dict) -> str:
    """Mock model: returns a string that never matches."""
    return "WRONG_PREDICTION_NEVER_MATCHES"


def _half_correct_model(item: dict) -> str:
    """Mock model: correct on even ids, wrong on odd ids."""
    rid_token = item["id"].split("-")[-1]
    rid = int(rid_token)
    if rid % 2 == 0:
        return item["expected"]
    return "WRONG"


@pytest.fixture
def real_benchmark() -> RetainedBenchmark:
    return load_retained(BENCH_DIR)


def test_evaluate_retained_empty_returns_one() -> None:
    """No items to evaluate → vacuous pass at accuracy 1.0."""
    empty_bench = RetainedBenchmark(
        items=[], hash_verified=True, source_hash="0" * 64
    )
    assert evaluate_retained(_always_correct_model, empty_bench) == 1.0


def test_evaluate_retained_all_correct_returns_one(
    real_benchmark: RetainedBenchmark,
) -> None:
    """Model that always returns expected → 1.0 accuracy."""
    acc = evaluate_retained(_always_correct_model, real_benchmark)
    assert acc == 1.0


def test_evaluate_retained_all_wrong_returns_zero(
    real_benchmark: RetainedBenchmark,
) -> None:
    """Model that never matches → 0.0 accuracy."""
    acc = evaluate_retained(_always_wrong_model, real_benchmark)
    assert acc == 0.0


def test_evaluate_retained_half_correct_returns_half(
    real_benchmark: RetainedBenchmark,
) -> None:
    """Model correct on even ids → ~0.5 accuracy on 50 items."""
    acc = evaluate_retained(_half_correct_model, real_benchmark)
    # 50 items with ids 0000..0049 — even ids: 25 → 0.5 exact
    assert acc == pytest.approx(0.5)


def test_evaluate_retained_accepts_seed_kwarg(
    real_benchmark: RetainedBenchmark,
) -> None:
    """Seed parameter is accepted and result unchanged for
    deterministic predictors (trace-only propagation contract)."""
    acc_no_seed = evaluate_retained(_half_correct_model, real_benchmark)
    acc_seed_42 = evaluate_retained(
        _half_correct_model, real_benchmark, seed=42
    )
    acc_seed_123 = evaluate_retained(
        _half_correct_model, real_benchmark, seed=123
    )
    assert acc_no_seed == acc_seed_42 == acc_seed_123
