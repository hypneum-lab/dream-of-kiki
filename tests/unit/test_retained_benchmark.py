"""Unit tests for retained benchmark loader (TDD, invariants S1+I1)."""
from pathlib import Path

import pytest

from harness.benchmarks.retained.retained import (
    RetainedBenchmark,
    RetainedIntegrityError,
    load_retained,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BENCH_DIR = REPO_ROOT / "harness" / "benchmarks" / "retained"


@pytest.fixture
def bench() -> RetainedBenchmark:
    return load_retained(BENCH_DIR)


def test_benchmark_loads(bench: RetainedBenchmark) -> None:
    assert len(bench.items) >= 50


def test_benchmark_hash_matches_frozen(bench: RetainedBenchmark) -> None:
    # Loader must verify items.jsonl SHA-256 against items.jsonl.sha256
    assert bench.hash_verified is True


def test_benchmark_items_have_required_fields(
    bench: RetainedBenchmark,
) -> None:
    for item in bench.items:
        assert "id" in item
        assert "context" in item
        assert "expected" in item
        assert "domain" in item


def test_benchmark_items_have_unique_ids(
    bench: RetainedBenchmark,
) -> None:
    ids = [item["id"] for item in bench.items]
    assert len(ids) == len(set(ids))


def test_loader_raises_on_hash_mismatch(tmp_path: Path) -> None:
    # Copy bench files to tmp, corrupt items.jsonl, expect
    # RetainedIntegrityError
    corrupt_items = tmp_path / "items.jsonl"
    corrupt_items.write_text('{"id": "fake", "context": "x", '
                             '"expected": "y", "domain": "z"}\n')
    original_hash = (BENCH_DIR / "items.jsonl.sha256").read_text()
    (tmp_path / "items.jsonl.sha256").write_text(original_hash)
    with pytest.raises(RetainedIntegrityError):
        load_retained(tmp_path)
