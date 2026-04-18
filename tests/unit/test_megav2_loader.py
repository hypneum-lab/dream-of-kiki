"""Unit tests for mega-v2 dataset loader bridge (S13.3)."""
from __future__ import annotations

from pathlib import Path

import pytest

from harness.benchmarks.mega_v2.adapter import (
    MegaV2NotAvailable,
    SYNTHETIC_DOMAINS,
    load_megav2_stratified,
)
from harness.benchmarks.retained.retained import RetainedBenchmark


def test_synthetic_fallback_returns_500_items(tmp_path: Path) -> None:
    """When real mega-v2 absent, synthetic fallback produces 500 items."""
    bench = load_megav2_stratified(
        real_path=None,  # explicit fallback
        items_per_domain=20,
        synthetic_seed=42,
    )
    assert isinstance(bench, RetainedBenchmark)
    assert len(bench.items) == 500


def test_synthetic_stratification_balanced() -> None:
    """Each of the 25 synthetic domains has exactly 20 items."""
    bench = load_megav2_stratified(
        real_path=None, items_per_domain=20, synthetic_seed=42
    )
    counts: dict[str, int] = {}
    for item in bench.items:
        counts[item["domain"]] = counts.get(item["domain"], 0) + 1
    assert len(counts) == 25
    assert all(c == 20 for c in counts.values())


def test_synthetic_items_have_required_fields() -> None:
    bench = load_megav2_stratified(
        real_path=None, items_per_domain=20, synthetic_seed=42
    )
    for item in bench.items:
        assert "id" in item
        assert "context" in item
        assert "expected" in item
        assert "domain" in item
        assert item["domain"] in SYNTHETIC_DOMAINS


def test_real_path_missing_raises_or_falls_back() -> None:
    """Non-existent real path should raise MegaV2NotAvailable
    when explicit_fallback=False."""
    with pytest.raises(MegaV2NotAvailable):
        load_megav2_stratified(
            real_path=Path("/nonexistent/mega-v2.jsonl"),
            items_per_domain=20,
            explicit_fallback=False,
        )
