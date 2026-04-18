"""Unit tests for ablation runner harness (S15.2)."""
from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from harness.benchmarks.retained.retained import RetainedBenchmark
from kiki_oniric.eval.ablation import (
    AblationRunner,
    ProfileSpec,
)


def _bench_fixture() -> RetainedBenchmark:
    """6-item benchmark for fast tests."""
    items = [
        {"id": f"r-{i}", "context": f"c{i}",
         "expected": f"y{i}", "domain": "test"}
        for i in range(6)
    ]
    return RetainedBenchmark(
        items=items, hash_verified=True, source_hash="0" * 64
    )


def _correct_predictor(item: dict) -> str:
    return item["expected"]


def _wrong_predictor(item: dict) -> str:
    return "WRONG"


def _half_predictor(item: dict) -> str:
    rid_token = item["id"].split("-")[-1]
    rid = int(rid_token)
    return item["expected"] if rid % 2 == 0 else "WRONG"


def _isolated_runner(
    tmp_path: Path,
    profile_specs: list[ProfileSpec],
    seeds: list[int],
) -> AblationRunner:
    return AblationRunner(
        profile_specs=profile_specs,
        seeds=seeds,
        benchmark=_bench_fixture(),
        registry_path=tmp_path / "registry.sqlite",
    )


def test_single_cell_run_produces_metric_row(tmp_path: Path) -> None:
    """Single (profile, seed) cell yields a one-row DataFrame."""
    spec = ProfileSpec(name="P_correct", predictor=_correct_predictor)
    runner = _isolated_runner(tmp_path, [spec], [42])
    df = runner.run()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 1
    row = df.iloc[0]
    assert row["profile"] == "P_correct"
    assert row["seed"] == 42
    assert row["accuracy"] == 1.0


def test_multi_cell_sweep_covers_full_grid(tmp_path: Path) -> None:
    """3 profiles × 3 seeds = 9 rows."""
    specs = [
        ProfileSpec(name="P_correct", predictor=_correct_predictor),
        ProfileSpec(name="P_half", predictor=_half_predictor),
        ProfileSpec(name="P_wrong", predictor=_wrong_predictor),
    ]
    runner = _isolated_runner(tmp_path, specs, [1, 2, 3])
    df = runner.run()
    assert len(df) == 9
    profile_seed_pairs = set(
        (row["profile"], row["seed"]) for _, row in df.iterrows()
    )
    expected = {
        (p.name, s) for p in specs for s in [1, 2, 3]
    }
    assert profile_seed_pairs == expected


def test_results_dataframe_schema(tmp_path: Path) -> None:
    """DataFrame has required columns for downstream stat tests."""
    spec = ProfileSpec(name="P_half", predictor=_half_predictor)
    runner = _isolated_runner(tmp_path, [spec], [42])
    df = runner.run()
    required_cols = {
        "run_id", "profile", "seed", "accuracy", "benchmark_hash",
    }
    assert required_cols <= set(df.columns)
    # Half-predictor: 3 of 6 items (even ids) correct
    assert df.iloc[0]["accuracy"] == pytest.approx(0.5)


def test_run_id_registered_and_broadcast(tmp_path: Path) -> None:
    """A single run_id is registered and stamped on every row."""
    specs = [
        ProfileSpec(name="P_correct", predictor=_correct_predictor),
        ProfileSpec(name="P_wrong", predictor=_wrong_predictor),
    ]
    runner = _isolated_runner(tmp_path, specs, [1, 2])
    df = runner.run()
    run_ids = set(df["run_id"].tolist())
    assert len(run_ids) == 1
    run_id = run_ids.pop()
    assert run_id and isinstance(run_id, str)
    # Repeating the same batch yields the same run_id (R1 contract).
    df_again = runner.run()
    assert set(df_again["run_id"].tolist()) == {run_id}
