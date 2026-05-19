"""Unit tests for multi-substrate ablation runner (C2.9 cycle 2).

Validates that the AblationRunner can iterate the cartesian product
``substrate × profile × seed`` instead of the cycle-1
``profile × seed`` matrix. The substrate axis is optional (None
preserves cycle-1 behaviour with a sentinel substrate name).

Reference : docs/specs/2026-04-17-dreamofkiki-master-design.md §5
            (cycle-2 C2.9 multi-substrate ablation)
"""
from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from harness.benchmarks.retained.retained import RetainedBenchmark
from kiki_oniric.eval.ablation import (
    AblationRunner,
    ProfileSpec,
    SubstrateSpec,
)


def _bench_fixture() -> RetainedBenchmark:
    items = [
        {"id": f"r-{i}", "context": f"c{i}",
         "expected": f"y{i}", "domain": "test"}
        for i in range(6)
    ]
    return RetainedBenchmark(
        items=items, hash_verified=True, source_hash="0" * 64
    )


def _correct(item: dict[str, Any]) -> str:
    return str(item["expected"])


def _wrong(item: dict[str, Any]) -> str:
    return "WRONG"


def _half(item: dict[str, Any]) -> str:
    rid = int(item["id"].split("-")[-1])
    return str(item["expected"]) if rid % 2 == 0 else "WRONG"


def test_substrate_spec_dataclass_holds_name() -> None:
    """SubstrateSpec is a frozen dataclass with .name."""
    spec = SubstrateSpec(name="mlx_kiki_oniric")
    assert spec.name == "mlx_kiki_oniric"


def test_run_with_substrate_adds_substrate_column(
    tmp_path: Path,
) -> None:
    """When substrate_specs is set, every row carries `substrate`."""
    runner = AblationRunner(
        profile_specs=[ProfileSpec("P_correct", _correct)],
        seeds=[1],
        benchmark=_bench_fixture(),
        substrate_specs=[SubstrateSpec("mlx_kiki_oniric")],
        registry_path=tmp_path / "reg.sqlite",
    )
    df = runner.run()
    assert "substrate" in df.columns
    assert df.iloc[0]["substrate"] == "mlx_kiki_oniric"


def test_two_substrates_doubles_row_count(tmp_path: Path) -> None:
    """2 substrates × 1 profile × 1 seed → 2 rows."""
    runner = AblationRunner(
        profile_specs=[ProfileSpec("P_correct", _correct)],
        seeds=[1],
        benchmark=_bench_fixture(),
        substrate_specs=[
            SubstrateSpec("mlx_kiki_oniric"),
            SubstrateSpec("esnn_thalamocortical"),
        ],
        registry_path=tmp_path / "reg.sqlite",
    )
    df = runner.run()
    assert len(df) == 2
    assert set(df["substrate"]) == {
        "mlx_kiki_oniric", "esnn_thalamocortical",
    }


def test_full_cartesian_grid_size(tmp_path: Path) -> None:
    """2 substrates × 3 profiles × 2 seeds → 12 rows."""
    profiles = [
        ProfileSpec("P_correct", _correct),
        ProfileSpec("P_half", _half),
        ProfileSpec("P_wrong", _wrong),
    ]
    runner = AblationRunner(
        profile_specs=profiles,
        seeds=[1, 2],
        benchmark=_bench_fixture(),
        substrate_specs=[
            SubstrateSpec("mlx_kiki_oniric"),
            SubstrateSpec("esnn_thalamocortical"),
        ],
        registry_path=tmp_path / "reg.sqlite",
    )
    df = runner.run()
    assert len(df) == 12
    triples = {
        (r["substrate"], r["profile"], r["seed"])
        for _, r in df.iterrows()
    }
    assert runner.substrate_specs is not None
    expected = {
        (sub.name, prof.name, seed)
        for sub in runner.substrate_specs
        for prof in profiles
        for seed in [1, 2]
    }
    assert triples == expected


def test_no_substrate_specs_preserves_cycle1_behaviour(
    tmp_path: Path,
) -> None:
    """Omitting substrate_specs keeps cycle-1 grid shape + adds the
    sentinel ``substrate`` column with value ``"unspecified"``."""
    runner = AblationRunner(
        profile_specs=[ProfileSpec("P_half", _half)],
        seeds=[1, 2, 3],
        benchmark=_bench_fixture(),
        registry_path=tmp_path / "reg.sqlite",
    )
    df = runner.run()
    assert len(df) == 3
    assert "substrate" in df.columns
    assert set(df["substrate"]) == {"unspecified"}


def test_run_id_unique_per_substrate_axis(tmp_path: Path) -> None:
    """All substrate rows of a single run() share the same run_id
    (single registered batch)."""
    runner = AblationRunner(
        profile_specs=[ProfileSpec("P_correct", _correct)],
        seeds=[1],
        benchmark=_bench_fixture(),
        substrate_specs=[
            SubstrateSpec("mlx_kiki_oniric"),
            SubstrateSpec("esnn_thalamocortical"),
        ],
        registry_path=tmp_path / "reg.sqlite",
    )
    df = runner.run()
    run_ids = set(df["run_id"].tolist())
    assert len(run_ids) == 1


def test_substrate_axis_preserves_accuracy_per_predictor(
    tmp_path: Path,
) -> None:
    """Predictor accuracy is independent of the substrate label
    (predictor is the same Python callable across substrates in
    this synthetic test)."""
    runner = AblationRunner(
        profile_specs=[
            ProfileSpec("P_half", _half),
        ],
        seeds=[42],
        benchmark=_bench_fixture(),
        substrate_specs=[
            SubstrateSpec("mlx_kiki_oniric"),
            SubstrateSpec("esnn_thalamocortical"),
        ],
        registry_path=tmp_path / "reg.sqlite",
    )
    df = runner.run()
    accuracies = df["accuracy"].tolist()
    assert len(accuracies) == 2
    # Half-predictor: 3 of 6 even ids → accuracy 0.5
    for acc in accuracies:
        assert acc == pytest.approx(0.5)
