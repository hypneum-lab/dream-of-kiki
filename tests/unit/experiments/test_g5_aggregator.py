"""Unit tests for `experiments.g5_cross_substrate.aggregator`."""
from __future__ import annotations

import json
from pathlib import Path

import pytest

from experiments.g5_cross_substrate.aggregator import (
    aggregate_cross_substrate_verdict,
    write_aggregate_dump,
)


def _write_milestone(
    path: Path,
    *,
    retention_by_arm: dict[str, list[float]],
    substrate: str,
) -> None:
    """Synthetic milestone fixture matching the run_g{4,5}.py shape."""
    payload = {
        "date": "2026-05-03",
        "substrate": substrate,
        "c_version": "C-v0.12.0+PARTIAL",
        "commit_sha": "abcdef0",
        "n_seeds": len(next(iter(retention_by_arm.values()))),
        "arms": ["baseline", "P_min", "P_equ", "P_max"],
        "data_dir": "fixture",
        "wall_time_s": 0.0,
        "cells": [],
        "verdict": {"retention_by_arm": retention_by_arm},
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True))


def test_aggregate_consistency_when_distributions_match(
    tmp_path: Path,
) -> None:
    """Identical retention by arm -> consistency_ok = True."""
    arms = {
        "baseline": [0.6, 0.61, 0.59, 0.6, 0.6],
        "P_min": [0.7, 0.71, 0.69, 0.7, 0.7],
        "P_equ": [0.8, 0.81, 0.79, 0.8, 0.8],
        "P_max": [0.9, 0.91, 0.89, 0.9, 0.9],
    }
    mlx_path = tmp_path / "mlx.json"
    esnn_path = tmp_path / "esnn.json"
    _write_milestone(
        mlx_path, retention_by_arm=arms, substrate="mlx_kiki_oniric"
    )
    _write_milestone(
        esnn_path, retention_by_arm=arms, substrate="esnn_thalamocortical"
    )
    verdict = aggregate_cross_substrate_verdict(mlx_path, esnn_path)
    assert verdict["dr3_cross_substrate_consistency_ok"] is True
    assert set(verdict["per_arm"].keys()) == {
        "baseline",
        "P_min",
        "P_equ",
        "P_max",
    }
    for arm in ("baseline", "P_min", "P_equ", "P_max"):
        assert verdict["per_arm"][arm]["consistency"] is True


def test_aggregate_divergence_when_distributions_differ(
    tmp_path: Path,
) -> None:
    """Strongly different P_max retention -> consistency_ok = False."""
    # Add tiny noise so pooled SD is non-zero (compute_hedges_g
    # rejects identical-constant samples to avoid undefined d).
    mlx_arms = {
        "baseline": [0.60, 0.61, 0.59, 0.60, 0.60],
        "P_min": [0.60, 0.61, 0.59, 0.60, 0.60],
        "P_equ": [0.60, 0.61, 0.59, 0.60, 0.60],
        "P_max": [0.95, 0.96, 0.94, 0.95, 0.95],
    }
    esnn_arms = {
        "baseline": [0.60, 0.61, 0.59, 0.60, 0.60],
        "P_min": [0.60, 0.61, 0.59, 0.60, 0.60],
        "P_equ": [0.60, 0.61, 0.59, 0.60, 0.60],
        "P_max": [0.10, 0.11, 0.09, 0.10, 0.10],
    }
    mlx_path = tmp_path / "mlx.json"
    esnn_path = tmp_path / "esnn.json"
    _write_milestone(
        mlx_path, retention_by_arm=mlx_arms, substrate="mlx_kiki_oniric"
    )
    _write_milestone(
        esnn_path,
        retention_by_arm=esnn_arms,
        substrate="esnn_thalamocortical",
    )
    verdict = aggregate_cross_substrate_verdict(mlx_path, esnn_path)
    assert verdict["dr3_cross_substrate_consistency_ok"] is False
    assert verdict["per_arm"]["P_max"]["consistency"] is False


def test_aggregate_writes_dump_files(tmp_path: Path) -> None:
    """`write_aggregate_dump` produces both .json and .md siblings."""
    arms = {
        "baseline": [0.6, 0.61, 0.59, 0.6, 0.6],
        "P_min": [0.7, 0.71, 0.69, 0.7, 0.7],
        "P_equ": [0.8, 0.81, 0.79, 0.8, 0.8],
        "P_max": [0.9, 0.91, 0.89, 0.9, 0.9],
    }
    mlx_path = tmp_path / "mlx.json"
    esnn_path = tmp_path / "esnn.json"
    _write_milestone(
        mlx_path, retention_by_arm=arms, substrate="mlx_kiki_oniric"
    )
    _write_milestone(
        esnn_path, retention_by_arm=arms, substrate="esnn_thalamocortical"
    )
    out_json = tmp_path / "agg.json"
    out_md = tmp_path / "agg.md"
    write_aggregate_dump(
        mlx_milestone=mlx_path,
        esnn_milestone=esnn_path,
        out_json=out_json,
        out_md=out_md,
    )
    assert out_json.exists()
    assert out_md.exists()
    body = json.loads(out_json.read_text())
    assert body["dr3_cross_substrate_consistency_ok"] is True
    md = out_md.read_text()
    assert "DR-3 cross-substrate consistency" in md


def test_aggregate_rejects_missing_arm(tmp_path: Path) -> None:
    """Milestone missing a required arm -> ValueError."""
    mlx_arms = {
        "baseline": [0.6, 0.6],
        "P_min": [0.6, 0.6],
        "P_equ": [0.6, 0.6],
        "P_max": [0.6, 0.6],
    }
    esnn_arms = {
        "baseline": [0.6, 0.6],
        "P_min": [0.6, 0.6],
        "P_equ": [0.6, 0.6],
        # P_max missing
    }
    mlx_path = tmp_path / "mlx.json"
    esnn_path = tmp_path / "esnn.json"
    _write_milestone(
        mlx_path, retention_by_arm=mlx_arms, substrate="mlx_kiki_oniric"
    )
    _write_milestone(
        esnn_path,
        retention_by_arm=esnn_arms,
        substrate="esnn_thalamocortical",
    )
    with pytest.raises(ValueError, match="P_max"):
        aggregate_cross_substrate_verdict(mlx_path, esnn_path)
