"""Unit tests for the G5-bis cross-substrate aggregator (Plan G5-bis Task 5)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def _write_g4ter_fixture(
    path: Path, retention: dict[str, list[float]]
) -> None:
    payload = {
        "verdict": {
            "h2_substrate_richer": {"hedges_g": 2.77},
            "retention_richer_by_arm": retention,
        }
    }
    path.write_text(json.dumps(payload))


def _write_g5bis_fixture(
    path: Path, retention: dict[str, list[float]]
) -> None:
    payload: dict[str, Any] = {
        "verdict": {
            "h7a_richer_esnn": {"hedges_g": 1.5},
            "retention_by_arm": retention,
        }
    }
    path.write_text(json.dumps(payload))


def _matched_arms(level_p_equ: float = 0.7) -> dict[str, list[float]]:
    return {
        "baseline": [0.50 + 0.001 * i for i in range(10)],
        "P_min": [0.55 + 0.001 * i for i in range(10)],
        "P_equ": [level_p_equ + 0.001 * i for i in range(10)],
        "P_max": [0.72 + 0.001 * i for i in range(10)],
    }


def _zero_effect_arms() -> dict[str, list[float]]:
    return {
        "baseline": [0.50 + 0.001 * i for i in range(10)],
        "P_min": [0.50 + 0.001 * i for i in range(10)],
        "P_equ": [0.50 + 0.001 * i for i in range(10)],
        "P_max": [0.50 + 0.001 * i for i in range(10)],
    }


def test_h7c_universal_when_means_match(tmp_path: Path) -> None:
    """Both substrates positive at P_equ ~0.7, baselines ~0.5 -> H7-C."""
    from experiments.g5_bis_richer_esnn.aggregator import (
        aggregate_g5bis_verdict,
    )

    mlx = tmp_path / "g4ter.json"
    esnn = tmp_path / "g5bis.json"
    _write_g4ter_fixture(mlx, _matched_arms(level_p_equ=0.7))
    _write_g5bis_fixture(esnn, _matched_arms(level_p_equ=0.7))
    verdict = aggregate_g5bis_verdict(mlx, esnn)
    assert verdict["h7_classification"] == "H7-C"
    assert verdict["per_arm"]["P_equ"]["consistency"] is True


def test_h7b_when_g_close_to_zero(tmp_path: Path) -> None:
    """E-SNN P_equ ~ baseline -> g_h7a near 0, no positive effect -> H7-B."""
    from experiments.g5_bis_richer_esnn.aggregator import (
        aggregate_g5bis_verdict,
    )

    mlx = tmp_path / "g4ter.json"
    esnn = tmp_path / "g5bis.json"
    _write_g4ter_fixture(mlx, _matched_arms(level_p_equ=0.7))
    _write_g5bis_fixture(esnn, _zero_effect_arms())
    verdict = aggregate_g5bis_verdict(mlx, esnn)
    assert verdict["h7_classification"] == "H7-B"


def test_h7a_when_positive_but_diverges(tmp_path: Path) -> None:
    """MLX P_equ=0.9 (g vs base ~ huge), E-SNN P_equ=0.6 (positive but level
    diverges from MLX) -> H7-A."""
    from experiments.g5_bis_richer_esnn.aggregator import (
        aggregate_g5bis_verdict,
    )

    mlx_arms = {
        "baseline": [0.50 + 0.001 * i for i in range(10)],
        "P_min": [0.55 + 0.001 * i for i in range(10)],
        "P_equ": [0.90 + 0.001 * i for i in range(10)],
        "P_max": [0.91 + 0.001 * i for i in range(10)],
    }
    esnn_arms = {
        "baseline": [0.50 + 0.001 * i for i in range(10)],
        "P_min": [0.55 + 0.001 * i for i in range(10)],
        "P_equ": [0.60 + 0.001 * i for i in range(10)],
        "P_max": [0.62 + 0.001 * i for i in range(10)],
    }
    mlx = tmp_path / "g4ter.json"
    esnn = tmp_path / "g5bis.json"
    _write_g4ter_fixture(mlx, mlx_arms)
    _write_g5bis_fixture(esnn, esnn_arms)
    verdict = aggregate_g5bis_verdict(mlx, esnn)
    assert verdict["h7_classification"] == "H7-A"
    assert verdict["per_arm"]["P_equ"]["consistency"] is False
