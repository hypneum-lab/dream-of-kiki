"""Unit tests for the G5-ter aggregator."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.g5_ter_spiking_cnn.aggregator import (
    aggregate_g5ter_verdict,
    write_aggregate_dump,
)


def _write(path: Path, retention: dict[str, list[float]]) -> None:
    path.write_text(
        json.dumps({"verdict": {"retention_by_arm": retention}})
    )


def _ret(b: float, p: float) -> dict[str, list[float]]:
    return {
        "baseline": [b + 0.001 * i for i in range(10)],
        "P_min": [b + 0.05 + 0.001 * i for i in range(10)],
        "P_equ": [p + 0.001 * i for i in range(10)],
        "P_max": [p - 0.05 + 0.001 * i for i in range(10)],
    }


def test_h8a_lif_washout(tmp_path: Path) -> None:
    """own-fail (esnn flat) + huge MLX-minus-ESNN gap -> LIF washout."""
    mlx, esnn = tmp_path / "mlx.json", tmp_path / "esnn.json"
    # MLX shows strong P_equ effect (0.95 vs 0.5 baseline) ;
    # E-SNN P_equ is essentially flat (0.5) — own-substrate fails.
    # Cross gap at P_equ is ~0.45 with tiny variance -> g >> 2.
    _write(mlx, _ret(0.5, 0.95))
    _write(esnn, _ret(0.5, 0.5))
    verdict = aggregate_g5ter_verdict(mlx, esnn)
    assert verdict["h8_classification"] == "H8-A"


def test_h8b_architecture_recovery(tmp_path: Path) -> None:
    """own-pass (esnn matches mlx) + closed gap -> CNN recovers signal."""
    mlx, esnn = tmp_path / "mlx.json", tmp_path / "esnn.json"
    # Both substrates show P_equ ~ 0.95 vs baseline 0.5 -> own-pass.
    # Cross gap at P_equ ~ 0 with tiny variance -> g < 1.
    _write(mlx, _ret(0.5, 0.95))
    _write(esnn, _ret(0.5, 0.95))
    verdict = aggregate_g5ter_verdict(mlx, esnn)
    assert verdict["h8_classification"] == "H8-B"


def test_h8c_partial(tmp_path: Path) -> None:
    """own-pass (positive but smaller) + persistent gap -> partial."""
    mlx, esnn = tmp_path / "mlx.json", tmp_path / "esnn.json"
    # E-SNN P_equ = 0.65 vs base 0.5 -> own-pass (g large), but cross
    # gap mlx 0.95 vs esnn 0.65 still huge -> not H8-B.
    _write(mlx, _ret(0.5, 0.95))
    _write(esnn, _ret(0.5, 0.65))
    verdict = aggregate_g5ter_verdict(mlx, esnn)
    assert verdict["h8_classification"] == "H8-C"


def test_write_aggregate_dump(tmp_path: Path) -> None:
    spread = {
        a: [0.5 + 0.001 * i for i in range(5)]
        for a in ("baseline", "P_min", "P_equ", "P_max")
    }
    mlx, esnn = tmp_path / "mlx.json", tmp_path / "esnn.json"
    _write(mlx, spread)
    _write(esnn, spread)
    out_json, out_md = tmp_path / "agg.json", tmp_path / "agg.md"
    write_aggregate_dump(
        mlx_milestone=mlx,
        esnn_milestone=esnn,
        out_json=out_json,
        out_md=out_md,
    )
    assert "H8" in out_md.read_text()
