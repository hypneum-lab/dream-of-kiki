"""Smoke + structural tests for the G4-quater Step 3 driver."""
from __future__ import annotations

from pathlib import Path

from experiments.g4_quater_test.run_step3_recombine_strategies import (
    ARMS,
    STRATEGIES,
    run_pilot,
)


def test_constants_match_prereg() -> None:
    assert ARMS == ("baseline", "P_min", "P_equ", "P_max")
    assert STRATEGIES == ("mog", "ae", "none")


def test_run_pilot_smoke(tmp_path: Path) -> None:
    data_dir = (
        Path(__file__).resolve().parents[3]
        / "experiments"
        / "g4_split_fmnist"
        / "data"
    )
    out_json = tmp_path / "step3.json"
    out_md = tmp_path / "step3.md"
    registry_db = tmp_path / "registry.sqlite"
    payload = run_pilot(
        data_dir=data_dir,
        seeds=(0,),
        strategies=("mog", "none"),
        out_json=out_json,
        out_md=out_md,
        registry_db=registry_db,
        epochs=1,
        batch_size=64,
        lr=0.01,
        smoke=True,
    )
    assert len(payload["cells"]) == 2 * len(ARMS)
    for cell in payload["cells"]:
        assert "run_id" in cell
        assert cell["recombine_strategy"] in ("mog", "none")
    assert out_json.exists()
    assert out_md.exists()
