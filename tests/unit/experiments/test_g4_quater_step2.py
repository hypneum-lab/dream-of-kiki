"""Smoke + structural tests for the G4-quater Step 2 driver."""
from __future__ import annotations

from pathlib import Path

from experiments.g4_quater_test.run_step2_restructure_sweep import (
    ARMS,
    RESTRUCTURE_FACTORS,
    run_pilot,
)

from .conftest import FMNIST_DATA, skip_if_no_fmnist


def test_constants_match_prereg() -> None:
    assert ARMS == ("baseline", "P_min", "P_equ", "P_max")
    assert RESTRUCTURE_FACTORS == (0.85, 0.95, 0.99)


@skip_if_no_fmnist
def test_run_pilot_smoke(tmp_path: Path) -> None:
    data_dir = FMNIST_DATA
    out_json = tmp_path / "step2.json"
    out_md = tmp_path / "step2.md"
    registry_db = tmp_path / "registry.sqlite"
    payload = run_pilot(
        data_dir=data_dir,
        seeds=(0,),
        factors=(0.95,),
        out_json=out_json,
        out_md=out_md,
        registry_db=registry_db,
        epochs=1,
        batch_size=64,
        lr=0.01,
        smoke=True,
    )
    assert len(payload["cells"]) == len(ARMS)
    for cell in payload["cells"]:
        assert "run_id" in cell
        assert cell["restructure_factor"] == 0.95
    assert out_json.exists()
    assert out_md.exists()
