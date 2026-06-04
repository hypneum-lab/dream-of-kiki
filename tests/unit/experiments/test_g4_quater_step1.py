"""Smoke + structural tests for the G4-quater Step 1 driver."""
from __future__ import annotations

from pathlib import Path

from experiments.g4_quater_test.run_step1_deeper import (
    ARMS,
    HIDDEN,
    run_pilot,
)

from .conftest import FMNIST_DATA, skip_if_no_fmnist


def test_constants_match_prereg() -> None:
    assert ARMS == ("baseline", "P_min", "P_equ", "P_max")
    assert HIDDEN == (64, 32, 16, 8)


@skip_if_no_fmnist
def test_run_pilot_smoke(tmp_path: Path) -> None:
    data_dir = FMNIST_DATA
    out_json = tmp_path / "step1.json"
    out_md = tmp_path / "step1.md"
    registry_db = tmp_path / "registry.sqlite"
    payload = run_pilot(
        data_dir=data_dir,
        seeds=(0,),
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
        assert "retention" in cell
    assert out_json.exists()
    assert out_md.exists()
