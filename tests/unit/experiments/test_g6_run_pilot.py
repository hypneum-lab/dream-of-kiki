"""Import-and-CLI smoke for the G6 pilot driver."""
from __future__ import annotations

import importlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_g6_module_importable() -> None:
    mod = importlib.import_module("experiments.g6_mmlu_stream.run_g6")
    assert hasattr(mod, "run_pilot"), "run_g6 must export run_pilot"
    assert hasattr(mod, "main"), "run_g6 must export main"
    assert hasattr(mod, "ARMS"), "run_g6 must export ARMS"
    assert tuple(mod.ARMS) == ("baseline", "P_min", "P_equ", "P_max")


def test_g6_help_smokes() -> None:
    result = subprocess.run(
        [sys.executable, "-m", "experiments.g6_mmlu_stream.run_g6", "--help"],
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
        timeout=15,
        check=False,
    )
    assert result.returncode == 0, (
        f"--help failed: stderr={result.stderr!r}"
    )
    assert "G6" in result.stdout or "g6" in result.stdout
    assert "--smoke" in result.stdout
    assert "--path" in result.stdout


# ----- Task 7 (cell runner + retention metric + R1 wiring) -----


def test_compute_retention_identity_when_no_forgetting() -> None:
    from experiments.g6_mmlu_stream.run_g6 import compute_retention

    matrix: dict[str, list[float | None]] = {
        "anatomy": [0.5, 0.5, 0.5, 0.5, 0.5],
        "astronomy": [None, 0.5, 0.5, 0.5, 0.5],
        "business_ethics": [None, None, 0.5, 0.5, 0.5],
        "clinical_knowledge": [None, None, None, 0.5, 0.5],
        "college_biology": [None, None, None, None, 0.5],
    }
    retention = compute_retention(
        matrix,
        subdomains=(
            "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology",
        ),
    )
    assert retention == 1.0


def test_compute_retention_drops_with_forgetting() -> None:
    from experiments.g6_mmlu_stream.run_g6 import compute_retention

    matrix: dict[str, list[float | None]] = {
        "anatomy": [0.8, 0.7, 0.6, 0.5, 0.4],
        "astronomy": [None, 0.8, 0.7, 0.6, 0.5],
        "business_ethics": [None, None, 0.8, 0.7, 0.6],
        "clinical_knowledge": [None, None, None, 0.8, 0.7],
        "college_biology": [None, None, None, None, 0.8],
    }
    # Per-subdomain ratios :
    #   anatomy            j=0  : 0.4 / 0.8 = 0.5
    #   astronomy          j=1  : 0.5 / 0.8 = 0.625
    #   business_ethics    j=2  : 0.6 / 0.8 = 0.75
    #   clinical_knowledge j=3  : 0.7 / 0.8 = 0.875
    # Mean = (0.5 + 0.625 + 0.75 + 0.875) / 4 = 2.75 / 4 = 0.6875
    retention = compute_retention(
        matrix,
        subdomains=(
            "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology",
        ),
    )
    assert retention == pytest.approx(0.6875, abs=1e-9)


def test_compute_retention_excludes_zero_initial() -> None:
    """When acc[S_j after S_j] is 0, the ratio is undefined."""
    from experiments.g6_mmlu_stream.run_g6 import compute_retention

    matrix: dict[str, list[float | None]] = {
        "anatomy": [0.0, 0.0, 0.0, 0.0, 0.0],  # underperforming
        "astronomy": [None, 0.8, 0.4, 0.4, 0.4],
        "business_ethics": [None, None, 0.8, 0.4, 0.4],
        "clinical_knowledge": [None, None, None, 0.8, 0.4],
        "college_biology": [None, None, None, None, 0.8],
    }
    retention = compute_retention(
        matrix,
        subdomains=(
            "anatomy", "astronomy", "business_ethics",
            "clinical_knowledge", "college_biology",
        ),
    )
    # Only 3 ratios contribute: 0.4/0.8 = 0.5 each -> mean 0.5
    assert retention == pytest.approx(0.5, abs=1e-9)


def test_run_pilot_path_b_smoke(tmp_path: Path) -> None:
    """End-to-end Path B smoke — 4 arms x 1 seed on synthetic fixture."""
    from experiments.g6_mmlu_stream.run_g6 import run_pilot

    fixture = tmp_path / "mmlu.jsonl"
    rows = []
    for subj in (
        "anatomy", "astronomy", "business_ethics",
        "clinical_knowledge", "college_biology",
    ):
        for i in range(8):
            rows.append({
                "question": f"{subj}-Q{i}?",
                "choices": ["A", "B", "C", "D"],
                "answer": i % 4,
                "subject": subj,
            })
    fixture.write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8",
    )
    out_json = tmp_path / "out.json"
    out_md = tmp_path / "out.md"
    db = tmp_path / "registry.sqlite"

    payload = run_pilot(
        fixture_path=fixture,
        out_json=out_json,
        out_md=out_md,
        registry_db=db,
        seeds=(0,),
        n_train=4,
        n_eval=4,
        inner_steps=2,
        lr=5e-5,
        rank=4,
        alpha=4.0,
        path="B",
        scale_slot="qwen3p5-1p5b-fp16",
    )
    assert "cells" in payload
    assert len(payload["cells"]) == 4  # 4 arms x 1 seed
    for cell in payload["cells"]:
        assert "run_id" in cell
        assert "retention" in cell
        assert "acc_matrix" in cell
        assert cell["arm"] in ("baseline", "P_min", "P_equ", "P_max")
    assert out_json.exists()
    assert out_md.exists()


def test_run_pilot_path_a_raises_not_implemented(tmp_path: Path) -> None:
    """Path A is locked-out on this M1 Max host (decisions doc)."""
    from experiments.g6_mmlu_stream.run_g6 import run_pilot

    fixture = tmp_path / "mmlu.jsonl"
    rows = []
    for subj in ("anatomy", "astronomy", "business_ethics",
                 "clinical_knowledge", "college_biology"):
        for i in range(8):
            rows.append({
                "question": f"{subj}-Q{i}?",
                "choices": ["A", "B", "C", "D"],
                "answer": i % 4,
                "subject": subj,
            })
    fixture.write_text(
        "\n".join(json.dumps(r) for r in rows), encoding="utf-8",
    )
    with pytest.raises(NotImplementedError, match="Path A"):
        run_pilot(
            fixture_path=fixture,
            out_json=tmp_path / "o.json",
            out_md=tmp_path / "o.md",
            registry_db=tmp_path / "r.sqlite",
            seeds=(0,),
            n_train=4, n_eval=4, inner_steps=2, lr=5e-5,
            rank=4, alpha=4.0,
            path="A", scale_slot="qwen3p6-35b-bf16-local",
        )
