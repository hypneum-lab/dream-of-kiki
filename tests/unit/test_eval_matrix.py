"""Unit tests for eval matrix config parser.

Validates that docs/interfaces/eval-matrix.yaml loads correctly
and exposes the stratification rules for the stratified matrix
dispatcher (T-Ops.2).
"""
from pathlib import Path

import pytest

from harness.config.eval_matrix import EvalMatrix, load_eval_matrix


REPO_ROOT = Path(__file__).resolve().parents[2]
MATRIX_PATH = REPO_ROOT / "docs" / "interfaces" / "eval-matrix.yaml"


@pytest.fixture
def matrix() -> EvalMatrix:
    return load_eval_matrix(MATRIX_PATH)


def test_matrix_version_starts_with_C_prefix(matrix: EvalMatrix) -> None:
    assert matrix.version.startswith("C-v")


def test_matrix_has_all_four_bump_rules(matrix: EvalMatrix) -> None:
    assert set(matrix.bump_rules.keys()) == {
        "PATCH",
        "MINOR",
        "MAJOR",
        "EC_change",
    }


def test_major_bump_requires_all_three_profiles(
    matrix: EvalMatrix,
) -> None:
    major = matrix.bump_rules["MAJOR"]
    obligatory = major["obligatory"][0]
    assert obligatory["profiles"] == ["P_min", "P_equ", "P_max"]
    assert obligatory["seeds"] == 3


def test_all_8_metrics_declared(matrix: EvalMatrix) -> None:
    expected = {"M1.a", "M1.b", "M2.b", "M3.a", "M3.b", "M3.c",
                "M4.a", "M4.b"}
    assert set(matrix.metrics.keys()) == expected


def test_m3_b_is_pivot_metric(matrix: EvalMatrix) -> None:
    assert matrix.metrics["M3.b"].get("pivot") is True


def test_m2_b_data_source_is_studyforrest(matrix: EvalMatrix) -> None:
    # G1 Branch A locked at S2.1
    assert matrix.metrics["M2.b"]["data_source"] == "studyforrest"


def test_loader_rejects_empty_yaml(tmp_path: Path) -> None:
    empty_yaml = tmp_path / "empty.yaml"
    empty_yaml.write_text("")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_eval_matrix(empty_yaml)


def test_loader_rejects_non_mapping_yaml(tmp_path: Path) -> None:
    list_yaml = tmp_path / "list.yaml"
    list_yaml.write_text("- item1\n- item2\n")
    with pytest.raises(ValueError, match="must be a YAML mapping"):
        load_eval_matrix(list_yaml)
