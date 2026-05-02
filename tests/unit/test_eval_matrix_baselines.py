"""TDD for the `baselines:` block in eval-matrix.yaml (FC-MINOR addition).

Reference :
  docs/superpowers/plans/2026-05-02-wake-sleep-cl-ablation-baseline.md
  Task 2.
"""
from __future__ import annotations

import textwrap
from pathlib import Path

import pytest

from harness.config.eval_matrix import load_eval_matrix


def _write_yaml(tmp_path: Path, body: str) -> Path:
    p = tmp_path / "eval-matrix.yaml"
    p.write_text(textwrap.dedent(body))
    return p


def test_baselines_block_parses(tmp_path: Path) -> None:
    p = _write_yaml(
        tmp_path,
        """
        version: "C-v0.12.0+PARTIAL"
        bump_rules: {}
        publication_ready_gate: {}
        metrics:
          M1.a: {name: forgetting_rate}
          M1.b: {name: avg_accuracy}
        baselines:
          wake_sleep_cl:
            name: wake_sleep_consolidated_learning
            bibkey: alfarano2024wakesleep
            arxiv: "2401.08623"
            scores_on: [M1.a, M1.b]
            variant: c
        """,
    )
    em = load_eval_matrix(p)
    assert "wake_sleep_cl" in em.baselines
    assert em.baselines["wake_sleep_cl"]["bibkey"] == "alfarano2024wakesleep"
    assert em.baselines["wake_sleep_cl"]["scores_on"] == ["M1.a", "M1.b"]


def test_baseline_scores_on_must_be_list(tmp_path: Path) -> None:
    """A scalar `scores_on:` must be rejected with a typed error."""
    p = _write_yaml(
        tmp_path,
        """
        version: "C-v0.12.0+PARTIAL"
        bump_rules: {}
        publication_ready_gate: {}
        metrics:
          M1.a: {name: forgetting_rate}
        baselines:
          bad:
            bibkey: foo2024
            scores_on: "M1.a"
            variant: c
        """,
    )
    with pytest.raises(ValueError, match="scores_on"):
        load_eval_matrix(p)


def test_baseline_scores_on_must_reference_known_metrics(
    tmp_path: Path,
) -> None:
    """Typo'd metric IDs in scores_on must be caught at load time."""
    p = _write_yaml(
        tmp_path,
        """
        version: "C-v0.12.0+PARTIAL"
        bump_rules: {}
        publication_ready_gate: {}
        metrics:
          M1.a: {name: forgetting_rate}
          M1.b: {name: avg_accuracy}
        baselines:
          typo:
            bibkey: foo2024
            scores_on: [M1.x]
            variant: c
        """,
    )
    with pytest.raises(ValueError, match="M1.x"):
        load_eval_matrix(p)


def test_baselines_block_optional_for_legacy_yaml(tmp_path: Path) -> None:
    """Older yamls without `baselines:` must still load (back-compat)."""
    p = _write_yaml(
        tmp_path,
        """
        version: "C-v0.7.0+PARTIAL"
        bump_rules: {}
        publication_ready_gate: {}
        metrics: {}
        """,
    )
    em = load_eval_matrix(p)
    assert em.baselines == {}


def test_baselines_each_entry_has_required_fields(tmp_path: Path) -> None:
    """Every baseline entry must declare bibkey + scores_on + variant."""
    p = _write_yaml(
        tmp_path,
        """
        version: "C-v0.12.0+PARTIAL"
        bump_rules: {}
        publication_ready_gate: {}
        metrics: {}
        baselines:
          incomplete:
            name: foo
        """,
    )
    with pytest.raises(ValueError, match="bibkey"):
        load_eval_matrix(p)


def test_repo_eval_matrix_has_wake_sleep_baseline() -> None:
    """The repo-level eval-matrix.yaml ships the wake-sleep CL baseline."""
    repo_root = Path(__file__).resolve().parents[2]
    matrix_path = repo_root / "docs" / "interfaces" / "eval-matrix.yaml"
    em = load_eval_matrix(matrix_path)
    assert "wake_sleep_cl" in em.baselines
    entry = em.baselines["wake_sleep_cl"]
    assert entry["bibkey"] == "alfarano2024wakesleep"
    assert entry["arxiv"] == "2401.08623"
    assert entry["variant"] == "c"
    assert "M1.a" in entry["scores_on"]
    assert "M1.b" in entry["scores_on"]
