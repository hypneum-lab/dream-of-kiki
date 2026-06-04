"""Q2 — pytest entry point for the structural-invariant audit.

Tests assert per-substrate FP rate against expected pre-registration
brackets. The audit allows non-zero FP — pytest pinning here ensures
the result file is reproducible and that the verdict-per-pre-reg is
emitted to JSON before paper §4 update (Task 10).
"""
from __future__ import annotations

import json
import subprocess
import sys
from typing import Any
from pathlib import Path

import pytest


REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_PATH = REPO_ROOT / "docs/milestones/q2-conformance-negative-results.json"


@pytest.fixture(scope="module")
def audit_results() -> dict[str, Any]:
    """Run the audit script if results don't exist; return the JSON."""
    if not RESULTS_PATH.exists():
        subprocess.run(
            [sys.executable, "scripts/run_q2_conformance_audit.py"],
            cwd=REPO_ROOT,
            check=True,
        )
    parsed: dict[str, Any] = json.loads(RESULTS_PATH.read_text())
    return parsed


def test_audit_results_exist(audit_results: dict[str, Any]) -> None:
    assert "audits" in audit_results
    assert "verdict" in audit_results
    assert len(audit_results["audits"]) == 15


def test_all_15_substrates_present(audit_results: dict[str, Any]) -> None:
    expected_names = {
        "IdentitySubstrate", "RandomNoiseSubstrate", "LookupSubstrate",
        "FrozenZerosSubstrate", "ConstantSubstrate",
        "ShapePreservingNoise", "BudgetGamingSubstrate", "PermutedReplay",
        "OverloadRecombine", "StatelessAccident", "CommutativityViolator",
        "BoundaryCheater",
        "RandomCoinFlip", "ShapeDistributionDependent",
        "SeedDependentSubstrate",
    }
    actual = {a["name"] for a in audit_results["audits"]}
    assert actual == expected_names


def test_verdict_in_known_brackets(audit_results: dict[str, Any]) -> None:
    assert audit_results["verdict"] in {
        "0_FP_robust", "1_to_2_FP_strengthen", "ge_3_FP_reformulate",
    }
