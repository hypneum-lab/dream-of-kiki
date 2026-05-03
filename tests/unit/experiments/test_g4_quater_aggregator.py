"""Smoke + structural tests for the G4-quater aggregator."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.g4_quater_test.aggregator import (
    aggregate_g4_quater_verdict,
)


def _stub_step1(path: Path) -> None:
    payload = {
        "verdict": {
            "h4a_deeper_substrate": {
                "j_statistic": 100.0,
                "p_value": 0.5,
                "reject_h0": False,
                "mean_p_min": 0.6,
                "mean_p_equ": 0.6,
                "mean_p_max": 0.6,
                "monotonic_observed": False,
                "alpha_per_test": 0.0167,
            },
            "retention_by_arm": {},
        }
    }
    path.write_text(json.dumps(payload))


def _stub_step2(path: Path) -> None:
    per_factor = [
        {
            "factor": 0.85,
            "j_statistic": 10.0,
            "p_value": 0.9,
            "reject_h0": False,
            "mean_p_min": 0.7,
            "mean_p_equ": 0.65,
            "mean_p_max": 0.65,
            "monotonic_observed": False,
            "alpha_per_test": 0.0056,
        },
    ]
    payload = {
        "verdict": {
            "h4b_per_factor": per_factor,
            "any_factor_recovers_ordering": False,
        }
    }
    path.write_text(json.dumps(payload))


def _stub_step3(path: Path, *, fail_to_reject: bool) -> None:
    p_value = 0.8 if fail_to_reject else 0.001
    h4c = {
        "n_p_max_mog": 95,
        "n_p_max_none": 95,
        "mean_p_max_mog": 0.70,
        "mean_p_max_none": 0.70 if fail_to_reject else 0.50,
        "welch_t": 0.1 if fail_to_reject else 5.0,
        "welch_p_two_sided": p_value,
        "alpha_per_test": 0.0167,
        "fail_to_reject_h0": fail_to_reject,
        "h4c_recombine_empty_confirmed": fail_to_reject,
        "hedges_g_mog_vs_none": 0.0 if fail_to_reject else 0.5,
    }
    payload = {
        "verdict": {
            "h4c_recombine_strategy": h4c,
            "ae_observation": {"insufficient_samples": True},
        }
    }
    path.write_text(json.dumps(payload))


def test_aggregate_no_confirmations(tmp_path: Path) -> None:
    s1 = tmp_path / "s1.json"
    s2 = tmp_path / "s2.json"
    s3 = tmp_path / "s3.json"
    _stub_step1(s1)
    _stub_step2(s2)
    _stub_step3(s3, fail_to_reject=False)
    v = aggregate_g4_quater_verdict(s1, s2, s3)
    assert v["summary"]["h4a_confirmed"] is False
    assert v["summary"]["h4b_confirmed"] is False
    assert v["summary"]["h4c_confirmed"] is False
    assert v["summary"]["any_confirmed"] is False


def test_aggregate_h4c_only(tmp_path: Path) -> None:
    s1 = tmp_path / "s1.json"
    s2 = tmp_path / "s2.json"
    s3 = tmp_path / "s3.json"
    _stub_step1(s1)
    _stub_step2(s2)
    _stub_step3(s3, fail_to_reject=True)
    v = aggregate_g4_quater_verdict(s1, s2, s3)
    assert v["summary"]["h4c_confirmed"] is True
    assert v["summary"]["any_confirmed"] is True
    assert v["summary"]["all_three_confirmed"] is False
