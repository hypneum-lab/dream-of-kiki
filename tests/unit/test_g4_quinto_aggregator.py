"""Aggregator unit tests using JSON fixtures (no real driver run)."""
from __future__ import annotations

import json
from pathlib import Path

from experiments.g4_quinto_test.aggregator import (
    aggregate_g4_quinto_verdict,
)


def _h5_jonckheere_payload(
    *, reject: bool, monotonic: bool
) -> dict[str, object]:
    return {
        "reject_h0": reject,
        "monotonic_observed": monotonic,
        "j_statistic": 100.0 if reject else 1.0,
        "p_value": 0.001 if reject else 0.5,
        "alpha_per_test": 0.0167,
        "mean_p_min": 0.5 if monotonic else 0.7,
        "mean_p_equ": 0.6 if monotonic else 0.7,
        "mean_p_max": 0.7,
    }


def _h5c_payload(*, confirmed: bool) -> dict[str, object]:
    return {
        "h5c_recombine_empty_confirmed": confirmed,
        "fail_to_reject_h0": confirmed,
        "welch_t": 0.01,
        "welch_p_two_sided": 0.99 if confirmed else 0.001,
        "alpha_per_test": 0.0167,
        "mean_p_max_mog": 0.7,
        "mean_p_max_none": 0.7,
        "hedges_g_mog_vs_none": 0.001,
        "n_p_max_mog": 30,
        "n_p_max_none": 30,
    }


def _ae_payload() -> dict[str, object]:
    return {
        "n_p_max_ae": 30,
        "n_p_max_none": 30,
        "mean_p_max_ae": 0.7,
        "mean_p_max_none": 0.7,
        "welch_t": 0.0,
        "welch_p_two_sided": 0.99,
    }


def _write(path: Path, key: str, payload: dict[str, object]) -> None:
    path.write_text(json.dumps({"verdict": {key: payload}}))


def _write_step3(
    path: Path, h5c: dict[str, object], ae: dict[str, object]
) -> None:
    path.write_text(
        json.dumps(
            {
                "verdict": {
                    "h5c_recombine_strategy": h5c,
                    "ae_observation": ae,
                }
            }
        )
    )


def test_aggregator_all_confirmed(tmp_path: Path) -> None:
    s1 = tmp_path / "s1.json"
    s2 = tmp_path / "s2.json"
    s3 = tmp_path / "s3.json"
    _write(
        s1, "h5a_mlp_cifar",
        _h5_jonckheere_payload(reject=True, monotonic=True),
    )
    _write(
        s2, "h5b_cnn_cifar",
        _h5_jonckheere_payload(reject=True, monotonic=True),
    )
    _write_step3(s3, _h5c_payload(confirmed=True), _ae_payload())
    v = aggregate_g4_quinto_verdict(s1, s2, s3)
    s = v["summary"]
    assert s["h5a_confirmed"] is True
    assert s["h5b_confirmed"] is True
    assert s["h5c_confirmed"] is True
    assert s["h4c_to_h5c_universality"] is True
    assert s["any_confirmed"] is True


def test_aggregator_step3_deferred(tmp_path: Path) -> None:
    s1 = tmp_path / "s1.json"
    s2 = tmp_path / "s2.json"
    _write(
        s1, "h5a_mlp_cifar",
        _h5_jonckheere_payload(reject=False, monotonic=False),
    )
    _write(
        s2, "h5b_cnn_cifar",
        _h5_jonckheere_payload(reject=False, monotonic=False),
    )
    v = aggregate_g4_quinto_verdict(s1, s2, None)
    s = v["summary"]
    assert s["h5c_deferred"] is True
    assert s["h5c_confirmed"] is False
    assert s["h4c_to_h5c_universality"] is False
    assert s["any_confirmed"] is False


def test_aggregator_h5c_only_confirmed(tmp_path: Path) -> None:
    """H5-A/B falsified, H5-C confirmed: universality flag fires."""
    s1 = tmp_path / "s1.json"
    s2 = tmp_path / "s2.json"
    s3 = tmp_path / "s3.json"
    _write(
        s1, "h5a_mlp_cifar",
        _h5_jonckheere_payload(reject=False, monotonic=False),
    )
    _write(
        s2, "h5b_cnn_cifar",
        _h5_jonckheere_payload(reject=False, monotonic=False),
    )
    _write_step3(s3, _h5c_payload(confirmed=True), _ae_payload())
    v = aggregate_g4_quinto_verdict(s1, s2, s3)
    s = v["summary"]
    assert s["h5a_confirmed"] is False
    assert s["h5b_confirmed"] is False
    assert s["h5c_confirmed"] is True
    assert s["h4c_to_h5c_universality"] is True
