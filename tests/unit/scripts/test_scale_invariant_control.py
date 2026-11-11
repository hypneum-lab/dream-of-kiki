"""Positive control of the C1 / C2 rule on a scale-invariant network.

docs/milestones/2026-11-11-downscale-dose-response.md, section 6 quater.
"""

from __future__ import annotations

import numpy as np

from scripts import scale_invariant_control as sc


def test_downscale_is_a_no_op_without_gain() -> None:
    p, x, _ = sc.build(gain=False)
    assert "gain" not in p
    assert np.allclose(sc.scores(sc.downscale(p, 0.5), x), sc.scores(p, x), atol=1e-9)


def test_downscale_is_a_temperature_with_gain() -> None:
    p, x, _ = sc.build(gain=True)
    base = sc.scores(p, x)
    assert np.allclose(sc.scores(sc.downscale(p, 0.5), x), base / 2.0, atol=1e-9)


def test_noise_has_the_requested_relative_norm_per_tensor() -> None:
    p, _, _ = sc.build(gain=True)
    q = sc.perturb_per_tensor(p, 0.2, 0)
    for k in p:
        assert np.isclose(np.linalg.norm(q[k] - p[k]) / np.linalg.norm(p[k]), 0.2)


def test_rule_returns_the_known_verdicts() -> None:
    plain = sc.control(False, n_boot=1000)
    gained = sc.control(True, n_boot=1000)
    assert plain["verdict"]["C1"] == "supported"
    assert plain["verdict"]["C2"] == "not_applicable"  # nothing to explain: KL is zero
    assert gained["verdict"]["C1"] == "supported"
    assert gained["verdict"]["C2"] == "supported"
    assert np.isclose(gained["verdict"]["temperature"]["T"], 2.0)
    assert plain["negative_control"]["C1"] == "refuted"
    assert gained["negative_control"]["C1"] == "refuted"
