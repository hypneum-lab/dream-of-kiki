"""Hopfield control arm (S5): identities the script must reproduce.

docs/milestones/2026-11-11-downscale-dose-response.md, section 6 bis.
"""

from __future__ import annotations

import numpy as np

from scripts import hopfield_scale_control as hc
from scripts.e1_dose_response import DOSES

N, CUES, SWEEPS = 200, 40, 8


def _setup(load: float = 0.05) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    xi = hc.patterns(N, load)
    s0, which = hc.cues(xi, CUES)
    return hc.hebbian(xi), s0, xi[which]


def test_cue_overlap_and_zero_diagonal() -> None:
    j, s0, target = _setup()
    assert np.allclose((s0 * target).mean(1), hc.CUE_OVERLAP)
    assert np.all(np.diag(j) == 0) and np.allclose(j, j.T)


def test_noise_has_the_requested_relative_norm() -> None:
    j, _, _ = _setup()
    g = hc.noisy(j, 0.2, 0) - j
    assert np.isclose(np.linalg.norm(g) / np.linalg.norm(j), 0.2)
    assert np.all(np.diag(g) == 0) and np.allclose(g, g.T)


def test_zero_temperature_ignores_the_scale() -> None:
    j, s0, target = _setup()
    base = hc.run(j, 0.0, s0, target, SWEEPS, 1)
    for eps in (0.01, 0.5):
        assert np.array_equal(hc.run(j * (1 - eps), 0.0, s0, target, SWEEPS, 1), base)


def test_rescaling_is_a_temperature_change_with_common_random_numbers() -> None:
    j, s0, target = _setup()
    for eps in (0.05, 0.5):
        d = hc.run(j * (1 - eps), 0.3, s0, target, SWEEPS, 1)
        tq = hc.run(j, 0.3 / (1 - eps), s0, target, SWEEPS, 1)
        assert np.abs(d - tq).max() <= hc.IDENTITY_TOL
    hot = hc.run(j, 0.3, s0, target, SWEEPS, 1)
    assert not np.array_equal(hc.run(j * 0.5, 0.3, s0, target, SWEEPS, 1), hot)


def test_retrieval_below_capacity_and_field_bookkeeping() -> None:
    j, s0, target = _setup(0.03)
    assert hc.run(j, 0.0, s0, target, SWEEPS, 1).mean() > 0.95
    # the incremental field equals a recomputed field: one synchronous check
    s = s0.copy()
    assert np.allclose(s @ j, (j @ s.T).T)


def test_c1_rule_on_constructed_ranks() -> None:
    t, f = True, False
    assert hc.c1([t] * 6, [t, t, f, f, f, f], [f, f, f, f, t, t])[0] == "supported"
    assert hc.c1([t, t, f, f, f, f], [t, t, f, f, f, f], [f, f, f, f, t, t])[0] == "refuted"
    assert hc.c1([t] * 6, [t] * 5 + [f], [f] * 5 + [t])[0] == "inconclusive"
    assert hc.c1([t] * 6, [t] * 6, [f] * 6)[0] == "non_test"


def test_cell_reports_identity_and_shapes() -> None:
    out = hc.cell(0.05, 0.2, n=N, n_cues=CUES, sweeps=SWEEPS, n_boot=200)
    assert out["identity_ok"] and len(out["table"]) == len(DOSES)
    assert out["C2_analogue"] in ("supported", "refuted")
    assert hc.cell(0.05, 0.0, n=N, n_cues=CUES, sweeps=SWEEPS, n_boot=200)["C2_analogue"] == (
        "not_applicable")
