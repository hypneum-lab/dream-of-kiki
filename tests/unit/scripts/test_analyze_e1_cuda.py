"""E1 decision code (task 6): every section 2 state, gate and section 7 row."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest

from scripts import analyze_e1_cuda as an
from scripts.e1_profiles import ALL_PROFILES


def _cells(shift: dict[str, float] | None = None, sd: float = 0.02,
           seed: int = 0) -> list[dict[str, Any]]:
    """141 seeds x 10 profiles; ``shift`` adds a mean delta per profile."""
    rng = np.random.default_rng(seed)
    shift = shift or {}
    base = rng.normal(0.0, 0.03, size=141)
    out = []
    for p in ALL_PROFILES:
        noise = rng.normal(0.0, sd / np.sqrt(2), size=141)
        for s in range(141):
            d = float(base[s] + noise[s] + shift.get(p, 0.0))
            out.append({"mode": "confirmatory", "profile": p, "seed": s, "pre": 0.5,
                        "post": 0.5 + d, "collapse": False, "opr_corpus_size": 10,
                        "budget": {"status": "matched"}, "corpus": {"sha": "x"},
                        "model": "m", "revision": "r", "amendment_sha256": "a",
                        "backend": "mlx",
                        "run_identity": "i"})
    return out


def _set(cells: list[dict[str, Any]], profile: str, seeds: range, **kw: Any) -> None:
    for c in cells:
        if c["profile"] == profile and c["seed"] in seeds:
            c.update(kw)


@pytest.mark.parametrize(("L", "U", "m", "state"), [
    (0.001, 0.03, 0.015, "SUPERIOR"),
    (0.001, 0.009, 0.005, "POSITIVE-BELOW-SESOI"),
    (0.001, 0.02, 0.009, "POSITIVE-BELOW-SESOI"),
    (-0.01, 0.009, 0.0, "BOUNDED-NO-GAIN"),
    (-0.01, 0.02, 0.005, "INCONCLUSIVE"),
])
def test_superiority_states(L: float, U: float, m: float, state: str) -> None:
    assert an.state_superiority(L, U, m, 0.01) == state


@pytest.mark.parametrize(("L", "U", "state"), [
    (-0.009, 0.01, "NON-INFERIOR"), (-0.05, -0.011, "INFERIOR"), (-0.02, 0.0, "INCONCLUSIVE"),
])
def test_noninferiority_states(L: float, U: float, state: str) -> None:
    assert an.state_noninferiority(L, U, 0.01) == state


def test_j1_all_supported() -> None:
    cells = _cells({"p_min": 0.03, "p_equ": 0.03})
    r = an.decide(cells)
    assert r["states"] == {"H14-A": "SUPERIOR", "H14-B": "SUPERIOR", "H14-C": "NON-INFERIOR"}
    assert r["joint_headline"] == "J1" and r["confirmatory"]
    assert r["hypotheses"]["H14-A"]["n"] == 141


def test_j2_j3_j4() -> None:
    r = an.decide(_cells({"p_min": 0.03}, sd=0.01))
    assert r["states"]["H14-A"] == "SUPERIOR" and r["states"]["H14-B"] == "BOUNDED-NO-GAIN"
    assert r["joint_headline"] == "J2"
    r = an.decide(_cells({}, sd=0.01))
    assert r["states"]["H14-A"] == "BOUNDED-NO-GAIN" and r["joint_headline"] == "J3"
    r = an.decide(_cells({"p_ctrl_onpolicy_replay": 0.05}, sd=0.01))
    assert r["states"]["H14-C"] == "INFERIOR"
    assert an.joint_headline({"H14-A": "INCONCLUSIVE", "H14-B": "SUPERIOR",
                              "H14-C": "NON-INFERIOR"}) == "J4"
    assert an.joint_headline({"H14-A": "SUPERIOR", "H14-B": "SUPERIOR",
                              "H14-C": "INCONCLUSIVE"}) == "J4"


def test_differential_exclusion_gate() -> None:
    cells = _cells({"p_min": 0.03})
    _set(cells, "p_ctrl_template", range(8), post=float("nan"))
    r = an.decide(cells)
    h = r["hypotheses"]["H14-A"]
    assert h["state"] == "UNINTERPRETABLE" and h["n_lost_cell_level"] == 8
    assert h["state_ungated"] == "SUPERIOR"
    cells = _cells({"p_min": 0.03})
    _set(cells, "p_ctrl_template", range(7), invariant_violation="S2")
    assert an.decide(cells)["states"]["H14-A"] == "SUPERIOR"


def test_opr_feasibility_and_overbudget() -> None:
    cells = _cells()
    _set(cells, an.OPR, range(8), opr_corpus_size=0)
    h = an.decide(cells)["hypotheses"]["H14-C"]
    assert h["state"] == "UNINTERPRETABLE" and h["opr_feasibility_dropped"] == 8
    assert h["n_lost_cell_level"] == 0 and h["n"] == 133
    cells = _cells()
    _set(cells, an.OPR, range(3), budget={"status": "scoring-overbudget"})
    h = an.decide(cells)["hypotheses"]["H14-C"]
    assert h["lost_by_reason"] == {"scoring_overbudget": 3}
    # overbudget only matters for matched-compute contrasts with OPR
    _set(cells, "p_min", range(3), budget={"status": "scoring-overbudget"})
    assert an.decide(cells)["hypotheses"]["H14-A"]["n_lost_cell_level"] == 0


def test_collapse_gate() -> None:
    cells = _cells({"p_min": 0.03, "p_equ": 0.03})
    _set(cells, "p_ctrl_vanilla_replay_tuned", range(29), collapse=True)  # 20.6 %
    r = an.decide(cells)
    assert r["states"]["H14-B"] == "UNINTERPRETABLE"
    assert r["hypotheses"]["H14-B"]["gates_failed"] == ["collapse"]
    cells = _cells({"p_min": 0.03, "p_equ": 0.03})
    _set(cells, "p_ctrl_vanilla_replay_tuned", range(28), collapse=True)  # 19.9 %
    assert an.decide(cells)["states"]["H14-B"] == "SUPERIOR"


def test_collapse_rate_is_per_profile_not_per_pair() -> None:
    cells = _cells({"p_min": 0.03, "p_equ": 0.03})
    # 29 collapsed vanilla cells, 8 of them on seeds where p_equ is excluded:
    # the profile rate (29/141) still fails the gate
    _set(cells, "p_ctrl_vanilla_replay_tuned", range(29), collapse=True)
    _set(cells, "p_equ", range(4), post=float("nan"))
    h = an.decide(cells)["hypotheses"]["H14-B"]
    assert h["collapse_rate"]["p_ctrl_vanilla_replay_tuned"] == pytest.approx(29 / 141)
    assert "collapse" in h["gates_failed"]


def test_opr_size_distribution() -> None:
    cells = _cells()
    _set(cells, an.OPR, range(3), opr_corpus_size=0)
    r = an.decide(cells)
    assert r["opr_corpus_size"] == {"0": 3, "10": 138}


def test_incomplete_run_is_exploratory() -> None:
    cells = [c for c in _cells({"p_min": 0.03})
             if not (c["profile"] == "p_max" and c["seed"] == 140)]
    r = an.decide(cells)
    assert not r["run_complete"] and not r["confirmatory"] and r["n_complete_seeds"] == 140


def test_refusals() -> None:
    cells = _cells()
    cells[0]["mode"] = "tuning"
    with pytest.raises(ValueError, match="non-confirmatory"):
        an.decide(cells)
    cells = _cells()
    cells.append(dict(cells[0]))
    with pytest.raises(ValueError, match="duplicate"):
        an.decide(cells)
    cells = _cells()
    cells[0]["corpus"] = {"sha": "y"}
    with pytest.raises(ValueError, match="corpus"):
        an.decide(cells)
    cells = _cells()
    del cells[7]["run_identity"]
    with pytest.raises(ValueError, match="lack"):
        an.decide(cells)
    cells = _cells()
    cells[5]["revision"] = "main"
    with pytest.raises(ValueError, match="revision"):
        an.decide(cells)
    cells = _cells()
    cells[0]["seed"] = 141
    with pytest.raises(ValueError, match="duplicate|outside"):
        an.decide(cells)


def test_degenerate_identical_differences() -> None:
    cells = _cells(sd=0.0)
    for c in cells:
        c["post"] = 0.5 + (0.02 if c["profile"] == "p_min" else 0.0)
    h = an.decide(cells)["hypotheses"]["H14-A"]
    assert h["degenerate"] and h["L"] == h["U"] == pytest.approx(0.02)
    assert h["state"] == "SUPERIOR"


def test_per_source_descriptive() -> None:
    cells = _cells()
    for c in cells:
        c["eval_sources"] = ["a", "a", "b", "b"]
        c["pre_correct"] = [False, False, False, False]
        c["post_correct"] = ([True, True, False, False] if c["profile"] == "p_min"
                             else [False, False, False, False])
    ps = an.decide(cells)["per_source"]["H14-A"]
    assert ps["a"]["mean"] == pytest.approx(1.0) and ps["b"]["mean"] == pytest.approx(0.0)


def test_h14c_infeasible_per_amendment() -> None:
    """PI decision 2026-10-25 (2a): the OPR arm is not run, H14-C is INFEASIBLE."""
    cells = [c for c in _cells({"p_min": 0.03, "p_equ": 0.03}) if c["profile"] != an.OPR]
    r = an.decide(cells, {"h14c_infeasible": True, "c": 2.0})
    assert r["states"] == {"H14-A": "SUPERIOR", "H14-B": "SUPERIOR", "H14-C": "INFEASIBLE"}
    assert r["joint_headline"] == "J4" and r["run_complete"] and r["arms_not_run"] == [an.OPR]
    assert r["secondary"][f"p_min-{an.OPR}"]["state"] == "INFEASIBLE"
    with pytest.raises(ValueError, match="not run"):
        an.decide(_cells(), {"h14c_infeasible": True, "c": 2.0})


def test_lr_matching_not_computable() -> None:
    """PI decision 2026-10-25 (2b)."""
    cells = [c for c in _cells() if c["profile"] != "p_ctrl_lr_matched"]
    r = an.decide(cells, {"h14c_infeasible": False, "c": None})
    assert r["exploratory_arms"]["p_equ-p_ctrl_lr_matched"]["state"] == "NOT-COMPUTABLE"
    assert r["run_complete"] and r["states"]["H14-C"] != "INFEASIBLE"
