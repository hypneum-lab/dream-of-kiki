"""E1 profiles, tuning selection, LR matching and n_eval (task 4)."""

from __future__ import annotations

import math
from typing import Any

import pytest

from scripts import e1_profiles as pr

AM = pr.Amendment((3e-4, 2), (3e-5, 5), 2.0, 140)


def _items(n: int = 4) -> list[dict[str, Any]]:
    return [{"prompt": f"Q{i}?\nAnswer:", "question_prompt": f"Q{i}?\nAnswer:",
             "choices": [" a", " b", " c", " d"], "answer": i % 4} for i in range(n)]


def test_profile_configs() -> None:
    c = {p: pr.profile_config(p, AM) for p in pr.ALL_PROFILES}
    assert (c["p_min"].lr, c["p_min"].m, c["p_min"].downscale) == (1e-4, 5, False)
    assert (c["p_equ"].lr, c["p_equ"].m, c["p_equ"].downscale) == (3e-4, 2, True)
    v = c["p_ctrl_vanilla_replay_tuned"]
    assert (v.lr, v.m, v.downscale, v.corpus) == (3e-4, 2, False, "facts")
    assert c["p_max"].restructure and c["p_max"].downscale and c["p_max"].lr == 1e-4
    o = c["p_ctrl_onpolicy_replay"]
    assert (o.lr, o.m, o.corpus, o.scoring_charged) == (3e-5, 5, "opr", True)
    lm = c["p_ctrl_lr_matched"]
    assert math.isclose(lm.lr or 0, 6e-4) and lm.m == 2 and not lm.downscale
    assert c["p_noise"].corpus == "none" and c["p_noise"].lr is None
    assert c["p_ctrl_template"].corpus == "prompts" and c["p_ctrl_template"].lr == 1e-4
    assert sum(x.scoring_charged for x in c.values()) == 1
    for p in pr.TUNED:
        with pytest.raises(ValueError, match="amendment"):
            pr.profile_config(p, None)
    assert pr.profile_config("p_min", None).lr == 1e-4
    with pytest.raises(ValueError):
        pr.profile_config("p_other", AM)


def test_profile_texts() -> None:
    it = _items()
    facts = pr.profile_texts("facts", it, 0)
    assert facts[1] == "Q1?\nAnswer: b"
    assert pr.profile_texts("prompts", it, 0) == [x["question_prompt"] for x in it]
    assert all(t.endswith("\nAnswer:") for t in pr.profile_texts("prompts", it, 0))  # no answer
    assert pr.profile_texts("none", it, 0) == []
    assert len(pr.profile_texts("neutral", it, 0)) == 4
    assert pr.profile_texts("shuffled", it, 3, shuffle=lambda t, s: t[::-1])[0] == facts[0][::-1]
    # OPR keeps only items whose pre-replay argmax is the gold
    assert pr.profile_texts("opr", it, 0, opr_choice=[0, 0, 2, 1]) == [facts[0], facts[2]]
    with pytest.raises(ValueError):
        pr.profile_texts("opr", it, 0)
    with pytest.raises(ValueError):
        pr.profile_texts("shuffled", it, 0)


def test_select_config_ties_and_grid() -> None:
    md = {cfg: 0.0 for cfg in pr.GRID}
    assert pr.select_config(md) == (3e-5, 2)  # all tied -> lowest LR, smaller m
    md[(1e-4, 5)] = 0.01
    md[(3e-4, 2)] = 0.01 + 5e-10  # within 1e-9 -> tie -> lower LR wins
    assert pr.select_config(md) == (1e-4, 5)
    md[(3e-4, 2)] = 0.0101
    assert pr.select_config(md) == (3e-4, 2)
    with pytest.raises(ValueError):
        pr.select_config({(1e-4, 5): 0.0})
    # ties are taken against the global maximum, not the running winner
    md = {cfg: -1.0 for cfg in pr.GRID}
    md[(3e-5, 5)], md[(1e-4, 2)], md[(1e-4, 5)] = 0.0, 0.75e-9, 1.5e-9
    assert pr.select_config(md) == (1e-4, 2)


def test_calibrate_c() -> None:
    ref = [1.0, 2.0, 3.0]  # median 2
    norms = {c: [c, c, c] for c in pr.C_GRID}
    assert pr.calibrate_c(norms, ref) == 2.0
    norms = {1.0: [1.0], 1.5: [4 / 1.5], 2.0: [1.5], 3.0: [10.0], 5.0: [20.0]}
    # c = 1.5 gives ratio 4/3, c = 2 gives 3/4: |log| ties -> the smaller c
    assert pr.calibrate_c(norms, [2.0]) == 1.5


def test_n_eval_rule() -> None:
    assert pr.n_eval_rule(0.10, 1000)["n_eval"] == 130  # planning value gives 128
    assert pr.n_eval_rule(0.20, 1000)["n_eval"] == 255
    r = pr.n_eval_rule(0.20, 250)
    assert r == {"n_eval_rule": 255, "n_eval": 210, "capped": True, "cap": 210}


def _cells(empty_opr: tuple[int, ...] = ()) -> list[dict[str, Any]]:
    cells: list[dict[str, Any]] = []
    for s in range(1000, 1010):
        for lr, m in pr.GRID:
            gain = 0.02 if (lr, m) == (3e-4, 2) else 0.0
            cells.append({"kind": "tune_vanilla", "seed": s, "lr": lr, "m": m,
                          "delta": gain + 0.001 * (s % 3),
                          "post_correct": [True, False, True, True]})
            ogain = 0.03 if (lr, m) == (3e-5, 5) else 0.0
            cells.append({"kind": "tune_opr", "seed": s, "lr": lr, "m": m,
                          "delta": ogain, "opr_corpus_size": 0 if s in empty_opr else 7,
                          "post_correct": [True, True, True, True]})
        for c in pr.C_GRID:
            cells.append({"kind": "calib_lr", "seed": s, "c": c, "lr": c * 3e-4, "m": 2,
                          "norm_change": c})
        cells.append({"kind": "calib_ref", "seed": s, "lr": 3e-4, "m": 2, "norm_change": 3.1})
    return cells


def test_tuning_amendment() -> None:
    a = pr.tuning_amendment(_cells(empty_opr=(1003,)), confirmatory_size=500)
    assert a["selection_vanilla"] == [3e-4, 2] and a["selection_opr"] == [3e-5, 5]
    assert a["c"] == 3.0 and a["opr_empty_tuning_seeds"] == [1003]
    assert a["p_disc"] == pytest.approx(0.25) and a["n_eval"] == math.ceil(0.25 / (0.039661 ** 2 / 2))
    assert len(a["tuning_seeds"]) == 10 and a["sd_d_proxy"] is not None
    am = pr.Amendment.from_dict(a)
    assert am.selection_vanilla == (3e-4, 2) and am.n_eval == a["n_eval"]


def test_tuning_amendment_opr_default_and_bad_calibration() -> None:
    a = pr.tuning_amendment(_cells(empty_opr=(1000, 1001, 1002)), confirmatory_size=500)
    assert a["opr_default_kept"] and a["selection_opr"] == [1e-4, 5]
    bad = _cells()
    for c in bad:
        if c["kind"] == "calib_lr":
            c["lr"] = 1e-4
    with pytest.raises(ValueError, match="c x LR"):
        pr.tuning_amendment(bad, confirmatory_size=500)


def test_amendment_input_validation() -> None:
    cells = _cells()
    with pytest.raises(ValueError, match="incomplete"):
        pr.tuning_amendment([c for c in cells if c["seed"] != 1009], confirmatory_size=500)
    with pytest.raises(ValueError, match="duplicate"):
        pr.tuning_amendment(cells + [dict(cells[0])], confirmatory_size=500)
    bad = _cells()
    bad[0]["delta"] = float("nan")
    with pytest.raises(ValueError, match="non-finite"):
        pr.tuning_amendment(bad, confirmatory_size=500)
    bad = _cells()
    bad[0]["seed"] = 999
    with pytest.raises(ValueError, match="outside"):
        pr.tuning_amendment(bad, confirmatory_size=500)
    with pytest.raises(ValueError, match="PI decision"):
        pr.tuning_amendment(_cells(empty_opr=tuple(range(1000, 1010))), confirmatory_size=500)


def test_validate_amendment() -> None:
    a = pr.tuning_amendment(_cells(), confirmatory_size=500)
    assert pr.validate_amendment(a, 500).n_eval == a["n_eval"]
    for field, value in (("c", 2.5), ("selection_vanilla", [2e-4, 5]), ("n_eval", 129)):
        with pytest.raises(ValueError):
            pr.validate_amendment({**a, field: value}, 500)
    with pytest.raises(ValueError, match="n_eval"):
        pr.validate_amendment(a, 300)  # cap differs for another corpus size


def test_validate_amendment_raw_types_and_ranges() -> None:
    a = pr.tuning_amendment(_cells(), confirmatory_size=500)
    for field, value in (("selection_vanilla", [3e-4, 2.9]), ("n_eval", float(a["n_eval"])),
                         ("p_disc", -1.0), ("p_disc", 2.0), ("n_eval", True)):
        with pytest.raises(ValueError):
            pr.validate_amendment({**a, field: value}, 500)


def test_tuning_rejects_bad_sizes_and_zero_norms() -> None:
    bad = _cells()
    for c in bad:
        if c["kind"] == "tune_opr":
            c["opr_corpus_size"] = -1
    with pytest.raises(ValueError, match="OPR corpus sizes"):
        pr.tuning_amendment(bad, confirmatory_size=500)
    bad = _cells()
    for c in bad:
        if c["kind"] == "calib_ref":
            c["norm_change"] = 0.0
    with pytest.raises(ValueError, match="PI decision"):
        pr.tuning_amendment(bad, confirmatory_size=500)
