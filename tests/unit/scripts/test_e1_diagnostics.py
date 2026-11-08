"""E1 section 6 diagnostics helpers (task 5)."""

from __future__ import annotations

import math

import pytest

from scripts import e1_diagnostics as dg


def test_relative_drift() -> None:
    assert dg.relative_drift(1.0, 4.0) == 0.5
    with pytest.raises(ValueError):
        dg.relative_drift(1.0, 0.0)


@pytest.mark.parametrize(("pre", "post", "flag"), [
    (0.60, 0.49, True), (0.60, 0.50, False), (0.30, 0.24, True), (0.30, 0.25, False),
])
def test_collapse_flag(pre: float, post: float, flag: bool) -> None:
    assert dg.collapse_flag(pre, post) is flag


def test_collapse_gate() -> None:
    assert not dg.collapse_gate_fails([True] * 2 + [False] * 8)  # 20 % exactly
    assert dg.collapse_gate_fails([True] * 3 + [False] * 7)
    assert dg.collapse_rate([]) == 0.0


def test_choice_kl() -> None:
    same = [[-1.0, -2.0, -3.0, -4.0]]
    assert dg.choice_kl(same, same) == pytest.approx(0.0)
    shifted = [[x + 5 for x in same[0]]]
    assert dg.choice_kl(same, shifted) == pytest.approx(0.0)  # softmax is shift-invariant
    p = [[0.0, 0.0, 0.0, 0.0]]
    q = [[math.log(1.0), math.log(1.0), math.log(1.0), math.log(5.0)]]
    exp = sum(0.25 * math.log(0.25 / v) for v in (1 / 8, 1 / 8, 1 / 8, 5 / 8))
    assert dg.choice_kl(p, q) == pytest.approx(exp)
    with pytest.raises(ValueError):
        dg.choice_kl([], [])


def test_choice_kl_underflow_guard() -> None:
    # Post score of choice 3 is 1000 below the others: softmax underflows to 0
    # (the p_ctrl_lr_matched|2 stop of 2026-11-01); the guard keeps it finite.
    ref = [[0.0, 0.0, 0.0, 0.0]]
    post = [[0.0, 0.0, 0.0, -1000.0]]
    lse = math.log(3.0)  # log-sum-exp of the post scores, exp(-1000) negligible
    exp = 0.75 * (math.log(0.25) - (0.0 - lse)) + 0.25 * (math.log(0.25) - (-1000.0 - lse))
    got = dg.choice_kl(ref, post)
    assert math.isfinite(got) and got == pytest.approx(exp)
    # terms with b > 0 are computed exactly as before the guard
    p = [[0.0, -1.0, -2.0, -3.0]]
    q = [[-0.5, -1.5, -2.0, -0.1]]
    pr_, pp_ = dg._softmax(p[0]), dg._softmax(q[0])
    old = sum(a * (math.log(a) - math.log(b)) for a, b in zip(pr_, pp_) if a > 0)
    assert dg.choice_kl(p, q) == old


def test_lost_correct() -> None:
    assert dg.lost_correct([True, True, False, True], [True, False, True, False]) == 2 / 3
    assert dg.lost_correct([False], [True]) is None


def test_usage_l1() -> None:
    assert dg.usage_l1({"0": 2, "1": 2}, {"0": 1, "1": 3}) == pytest.approx(0.5)
    assert dg.usage_l1({"0": 1}, {"1": 1}) == pytest.approx(2.0)
    assert dg.usage_l1({}, {"1": 1}) is None


def test_mean_ci95() -> None:
    r = dg.mean_ci95([1.0, 2.0, 3.0, None])
    assert r["n"] == 3 and r["mean"] == 2.0
    hi = r["hi"]
    assert isinstance(hi, float) and r["lo"] is not None
    assert hi - 2.0 == pytest.approx(4.302652729911275 * 1 / math.sqrt(3))
    assert dg.mean_ci95([]) == {"n": 0, "mean": None, "lo": None, "hi": None}
