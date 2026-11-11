"""Verdict of the DOWNSCALE dose-response on constructed tables.

docs/milestones/2026-11-11-downscale-dose-response.md, section 4.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from scripts import e1_dose_response as dr

N, K = 760, 4


def _base(seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    ans = rng.integers(0, K, N)
    s = rng.normal(0.0, 1.0, (N, K))
    known = rng.random(N) < 0.35  # about 0.51 accuracy
    s[known, ans[known]] += 6.0
    return s, ans


def _noise(seed: int) -> np.ndarray:
    return np.random.default_rng(seed).normal(0.0, 1.0, (N, K))


def _passes(s0: np.ndarray, d: dict[float, np.ndarray], rt_from: int) -> list[dict[str, Any]]:
    """D scores per dose; Rt equal to the base below dose index rt_from, pure noise from it."""
    out: list[dict[str, Any]] = []
    for arm, eps, seed in dr.plan():
        if arm == "base" or eps is None:
            s = s0
        elif arm == "D":
            s = d[eps]
        else:
            s = s0 if dr.DOSES.index(eps) < rt_from else _noise(1000 + (seed or 0))
        out.append({"arm": arm, "eps": eps, "seed": seed, "scores": s})
    return out


def _run(passes: list[dict[str, Any]], ans: np.ndarray) -> dict[str, Any]:
    return dr.analyse(passes, ans, np.ones(N, dtype=bool), n_boot=2000)


def test_plan_has_28_passes() -> None:
    p = dr.plan()
    assert len(p) == 28 and p[0] == ("base", None, None)
    assert sum(a == "D" for a, _, _ in p) == 6 and sum(a == "Rt" for a, _, _ in p) == 18
    assert sum(a == "Rg" for a, _, _ in p) == 3


def test_pure_symmetry_supports_c1() -> None:
    s0, ans = _base()
    r = _run(_passes(s0, {e: s0 for e in dr.DOSES}, rt_from=2), ans)
    assert (r["rank_D"], r["rank_Rt"], r["C1"]) == (6, 2, "supported")
    assert r["C2"] == "not_applicable"  # identical distributions: nothing to explain


def test_same_tolerance_refutes_c1() -> None:
    s0, ans = _base()
    d = {e: (s0 if i < 2 else _noise(7)) for i, e in enumerate(dr.DOSES)}
    r = _run(_passes(s0, d, rt_from=2), ans)
    assert r["rank_D"] == r["rank_Rt"] == 2 and r["C1"] == "refuted"


def test_one_step_apart_is_inconclusive() -> None:
    s0, ans = _base()
    d = {e: (s0 if i < 3 else _noise(7)) for i, e in enumerate(dr.DOSES)}
    assert _run(_passes(s0, d, rt_from=2), ans)["C1"] == "inconclusive"


def test_gate_returns_non_test() -> None:
    s0, ans = _base()
    never = _run(_passes(s0, {e: s0 for e in dr.DOSES}, rt_from=6), ans)  # noise never breaks
    at_once = _run(_passes(s0, {e: s0 for e in dr.DOSES}, rt_from=0), ans)  # breaks at 0.01
    assert never["C1"] == at_once["C1"] == "non_test"
    assert never["C2"] == "not_applicable"


def test_temperature_supports_c2() -> None:
    s0, ans = _base()
    r = _run(_passes(s0, {e: s0 / 2.0 for e in dr.DOSES}, rt_from=2), ans)
    assert r["C1"] == "supported" and r["C2"] == "supported"
    assert abs(r["temperature"]["T"] - 2.0) < 0.1 and r["temperature"]["share_removed"] > 0.99


def test_reshuffled_losers_refute_c2() -> None:
    s0, ans = _base()
    s = s0.copy()
    for i in range(N):  # keep the argmax and the sharpness, rotate the other options' scores
        top = int(s[i].argmax())
        rest = [k for k in range(K) if k != top]
        s[i, rest] = np.roll(s0[i, rest], 1)
    r = _run(_passes(s0, {e: s for e in dr.DOSES}, rt_from=2), ans)
    assert r["rank_D"] == 6 and r["temperature"]["kl_at_1"] > dr.KL_FLOOR
    assert r["C2"] == "refuted"
