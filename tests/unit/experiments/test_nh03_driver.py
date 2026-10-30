"""Unit tests for the NH-03 driver (budget equality, R1, schedule)."""
from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("mlx.core")

from experiments.g4_split_fmnist.dataset import SplitFMNISTTask  # noqa: E402
from experiments.nh03_ttt_vs_episodes import run_nh03  # noqa: E402
from experiments.nh03_ttt_vs_episodes.analysis import (  # noqa: E402
    bootstrap_ci,
    classify,
    contrast,
)


def _tasks() -> list[SplitFMNISTTask]:
    rng = np.random.default_rng(0)
    out: list[SplitFMNISTTask] = []
    for t in range(5):
        x = rng.standard_normal((96, 12)).astype(np.float32) + t
        y = (x[:, t % 12] > t).astype(np.int64)
        out.append(SplitFMNISTTask(x_train=x, y_train=y,
                                   x_test=x[:40].copy(), y_test=y[:40].copy()))
    return out


@pytest.fixture(scope="module")
def tasks() -> list[SplitFMNISTTask]:
    return _tasks()


@pytest.mark.parametrize("arm", run_nh03.ALL_ARMS)
def test_every_arm_spends_exact_budget(arm: str, tasks: list[SplitFMNISTTask]) -> None:
    cell = run_nh03.run_cell(arm, 3, tasks, ttt_lr=0.1)
    expected = 0 if arm == "baseline" else run_nh03.WRITE_BUDGET
    assert cell["writes_total"] == expected
    assert cell["dr0_all_writes_traced"]
    if arm in ("A_discrete", "B_ttt_chunked"):
        assert cell["n_episodes_logged"] == run_nh03.N_INTERVALS
    if arm in ("A_continuous", "B_ttt_continuous"):
        assert cell["n_episodes_logged"] == run_nh03.WRITE_BUDGET
    if arm.startswith("B_"):
        assert cell["writes_by_op"] == {"replay": 20, "downscale": 4}
        assert cell["fast_weight_fro_norm"] > 0.0
    else:
        assert cell["fast_weight_fro_norm"] == 0.0


def test_r1_same_seed_same_cell(tasks: list[SplitFMNISTTask]) -> None:
    a = run_nh03.run_cell("B_ttt_continuous", 5, tasks, ttt_lr=0.1)
    b = run_nh03.run_cell("B_ttt_continuous", 5, tasks, ttt_lr=0.1)
    for k in ("acc_task1_initial", "acc_task1_final", "retention",
              "fast_weight_fro_norm"):
        assert a[k] == b[k]


def test_task0_identical_across_arms(tasks: list[SplitFMNISTTask]) -> None:
    """Consolidation starts after task 0 : acc_initial is arm-free."""
    accs = {run_nh03.run_cell(a, 7, tasks, ttt_lr=0.1)["acc_task1_initial"]
            for a in run_nh03.ALL_ARMS}
    assert len(accs) == 1


def test_positions_discrete_vs_continuous() -> None:
    assert run_nh03._positions(600, False) == [0] * 6
    pos = run_nh03._positions(600, True)
    assert pos == [0, 100, 200, 300, 400, 500]


def test_classify_rules() -> None:
    assert classify(4.0, (1.0, 7.0))["verdict"] == "continuous_gt_bounded"
    assert classify(0.5, (-2.0, 2.5))["verdict"] == "equivalent_at_this_scale"
    assert classify(0.5, (-3.5, 2.5))["verdict"] == "inconclusive"
    r = classify(-5.0, (-8.0, -2.0))
    assert r["verdict"] == "inconclusive"
    assert r["direction"] == "bounded_gt_continuous_descriptive"


def test_bootstrap_ci_deterministic_and_contains_mean() -> None:
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    assert bootstrap_ci(d) == bootstrap_ci(d)
    lo, hi = bootstrap_ci(d)
    assert lo <= d.mean() <= hi


def test_contrast_pairs_by_seed_and_excludes() -> None:
    cells = [
        {"arm": "X", "seed": s, "retention": 0.6 + 0.01 * s,
         "excluded_underperforming_baseline": s == 2}
        for s in range(4)
    ] + [
        {"arm": "Y", "seed": s, "retention": 0.5,
         "excluded_underperforming_baseline": False}
        for s in range(4)
    ]
    c = contrast(cells, "X", "Y")
    assert c["n"] == 3 and c["seeds"] == [0, 1, 3]
    assert c["mean_points"] == pytest.approx(100 * (0.1 + 0.01 * 4 / 3))


# ----- schedule ablation (pre-reg 2026-10-27) -----


def test_schedule_positions_windows() -> None:
    sp = run_nh03.schedule_positions
    assert sp(600, "early_block") == [0] * 6
    assert sp(600, "late_block") == [600] * 6
    assert sp(600, "uniform") == [0, 100, 200, 300, 400, 500]
    assert sp(600, "early_spread") == [0, 33, 66, 100, 133, 166]
    assert sp(600, "late_spread") == [433, 466, 500, 533, 566, 600]
    with pytest.raises(ValueError):
        sp(600, "bogus")


@pytest.mark.parametrize("arm", run_nh03.ABLATION_ARMS)
def test_ablation_arms_spend_exact_budget(
    arm: str, tasks: list[SplitFMNISTTask]
) -> None:
    cell = run_nh03.run_cell(arm, 4, tasks, ttt_lr=0.1)
    assert cell["writes_total"] == run_nh03.WRITE_BUDGET
    assert cell["dr0_all_writes_traced"]
    block = arm.endswith("_block")
    expected_des = run_nh03.N_INTERVALS if block else run_nh03.WRITE_BUDGET
    assert cell["n_episodes_logged"] == expected_des


@pytest.mark.parametrize(
    ("old", "new"),
    [("A_discrete", "slow_early_block"), ("A_continuous", "slow_uniform"),
     ("B_ttt_continuous", "ttt_uniform"), ("B_ttt_chunked", "ttt_early_block")],
)
def test_legacy_arms_equal_ablation_arms(
    old: str, new: str, tasks: list[SplitFMNISTTask]
) -> None:
    """R1 : the refactor leaves the NH-03 arms bit-identical."""
    a = run_nh03.run_cell(old, 6, tasks, ttt_lr=0.1)
    b = run_nh03.run_cell(new, 6, tasks, ttt_lr=0.1)
    assert a["retention"] == b["retention"]
    assert a["fast_weight_fro_norm"] == b["fast_weight_fro_norm"]


def test_ablation_decision_rules() -> None:
    from experiments.nh03_ttt_vs_episodes.ablation_analysis import decide

    def e(m: float, lo: float, hi: float) -> dict[str, object]:
        return {"mean": m, "ci95": [lo, hi]}

    assert decide(e(5, 2, 8), e(0, -1, 1), e(5, 2, 8)) == "recency"
    assert decide(e(0, -1, 1), e(5, 2, 8), e(-5, -8, -2)) == "spreading"
    assert decide(e(5, 2, 8), e(5, 2, 8), e(0, -2, 2)) == "both"
    assert decide(e(2, 1, 3), e(1, 0.5, 2), e(1, -1, 3)) == "inconclusive"


def test_ablation_analysis_on_synthetic_cells() -> None:
    from experiments.nh03_ttt_vs_episodes.ablation_analysis import (
        analyse_ablation,
    )

    rng = np.random.default_rng(0)
    base = {"early_block": 0.60, "early_spread": 0.61, "late_block": 0.70,
            "late_spread": 0.71, "uniform": 0.66}
    cells = [
        {"arm": f"{fam}_{k}", "seed": s,
         "retention": v + 0.005 * rng.standard_normal(),
         "excluded_underperforming_baseline": False}
        for fam in ("ttt", "slow") for k, v in base.items() for s in range(10)
    ]
    res = analyse_ablation(cells)
    assert res["primary"]["verdict"] == "recency"
    assert res["primary"]["recency_R"]["mean"] == pytest.approx(10, abs=1)
