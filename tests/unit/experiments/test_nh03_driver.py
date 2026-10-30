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
