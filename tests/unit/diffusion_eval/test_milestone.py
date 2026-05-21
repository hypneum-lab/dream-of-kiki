"""M5 milestone aggregation — deterministic bytes, per-profile stats."""

from __future__ import annotations

import json

from harness.diffusion_eval.milestone import aggregate_cells


def _fake_cells() -> list[dict]:
    cells = []
    for profile in ("p_min", "p_equ", "p_max"):
        for seed in range(30):
            for task in range(5):
                cells.append({
                    "profile": profile, "seed": seed, "task_idx": task,
                    "replay_rate": 0.5 + 0.001 * seed,
                    "downscale_norm": 0.9,
                    "restructure_sum": 1.0,
                    "recombine_rate": 0.3,
                    "delta_acc": 0.1,
                    "wall_s": 40.0, "slow_cell": False,
                })
    return cells


def test_aggregate_is_deterministic() -> None:
    cells = _fake_cells()
    a = json.dumps(aggregate_cells(cells), sort_keys=True)
    b = json.dumps(aggregate_cells(cells), sort_keys=True)
    assert a == b


def test_aggregate_has_per_profile_stats() -> None:
    summary = aggregate_cells(_fake_cells())
    assert set(summary["profiles"]) == {"p_min", "p_equ", "p_max"}
    p_min = summary["profiles"]["p_min"]
    assert p_min["n_cells"] == 150
    assert "replay_rate" in p_min["stats"]
    assert "mean" in p_min["stats"]["replay_rate"]
