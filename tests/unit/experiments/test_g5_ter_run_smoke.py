"""Smoke test for the G5-ter pilot driver."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from experiments.g5_ter_spiking_cnn import run_g5_ter


def _synthetic_cifar_tasks() -> list[dict[str, np.ndarray]]:
    rng = np.random.default_rng(0)
    out: list[dict[str, np.ndarray]] = []
    for _ in range(5):
        n = 8
        x = rng.standard_normal((n, 32, 32, 3)).astype(np.float32)
        y = rng.integers(0, 2, size=n, dtype=np.int64)
        out.append(
            {
                "x_train": x.reshape(n, -1),
                "x_train_nhwc": x,
                "y_train": y,
                "x_test": x.reshape(n, -1),
                "x_test_nhwc": x,
                "y_test": y,
            }
        )
    return out


def test_run_pilot_smoke(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        "experiments.g5_ter_spiking_cnn.run_g5_ter."
        "load_split_cifar10_5tasks_auto",
        lambda _d: _synthetic_cifar_tasks(),
    )
    out_json = tmp_path / "smoke.json"
    out_md = tmp_path / "smoke.md"
    run_g5_ter.run_pilot(
        data_dir=tmp_path,
        seeds=(0,),
        out_json=out_json,
        out_md=out_md,
        registry_db=tmp_path / "reg.sqlite",
        epochs=1,
        batch_size=4,
        lr=0.01,
        n_steps=2,
    )
    body = json.loads(out_json.read_text())
    assert out_md.exists()
    assert len(body["cells"]) == len(run_g5_ter.ARMS)
    assert "h8a_spiking_cnn" in body["verdict"]
    assert "retention_by_arm" in body["verdict"]
