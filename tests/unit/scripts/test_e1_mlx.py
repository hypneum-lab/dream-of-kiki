"""E1 MLX port: split replay, weight pins, RMSNorm parameterisation, parity.

Skipped where torch / mlx are not installed (CI). The full PyTorch vs MLX
parity run needs the pinned Qwen tokenizer files: set
``E1_PARITY_TOKENIZER_DIR``.
"""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import pytest

torch = pytest.importorskip("torch")

from scripts import e1_splits
from scripts import pilot_cycle3_cuda as pt


def _bench(n: int = 70) -> dict[str, list[dict[str, Any]]]:
    return {"facts": [{"prompt": f"Q{i} has some words in it?\nAnswer:",
                       "question_prompt": f"Q{i}?\nAnswer:",
                       "choices": [f" a{i}", " b", " c c", " d d d"], "answer": i % 4,
                       "source": "mmlu_cf", "item_sha256": f"{i:064x}"} for i in range(n)]}


def test_file_splits_replay_torch_draws_exactly() -> None:
    b = _bench()
    seeds = [0, 1, 140, 1000]
    fs = e1_splits.FileSplits(e1_splits.build(b, seeds, {}), b)
    for s in seeds:
        for seed in (s, s + pt.NOISE_SEED_OFFSET):
            for n_eval in (0, 12):
                assert fs.cell_split(b, seed, 40, n_eval) == pt._cell_split(b, seed, 40, n_eval)
        _, rep = pt._cell_split(b, s, 40, 0)
        for it in rep:
            text = it["prompt"] + it["choices"][it["answer"]]
            assert fs.shuffle(text, s) == pt._shuffle_text(text, s)


def test_file_splits_refuse_other_items_and_seeds() -> None:
    b = _bench()
    doc = e1_splits.build(b, [5], {})
    other = {"facts": list(reversed(b["facts"]))}
    with pytest.raises(SystemExit, match="other items"):
        e1_splits.FileSplits(doc, other)
    fs = e1_splits.FileSplits(doc, b)
    with pytest.raises(SystemExit, match="not in the split file"):
        fs.cell_split(b, 6, 40, 0)


mx = pytest.importorskip("mlx.core")

from scripts import e1_mlx


def test_verify_weights(tmp_path: Path) -> None:
    (tmp_path / "w").mkdir()
    a, b = tmp_path / "w" / "a.safetensors", tmp_path / "w" / "config.json"
    a.write_bytes(b"x" * 10)
    b.write_bytes(b"{}")
    pins = {"repo": "r", "revision": "v", "files": {
        "a.safetensors": {"size": 10, "sha256": hashlib.sha256(b"x" * 10).hexdigest()},
        "config.json": {"size": 2, "git_blob": hashlib.sha1(b"blob 2\x00{}").hexdigest()}}}
    p = tmp_path / "pins.json"
    p.write_text(json.dumps(pins))
    assert e1_mlx.verify_weights(tmp_path / "w", p)["n_files"] == 2
    a.write_bytes(b"y" * 10)
    with pytest.raises(SystemExit, match="differ"):
        e1_mlx.verify_weights(tmp_path / "w", p)
    (tmp_path / "w" / "extra").write_text("")
    with pytest.raises(SystemExit, match="exactly"):
        e1_mlx.verify_weights(tmp_path / "w", p)


def test_zero_centered_rmsnorm_matches_hf_formula() -> None:
    import numpy as np

    w = mx.array([0.05, -0.1, 0.2, 0.0], dtype=mx.bfloat16)
    x = mx.array([[1.0, 2.0, -1.0, 0.5]], dtype=mx.bfloat16)
    y = e1_mlx.ZeroCenteredRMSNorm(w, 1e-6)(x)
    xf = np.asarray(x.astype(mx.float32))
    ref = xf / np.sqrt((xf ** 2).mean(-1, keepdims=True) + 1e-6) * (1 + np.asarray(
        w.astype(mx.float32)))
    assert y.dtype == mx.bfloat16
    assert np.allclose(np.asarray(y.astype(mx.float32)), ref, atol=1e-2)


@pytest.mark.skipif(not os.environ.get("E1_PARITY_TOKENIZER_DIR"),
                    reason="needs the pinned Qwen tokenizer files")
def test_torch_mlx_parity_float32(tmp_path: Path) -> None:
    pytest.importorskip("transformers")
    from scripts import e1_mlx_parity

    rep = e1_mlx_parity.run(Path(os.environ["E1_PARITY_TOKENIZER_DIR"]), "float32", tmp_path)
    assert rep["tokens_identical"] and rep["cells_exact_ok"] and rep["sgd_step"]["ok"]
    assert rep["scores"]["ok"] and rep["token_ce"]["n_positions_diff_gt_1e-2"] == 0
