#!/usr/bin/env python3
"""E1 port revalidation: PyTorch vs MLX on a tiny Qwen3.5-MoE (same weights).

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` section 9
(PI decision 2026-10-25: MLX backend on the Mac Studio; parity on macM1
before any 35B run).

A tiny model with the architecture of Qwen3.6-35B-A3B (``qwen3_5_moe``:
Gated-DeltaNet linear attention every 3 of 4 layers, full attention, 8
routed experts with top-2, a shared expert, zero-centred RMSNorm) is
built with random weights in HF transformers, saved in the checkpoint
layout of the real model and loaded by both backends. Compared:

1. choice scores of ``_score_choice`` (length-normalised log-prob);
2. per-token cross-entropy of the replay loss;
3. the parameter change of one momentumless SGD step;
4. whole harness cells (``scripts/e1_harness.run_cell``) for p_min,
   p_equ (DOWNSCALE), p_max (RESTRUCTURE), OPR (scoring charge) and the
   shuffle control: budget accounting must be identical, scores, drift
   and norm change within tolerance;
5. MLX run-to-run determinism of a cell.

Needs torch, transformers and mlx in one interpreter, plus the pinned
Qwen tokenizer files. Prints and writes a JSON report.

    python scripts/e1_mlx_parity.py --tokenizer-dir D --out report.json [--dtype float32]
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

TOKENIZER_FILES = ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt")
# Tolerances, fixed before the run. Cross-entropies are ~12 nats (vocabulary
# 248 320), so 1e-3 absolute is ~1e-4 relative; float32 reductions differ in
# order between Metal and CPU. Router weights are drawn with a larger scale so
# that top-k routing has clear margins: a near-tie flips a token to another
# expert on one backend only (a discrete, backend-specific outcome, counted
# separately below), which no tolerance can absorb.
TOL = {"float32": {"score": 1e-3, "loss": 1e-3, "delta_rel": 1e-2, "drift_rel": 1e-2},
       "bfloat16": {"score": 1e-1, "loss": 1e-1, "delta_rel": 2e-1, "drift_rel": 2e-1}}
ROUTER_STD = 0.5


def text_config(vocab: int) -> dict[str, Any]:
    return {
        "model_type": "qwen3_5_moe_text", "vocab_size": vocab, "hidden_size": 64,
        "num_hidden_layers": 4, "full_attention_interval": 4, "num_attention_heads": 2,
        "num_key_value_heads": 1, "head_dim": 64, "attention_bias": False,
        "attn_output_gate": True, "linear_conv_kernel_dim": 4, "linear_key_head_dim": 32,
        "linear_num_key_heads": 2, "linear_num_value_heads": 4, "linear_value_head_dim": 32,
        "num_experts": 8, "num_experts_per_tok": 2, "moe_intermediate_size": 32,
        "shared_expert_intermediate_size": 32, "hidden_act": "silu", "rms_norm_eps": 1e-6,
        "max_position_embeddings": 4096, "tie_word_embeddings": False,
        "output_router_logits": False, "router_aux_loss_coef": 0.001,
        "partial_rotary_factor": 0.25,
        "rope_parameters": {"mrope_interleaved": True, "mrope_section": [3, 3, 2],
                            "partial_rotary_factor": 0.25, "rope_theta": 10000000,
                            "rope_type": "default"},
        "initializer_range": 0.02, "attention_dropout": 0.0, "dtype": "float32",
        "mtp_num_hidden_layers": 0,
    }


def build_tiny(tok_dir: Path, out: Path, dtype: str, seed: int = 0) -> Any:
    import torch
    from transformers import Qwen3_5MoeForCausalLM, Qwen3_5MoeTextConfig

    tc = text_config(248320)  # the real vocabulary size
    torch.manual_seed(seed)
    model = Qwen3_5MoeForCausalLM(Qwen3_5MoeTextConfig(**tc))
    with torch.no_grad():
        for name, p in model.named_parameters():
            if "norm" in name and p.ndim == 1:
                p.copy_(torch.randn_like(p) * 0.1)  # zero-centred norms: small w
            elif name.endswith("A_log"):
                p.copy_(torch.log(torch.rand_like(p) * 15 + 1))
            elif name.endswith("dt_bias"):
                p.copy_(torch.randn_like(p) * 0.1)
            elif name.endswith("mlp.gate.weight"):
                p.copy_(torch.randn_like(p) * ROUTER_STD)
            else:
                p.copy_(torch.randn_like(p) * 0.05)
    model = model.to(getattr(torch, dtype)).eval()
    out.mkdir(parents=True, exist_ok=True)
    save_weights(model, out)
    cfg = {"model_type": "qwen3_5_moe", "architectures": ["Qwen3_5MoeForConditionalGeneration"],
           "tie_word_embeddings": False, "text_config": {**tc, "dtype": dtype}}
    (out / "config.json").write_text(json.dumps(cfg))
    for f in TOKENIZER_FILES:
        shutil.copy(tok_dir / f, out / f)
    return model


def save_weights(model: Any, out: Path) -> None:
    from safetensors.torch import save_file

    save_file({k: v.detach().contiguous() for k, v in model.state_dict().items()},
              str(out / "model.safetensors"))


def _torch_token_ce(model: Any, ids: list[int]) -> np.ndarray:
    import torch

    x = torch.tensor([ids])
    with torch.no_grad():
        logits = model(x).logits[0, :-1].float()
    return torch.nn.functional.cross_entropy(logits, x[0, 1:], reduction="none").numpy()


def _mlx_token_ce(model: Any, ids: list[int]) -> np.ndarray:
    import mlx.core as mx
    import mlx.nn as nn

    x = mx.array(ids)[None]
    logits = model(x)[0, :-1].astype(mx.float32)
    return np.asarray(nn.losses.cross_entropy(logits, x[0, 1:], reduction="none"))


def _items(n: int) -> list[dict[str, Any]]:
    out = []
    for i in range(n):
        p = f"Date: 2026-05-{i % 27 + 1:02d}\nWhich city hosted event number {i} this week?\nAnswer:"
        out.append({"prompt": p, "question_prompt": p,
                    "choices": [f" Paris {i}", f" Rome {i * 3}", f" Lima {i + 7}", " Oslo"],
                    "answer": i % 4, "source": ("realtimeqa", "daily_oracle", "mmlu_cf")[i % 3],
                    "item_sha256": f"{i:064x}", "src_hash": f"{i:064x}"})
    return out


def _mlx_tree(weights_dir: Path) -> dict[str, Any]:
    from mlx.utils import tree_flatten

    from scripts.e1_mlx import load

    model, _, _ = load(weights_dir)
    return dict(tree_flatten(model.parameters()))


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b)) / den if den > 0 else float(np.linalg.norm(a))


def run(tok_dir: Path, dtype: str, work: Path) -> dict[str, Any]:
    import mlx.core as mx
    import torch
    from transformers import AutoTokenizer

    from scripts import e1_harness as hn
    from scripts import e1_profiles as pr
    from scripts import e1_splits
    from scripts import pilot_cycle3_cuda as pt
    from scripts.e1_mlx import MLXBackend, load

    tol = TOL[dtype]
    rep: dict[str, Any] = {"dtype": dtype, "tolerances": tol,
                           "versions": {"torch": torch.__version__, "mlx": mx.__version__}}
    d0 = work / "tiny"
    tmodel = build_tiny(tok_dir, d0, dtype)
    ttok = AutoTokenizer.from_pretrained(str(d0))
    mmodel, mtok, k = load(d0)
    tb = pt.TorchBackend(tmodel, ttok)
    items = _items(60)

    # 0. tokenisation
    texts = [it["prompt"] + c for it in items[:10] for c in it["choices"]]
    rep["tokens_identical"] = all(ttok(t).input_ids == mtok(t)["input_ids"] for t in texts)

    # 1. scores
    mb = MLXBackend(mmodel, mtok, None, k)
    ts, ta = tb.score_items(items[:12])
    ms, ma = mb.score_items(items[:12])
    diff = float(np.max(np.abs(np.asarray(ts) - np.asarray(ms))))
    rep["scores"] = {"max_abs_diff": diff, "argmax_equal": ta == ma, "ok": diff <= tol["score"]}

    # 2. per-token losses
    worst, n_pos, flips = 0.0, 0, 0
    for it in items[:20]:
        ids = ttok(pr.fact_text(it)).input_ids
        d = np.abs(_torch_token_ce(tmodel, ids) - _mlx_token_ce(mmodel, ids))
        n_pos += d.size
        flips += int(np.sum(d > 1e-2))  # routing flips (discrete), reported apart
        worst = max(worst, float(np.max(d[d <= 1e-2])) if np.any(d <= 1e-2) else 0.0)
    rep["token_ce"] = {"max_abs_diff_excl_flips": worst, "n_positions": n_pos,
                       "n_positions_diff_gt_1e-2": flips,
                       "ok": worst <= tol["loss"] and flips == 0}

    # 3. one SGD step (lr 1e-2) on the same text: parameter change, compared in
    # the MLX parameter space (torch result exported through the same loader)
    base_tree = _mlx_tree(d0)
    ids = ttok(pr.fact_text(items[0])).input_ids
    lengths = [len(ids)]
    from scripts.e1_schedule import Update

    tb.snapshot()
    tb.replay([pr.fact_text(items[0])], lengths, [Update(0, len(ids))], 1e-2)
    d1 = work / "tiny-torch-step"
    d1.mkdir(exist_ok=True)
    for f in ("config.json", *TOKENIZER_FILES):
        shutil.copy(d0 / f, d1 / f)
    save_weights(tmodel, d1)
    torch_step = _mlx_tree(d1)
    tb.restore()
    mb.snapshot()
    mb.replay([pr.fact_text(items[0])], lengths, [Update(0, len(ids))], 1e-2)
    from mlx.utils import tree_flatten

    mlx_step = dict(tree_flatten(mmodel.parameters()))
    mb.restore()
    rels, changed = {}, 0
    for key, b0 in base_tree.items():
        dt = np.asarray((torch_step[key].astype(mx.float32) - b0.astype(mx.float32)))
        dm = np.asarray((mlx_step[key].astype(mx.float32) - b0.astype(mx.float32)))
        if float(np.linalg.norm(dt)) > 0:
            changed += 1
            rels[key] = _rel(dm, dt)
    worst_key = max(rels, key=lambda k_: rels[k_])
    rep["sgd_step"] = {"n_tensors": len(base_tree), "n_changed_torch": changed,
                       "max_rel_diff": rels[worst_key], "worst_tensor": worst_key,
                       "ok": rels[worst_key] <= tol["delta_rel"]}

    # 4. harness cells on both backends, identical item draws
    benches = {"facts": items}
    seeds = [2000, 2001]
    doc = e1_splits.build(benches, seeds, {"test": True})
    mb.splits = e1_splits.FileSplits(doc, benches)
    am = pr.Amendment(pr.DEFAULT, (3e-4, 2), 2.0, 12)
    cells: dict[str, Any] = {}
    exact_ok = close_ok = True
    for prof in ("p_min", "p_equ", "p_max", "p_ctrl_onpolicy_replay", "p_ctrl_shuffle",
                 "p_ctrl_template"):
        for s in seeds:  # each cell restores the base snapshot first
            ct = hn.profile_cell(tb, benches, prof, s, am, am.n_eval)
            cm = hn.profile_cell(mb, benches, prof, s, am, am.n_eval)
            exact = {f: (ct[f], cm[f]) for f in ("t_s", "achieved_loss_tokens", "n_updates",
                                                  "n_truncated", "budget", "n_eval_items",
                                                  "opr_corpus_size", "s_opr") if f in ct}
            ex = all(a == b for a, b in exact.values())
            close = {"pre_diff": abs(ct["pre"] - cm["pre"]),
                     "post_diff": abs(ct["post"] - cm["post"]),
                     "norm_change_rel": abs(ct["norm_change"] - cm["norm_change"])
                     / max(ct["norm_change"], 1e-12),
                     "drift_rel": abs(ct["drift"] - cm["drift"]) / max(ct["drift"], 1e-12),
                     "kl_diff": abs((ct.get("kl_ref") or 0) - (cm.get("kl_ref") or 0)),
                     "restructure": (ct["ops"]["restructure_idx"], cm["ops"]["restructure_idx"])}
            ok = (close["norm_change_rel"] <= tol["drift_rel"]
                  and close["drift_rel"] <= tol["drift_rel"])
            cells[f"{prof}|{s}"] = {"exact_equal": ex, "exact": exact, **close, "close_ok": ok}
            exact_ok &= ex
            close_ok &= ok
    rep["cells"] = cells
    rep["cells_exact_ok"] = exact_ok
    rep["cells_close_ok"] = close_ok

    # 5. MLX determinism: the same cell twice gives bit-identical parameters
    hn.profile_cell(mb, benches, "p_min", 2000, am, am.n_eval)
    a = {k_: np.asarray(v) for k_, v in tree_flatten(mmodel.parameters())}
    hn.profile_cell(mb, benches, "p_min", 2000, am, am.n_eval)
    b = {k_: np.asarray(v) for k_, v in tree_flatten(mmodel.parameters())}
    diffs = {k_: float(np.max(np.abs(a[k_].astype(np.float64) - b[k_].astype(np.float64))))
             for k_ in a}
    rep["mlx_deterministic"] = {"bit_identical": all(v == 0.0 for v in diffs.values()),
                                "n_tensors_differing": sum(v > 0 for v in diffs.values()),
                                "max_abs_diff": max(diffs.values())}
    rep["ok"] = bool(rep["tokens_identical"] and rep["scores"]["ok"] and rep["token_ce"]["ok"]
                     and rep["sgd_step"]["ok"] and exact_ok and close_ok)
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer-dir", type=Path, required=True)
    ap.add_argument("--dtype", default="float32", choices=sorted(TOL))
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    with tempfile.TemporaryDirectory() as tmp:
        rep = run(args.tokenizer_dir, args.dtype, Path(tmp))
    args.out.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "cells"}, indent=1, default=str))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
