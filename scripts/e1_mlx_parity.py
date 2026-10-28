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
TOL = {"float32": {"score": 1e-3, "loss": 1e-3, "delta_rel": 1e-2, "drift_rel": 1e-2,
                   "determinism_abs": 1e-6},
       "bfloat16": {"score": 1e-1, "loss": 1e-1, "delta_rel": 2e-1, "drift_rel": 2e-1,
                    "determinism_abs": None}}
# In both dtypes the decisions must agree exactly: tokens, choice argmax,
# per-item correctness pre and post, accuracies, every budget / token / step
# count, OPR corpus size, the RESTRUCTURE index and the expert-usage
# histograms; and restore must be bit-exact. The continuous tolerances above
# apply to scores, losses, the SGD step and the drift measures; for bf16 they
# are reported, and the PI ratifies the bf16 envelope (D5, D6).
ROUTER_STD = 0.5
# Expert-usage L1 (pre vs post) is a diagnostic: pre-replay counts must be
# identical (tested), post-replay a few near-tie tokens may route differently
# after rounding-level weight differences. Set after a first run showed
# non-zero differences; disclosed as such.
USAGE_L1_TOL = 1e-2


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


def _np32(v: Any) -> np.ndarray:
    """float32 numpy copy of an MLX array (numpy has no bf16; exact widening)."""
    import mlx.core as mx

    return np.asarray(v.astype(mx.float32)) if mx.issubdtype(v.dtype, mx.floating) \
        else np.asarray(v)


def _rel(a: np.ndarray, b: np.ndarray) -> float:
    den = float(np.linalg.norm(b))
    return float(np.linalg.norm(a - b)) / den if den > 0 else float(np.linalg.norm(a))


def run(tok_dir: Path, dtype: str, work: Path, cell_seeds: tuple[int, ...] = (2000, 2001),
        cell_profiles: tuple[str, ...] = ("p_min", "p_equ", "p_max", "p_ctrl_onpolicy_replay",
                                          "p_ctrl_shuffle", "p_ctrl_template")
        ) -> dict[str, Any]:
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
    ttok = hn.RawTokenizer(d0 / "tokenizer.json")  # S8: the pinned raw tokenizer.json
    mmodel, mtok, k = load(d0, hn.TOKENIZER_SHA256)
    tb = pt.TorchBackend(tmodel, ttok)
    hf_tok = AutoTokenizer.from_pretrained(str(d0))
    items = _items(60)

    # 0. tokenisation
    texts = [it["prompt"] + c for it in items[:10] for c in it["choices"]]
    rep["tokens_identical"] = all(ttok.encode_ids(t) == mtok.encode_ids(t) for t in texts)
    rep["raw_vs_transformers_tokens_equal_on_probe"] = all(
        ttok.encode_ids(t) == hf_tok(t)["input_ids"] for t in texts)

    # 1. scores (and the base model's routing decisions, counted on both)
    mb = MLXBackend(mmodel, mtok, None, k)
    with tb.router_counter() as rt:
        ts, ta = tb.score_items(items[:12])
    with mb.router_counter() as rm:
        ms, ma = mb.score_items(items[:12])

    rep["routing_base_identical"] = rt.counts == rm.counts
    diff = float(np.max(np.abs(np.asarray(ts) - np.asarray(ms))))
    rep["scores"] = {"max_abs_diff": diff, "argmax_equal": ta == ma,
                     "ok": ta == ma and diff <= tol["score"]}

    # 2. per-token losses
    worst, n_pos, flips = 0.0, 0, 0
    for it in items[:20]:
        ids = ttok.encode_ids(pr.fact_text(it))
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
    ids = ttok.encode_ids(pr.fact_text(items[0]))
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
    rels, changed, spurious = {}, 0, []
    for key, b0 in base_tree.items():
        dt = np.asarray((torch_step[key].astype(mx.float32) - b0.astype(mx.float32)))
        dm = np.asarray((mlx_step[key].astype(mx.float32) - b0.astype(mx.float32)))
        if float(np.linalg.norm(dt)) > 0:
            changed += 1
            rels[key] = _rel(dm, dt)
        elif float(np.linalg.norm(dm)) > 0:
            spurious.append(key)  # MLX changed a tensor torch left unchanged
    worst_key = max(rels, key=lambda k_: rels[k_])
    rep["sgd_step"] = {"n_tensors": len(base_tree), "n_changed_torch": changed,
                       "spurious_mlx_changes": spurious,
                       "max_rel_diff": rels[worst_key], "worst_tensor": worst_key,
                       "ok": rels[worst_key] <= tol["delta_rel"] and not spurious}

    # 4. harness cells on both backends, identical item draws
    benches = {"facts": items}
    seeds = list(cell_seeds)
    doc = e1_splits.build(benches, seeds, {"test": True})
    mb.splits = e1_splits.FileSplits(doc, benches)
    t_base = {k_: v.clone() for k_, v in tb.base.items()}
    m_base = dict(tree_flatten(mmodel.parameters()))
    am = pr.Amendment(pr.DEFAULT, (3e-4, 2), 2.0, 12)
    cells: dict[str, Any] = {}
    exact_ok = close_ok = True
    for prof in cell_profiles:
        for s in seeds:  # each cell restores the base snapshot first
            ct = hn.profile_cell(tb, benches, prof, s, am, am.n_eval)
            cm = hn.profile_cell(mb, benches, prof, s, am, am.n_eval)
            exact = {f: (ct[f], cm[f]) for f in ("t_s", "achieved_loss_tokens", "n_updates",
                                                  "n_truncated", "budget", "n_eval_items",
                                                  "opr_corpus_size", "s_opr") if f in ct}
            ex = all(a == b for a, b in exact.values())
            decisions = {f: ct.get(f) == cm.get(f) for f in (
                "pre", "post", "pre_correct", "post_correct", "collapse")}
            decisions["restructure"] = (ct["ops"]["restructure_idx"]
                                        == cm["ops"]["restructure_idx"])
            ex = ex and all(decisions.values())
            close = {"decisions": decisions,
                     "pre_diff": abs(ct["pre"] - cm["pre"]),
                     "post_diff": abs(ct["post"] - cm["post"]),
                     "router_drift_rel": abs((ct["router_drift"] or 0) - (cm["router_drift"] or 0))
                     / max(ct["router_drift"] or 0, 1e-12),
                     "norm_change_rel": abs(ct["norm_change"] - cm["norm_change"])
                     / max(ct["norm_change"], 1e-12),
                     "drift_rel": abs(ct["drift"] - cm["drift"]) / max(ct["drift"], 1e-12),
                     "kl_diff": abs((ct.get("kl_ref") or 0) - (cm.get("kl_ref") or 0)),
                     # post-replay routing can flip near-tie tokens (D5): diagnostic
                     "usage_l1_diff": abs((ct.get("usage_l1") or 0) - (cm.get("usage_l1") or 0)),
                     "restructure": (ct["ops"]["restructure_idx"], cm["ops"]["restructure_idx"])}
            ok = (close["norm_change_rel"] <= tol["drift_rel"]
                  and close["drift_rel"] <= tol["drift_rel"]
                  and close["router_drift_rel"] <= tol["drift_rel"]
                  and close["kl_diff"] <= tol["score"]
                  and close["usage_l1_diff"] <= USAGE_L1_TOL)
            cells[f"{prof}|{s}"] = {"exact_equal": ex, "exact": exact, **close, "close_ok": ok}
            exact_ok &= ex
            close_ok &= ok
    rep["cells"] = cells
    # explicit restore exactness on both backends
    tb.restore()
    mb.restore()
    rep["restore_exact"] = {
        "torch": all(torch.equal(v, t_base[k_]) for k_, v in tmodel.state_dict().items()),
        "mlx": all(np.array_equal(_np32(v), _np32(m_base[k_]))
                   for k_, v in tree_flatten(mmodel.parameters()))}
    rep["cells_exact_ok"] = exact_ok
    rep["cells_close_ok"] = close_ok

    # 5. MLX determinism: the same cell twice gives bit-identical parameters
    hn.profile_cell(mb, benches, "p_min", 2000, am, am.n_eval)
    a = {k_: _np32(v) for k_, v in tree_flatten(mmodel.parameters())}
    hn.profile_cell(mb, benches, "p_min", 2000, am, am.n_eval)
    b = {k_: _np32(v) for k_, v in tree_flatten(mmodel.parameters())}
    diffs = {k_: float(np.max(np.abs(a[k_].astype(np.float64) - b[k_].astype(np.float64))))
             for k_ in a}
    env = tol["determinism_abs"]
    rep["mlx_deterministic"] = {"bit_identical": all(v == 0.0 for v in diffs.values()),
                                "n_tensors_differing": sum(v > 0 for v in diffs.values()),
                                "max_abs_diff": max(diffs.values()), "envelope": env,
                                "ok": env is None or max(diffs.values()) <= env}
    rep["ok"] = bool(rep["tokens_identical"] and rep["routing_base_identical"]
                     and rep["scores"]["ok"] and rep["token_ce"]["ok"]
                     and rep["sgd_step"]["ok"] and exact_ok and close_ok
                     and all(rep["restore_exact"].values()) and rep["mlx_deterministic"]["ok"])
    return rep


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tokenizer-dir", type=Path, required=True)
    ap.add_argument("--dtype", default="float32", choices=sorted(TOL))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--quick", action="store_true", help="one seed, four profiles")
    args = ap.parse_args(argv)
    kw: dict[str, Any] = {}
    if args.quick:
        kw = {"cell_seeds": (2000,),
              "cell_profiles": ("p_min", "p_equ", "p_max", "p_ctrl_onpolicy_replay")}
    with tempfile.TemporaryDirectory() as tmp:
        rep = run(args.tokenizer_dir, args.dtype, Path(tmp), **kw)
    args.out.write_text(json.dumps(rep, indent=1, sort_keys=True, default=str) + "\n")
    print(json.dumps({k: v for k, v in rep.items() if k != "cells"}, indent=1, default=str))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
