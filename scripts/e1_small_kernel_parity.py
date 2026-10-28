#!/usr/bin/env python3
"""E1-small S6 check: fla + causal-conv1d kernels vs the transformers torch
fallback for Gated DeltaNet, on the pinned Qwen3.5-4B (one CUDA GPU).

**Gate** : ``docs/osf-prereg-e1-small-2026-10-25.md`` section 9 (S6). Run
with ``HL_STUDY=e1-small`` after the GPU is free; never part of a cell.

The same short inputs go through the model loaded with the kernels
(registered) and then with the torch fallback (``force_torch_linear_attention``),
one model at a time, bf16, raw pinned tokenizer:

- per-token log-probabilities of the full sequences (pilot MMLU prompts with
  each choice) and the ``_score_choice`` scores and argmax of every item;
- the replay loss and the full gradient of one SGD step on two texts.

``TOL`` is fixed before the run. The verdict is PASS only if every metric
is within it.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Tolerances (bf16, two summation orders of the same mathematics), fixed
# before the first run on 2026-10-25.
TOL = {"logprob_max_abs": 0.25, "logprob_mean_abs": 0.02, "choice_argmax_agree_min": 0.95,
       "choice_score_max_abs": 0.05, "loss_rel": 0.01, "grad_cos_min": 0.99,
       "grad_norm_ratio_dev": 0.02, "grad_rel_err": 0.15}


def verdict(m: Mapping[str, float], tol: Mapping[str, float] = TOL) -> dict[str, Any]:
    """Compare the metrics with the tolerances (pure, tested without torch)."""
    checks = {
        "logprob_max_abs": m["logprob_max_abs"] <= tol["logprob_max_abs"],
        "logprob_mean_abs": m["logprob_mean_abs"] <= tol["logprob_mean_abs"],
        "choice_argmax_agree": m["choice_argmax_agree"] >= tol["choice_argmax_agree_min"],
        "choice_score_max_abs": m["choice_score_max_abs"] <= tol["choice_score_max_abs"],
        "loss_rel": m["loss_rel"] <= tol["loss_rel"],
        "grad_cos": m["grad_cos"] >= tol["grad_cos_min"],
        "grad_norm_ratio": abs(m["grad_norm_ratio"] - 1.0) <= tol["grad_norm_ratio_dev"],
        "grad_rel_err": m["grad_rel_err"] <= tol["grad_rel_err"],
        "grad_keys": m["grad_keys_equal"] == 1.0,
    }
    ok = all(checks.values()) and all(math.isfinite(float(v)) for v in m.values())
    return {"checks": checks, "verdict": "PASS" if ok else "FAIL"}


def _items(n: int) -> list[dict[str, Any]]:
    from scripts import e1_corpus_io as cio

    fix = ROOT / "tests" / "fixtures"
    benches, _ = cio.load_pilot(fix, fix / "e1_partition_2026-10.json", "confirmatory")
    return benches["mmlu"][:n]


def _measure(h: Any, model: Any, tok: Any, items: list[dict[str, Any]],
             texts: list[str]) -> dict[str, Any]:
    import torch

    dev = h._input_device(model)
    lp, scores = [], []
    with torch.no_grad():
        for it in items:
            scores.append([h._score_choice(model, tok, it["prompt"], c) for c in it["choices"]])
            for c in it["choices"]:
                ids = h._ids(tok, it["prompt"] + c).to(dev)
                out = torch.log_softmax(model(ids).logits[0, :-1].float(), dim=-1)
                lp.append(out[range(ids.shape[1] - 1), ids[0, 1:]].cpu())
    model.train()
    model.zero_grad(set_to_none=True)
    losses = []
    for t in texts:
        x = h._ids(tok, t).to(dev)
        out = model(x, labels=x)
        (out.loss / len(texts)).backward()
        losses.append(float(out.loss.detach()))
    grads = {n: p.grad.detach().to("cpu", copy=True) for n, p in model.named_parameters()
             if p.grad is not None}
    model.zero_grad(set_to_none=True)
    model.eval()
    return {"lp": lp, "scores": scores, "loss": sum(losses) / len(losses), "grads": grads,
            "peak": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}


def compare(a: Mapping[str, Any], b: Mapping[str, Any]) -> dict[str, float]:
    import torch

    d = torch.cat([(x - y).abs() for x, y in zip(a["lp"], b["lp"], strict=True)])
    sa, sb = torch.tensor(a["scores"]), torch.tensor(b["scores"])
    same_keys = set(a["grads"]) == set(b["grads"])
    dot = na = nb = nd = 0.0
    for k in set(a["grads"]) & set(b["grads"]):
        g, o = a["grads"][k].float(), b["grads"][k].float()
        dot += float((g * o).sum())
        na += float((g * g).sum())
        nb += float((o * o).sum())
        nd += float(((g - o) ** 2).sum())
    ok = na > 0 and nb > 0
    return {"logprob_max_abs": float(d.max()), "logprob_mean_abs": float(d.mean()),
            "choice_score_max_abs": float((sa - sb).abs().max()),
            "choice_argmax_agree": float((sa.argmax(1) == sb.argmax(1)).float().mean()),
            "loss_rel": abs(a["loss"] - b["loss"]) / abs(b["loss"]),
            "grad_cos": dot / math.sqrt(na * nb) if ok else float("nan"),
            "grad_norm_ratio": math.sqrt(na / nb) if ok else float("nan"),
            "grad_rel_err": math.sqrt(nd / nb) if ok else float("nan"),
            "grad_keys_equal": 1.0 if same_keys else 0.0}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--n-items", type=int, default=25)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import gc

    import torch

    from scripts import e1_harness as hn
    from scripts import pilot_cycle3_cuda as h

    if hn.STUDY != "e1-small" or not torch.cuda.is_available():
        raise SystemExit("run with HL_STUDY=e1-small on the CUDA node")
    items = _items(args.n_items)
    texts = [items[0]["prompt"] + items[0]["choices"][items[0]["answer"]],
             items[1]["prompt"] + items[1]["choices"][items[1]["answer"]]]
    res: dict[str, Any] = {}
    for impl in ("fla+causal_conv1d", "torch"):
        if impl == "torch":
            h.force_torch_linear_attention()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model, tok = h._load_model(hn.MODEL_REPO, hn.MODEL_REVISION)
        got = h.linear_attn_impl(model)
        if got != impl:
            raise SystemExit(f"expected {impl}, loaded {got}")
        res[impl] = _measure(h, model, tok, items, texts)
        res[impl]["wall_s"] = round(time.time() - t0, 1)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    m = compare(res["fla+causal_conv1d"], res["torch"])
    v = verdict(m)
    doc = {"metrics": m, "tol": TOL, **v, "n_items": len(items), "n_texts": len(texts),
           "n_sequences": len(res["torch"]["lp"]),
           "loss": {k: r["loss"] for k, r in res.items()},
           "wall_s": {k: r["wall_s"] for k, r in res.items()},
           "peak_bytes": {k: r["peak"] for k, r in res.items()},
           "model": hn.MODEL_REPO, "revision": hn.MODEL_REVISION,
           "versions": {d: h._dist_version(d) for d in ("torch", "transformers", "tokenizers",
                                                        "fla-core", "causal-conv1d")},
           "gpu": torch.cuda.get_device_name(0)}
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"verdict": v["verdict"], **m}, sort_keys=True))
    return 0 if v["verdict"] == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
