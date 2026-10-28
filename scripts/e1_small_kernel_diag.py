#!/usr/bin/env python3
"""E1-small S6 diagnostic (PI decision 2026-10-26 22:09): which bf16
linear-attention implementation is closer to an fp32 reference?

**Gate** : ``docs/osf-prereg-e1-small-2026-10-25.md`` section 9.2. Run with
``HL_STUDY=e1-small`` on the CUDA node, after the failed parity check; forward
passes only, never part of a cell.

Inputs are those of ``scripts/e1_small_kernel_parity.py``: 25 pilot MMLU items
(100 prompt + choice sequences, with their ``_score_choice`` scores) and the
same 2 replay texts (their per-token log-probabilities join the token set).
Three loads of the pinned 4B, one at a time:

1. bf16 with fla + causal-conv1d (the kernels);
2. fp32 with the transformers torch fallback (the reference);
3. bf16 with the torch fallback.

For each bf16 implementation, against fp32: max and mean |Δ log p| per token,
max |Δ choice score| and the fraction of items whose argmax equals fp32's.

**Decision rule (fixed before the run).** The registered instrument is the
kernels only if their mean |Δ log p| to fp32 is <= the fallback's AND their
argmax agreement with fp32 is >= the fallback's; otherwise the torch fallback
is registered and fla / causal-conv1d are dropped. The original parity bounds
and their failure are unchanged.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections.abc import Mapping
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def decide(kern: Mapping[str, float], fall: Mapping[str, float]) -> str:
    """The rule above (pure, tested without torch)."""
    ok = (kern["logprob_mean_abs"] <= fall["logprob_mean_abs"]
          and kern["argmax_agree"] >= fall["argmax_agree"])
    return "fla+causal_conv1d" if ok else "torch"


def _forward(h: Any, model: Any, tok: Any, items: list[dict[str, Any]],
             texts: list[str]) -> dict[str, Any]:
    import torch

    dev = h._input_device(model)
    lp, scores = [], []
    with torch.no_grad():
        for it in items:
            scores.append([h._score_choice(model, tok, it["prompt"], c) for c in it["choices"]])
        seqs = [it["prompt"] + c for it in items for c in it["choices"]] + texts
        for s in seqs:
            ids = h._ids(tok, s).to(dev)
            out = torch.log_softmax(model(ids).logits[0, :-1].float(), dim=-1)
            lp.append(out[range(ids.shape[1] - 1), ids[0, 1:]].double().cpu())
    return {"lp": lp, "scores": scores,
            "peak": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0}


def distance(x: Mapping[str, Any], ref: Mapping[str, Any]) -> dict[str, float]:
    import torch

    d = torch.cat([(a - b).abs() for a, b in zip(x["lp"], ref["lp"], strict=True)])
    sx, sr = torch.tensor(x["scores"]), torch.tensor(ref["scores"])
    return {"logprob_max_abs": float(d.max()), "logprob_mean_abs": float(d.mean()),
            "choice_score_max_abs": float((sx - sr).abs().max()),
            "argmax_agree": float((sx.argmax(1) == sr.argmax(1)).double().mean())}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args(argv)
    import gc

    import torch

    from scripts import e1_harness as hn
    from scripts import e1_small_kernel_parity as kp
    from scripts import pilot_cycle3_cuda as h

    if hn.STUDY != "e1-small" or not torch.cuda.is_available():
        raise SystemExit("run with HL_STUDY=e1-small on the CUDA node")
    items = kp._items(25)
    texts = [items[0]["prompt"] + items[0]["choices"][items[0]["answer"]],
             items[1]["prompt"] + items[1]["choices"][items[1]["answer"]]]
    runs = (("bf16_kernels", torch.bfloat16, "fla+causal_conv1d"),
            ("fp32_fallback", torch.float32, "torch"), ("bf16_fallback", torch.bfloat16, "torch"))
    res: dict[str, Any] = {}
    for name, dtype, impl in runs:
        if impl == "torch":
            h.force_torch_linear_attention()
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        model, tok = h._load_model(hn.MODEL_REPO, hn.MODEL_REVISION, dtype_override=dtype)
        if h.linear_attn_impl(model) != impl or next(model.parameters()).dtype != dtype:
            raise SystemExit(f"{name}: loaded {h.linear_attn_impl(model)} / "
                             f"{next(model.parameters()).dtype}")
        res[name] = _forward(h, model, tok, items, texts)
        res[name]["wall_s"] = round(time.time() - t0, 1)
        del model
        gc.collect()
        torch.cuda.empty_cache()
    dist = {k: distance(res[k], res["fp32_fallback"]) for k in ("bf16_kernels", "bf16_fallback")}
    dist["kernels_vs_fallback"] = distance(res["bf16_kernels"], res["bf16_fallback"])
    instrument = decide(dist["bf16_kernels"], dist["bf16_fallback"])
    doc = {"distance_to_fp32": dist, "registered_linear_attn": instrument,
           "n_items": len(items), "n_sequences": len(res["fp32_fallback"]["lp"]),
           "n_tokens": int(sum(x.numel() for x in res["fp32_fallback"]["lp"])),
           "wall_s": {k: r["wall_s"] for k, r in res.items()},
           "peak_bytes": {k: r["peak"] for k, r in res.items()},
           "model": hn.MODEL_REPO, "revision": hn.MODEL_REVISION,
           "versions": {d: h._dist_version(d) for d in ("torch", "transformers", "tokenizers",
                                                        "fla-core", "causal-conv1d")},
           "gpu": torch.cuda.get_device_name(0)}
    args.out.write_text(json.dumps(doc, indent=1, sort_keys=True) + "\n")
    print(json.dumps({"registered_linear_attn": instrument, **dist}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
