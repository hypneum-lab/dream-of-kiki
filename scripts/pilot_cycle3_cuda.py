#!/usr/bin/env python3
"""E1 dream-ablation — CUDA/PyTorch port (first vertical slice).

Mirrors the MLX ``pilot_cycle3_real.py`` cell logic on NVIDIA GPUs so the full
control suite can run on sovereign cloud (OVH H200 / DataCrunch / Scaleway).

This first slice implements the cost-critical pattern and the two profiles that
matter most for the scientific defence:

  * **load-once + restore** — the 35B is loaded ONCE; a CPU copy of the baseline
    weights is restored before every cell (a few-second GPU copy instead of a
    30-60 s reload from disk → ~halves the per-cell time vs the MLX harness).
  * ``p_min`` — DREAM: SGD replay of experiences (the core effect).
  * ``p_noise`` — NOISE-FLOOR control: no dream; re-eval on a different sample
    draw → delta = test-retest variance the dream must beat.

Downscale / recombine / restructure (p_equ / p_max) land in the next slice.

R1: results are a DIFFERENT backend from the MLX runs — never mix the numbers.

  HL_MODEL / --scale  HF id or local path (default a tiny model for a CPU/GPU smoke)
  HL_PROFILES         comma list (default "p_min,p_noise")
  --n-seeds N         seeds 0..N-1
  --smoke             1 cell per profile, tiny fixture
  HL_LR / HL_INNER    replay LR / inner SGD steps per record
  output: pilot-cycle3-cuda-<slug>.cells.jsonl (resumable) + final json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import time
from pathlib import Path

import torch

# --- real benchmark fixtures (dream-of-kiki tests/fixtures), backend-agnostic
#     JSONL reused as-is; only the scoring is ported to torch. ----------------
_FIX = Path(__file__).resolve().parents[1] / "tests" / "fixtures"


def _load_mmlu(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            items.append({"prompt": r["question"].strip() + "\nAnswer:",
                          "choices": [" " + c for c in r["choices"]],
                          "answer": int(r["answer"])})
    return items


def _load_hellaswag(path: Path) -> list[dict]:
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            items.append({"prompt": r["ctx"].strip(),
                          "choices": [" " + e for e in r["endings"]],
                          "answer": int(r["label"])})
    return items


_REPLAY_TEXTS = [
    "The capital of France is Paris, a city on the Seine.",
    "Two plus two equals four; addition combines quantities.",
    "Water (H2O) is two hydrogen atoms bonded to one oxygen atom.",
    "Cold is the opposite of hot; temperature measures heat.",
]


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = _device()
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    tok = AutoTokenizer.from_pretrained(model_id)
    # device_map="auto" (sharding a 35B over GPUs) needs `accelerate`; only use
    # it for multi-GPU. Single GPU/CPU: plain load + .to(dev). HL_DEVICE_MAP
    # overrides (e.g. "auto" to force sharding on a multi-H200 node).
    default_map = "auto" if (dev == "cuda" and torch.cuda.device_count() > 1) else ""
    use_map = os.environ.get("HL_DEVICE_MAP", default_map)
    kw = {"dtype": dtype}
    if use_map:
        kw["device_map"] = use_map
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if not use_map:
        model = model.to(dev)
    return model, tok


def _snapshot_cpu(model) -> dict:
    """Baseline weights on CPU, restored before each cell (load-once pattern)."""
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _restore(model, base: dict) -> None:
    model.load_state_dict(base, strict=True)


@torch.no_grad()
def _weight_sig(model):
    """Cheap global fingerprint: mean of the first few large weight tensors.
    Any dream update shifts these, so sig_pre != sig_post proves the model
    changed (more robust than a single scalar, far cheaper than a full norm)."""
    out = []
    for v in model.parameters():
        if v.is_floating_point() and v.numel() > 10000:
            out.append(round(v.detach().float().mean().item(), 9))
            if len(out) >= 5:
                break
    return out


@torch.no_grad()
def _score_choice(model, tok, prompt: str, choice: str) -> float:
    """Length-normalised log-prob of `choice` continued after `prompt` (the
    standard cloze score for MMLU/HellaSwag multiple-choice)."""
    dev = next(model.parameters()).device
    p_ids = tok(prompt, return_tensors="pt").input_ids
    full = tok(prompt + choice, return_tensors="pt").input_ids.to(dev)
    out = model(full)
    logp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
    tgt = full[0, 1:]
    cont = logp[range(tgt.numel()), tgt][p_ids.shape[1] - 1:]   # continuation tokens only
    return cont.mean().item() if cont.numel() else -1e9


@torch.no_grad()
def _eval_bench(model, tok, items, n_samples: int, seed: int) -> float:
    """Accuracy = fraction of subsampled items whose correct choice wins."""
    if not items:
        return 0.0
    g = torch.Generator().manual_seed(seed)
    n = min(n_samples, len(items)) if n_samples > 0 else len(items)
    idx = torch.randperm(len(items), generator=g)[:n].tolist()
    correct = 0
    for i in idx:
        ex = items[i]
        scores = [_score_choice(model, tok, ex["prompt"], c) for c in ex["choices"]]
        if int(torch.tensor(scores).argmax()) == ex["answer"]:
            correct += 1
    return correct / max(1, len(idx))


def _eval_composite(model, tok, benches: dict, n_samples: int, seed: int) -> dict:
    """Per-benchmark accuracy + composite mean (mirrors the MLX harness)."""
    accs = {name: _eval_bench(model, tok, items, n_samples, seed)
            for name, items in benches.items()}
    accs["composite"] = sum(accs.values()) / max(1, len(accs))
    return accs


def _replay(model, tok, texts, lr: float, inner: int) -> None:
    """DREAM core: SGD next-token replay over experiences (momentumless SGD =
    same family as the MLX harness; no optimizer state to bloat the 35B)."""
    dev = next(model.parameters()).device
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    for text in texts:
        ids = tok(text, return_tensors="pt").input_ids.to(dev)
        if ids.shape[1] < 2:
            continue
        for _ in range(inner):
            out = model(ids, labels=ids)
            opt.zero_grad()
            out.loss.backward()
            opt.step()
    model.eval()


def _run_cell(model, tok, base, benches, profile, seed, *, n_samples, lr, inner):
    t0 = time.time()
    _restore(model, base)
    pre = _eval_composite(model, tok, benches, n_samples, seed)
    w_pre = _weight_sig(model)
    if profile != "p_noise":
        _replay(model, tok, _REPLAY_TEXTS, lr, inner)
    w_post = _weight_sig(model)
    # noise floor re-evals the UNCHANGED model on a different draw
    post_seed = seed + 9973 if profile == "p_noise" else seed
    post = _eval_composite(model, tok, benches, n_samples, post_seed)
    return {
        "profile": profile, "seed": seed,
        "pre": pre["composite"], "post": post["composite"],
        "delta": post["composite"] - pre["composite"],
        "pre_by_bench": pre, "post_by_bench": post,
        "w_pre": w_pre, "w_post": w_post,
        "weight_changed": w_pre != w_post,
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", default=os.environ.get("HL_MODEL", "sshleifer/tiny-gpt2"))
    ap.add_argument("--n-seeds", type=int, default=int(os.environ.get("HL_NSEEDS", "5")))
    ap.add_argument("--n-samples", type=int, default=int(os.environ.get("HL_NSAMPLES", "4")))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()

    lr = float(os.environ.get("HL_LR", "1e-4"))
    inner = int(os.environ.get("HL_INNER", "3"))
    profiles = tuple((os.environ.get("HL_PROFILES", "p_min,p_noise")).split(","))
    seeds = (0,) if args.smoke else tuple(range(args.n_seeds))
    slug = args.scale.replace("/", "_")
    out_dir = Path(os.environ.get("HL_OUT", ".")); out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"pilot-cycle3-cuda-{slug}.cells.jsonl"

    print(f"[e1-cuda] device={_device()} model={args.scale} "
          f"profiles={profiles} seeds={len(seeds)} lr={lr} inner={inner}", flush=True)
    # real benchmark fixtures (env overrides for custom/larger sets)
    benches = {}
    mmlu_p = Path(os.environ.get("HL_MMLU", _FIX / "mmlu_real.jsonl"))
    hella_p = Path(os.environ.get("HL_HELLASWAG", _FIX / "hellaswag_sanity.jsonl"))
    if mmlu_p.exists():
        benches["mmlu"] = _load_mmlu(mmlu_p)
    if hella_p.exists():
        benches["hellaswag"] = _load_hellaswag(hella_p)
    if not benches:
        raise SystemExit(f"[e1-cuda] no fixtures found ({mmlu_p}, {hella_p})")
    print(f"[e1-cuda] benches: " + ", ".join(f"{k}={len(v)}" for k, v in benches.items()), flush=True)

    model, tok = _load_model(args.scale)
    base = _snapshot_cpu(model)              # load ONCE, restore per cell
    print(f"[e1-cuda] model loaded + baseline snapshotted "
          f"({sum(v.numel() for v in base.values())/1e9:.2f} G params)", flush=True)

    done = set(); cells = []
    if cache.exists() and not os.getenv("HL_E1_FRESH"):
        for line in cache.open():
            if line.strip():
                r = json.loads(line); cells.append(r); done.add((r["profile"], r["seed"]))
        print(f"[e1-cuda] resume: {len(done)} cached cells", flush=True)

    for profile in profiles:
        for seed in seeds:
            if (profile, seed) in done:
                continue
            cell = _run_cell(model, tok, base, benches, profile, seed,
                             n_samples=args.n_samples, lr=lr, inner=inner)
            cells.append(cell)
            with cache.open("a") as fh:
                fh.write(json.dumps(cell) + "\n")
            print(f"[e1-cuda] {profile} seed={seed} pre={cell['pre']:.3f} "
                  f"post={cell['post']:.3f} delta={cell['delta']:+.3f} "
                  f"wchg={cell['weight_changed']} wall={cell['wall_s']}s", flush=True)

    # per-profile summary
    import statistics as st
    summary = {}
    for profile in profiles:
        ds = [c["delta"] for c in cells if c["profile"] == profile]
        if ds:
            summary[profile] = {"n": len(ds), "mean_delta": round(st.mean(ds), 4),
                                "sd": round(st.pstdev(ds), 4) if len(ds) > 1 else 0.0}
    (out_dir / f"pilot-cycle3-cuda-{slug}.json").write_text(json.dumps(
        {"model": args.scale, "profiles": list(profiles), "summary": summary,
         "cells": cells}, indent=2))
    print(f"[e1-cuda] summary: {summary}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
