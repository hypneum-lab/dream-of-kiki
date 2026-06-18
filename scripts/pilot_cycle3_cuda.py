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


# Neutral filler for the matched-compute SFT control — content unrelated to the
# benchmarks, so it isolates "any gradient step helps" from "relevant replay".
_NEUTRAL_TEXTS = [
    "The weather changed slowly from one day to the next.",
    "A long road wound quietly through the empty valley.",
    "She closed the wooden door and walked down the street.",
    "Clouds drifted above the distant grey hills at dusk.",
    "The old clock kept its steady rhythm through the night.",
]


def _shuffle_text(text: str, seed: int) -> str:
    """Word-shuffle: keeps the tokens (same updates) but destroys meaning."""
    w = text.split()
    if len(w) < 2:
        return text
    g = torch.Generator().manual_seed(seed + len(w))
    return " ".join(w[i] for i in torch.randperm(len(w), generator=g).tolist())


def _cell_split(benches: dict, seed: int, n_replay: int, n_eval: int):
    """Disjoint per-(bench, seed) split — replay pool vs eval subset — so the
    dream never replays an item it is scored on (leakage control built in)."""
    eval_sets, replay_items = {}, []
    for name, items in benches.items():
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(items), generator=g).tolist()
        replay_items += [items[i] for i in order[:n_replay]]
        ev = order[n_replay:]
        eval_sets[name] = [items[i] for i in (ev[:n_eval] if n_eval > 0 else ev)]
    return eval_sets, replay_items


def _profile_replay(profile: str, replay_items: list, seed: int) -> list:
    """Replay corpus per profile: relevant facts (dream), neutral (SFT control),
    shuffled (content-destroyed control), or none (noise floor)."""
    if profile == "p_noise":
        return []
    facts = [it["prompt"] + it["choices"][it["answer"]] for it in replay_items]
    if profile == "p_ctrl_sft":
        reps = (len(facts) // len(_NEUTRAL_TEXTS)) + 1
        return (_NEUTRAL_TEXTS * reps)[:max(1, len(facts))]
    if profile == "p_ctrl_shuffle":
        return [_shuffle_text(f, seed) for f in facts]
    return facts  # p_min: relevant, coherent, held-out replay


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_id: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer
    dev = _device()
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    # trust_remote_code lets newer archs (e.g. qwen3_5_moe) load when the
    # installed transformers lacks native support / the repo ships modeling code.
    trc = os.environ.get("HL_TRUST_REMOTE_CODE", "0") == "1"
    tok = AutoTokenizer.from_pretrained(model_id, trust_remote_code=trc)
    # device_map="auto" (sharding a 35B over GPUs) needs `accelerate`; only use
    # it for multi-GPU. Single GPU/CPU: plain load + .to(dev). HL_DEVICE_MAP
    # overrides (e.g. "auto" to force sharding on a multi-H200 node).
    default_map = "auto" if (dev == "cuda" and torch.cuda.device_count() > 1) else ""
    use_map = os.environ.get("HL_DEVICE_MAP", default_map)
    kw = {"dtype": dtype, "trust_remote_code": trc}
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


def _input_device(model):
    """Device where inputs must land. On a device_map-sharded 35B the input
    embedding lives on one GPU and accelerate hooks move activations across the
    others, so inputs go to the embedding's device, not next(parameters())."""
    try:
        return model.get_input_embeddings().weight.device
    except Exception:  # noqa: BLE001
        return next(model.parameters()).device


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
    dev = _input_device(model)
    p_ids = tok(prompt, return_tensors="pt").input_ids
    full = tok(prompt + choice, return_tensors="pt").input_ids.to(dev)
    out = model(full)
    logp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
    tgt = full[0, 1:]
    cont = logp[range(tgt.numel()), tgt][p_ids.shape[1] - 1:]   # continuation tokens only
    return cont.mean().item() if cont.numel() else -1e9


@torch.no_grad()
def _eval_items(model, tok, items) -> float:
    """Accuracy over a fixed item list (already subsampled by _cell_split)."""
    if not items:
        return 0.0
    correct = 0
    for ex in items:
        scores = [_score_choice(model, tok, ex["prompt"], c) for c in ex["choices"]]
        if int(torch.tensor(scores).argmax()) == ex["answer"]:
            correct += 1
    return correct / len(items)


def _eval_sets(model, tok, eval_sets: dict) -> dict:
    """Per-benchmark accuracy + composite mean over pre-split eval sets."""
    accs = {name: _eval_items(model, tok, items) for name, items in eval_sets.items()}
    accs["composite"] = sum(accs.values()) / max(1, len(accs))
    return accs


def _replay(model, tok, texts, lr: float, inner: int) -> None:
    """DREAM core: SGD next-token replay over experiences (momentumless SGD =
    same family as the MLX harness; no optimizer state to bloat the 35B)."""
    dev = _input_device(model)
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


def _get_layers(model):
    """Locate the transformer block list across common architectures."""
    for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        ok = True
        for a in path.split("."):
            if hasattr(obj, a):
                obj = getattr(obj, a)
            else:
                ok = False
                break
        if ok:
            return obj
    return None


@torch.no_grad()
def _downscale(model, factor: float) -> None:
    """DOWNSCALE op: shrink every weight by `factor` (mirrors the MLX recursive
    downscale that multiplies each leaf tensor)."""
    for p in model.parameters():
        if p.is_floating_point():
            p.mul_(factor)


def _swap_layers(model, i: int) -> None:
    layers = _get_layers(model)
    layers[i], layers[i + 1] = layers[i + 1], layers[i]


def _restructure(model, seed: int):
    """RESTRUCTURE op (p_max): swap two adjacent transformer blocks (NAS-style
    perturbation, i = seed % (n-1)). Returns the swapped index (so the caller
    can undo it before the next baseline restore), or None. Skipped if the two
    blocks live on different devices (sharded) to avoid a device mismatch."""
    layers = _get_layers(model)
    if layers is None or len(layers) < 2:
        return None
    i = seed % (len(layers) - 1)
    try:
        if next(layers[i].parameters()).device != next(layers[i + 1].parameters()).device:
            return None
        _swap_layers(model, i)
        return i
    except Exception:  # noqa: BLE001
        return None


def _apply_dream_ops(model, profile: str, seed: int) -> dict:
    """Structural dream ops beyond replay: downscale (p_equ/p_max) shrinks all
    weights; restructure (p_max) swaps two adjacent blocks; recombine is a
    weight-neutral latent op (no-op here, as in the MLX harness)."""
    info = {"restructure_idx": None}
    if profile in ("p_equ", "p_max"):
        factor = float(os.environ.get("HL_SHRINK", "0.99"))
        _downscale(model, factor)
        info["downscale"] = factor
    if profile == "p_max":
        info["restructure_idx"] = _restructure(model, seed)
    return info


def _run_cell(model, tok, base, benches, profile, seed, *, n_samples, n_replay, lr, inner):
    t0 = time.time()
    _restore(model, base)
    eval_sets, replay_items = _cell_split(benches, seed, n_replay, n_samples)
    pre = _eval_sets(model, tok, eval_sets)
    w_pre = _weight_sig(model)
    texts = _profile_replay(profile, replay_items, seed)
    if texts:
        _replay(model, tok, texts, lr, inner)
    ops = _apply_dream_ops(model, profile, seed)
    w_post = _weight_sig(model)
    # dream/control profiles re-eval the SAME held-out items (isolate the model
    # change); the noise floor uses a fresh disjoint draw (sampling variance).
    if profile == "p_noise":
        eval_post, _ = _cell_split(benches, seed + 9973, n_replay, n_samples)
    else:
        eval_post = eval_sets
    post = _eval_sets(model, tok, eval_post)
    if ops.get("restructure_idx") is not None:    # undo the block swap so the
        _swap_layers(model, ops["restructure_idx"])  # next _restore is canonical
    return {
        "profile": profile, "seed": seed,
        "pre": pre["composite"], "post": post["composite"],
        "delta": post["composite"] - pre["composite"],
        "pre_by_bench": pre, "post_by_bench": post, "n_replay": len(replay_items),
        "ops": ops,
        "w_pre": w_pre, "w_post": w_post, "weight_changed": w_pre != w_post,
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
    n_replay = int(os.environ.get("HL_NREPLAY", "10"))
    profiles = tuple((os.environ.get(
        "HL_PROFILES",
        "p_min,p_equ,p_max,p_ctrl_sft,p_ctrl_shuffle,p_noise")).split(","))
    seeds = (0,) if args.smoke else tuple(range(args.n_seeds))
    slug = args.scale.replace("/", "_")
    out_dir = Path(os.environ.get("HL_OUT", "."))
    out_dir.mkdir(parents=True, exist_ok=True)
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
    print("[e1-cuda] benches: " + ", ".join(f"{k}={len(v)}" for k, v in benches.items()), flush=True)

    model, tok = _load_model(args.scale)
    base = _snapshot_cpu(model)              # load ONCE, restore per cell
    print(f"[e1-cuda] model loaded + baseline snapshotted "
          f"({sum(v.numel() for v in base.values())/1e9:.2f} G params)", flush=True)

    done, cells = set(), []
    if cache.exists() and not os.getenv("HL_E1_FRESH"):
        for line in cache.open():
            if line.strip():
                r = json.loads(line)
                cells.append(r)
                done.add((r["profile"], r["seed"]))
        print(f"[e1-cuda] resume: {len(done)} cached cells", flush=True)

    for profile in profiles:
        for seed in seeds:
            if (profile, seed) in done:
                continue
            cell = _run_cell(model, tok, base, benches, profile, seed,
                             n_samples=args.n_samples, n_replay=n_replay, lr=lr, inner=inner)
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
