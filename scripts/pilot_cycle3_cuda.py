#!/usr/bin/env python3
"""E1 35B replay controls — CUDA / PyTorch harness.

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` (sections
4, 5, 6, 9). R1: a different backend from the MLX runs; numbers are never
pooled across backends.

Load-once + restore: the model is loaded once, a CPU copy of the base
weights is restored before every cell.

Modes (``--mode``), each with its fixed seeds and the only partition it
may read (``scripts.e1_corpus_io``):

==============  ==================  ===========================================
mode            seeds               reads
==============  ==================  ===========================================
smoke           2000-2002           pilot fixtures (section 5.3), all profiles
headroom        none (base model)   TUNING of the PENDING-HEADROOM corpus
tuning          1000-1009           TUNING of the FROZEN corpus (120 cells)
calibrate       1000-1009           TUNING of the FROZEN corpus (60 cells)
amend           -                   tuning + calibration cells (no model)
confirmatory    0-140               CONFIRMATORY of the FROZEN corpus
==============  ==================  ===========================================

Confirmatory cells need the frozen tuning amendment (``--amendment``).
The model revision is pinned (section 10 item 1): loading
``Qwen/Qwen3.6-35B-A3B`` at any other revision is refused.

    python pilot_cycle3_cuda.py --mode smoke --model <id> [--revision R]
    python pilot_cycle3_cuda.py --mode headroom --corpus C --manifest M
    python pilot_cycle3_cuda.py --mode tuning --corpus C --manifest M
    python pilot_cycle3_cuda.py --mode calibrate --corpus C --manifest M --tuning-cells T
    python pilot_cycle3_cuda.py --mode amend --manifest M --tuning-cells T --calib-cells K
    python pilot_cycle3_cuda.py --mode confirmatory --corpus C --manifest M --amendment A
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e1_corpus_io as cio
from scripts import e1_diagnostics as dg
from scripts import e1_profiles as pr
from scripts.e1_budget import budget_status, scoring_pass_tokens
from scripts.e1_schedule import schedule, summary, token_budget

MODEL_REPO = "Qwen/Qwen3.6-35B-A3B"
MODEL_REVISION = "995ad96eacd98c81ed38be0c5b274b04031597b0"
SEEDS = {"smoke": tuple(range(2000, 2003)), "tuning": tuple(range(1000, 1010)),
         "calibrate": tuple(range(1000, 1010)), "confirmatory": tuple(range(141))}
N_REPLAY = 40
NOISE_SEED_OFFSET = 9973
_FIX = ROOT / "tests" / "fixtures"


def _shuffle_text(text: str, seed: int) -> str:
    """Word-shuffle: keeps the tokens (same updates) but destroys meaning."""
    w = text.split()
    if len(w) < 2:
        return text
    g = torch.Generator().manual_seed(seed + len(w))
    return " ".join(w[i] for i in torch.randperm(len(w), generator=g).tolist())


def _cell_split(benches: dict[str, list[dict[str, Any]]], seed: int, n_replay: int,
                n_eval: int) -> tuple[dict[str, list[dict[str, Any]]], list[dict[str, Any]]]:
    """Disjoint per-(bench, seed) split: replay items vs eval items."""
    eval_sets, replay_items = {}, []
    for name, items in benches.items():
        g = torch.Generator().manual_seed(seed)
        order = torch.randperm(len(items), generator=g).tolist()
        replay_items += [items[i] for i in order[:n_replay]]
        ev = order[n_replay:]
        eval_sets[name] = [items[i] for i in (ev[:n_eval] if n_eval > 0 else ev)]
    return eval_sets, replay_items


# --- model -------------------------------------------------------------------


def _device() -> str:
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_model(model_id: str, revision: str | None) -> tuple[Any, Any]:
    if model_id == MODEL_REPO and revision != MODEL_REVISION:
        raise SystemExit(f"{MODEL_REPO} must be loaded at the pinned revision {MODEL_REVISION}")
    from transformers import AutoModelForCausalLM, AutoTokenizer

    dev = _device()
    dtype = torch.bfloat16 if dev == "cuda" else torch.float32
    trc = os.environ.get("HL_TRUST_REMOTE_CODE", "0") == "1"
    tok = AutoTokenizer.from_pretrained(model_id, revision=revision, trust_remote_code=trc)
    default_map = "auto" if (dev == "cuda" and torch.cuda.device_count() > 1) else ""
    use_map = os.environ.get("HL_DEVICE_MAP", default_map)
    kw: dict[str, Any] = {"dtype": dtype, "trust_remote_code": trc, "revision": revision}
    if use_map:
        kw["device_map"] = use_map
    model = AutoModelForCausalLM.from_pretrained(model_id, **kw)
    if not use_map:
        model = model.to(dev)
    model.eval()
    return model, tok


def _snapshot_cpu(model: Any) -> dict[str, torch.Tensor]:
    return {k: v.detach().to("cpu", copy=True) for k, v in model.state_dict().items()}


def _restore(model: Any, base: dict[str, torch.Tensor]) -> None:
    model.load_state_dict(base, strict=True)


def _input_device(model: Any) -> torch.device:
    try:
        return model.get_input_embeddings().weight.device  # type: ignore[no-any-return]
    except Exception:  # noqa: BLE001
        return next(model.parameters()).device  # type: ignore[no-any-return]


def _ids(tok: Any, text: str) -> torch.Tensor:
    return tok(text, return_tensors="pt").input_ids  # type: ignore[no-any-return]


@torch.no_grad()
def _score_choice(model: Any, tok: Any, prompt: str, choice: str) -> float:
    """Length-normalised log-prob of ``choice`` after ``prompt`` (cloze score)."""
    dev = _input_device(model)
    p_ids = _ids(tok, prompt)
    full = _ids(tok, prompt + choice).to(dev)
    out = model(full)
    logp = torch.log_softmax(out.logits[0, :-1].float(), dim=-1)
    tgt = full[0, 1:]
    cont = logp[range(tgt.numel()), tgt][p_ids.shape[1] - 1:]
    return cont.mean().item() if cont.numel() else -1e9


@torch.no_grad()
def _score_items(model: Any, tok: Any, items: Sequence[dict[str, Any]]) -> tuple[
        list[list[float]], list[int]]:
    """Choice scores and argmax (ties -> lowest index, ``torch.argmax``)."""
    scores, argmax = [], []
    for ex in items:
        s = [_score_choice(model, tok, ex["prompt"], c) for c in ex["choices"]]
        scores.append(s)
        argmax.append(int(torch.tensor(s).argmax()))
    return scores, argmax


def _eval(model: Any, tok: Any, eval_sets: dict[str, list[dict[str, Any]]]) -> dict[str, Any]:
    by, correct, scores, sources = {}, [], [], []
    for name, items in eval_sets.items():
        sc, am = _score_items(model, tok, items)
        cor = [a == int(it["answer"]) for a, it in zip(am, items, strict=True)]
        by[name] = sum(cor) / len(cor) if cor else 0.0
        correct += cor
        scores += sc
        sources += [str(it.get("source", name)) for it in items]
    acc = sum(by.values()) / max(1, len(by))
    return {"acc": acc, "by_bench": by, "correct": correct, "scores": scores,
            "sources": sources}


# --- MoE router ---------------------------------------------------------------


def _router_modules(model: Any) -> list[tuple[str, Any]]:
    return [(n, m) for n, m in model.named_modules()
            if n.endswith("mlp.gate") and isinstance(m, torch.nn.Linear)]


class _RouterCounter:
    """Counts top-k routed tokens per (layer, expert) while attached."""

    def __init__(self, model: Any) -> None:
        self.k = int(getattr(getattr(model, "config", None), "num_experts_per_tok", 0) or 0)
        self.mods = _router_modules(model) if self.k else []
        self.counts: dict[str, int] = {}
        self.handles: list[Any] = []

    def __enter__(self) -> _RouterCounter:
        for name, mod in self.mods:
            self.handles.append(mod.register_forward_hook(self._hook(name)))
        return self

    def _hook(self, name: str) -> Any:
        def fn(_m: Any, _i: Any, out: Any) -> None:
            logits = out[0] if isinstance(out, tuple) else out
            top = torch.topk(logits.reshape(-1, logits.shape[-1]).float(), self.k, dim=-1).indices
            for e, c in zip(*torch.unique(top, return_counts=True), strict=True):
                key = f"{name}:{int(e)}"
                self.counts[key] = self.counts.get(key, 0) + int(c)
        return fn

    def __exit__(self, *exc: Any) -> None:
        for h in self.handles:
            h.remove()


@torch.no_grad()
def _drift(model: Any, base: dict[str, torch.Tensor], router_only: bool = False) -> tuple[
        float, float]:
    """(sum of squared differences, sum of squared pre values) over float tensors."""
    sq_d = sq_p = 0.0
    for k, v in model.state_dict().items():
        if not v.is_floating_point() or (router_only and not k.endswith("mlp.gate.weight")):
            continue
        b = base[k].to(v.device, dtype=torch.float32)
        sq_d += float(((v.float() - b) ** 2).sum())
        sq_p += float((b ** 2).sum())
    return sq_d, sq_p


# --- replay and dream ops -------------------------------------------------------


def _replay(model: Any, tok: Any, texts: Sequence[str], lr: float, m: int,
            budget: int) -> dict[str, Any]:
    """Token-budget replay (section 5.4): momentumless SGD, batch 1."""
    ids = [_ids(tok, t) for t in texts]
    lengths = [int(x.shape[1]) for x in ids]
    updates = schedule(lengths, m, budget) if texts else []
    dev = _input_device(model)
    opt = torch.optim.SGD(model.parameters(), lr=lr)
    model.train()
    losses: list[float] = []
    violation = None
    done: list[Any] = []
    for u in updates:
        x = ids[u.record][:, : u.n_input].to(dev)
        opt.zero_grad(set_to_none=True)
        out = model(x, labels=x)
        if not torch.isfinite(out.loss):
            violation = "S2"  # BLOCKING: no NaN / Inf (docs/invariants/registry.md)
            break
        out.loss.backward()
        opt.step()
        losses.append(float(out.loss.detach()))
        done.append(u)
    opt.zero_grad(set_to_none=True)
    model.eval()
    return {**summary(done, lengths), "planned_loss_tokens": budget,
            "planned_updates": len(updates),
            "mean_replay_loss": sum(losses) / len(losses) if losses else None,
            "invariant_violation": violation}


def _get_layers(model: Any) -> Any:
    for path in ("model.layers", "model.model.layers", "transformer.h", "gpt_neox.layers"):
        obj = model
        for a in path.split("."):
            if not hasattr(obj, a):
                break
            obj = getattr(obj, a)
        else:
            return obj
    return None


@torch.no_grad()
def _downscale(model: Any, factor: float) -> None:
    for p in model.parameters():
        if p.is_floating_point():
            p.mul_(factor)


def _swap_layers(model: Any, i: int) -> None:
    layers = _get_layers(model)
    layers[i], layers[i + 1] = layers[i + 1], layers[i]


def _restructure(model: Any, seed: int) -> int | None:
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


def _apply_dream_ops(model: Any, seed: int, downscale: bool, restructure: bool) -> dict[str, Any]:
    """DOWNSCALE (p_equ, p_max: shrink 0.99), RESTRUCTURE (p_max: swap two
    adjacent blocks). RECOMBINE is a weight-neutral no-op on this backend."""
    ops: dict[str, Any] = {"downscale": None, "restructure_idx": None, "recombine": "no-op"}
    if downscale:
        ops["downscale"] = pr.SHRINK
        _downscale(model, pr.SHRINK)
    if restructure:
        ops["restructure_idx"] = _restructure(model, seed)
    return ops


@torch.no_grad()
def _weights_finite(model: Any) -> bool:
    return all(bool(torch.isfinite(p).all()) for p in model.parameters() if p.is_floating_point())


# --- one cell ------------------------------------------------------------------


def run_cell(model: Any, tok: Any, base: dict[str, torch.Tensor],
             benches: dict[str, list[dict[str, Any]]], *, seed: int, n_eval: int,
             corpus_kind: str, lr: float | None, m: int | None, downscale: bool,
             restructure: bool, charge_opr: bool, noise: bool = False,
             diagnostics: bool = True) -> dict[str, Any]:
    """One E1 cell: restore, pre-eval, budgeted replay, ops, post-eval, diagnostics."""
    t0 = time.time()
    _restore(model, base)
    if torch.cuda.is_available():
        for d in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(d)
    eval_sets, replay_items = _cell_split(benches, seed, N_REPLAY, n_eval)
    with _RouterCounter(model) as rc_pre:
        pre = _eval(model, tok, eval_sets)
    facts = pr.profile_texts("facts", replay_items, seed)
    t_s = token_budget([int(_ids(tok, f).shape[1]) for f in facts])
    rec: dict[str, Any] = {"seed": seed, "n_replay": len(replay_items), "t_s": t_s,
                           "n_eval_items": len(pre["correct"])}
    opr_choice = None
    arm_scoring = 0
    if corpus_kind == "opr":  # pre-replay checkpoint scores the replay prompts
        _, opr_choice = _score_items(model, tok, replay_items)
        arm_scoring = scoring_pass_tokens(
            (0, [int(_ids(tok, it["prompt"] + c).shape[1]) for c in it["choices"]])
            for it in replay_items)
    texts = pr.profile_texts(corpus_kind, replay_items, seed, _shuffle_text, opr_choice)
    if corpus_kind == "opr":
        rec["opr_corpus_size"] = len(texts)
        rec["s_opr"] = arm_scoring
    budget = budget_status(t_s, arm_scoring if charge_opr else 0)
    rec["budget"] = budget
    if noise or not texts or lr is None or m is None:
        rep = {"achieved_loss_tokens": 0, "n_updates": 0, "n_truncated": 0,
               "planned_loss_tokens": 0, "planned_updates": 0, "mean_replay_loss": None,
               "invariant_violation": None}
    else:
        rep = _replay(model, tok, texts, lr, m, int(budget["update_tokens"]))
    rec.update(rep)
    ops = _apply_dream_ops(model, seed, downscale, restructure)
    rec["ops"] = ops
    if rec["invariant_violation"] is None and not _weights_finite(model):
        rec["invariant_violation"] = "S2"
    post_sets = _cell_split(benches, seed + NOISE_SEED_OFFSET, N_REPLAY, n_eval)[0] if noise \
        else eval_sets
    with _RouterCounter(model) as rc_post:
        post = _eval(model, tok, post_sets)
    if diagnostics:
        sq_d, sq_p = _drift(model, base)
        rq_d, rq_p = _drift(model, base, router_only=True)
        rec["norm_change"] = math.sqrt(sq_d)
        rec["drift"] = dg.relative_drift(sq_d, sq_p) if sq_p > 0 else None
        rec["router_drift"] = dg.relative_drift(rq_d, rq_p) if rq_p > 0 else None
        if not noise:  # same probe items pre and post
            rec["kl_ref"] = dg.choice_kl(pre["scores"], post["scores"]) if pre["scores"] else None
            rec["lost_correct"] = dg.lost_correct(pre["correct"], post["correct"])
            rec["usage_l1"] = dg.usage_l1(rc_pre.counts, rc_post.counts)
    if ops["restructure_idx"] is not None:
        _swap_layers(model, ops["restructure_idx"])
    rec.update({
        "pre": pre["acc"], "post": post["acc"], "delta": post["acc"] - pre["acc"],
        "pre_by_bench": pre["by_bench"], "post_by_bench": post["by_bench"],
        "pre_correct": pre["correct"], "post_correct": post["correct"],
        "eval_sources": pre["sources"],
        "collapse": dg.collapse_flag(pre["acc"], post["acc"]),
        "wall_s": round(time.time() - t0, 1),
        "cuda_max_memory_allocated": ([torch.cuda.max_memory_allocated(d)
                                       for d in range(torch.cuda.device_count())]
                                      if torch.cuda.is_available() else []),
        "host_maxrss": _host_maxrss(),
    })
    return rec


def _host_maxrss() -> int:
    """Peak resident set size of this process in bytes (ru_maxrss)."""
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def profile_cell(model: Any, tok: Any, base: dict[str, torch.Tensor],
                 benches: dict[str, list[dict[str, Any]]], profile: str, seed: int,
                 amendment: pr.Amendment, n_eval: int) -> dict[str, Any]:
    cfg = pr.profile_config(profile, amendment)
    rec = run_cell(model, tok, base, benches, seed=seed, n_eval=n_eval,
                   corpus_kind=cfg.corpus, lr=cfg.lr, m=cfg.m, downscale=cfg.downscale,
                   restructure=cfg.restructure, charge_opr=cfg.scoring_charged,
                   noise=profile == "p_noise")
    return {"profile": profile, "lr": cfg.lr, "m": cfg.m, **rec}


# --- modes ---------------------------------------------------------------------


IDENTITY_KEYS = ("mode", "model", "revision", "corpus", "amendment_sha256",
                 "smoke_placeholder_amendment", "selection_vanilla", "tuning_cells_sha256",
                 "code_commit")


def _code_commit() -> str | None:
    env = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env:
        return env
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def _sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _cache(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    cells: list[dict[str, Any]] = []
    if path.exists():
        cells = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return cells, {c["key"] for c in cells}


def _append(path: Path, cell: dict[str, Any]) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(cell) + "\n")


def _read_cells(path: Path) -> list[dict[str, Any]]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def check_provenance(cells: Sequence[dict[str, Any]], mode: str, corpus_sha256: str,
                     tuning_cells_sha256: str | None = None) -> None:
    """Tuning / calibration records must come from one run of ``mode`` on the
    pinned model and on the TUNING partition of the given (FROZEN) corpus."""
    if not cells:
        raise SystemExit(f"no {mode} cells")
    ids = {c.get("run_identity") for c in cells}
    for c in cells:
        corp = c.get("corpus") or {}
        ok = (c.get("mode") == mode and c.get("model") == MODEL_REPO
              and c.get("revision") == MODEL_REVISION and corp.get("partition") == "tuning"
              and corp.get("corpus_sha256") == corpus_sha256 and c.get("run_identity"))
        if mode == "calibrate":
            ok = ok and c.get("tuning_cells_sha256") == tuning_cells_sha256
        if not ok:
            raise SystemExit(f"{mode} cell {c.get('key')} does not come from this corpus, "
                             "the pinned model or the given tuning cells")
    if len(ids) != 1:
        raise SystemExit(f"{mode} cells come from more than one run identity")


def _selection_i(tuning_cells: Path, corpus_sha256: str) -> pr.Config:
    cells = _read_cells(tuning_cells)
    check_provenance(cells, "tuning", corpus_sha256)
    seeds = SEEDS["tuning"]
    by = {(c["seed"], float(c["lr"]), int(c["m"])): float(c["delta"]) for c in cells
          if c.get("kind") == "tune_vanilla"}
    if set(by) != {(s, lr, m) for s in seeds for lr, m in pr.GRID}:
        raise SystemExit("selection (i) needs the complete tune_vanilla grid for seeds 1000-1009")
    return pr.select_config({(lr, m): sum(by[(s, lr, m)] for s in seeds) / len(seeds)
                             for lr, m in pr.GRID})


def main(argv: list[str] | None = None) -> int:  # noqa: C901 - one dispatch per mode
    ap = argparse.ArgumentParser(description="E1 CUDA harness")
    ap.add_argument("--mode", required=True, choices=("smoke", "headroom", "tuning", "calibrate",
                                                       "amend", "confirmatory"))
    ap.add_argument("--model", default=os.environ.get("HL_MODEL", MODEL_REPO))
    ap.add_argument("--revision", default=os.environ.get("HL_REVISION", MODEL_REVISION))
    ap.add_argument("--corpus", type=Path)
    ap.add_argument("--manifest", type=Path)
    ap.add_argument("--amendment", type=Path)
    ap.add_argument("--tuning-cells", type=Path)
    ap.add_argument("--calib-cells", type=Path)
    ap.add_argument("--seeds", help="subset of the mode's seeds, e.g. 0-46 (sharding)")
    ap.add_argument("--smoke-n-eval", type=int, default=50)
    ap.add_argument("--pilot-partition", default="confirmatory", choices=cio.PARTITIONS)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("HL_OUT", ".")))
    args = ap.parse_args(argv)
    if args.mode not in ("smoke", "amend") and (args.model != MODEL_REPO
                                                or args.revision != MODEL_REVISION):
        raise SystemExit(f"{args.mode} runs only on {MODEL_REPO}@{MODEL_REVISION}")
    args.out.mkdir(parents=True, exist_ok=True)

    if args.mode == "amend":
        if not (args.manifest and args.tuning_cells and args.calib_cells):
            raise SystemExit("amend needs --manifest, --tuning-cells and --calib-cells")
        man = json.loads(args.manifest.read_text())
        if man.get("status") != cio.FROZEN:
            raise SystemExit("amend needs the FROZEN manifest")
        tcells, ccells = _read_cells(args.tuning_cells), _read_cells(args.calib_cells)
        check_provenance(tcells, "tuning", str(man["corpus_sha256"]))
        check_provenance(ccells, "calibrate", str(man["corpus_sha256"]),
                         _sha_file(args.tuning_cells))
        sel = _selection_i(args.tuning_cells, str(man["corpus_sha256"]))
        if any(c.get("selection_vanilla") != list(sel) for c in ccells):
            raise SystemExit("calibration cells were not run at selection (i)")
        cells = tcells + ccells
        am = pr.tuning_amendment(cells, int(man["partitions"]["confirmatory"]["n_items"]))
        am.update({"manifest_sha256": _sha_file(args.manifest),
                   "tuning_cells_sha256": _sha_file(args.tuning_cells),
                   "calib_cells_sha256": _sha_file(args.calib_cells),
                   "date": time.strftime("%Y-%m-%d")})
        out = args.out / "e1-tuning-amendment.json"
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        out.write_text(json.dumps(am, indent=1, sort_keys=True) + "\n")
        print(json.dumps(am, sort_keys=True))
        return 0

    if args.mode == "smoke":
        benches, prov = cio.load_pilot(_FIX, _FIX / "e1_partition_2026-10.json",
                                       args.pilot_partition)
    else:
        if not (args.corpus and args.manifest):
            raise SystemExit(f"{args.mode} needs --corpus and --manifest")
        items, prov = cio.load_corpus(args.corpus, args.manifest, args.mode)
        benches = {"facts": items}

    seeds = SEEDS.get(args.mode, ())
    if args.seeds:
        lo, _, hi = args.seeds.partition("-")
        want = set(range(int(lo), int(hi or lo) + 1))
        if not want <= set(seeds):
            raise SystemExit(f"--seeds outside the {args.mode} seeds")
        seeds = tuple(s for s in seeds if s in want)

    model, tok = _load_model(args.model, args.revision or None)
    base = _snapshot_cpu(model)
    slug = args.model.replace("/", "_")
    common = {"mode": args.mode, "model": args.model, "revision": args.revision,
              "corpus": prov, "device": _device(),
              "torch": torch.__version__}

    if args.mode == "headroom":
        _restore(model, base)
        _, am_ = _score_items(model, tok, items)
        correct = {it["src_hash"]: a == int(it["answer"]) for it, a in zip(items, am_, strict=True)}
        out = args.out / "e1-headroom.json"
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        out.write_text(json.dumps({"correct": correct, **common}, indent=1, sort_keys=True) + "\n")
        print(f"[e1] headroom: {sum(correct.values())}/{len(correct)} tuning items correct")
        return 0

    common["code_commit"] = _code_commit()
    if args.mode == "calibrate":
        assert args.tuning_cells is not None
        common["tuning_cells_sha256"] = _sha_file(args.tuning_cells)
    cache = args.out / f"e1-{args.mode}-{slug}.cells.jsonl"
    cached, done = _cache(cache)
    jobs: list[tuple[str, dict[str, Any]]] = []
    if args.mode in ("smoke", "confirmatory"):
        if args.mode == "confirmatory":
            if not args.amendment:
                raise SystemExit("confirmatory cells need the frozen tuning amendment")
            assert args.manifest is not None
            doc = json.loads(args.amendment.read_text())
            if doc.get("manifest_sha256") != _sha_file(args.manifest):
                raise SystemExit("the amendment was frozen against another corpus manifest")
            conf_size = int(json.loads(args.manifest.read_text())
                            ["partitions"]["confirmatory"]["n_items"])
            am = pr.validate_amendment(doc, conf_size)
            common["amendment_sha256"] = _sha_file(args.amendment)
            n_eval = am.n_eval
        else:  # smoke: defaults stand in for the not-yet-tuned arms
            am = pr.Amendment(pr.DEFAULT, pr.DEFAULT, 1.0, args.smoke_n_eval)
            common["smoke_placeholder_amendment"] = am.as_dict()
            n_eval = args.smoke_n_eval
        skipped = [p for p in pr.ALL_PROFILES if not am.runs(p)]
        if skipped:  # declared in the amendment, before any confirmatory seed
            print(f"[e1] not run per the tuning amendment: {skipped}", flush=True)
        for p in pr.ALL_PROFILES:
            if p in skipped:
                continue
            for s in seeds:
                jobs.append((f"{p}|{s}", {"profile": p, "seed": s, "am": am, "n_eval": n_eval}))
    elif args.mode == "tuning":
        for s in seeds:
            for lr, m in pr.GRID:
                for kind in ("tune_vanilla", "tune_opr"):
                    jobs.append((f"{kind}|{s}|{lr}|{m}", {"kind": kind, "seed": s, "lr": lr,
                                                          "m": m}))
    elif args.mode == "calibrate":
        if not args.tuning_cells:
            raise SystemExit("calibrate needs --tuning-cells (selection (i))")
        lr_i, m_i = _selection_i(args.tuning_cells, str(prov["corpus_sha256"]))
        common["selection_vanilla"] = [lr_i, m_i]
        for s in seeds:
            for c in pr.C_GRID:
                jobs.append((f"calib_lr|{s}|{c}", {"kind": "calib_lr", "seed": s, "c": c,
                                                   "lr": c * lr_i, "m": m_i}))
            jobs.append((f"calib_ref|{s}", {"kind": "calib_ref", "seed": s, "lr": lr_i,
                                            "m": m_i}))

    identity = json.dumps({k: common.get(k) for k in IDENTITY_KEYS}, sort_keys=True)
    run_id = hashlib.sha256(identity.encode()).hexdigest()
    stale = sorted({c.get("run_identity") for c in cached} - {run_id}, key=str)
    if stale:
        raise SystemExit(f"{cache} holds cells of another run identity; use a fresh --out")
    for key, job in jobs:
        if key in done:
            continue
        s = int(job["seed"])
        if "profile" in job:
            cell = profile_cell(model, tok, base, benches, job["profile"], s, job["am"],
                                job["n_eval"])
        else:
            kind = job["kind"]
            cell = run_cell(model, tok, base, benches, seed=s, n_eval=0,
                            corpus_kind="opr" if kind == "tune_opr" else "facts",
                            lr=job["lr"], m=job["m"], downscale=kind == "calib_ref",
                            restructure=False, charge_opr=kind == "tune_opr")
            cell.update({k: v for k, v in job.items() if k != "seed"})
        cell.update(common)
        cell["key"] = key
        cell["run_identity"] = run_id
        _append(cache, cell)
        print(f"[e1] {args.mode} {key} pre={cell['pre']:.3f} post={cell['post']:.3f} "
              f"tokens={cell['achieved_loss_tokens']}/{cell['planned_loss_tokens']} "
              f"wall={cell['wall_s']}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
