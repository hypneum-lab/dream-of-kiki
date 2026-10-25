"""E1 harness, backend-agnostic part: one cell, the modes and the run identity.

**Gate** : ``docs/osf-prereg-e1-replay-controls-2026-10-23.md`` (sections
4, 5, 6, 9). No framework import here: the model-specific operations are
supplied by a backend object (``scripts/pilot_cycle3_cuda.py``: PyTorch /
HF transformers; ``scripts/e1_mlx.py``: Apple MLX). R1: numbers from
different backends are never pooled; the backend name is part of the run
identity of every cell.

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
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
import time
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, Protocol

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts import e1_corpus_io as cio
from scripts import e1_diagnostics as dg
from scripts import e1_profiles as pr
from scripts import e1_sources as src
from scripts.e1_budget import budget_status, scoring_pass_tokens
from scripts.e1_schedule import schedule, summary, token_budget

# The pinned model of the study selected by HL_STUDY (default ``e1``, the
# 35B draft; ``e1-small``: docs/osf-prereg-e1-small-2026-10-25.md). The pin
# table is ``scripts.e1_sources.STUDY_MODELS``, shared with the corpus freeze;
# every model check of the harness and its backends reads these two names.
STUDY: str = os.environ.get("HL_STUDY", "e1")
if STUDY not in src.STUDY_MODELS:
    raise SystemExit(f"HL_STUDY={STUDY!r} is not one of {sorted(src.STUDY_MODELS)}")
MODEL_REPO, MODEL_REVISION = src.STUDY_MODELS[STUDY]
# Registered (backend, device) of a study, enforced on every cell, the
# amendment and the backend itself; None leaves it to the study's draft.
STUDY_BACKEND: dict[str, tuple[str, str] | None] = {"e1": None, "e1-small": ("torch", "cuda")}


def _study_backend_ok(backend: Any, versions: Any, study: Any) -> bool:
    """Backend, device, actual parameter placement (one device of that type)
    and study label match the study's registered instrument."""
    req = STUDY_BACKEND[STUDY]
    if req is None:
        return True
    v = versions or {}
    where = v.get("placement") or []
    one_device = len(where) == 1 and str(where[0]).split(":")[0] == req[1]
    return (backend, v.get("device")) == req and one_device and study == STUDY


def check_manifest_study(man: dict[str, Any]) -> None:
    """A FROZEN manifest must have been frozen with this study's headroom model
    (manifests without that record are accepted for the default study only)."""
    hr = man.get("headroom") or {}
    if "model" not in hr and STUDY == "e1":
        return
    if (hr.get("model"), hr.get("revision")) != (MODEL_REPO, MODEL_REVISION):
        raise SystemExit(f"the FROZEN corpus was not frozen with the {STUDY} headroom "
                         f"of {MODEL_REPO}@{MODEL_REVISION}")
SEEDS = {"smoke": tuple(range(2000, 2003)), "tuning": tuple(range(1000, 1010)),
         "calibrate": tuple(range(1000, 1010)), "confirmatory": tuple(range(141))}
N_REPLAY = 40
NOISE_SEED_OFFSET = 9973
FIX = ROOT / "tests" / "fixtures"
Item = dict[str, Any]
Benches = dict[str, list[Item]]


class Backend(Protocol):
    """Model operations a harness cell needs (one loaded model per backend)."""

    name: str

    def versions(self) -> dict[str, Any]: ...
    def snapshot(self) -> None: ...
    def restore(self) -> None: ...
    def n_tokens(self, text: str) -> int: ...
    def score_items(self, items: Sequence[Item]) -> tuple[list[list[float]], list[int]]: ...
    def replay(self, texts: Sequence[str], lengths: Sequence[int], updates: Sequence[Any],
               lr: float) -> dict[str, Any]: ...
    def downscale(self, factor: float) -> None: ...
    def restructure(self, seed: int) -> int | None: ...
    def swap_layers(self, i: int) -> None: ...
    def weights_finite(self) -> bool: ...
    def drift(self, router_only: bool) -> tuple[float, float]: ...
    def router_counter(self) -> Any: ...
    def reset_peak_memory(self) -> None: ...
    def peak_memory(self) -> dict[str, Any]: ...
    def cell_split(self, benches: Benches, seed: int, n_replay: int,
                   n_eval: int) -> tuple[Benches, list[Item]]: ...
    def shuffle(self, text: str, seed: int) -> str: ...


def host_maxrss() -> int:
    """Peak resident set size of this process in bytes (ru_maxrss)."""
    import resource

    r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return int(r if sys.platform == "darwin" else r * 1024)


def evaluate(b: Backend, eval_sets: Benches) -> dict[str, Any]:
    by, correct, scores, sources = {}, [], [], []
    for name, items in eval_sets.items():
        sc, am = b.score_items(items)
        cor = [a == int(it["answer"]) for a, it in zip(am, items, strict=True)]
        by[name] = sum(cor) / len(cor) if cor else 0.0
        correct += cor
        scores += sc
        sources += [str(it.get("source", name)) for it in items]
    acc = sum(by.values()) / max(1, len(by))
    return {"acc": acc, "by_bench": by, "correct": correct, "scores": scores,
            "sources": sources}


def replay(b: Backend, texts: Sequence[str], lr: float, m: int, budget: int) -> dict[str, Any]:
    """Token-budget replay (section 5.4): momentumless SGD, batch 1."""
    lengths = [b.n_tokens(t) for t in texts]
    updates = schedule(lengths, m, budget) if texts else []
    res = b.replay(texts, lengths, updates, lr)
    done = updates[: int(res["n_executed"])]
    losses = res["losses"]
    return {**summary(done, lengths), "planned_loss_tokens": budget,
            "planned_updates": len(updates),
            "mean_replay_loss": sum(losses) / len(losses) if losses else None,
            "invariant_violation": res["invariant_violation"]}


def apply_dream_ops(b: Backend, seed: int, downscale: bool, restructure: bool) -> dict[str, Any]:
    """DOWNSCALE (p_equ, p_max: shrink 0.99), RESTRUCTURE (p_max: swap two
    adjacent blocks). RECOMBINE is a weight-neutral no-op on this backend."""
    ops: dict[str, Any] = {"downscale": None, "restructure_idx": None, "recombine": "no-op"}
    if downscale:
        ops["downscale"] = pr.SHRINK
        b.downscale(pr.SHRINK)
    if restructure:
        ops["restructure_idx"] = b.restructure(seed)
    return ops


def run_cell(b: Backend, benches: Benches, *, seed: int, n_eval: int, corpus_kind: str,
             lr: float | None, m: int | None, downscale: bool, restructure: bool,
             charge_opr: bool, noise: bool = False, diagnostics: bool = True) -> dict[str, Any]:
    """One E1 cell: restore, pre-eval, budgeted replay, ops, post-eval, diagnostics."""
    t0 = time.time()
    b.restore()
    b.reset_peak_memory()
    eval_sets, replay_items = b.cell_split(benches, seed, N_REPLAY, n_eval)
    with b.router_counter() as rc_pre:
        pre = evaluate(b, eval_sets)
    facts = pr.profile_texts("facts", replay_items, seed)
    t_s = token_budget([b.n_tokens(f) for f in facts])
    rec: dict[str, Any] = {"seed": seed, "n_replay": len(replay_items), "t_s": t_s,
                           "n_eval_items": len(pre["correct"])}
    opr_choice = None
    arm_scoring = 0
    if corpus_kind == "opr":  # pre-replay checkpoint scores the replay prompts
        _, opr_choice = b.score_items(replay_items)
        arm_scoring = scoring_pass_tokens(
            (0, [b.n_tokens(it["prompt"] + c) for c in it["choices"]]) for it in replay_items)
    texts = pr.profile_texts(corpus_kind, replay_items, seed, b.shuffle, opr_choice)
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
        rep = replay(b, texts, lr, m, int(budget["update_tokens"]))
    rec.update(rep)
    ops = apply_dream_ops(b, seed, downscale, restructure)
    rec["ops"] = ops
    if rec["invariant_violation"] is None and not b.weights_finite():
        rec["invariant_violation"] = "S2"  # BLOCKING: no NaN / Inf (docs/invariants)
    post_sets = (b.cell_split(benches, seed + NOISE_SEED_OFFSET, N_REPLAY, n_eval)[0]
                 if noise else eval_sets)
    with b.router_counter() as rc_post:
        post = evaluate(b, post_sets)
    if ops["restructure_idx"] is not None:  # undo the block swap before comparing
        b.swap_layers(ops["restructure_idx"])  # parameters with the snapshot by name
    if diagnostics:
        sq_d, sq_p = b.drift(False)
        rq_d, rq_p = b.drift(True)
        rec["norm_change"] = math.sqrt(sq_d)
        rec["drift"] = dg.relative_drift(sq_d, sq_p) if sq_p > 0 else None
        rec["router_drift"] = dg.relative_drift(rq_d, rq_p) if rq_p > 0 else None
        if not noise:  # same probe items pre and post
            rec["kl_ref"] = dg.choice_kl(pre["scores"], post["scores"]) if pre["scores"] else None
            rec["lost_correct"] = dg.lost_correct(pre["correct"], post["correct"])
            rec["usage_l1"] = dg.usage_l1(rc_pre.counts, rc_post.counts)
    rec.update({
        "pre": pre["acc"], "post": post["acc"], "delta": post["acc"] - pre["acc"],
        "pre_by_bench": pre["by_bench"], "post_by_bench": post["by_bench"],
        "pre_correct": pre["correct"], "post_correct": post["correct"],
        "eval_sources": pre["sources"],
        "collapse": dg.collapse_flag(pre["acc"], post["acc"]),
        "wall_s": round(time.time() - t0, 1),
        **b.peak_memory(),
        "host_maxrss": host_maxrss(),
    })
    return rec


def profile_cell(b: Backend, benches: Benches, profile: str, seed: int,
                 amendment: pr.Amendment, n_eval: int) -> dict[str, Any]:
    cfg = pr.profile_config(profile, amendment)
    rec = run_cell(b, benches, seed=seed, n_eval=n_eval, corpus_kind=cfg.corpus, lr=cfg.lr,
                   m=cfg.m, downscale=cfg.downscale, restructure=cfg.restructure,
                   charge_opr=cfg.scoring_charged, noise=profile == "p_noise")
    return {"profile": profile, "lr": cfg.lr, "m": cfg.m, **rec}


# --- modes ---------------------------------------------------------------------

IDENTITY_KEYS = ("mode", "backend", "model", "revision", "corpus", "amendment_sha256",
                 "smoke_placeholder_amendment", "selection_vanilla", "tuning_cells_sha256",
                 "code_commit", "versions", "splits_sha256")


def code_commit() -> str | None:
    env = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env:
        return env
    import subprocess

    try:
        return subprocess.run(["git", "-C", str(ROOT), "rev-parse", "HEAD"], check=True,
                              capture_output=True, text=True).stdout.strip()
    except Exception:  # noqa: BLE001
        return None


def sha_file(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()


def _cache(path: Path) -> tuple[list[dict[str, Any]], set[str]]:
    cells: list[dict[str, Any]] = []
    if path.exists():
        cells = [json.loads(x) for x in path.read_text().splitlines() if x.strip()]
    return cells, {c["key"] for c in cells}


def _append(path: Path, cell: dict[str, Any]) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(cell) + "\n")


def read_cells(path: Path) -> list[dict[str, Any]]:
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
        ok = ok and _study_backend_ok(c.get("backend"), c.get("versions"), c.get("study"))
        if mode == "calibrate":
            ok = ok and c.get("tuning_cells_sha256") == tuning_cells_sha256
        if not ok:
            raise SystemExit(f"{mode} cell {c.get('key')} does not come from this corpus, "
                             "the pinned model or the given tuning cells")
    if len(ids) != 1:
        raise SystemExit(f"{mode} cells come from more than one run identity")


def selection_i(tuning_cells: Path, corpus_sha256: str) -> pr.Config:
    cells = read_cells(tuning_cells)
    check_provenance(cells, "tuning", corpus_sha256)
    seeds = SEEDS["tuning"]
    by = {(c["seed"], float(c["lr"]), int(c["m"])): float(c["delta"]) for c in cells
          if c.get("kind") == "tune_vanilla"}
    if set(by) != {(s, lr, m) for s in seeds for lr, m in pr.GRID}:
        raise SystemExit("selection (i) needs the complete tune_vanilla grid for seeds 1000-1009")
    return pr.select_config({(lr, m): sum(by[(s, lr, m)] for s in seeds) / len(seeds)
                             for lr, m in pr.GRID})


BackendFactory = Callable[[argparse.Namespace, Benches, dict[str, Any]], Backend]


def add_args(ap: argparse.ArgumentParser) -> None:
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
    ap.add_argument("--smoke-profiles", help="comma list (first smoke: a single cell)")
    ap.add_argument("--pilot-partition", default="confirmatory", choices=cio.PARTITIONS)
    ap.add_argument("--out", type=Path, default=Path(os.environ.get("HL_OUT", ".")))


def run(args: argparse.Namespace, make_backend: BackendFactory) -> int:  # noqa: C901
    """Dispatch one mode with the given backend factory."""
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
        check_manifest_study(man)
        tcells, ccells = read_cells(args.tuning_cells), read_cells(args.calib_cells)
        check_provenance(tcells, "tuning", str(man["corpus_sha256"]))
        check_provenance(ccells, "calibrate", str(man["corpus_sha256"]),
                         sha_file(args.tuning_cells))
        sel = selection_i(args.tuning_cells, str(man["corpus_sha256"]))
        if any(c.get("selection_vanilla") != list(sel) for c in ccells):
            raise SystemExit("calibration cells were not run at selection (i)")
        backends = {c.get("backend") for c in tcells + ccells}
        if len(backends) != 1:
            raise SystemExit("tuning and calibration cells come from different backends")
        am = pr.tuning_amendment(tcells + ccells,
                                 int(man["partitions"]["confirmatory"]["n_items"]))
        am.update({"manifest_sha256": sha_file(args.manifest),
                   "tuning_cells_sha256": sha_file(args.tuning_cells),
                   "calib_cells_sha256": sha_file(args.calib_cells),
                   "backend": backends.pop(), "study": STUDY,
                   "date": time.strftime("%Y-%m-%d")})
        out = args.out / "e1-tuning-amendment.json"
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        out.write_text(json.dumps(am, indent=1, sort_keys=True) + "\n")
        print(json.dumps(am, sort_keys=True))
        return 0

    items: list[Item] = []
    if args.mode == "smoke":
        benches, prov = cio.load_pilot(FIX, FIX / "e1_partition_2026-10.json",
                                       args.pilot_partition)
    else:
        if not (args.corpus and args.manifest):
            raise SystemExit(f"{args.mode} needs --corpus and --manifest")
        items, prov = cio.load_corpus(args.corpus, args.manifest, args.mode)
        if prov["status"] == cio.FROZEN:
            check_manifest_study(json.loads(args.manifest.read_text()))
        benches = {"facts": items}

    seeds = SEEDS.get(args.mode, ())
    if args.seeds:
        lo, _, hi = args.seeds.partition("-")
        want = set(range(int(lo), int(hi or lo) + 1))
        if not want <= set(seeds):
            raise SystemExit(f"--seeds outside the {args.mode} seeds")
        seeds = tuple(s for s in seeds if s in want)

    common: dict[str, Any] = {"mode": args.mode, "model": args.model, "revision": args.revision,
                              "corpus": prov, "study": STUDY}
    b = make_backend(args, benches, common)
    b.snapshot()
    common.update({"backend": b.name, "versions": b.versions()})
    if not _study_backend_ok(b.name, common["versions"], STUDY):
        raise SystemExit(f"{STUDY} runs only on the {STUDY_BACKEND[STUDY]} backend")
    slug = args.model.replace("/", "_")

    if args.mode == "headroom":
        b.restore()
        _, am_ = b.score_items(items)
        correct = {it["src_hash"]: a == int(it["answer"]) for it, a in zip(items, am_, strict=True)}
        out = args.out / "e1-headroom.json"
        if out.exists():
            raise SystemExit(f"refusing to overwrite {out}")
        out.write_text(json.dumps({"correct": correct, **common}, indent=1, sort_keys=True) + "\n")
        print(f"[e1] headroom: {sum(correct.values())}/{len(correct)} tuning items correct")
        return 0

    common["code_commit"] = code_commit()
    if args.mode == "calibrate":
        assert args.tuning_cells is not None
        common["tuning_cells_sha256"] = sha_file(args.tuning_cells)
    cache = args.out / f"e1-{args.mode}-{b.name}-{slug}.cells.jsonl"
    cached, done = _cache(cache)
    jobs: list[tuple[str, dict[str, Any]]] = []
    if args.mode in ("smoke", "confirmatory"):
        if args.mode == "confirmatory":
            if not args.amendment:
                raise SystemExit("confirmatory cells need the frozen tuning amendment")
            assert args.manifest is not None
            doc = json.loads(args.amendment.read_text())
            if doc.get("manifest_sha256") != sha_file(args.manifest):
                raise SystemExit("the amendment was frozen against another corpus manifest")
            if doc.get("backend") not in (None, b.name) or (
                    STUDY_BACKEND[STUDY] is not None and (doc.get("backend"), doc.get("study"))
                    != (b.name, STUDY)):
                raise SystemExit("the amendment was tuned on another backend or study")
            conf_size = int(json.loads(args.manifest.read_text())
                            ["partitions"]["confirmatory"]["n_items"])
            am = pr.validate_amendment(doc, conf_size)
            common["amendment_sha256"] = sha_file(args.amendment)
            n_eval = am.n_eval
            profiles = list(pr.ALL_PROFILES)
        else:  # smoke: defaults stand in for the not-yet-tuned arms
            am = pr.Amendment(pr.DEFAULT, pr.DEFAULT, 1.0, args.smoke_n_eval)
            common["smoke_placeholder_amendment"] = am.as_dict()
            n_eval = args.smoke_n_eval
            profiles = (args.smoke_profiles.split(",") if args.smoke_profiles
                        else list(pr.ALL_PROFILES))
            if not set(profiles) <= set(pr.ALL_PROFILES):
                raise SystemExit(f"unknown smoke profile in {profiles}")
        skipped = [p for p in profiles if not am.runs(p)]
        if skipped:  # declared in the amendment, before any confirmatory seed
            print(f"[e1] not run per the tuning amendment: {skipped}", flush=True)
        for p in profiles:
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
        lr_i, m_i = selection_i(args.tuning_cells, str(prov["corpus_sha256"]))
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
            cell = profile_cell(b, benches, job["profile"], s, job["am"], job["n_eval"])
        else:
            kind = job["kind"]
            cell = run_cell(b, benches, seed=s, n_eval=0,
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
