"""G6-Studio Path A driver — real LoRA SpikingKiki-V4 35B-A3B-V4 × MMLU CL stream.

First real-LLM-scale validation of framework C 4-channel coupling.
Per (arm, seed) cell : load ``QwenLoRAWrapper`` → for i in 1..N
[real LoRA fine-tune via ``mlx_lm.tuner.lora`` →
:func:`dream_episode_real` mutates the **live** LoRA delta (load-bearing
fix vs G6 Path B spectator) → letter-argmax MMLU eval via
``mlx_lm.generate``] → retention metric → R1 register.

Resumability : per-subdomain partial JSON dumps under ::

    docs/milestones/g6-studio-path-a-partial-<subdomain>-2026-05-04.json

(gitignored). On watchdog kill, the next launch can glob existing
partials, build a set of completed ``(arm, seed, idx)`` tuples, and
skip those — see plan Task 9 step 4.

Usage on Studio (production) ::

    DREAM_MICRO_KIKI_REAL=1 \\
    DREAM_MICRO_KIKI_REAL_BACKEND_PATH=\\
        /Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4 \\
    uv run python -m experiments.g6_studio_path_a.run_g6_studio_path_a \\
        --n-seeds 5

Usage on M1 Max conductor (smoke) ::

    uv run python -m experiments.g6_studio_path_a.run_g6_studio_path_a \\
        --smoke --n-seeds 1

Reference :
- Plan : ``docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md`` Task 6
- Pre-reg : ``docs/osf-prereg-g6-studio-path-a.md`` §5-§6-§8
- Path B template : ``experiments/g6_mmlu_stream/run_g6.py``
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Optional, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from experiments.g6_mmlu_stream.run_g6 import (  # noqa: E402
    AccMatrix,
    UNDERPERFORM_THRESHOLD,
    compute_retention,
)
from experiments.g6_mmlu_stream.stream import (  # noqa: E402
    SubdomainSplit,
    build_subdomain_stream,
)
from experiments.g6_studio_path_a.aggregator_h9 import (  # noqa: E402
    classify_h9,
)
from experiments.g6_studio_path_a.dream_episode_real import (  # noqa: E402
    DEFAULT_PRIMARY_KEY,
    dream_episode_real,
)
from experiments.g6_studio_path_a.lora_loader import (  # noqa: E402
    QwenLoRAWrapper,
    load_qwen_with_adapters,
)
from experiments.g6_studio_path_a.lora_train_step import (  # noqa: E402
    TrainHyperparams,
    train_subdomain_lora,
)
from experiments.g6_studio_path_a.mmlu_eval import (  # noqa: E402
    evaluate_mmlu_subdomain,
)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.substrates.micro_kiki import (  # noqa: E402
    MicroKikiSubstrate,
)


C_VERSION = "C-v0.12.0+PARTIAL"
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
DEFAULT_SUBDOMAINS: tuple[str, ...] = (
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
)
SMOKE_SUBDOMAINS: tuple[str, ...] = ("anatomy",)
DATE_TAG = "2026-05-04"
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "milestones" / f"g6-studio-path-a-{DATE_TAG}.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "docs" / "milestones" / f"g6-studio-path-a-{DATE_TAG}.md"
)
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
DEFAULT_BASE_PATH = (
    "/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4"
)
DEFAULT_ADAPTER_PATH = Path(
    "/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-V4-adapters",
)
DEFAULT_FIXTURE_PATH = (
    REPO_ROOT / "tests" / "fixtures" / "mmlu_g6_synthetic.jsonl"
)
PARTIAL_TEMPLATE = (
    "g6-studio-path-a-partial-{subdomain}-{arm}-seed{seed}-step{idx:02d}-"
    f"{DATE_TAG}.json"
)


class CellResult(TypedDict):
    """One (arm, seed) cell ; mirrors the Path B CellResult shape."""

    arm: str
    seed: int
    retention: float
    excluded_underperforming_baseline: bool
    wall_time_s: float
    acc_matrix: AccMatrix
    run_id: str


def _resolve_commit_sha() -> str:
    """Mirror :func:`experiments.g6_mmlu_stream.run_g6._resolve_commit_sha`."""
    env_sha = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=False,
            timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _write_partial(
    *,
    out_dir: Path,
    arm: str,
    seed: int,
    idx: int,
    subdomain: str,
    payload: dict[str, Any],
) -> Path:
    """Write a per-subdomain partial JSON dump (resumability)."""
    target = out_dir / PARTIAL_TEMPLATE.format(
        subdomain=subdomain, arm=arm, seed=seed, idx=idx,
    )
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    return target


def _stub_generate(
    model: Any,
    tokenizer: Any,
    *,
    prompt: str,
    max_tokens: int,
) -> str:
    """Deterministic stub generator for ``--smoke`` runs.

    Used when ``DREAM_MICRO_KIKI_REAL`` is unset — emits a single
    letter derived from a hash of the prompt so the smoke path
    exercises the eval pipeline shape without touching MLX.
    """
    import hashlib

    digest = int(
        hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8], 16,
    )
    return "ABCD"[digest % 4]


def _stub_train_subdomain_lora(
    *,
    model: Any,
    tokenizer: Any,
    train_records: Any,
    hyperparams: TrainHyperparams,
    adapter_keys: tuple[str, ...],
    seed: int,
    subdomain: str,
) -> dict[str, np.ndarray]:
    """Smoke-path replacement for :func:`train_subdomain_lora`.

    Generates a deterministic random LoRA delta keyed on
    ``(seed, subdomain)`` so the dream pipeline has a non-trivial
    tensor to mutate. Only used when ``--smoke`` is set or the
    real backend env flag is unset.
    """
    import hashlib

    raw = f"g6s-stub|{seed}|{subdomain}".encode("utf-8")
    sub_seed = int(hashlib.sha256(raw).hexdigest()[:8], 16)
    rng = np.random.default_rng(sub_seed)
    out_dim = 8
    return {
        key: rng.standard_normal(
            (out_dim, hyperparams.rank),
        ).astype(np.float32)
        for key in adapter_keys
    }


def _run_cell_real(
    *,
    arm: str,
    seed: int,
    splits: list[SubdomainSplit],
    wrapper: QwenLoRAWrapper,
    substrate: MicroKikiSubstrate,
    hp: TrainHyperparams,
    adapter_keys: tuple[str, ...],
    out_dir: Path,
    smoke: bool,
) -> CellResult:
    """Execute one (arm, seed) cell — real LoRA + live-delta coupling.

    The smoke path swaps ``train_subdomain_lora`` and
    ``evaluate_mmlu_subdomain`` for deterministic stubs so the
    pipeline runs end-to-end on M1 Max in seconds. The real path
    only fires when ``smoke=False`` and ``DREAM_MICRO_KIKI_REAL=1``.
    """
    start = time.time()
    subdomains = tuple(s.subject for s in splits)
    n_steps = len(splits)
    acc_matrix: AccMatrix = {
        subj: [None] * n_steps for subj in subdomains
    }
    live_delta: dict[str, np.ndarray] = {}
    prior_deltas: list[np.ndarray] = []
    sibling_deltas: list[np.ndarray] = []

    for idx, split in enumerate(splits):
        # 1. Per-subdomain real LoRA fine-tune (or smoke stub).
        if smoke:
            post = _stub_train_subdomain_lora(
                model=wrapper.model,
                tokenizer=wrapper.tokenizer,
                train_records=split.train,
                hyperparams=hp,
                adapter_keys=adapter_keys,
                seed=seed,
                subdomain=split.subject,
            )
        else:
            # Apply LoRA layers only on the FIRST subdomain of a
            # ``fresh_init`` cell. Subsequent subdomains inherit the
            # already-wrapped layers ; re-wrapping would double-stack
            # LoRA modules and break the live-delta key contract.
            apply_lora = wrapper.fresh_init and idx == 0
            post = train_subdomain_lora(
                model=wrapper.model,
                tokenizer=wrapper.tokenizer,
                train_records=split.train,
                hyperparams=hp,
                adapter_keys=adapter_keys,
                apply_lora_layers=apply_lora,
            )

        # Promote post-step delta into the live_delta dict that the
        # dream handlers mutate in place. When the trainer returned
        # nothing (degenerate early-iter case), keep the prior live
        # delta — the eval continues against the last known good
        # state.
        if DEFAULT_PRIMARY_KEY in post:
            live_delta[DEFAULT_PRIMARY_KEY] = post[
                DEFAULT_PRIMARY_KEY
            ].copy()
        elif post:
            first_key = next(iter(post))
            live_delta[DEFAULT_PRIMARY_KEY] = post[first_key].copy()

        # 2. Optional dream episode (skip on baseline arm).
        if arm != "baseline" and DEFAULT_PRIMARY_KEY in live_delta:
            dream_episode_real(
                substrate=substrate,
                profile=arm,
                live_delta=live_delta,
                seed=seed,
                subdomain=split.subject,
                prior_deltas=list(prior_deltas),
                sibling_deltas=list(sibling_deltas),
            )

        # 3. Evaluate on S_1..S_idx.
        for j in range(idx + 1):
            past = splits[j]
            if smoke:
                acc = evaluate_mmlu_subdomain(
                    model=wrapper.model,
                    tokenizer=wrapper.tokenizer,
                    records=past.eval_,
                    seed=seed,
                    generate_fn=_stub_generate,
                )
            else:
                acc = evaluate_mmlu_subdomain(
                    model=wrapper.model,
                    tokenizer=wrapper.tokenizer,
                    records=past.eval_,
                    seed=seed,
                )
            acc_matrix[past.subject][idx] = acc

        # 4. Per-subdomain partial dump (resumability).
        _write_partial(
            out_dir=out_dir,
            arm=arm,
            seed=seed,
            idx=idx,
            subdomain=split.subject,
            payload={
                "arm": arm,
                "seed": seed,
                "idx": idx,
                "subdomain": split.subject,
                "acc_matrix": acc_matrix,
                "wall_time_s": time.time() - start,
                "smoke": smoke,
            },
        )

        # 5. Carry the in-flight delta forward to the next subdomain
        # as both a prior (OPLoRA range to project away) and a
        # sibling (TIES-Merge contributor).
        if DEFAULT_PRIMARY_KEY in live_delta:
            prior_deltas.append(
                live_delta[DEFAULT_PRIMARY_KEY].copy(),
            )
            sibling_deltas.append(
                live_delta[DEFAULT_PRIMARY_KEY].copy(),
            )

    initial_first = acc_matrix[subdomains[0]][0]
    excluded = bool(
        initial_first is not None
        and initial_first < UNDERPERFORM_THRESHOLD,
    )
    return {
        "arm": arm,
        "seed": seed,
        "retention": float(
            compute_retention(acc_matrix, subdomains=subdomains),
        ),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
        "acc_matrix": acc_matrix,
        "run_id": "",
    }


def _retention_by_arm(
    cells: list[CellResult],
) -> dict[str, list[float]]:
    """Group retention scores by arm, dropping excluded cells."""
    out: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for cell in cells:
        if cell["excluded_underperforming_baseline"]:
            continue
        out[cell["arm"]].append(cell["retention"])
    return out


def _render_md_report(payload: dict[str, Any]) -> str:
    """Render the canonical milestone MD report (mirror Path B style)."""
    lines = [
        "# G6-Studio Path A — real LoRA SpikingKiki-V4 × MMLU CL stream",
        "",
        f"**Date** : {payload['date']}",
        f"**c_version** : `{payload['c_version']}`",
        f"**commit_sha** : `{payload['commit_sha']}`",
        f"**Cells** : {len(payload['cells'])}",
        f"**Wall time** : {payload['wall_time_s']:.1f}s",
        f"**Smoke** : {payload['smoke']}",
        "",
        "## Pre-registered hypotheses (LOCKED)",
        "",
        "Pre-registration : `docs/osf-prereg-g6-studio-path-a.md`.",
        "Decision rules at α/3 = 0.0167 (Bonferroni over "
        "{H9-A, H9-B, H9-C}).",
        "",
        "### Verdict",
        f"```\n{json.dumps(payload['verdict'], indent=2, sort_keys=True)}"
        "\n```",
        "",
        "## Cells (R1 traceability)",
        "",
        "| arm | seed | retention | excluded | run_id |",
        "|-----|------|-----------|----------|--------|",
    ]
    for cell in payload["cells"]:
        lines.append(
            f"| {cell['arm']} | {cell['seed']} | "
            f"{cell['retention']:.4f} | "
            f"{cell['excluded_underperforming_baseline']} | "
            f"`{cell['run_id']}` |",
        )
    lines += [
        "",
        "## Honest reporting",
        "",
        "Per pre-reg §6, EC stays PARTIAL across all H9-{A,B,C} rows. "
        "FC stays at C-v0.12.0. H9-A confirmation queues an "
        "Option-A (N >= 10) follow-up pre-reg before any STABLE bump. "
        "H9-B confirms G5-bis MLX-only artefact extends from toy "
        "E-SNN to real-LLM tier. H9-C confirms DR-4 inversion "
        "universalises at real-LLM scale.",
    ]
    return "\n".join(lines)


def run_pilot(
    *,
    fixture_path: Path,
    out_json: Path,
    out_md: Path,
    registry_db: Path,
    seeds: tuple[int, ...],
    n_train: int,
    n_eval: int,
    hp: TrainHyperparams,
    adapter_keys: tuple[str, ...],
    base_path: str,
    adapter_path: Path,
    subdomains: tuple[str, ...],
    smoke: bool,
) -> dict[str, Any]:
    """Execute the G6-Studio Path A sweep ; return the verdict payload.

    Real backend (``smoke=False`` + ``DREAM_MICRO_KIKI_REAL=1``)
    requires ~8-12 h on Studio M3 Ultra at the locked Option-B
    hyperparams (5 seeds × 4 arms × 5 subdomains = 100 cells).
    """
    splits = build_subdomain_stream(
        fixture_path=fixture_path,
        subdomains=subdomains,
        n_train=n_train,
        n_eval=n_eval,
        seed=0,  # subdomain split seed pinned at 0 for R1
    )
    registry = RunRegistry(registry_db)
    commit_sha = _resolve_commit_sha()
    out_dir = out_json.parent
    cells: list[CellResult] = []
    sweep_start = time.time()

    for arm in ARMS:
        for seed in seeds:
            if smoke:
                # Smoke path : skip the 58 GB load, build a stub
                # wrapper carrying ``None`` handles. The cell
                # runner's smoke branch routes around all MLX
                # surfaces.
                wrapper = QwenLoRAWrapper(
                    model=None,
                    tokenizer=None,
                    adapter_path=None,
                    rank=hp.rank,
                    fresh_init=True,
                )
            else:
                wrapper = load_qwen_with_adapters(
                    base_path=base_path,
                    adapter_path=adapter_path,
                    rank=hp.rank,
                )
            substrate = MicroKikiSubstrate(
                num_layers=20, rank=hp.rank, seed=seed,
            )
            substrate.load()  # honours DREAM_MICRO_KIKI_REAL env
            cell = _run_cell_real(
                arm=arm,
                seed=seed,
                splits=splits,
                wrapper=wrapper,
                substrate=substrate,
                hp=hp,
                adapter_keys=adapter_keys,
                out_dir=out_dir,
                smoke=smoke,
            )
            cell["run_id"] = registry.register(
                c_version=C_VERSION,
                profile=f"g6-studio-path-a/{arm}",
                seed=seed,
                commit_sha=commit_sha,
            )
            cells.append(cell)

    wall = time.time() - sweep_start
    retention = _retention_by_arm(cells)
    verdict = classify_h9(retention)
    payload: dict[str, Any] = {
        "date": DATE_TAG,
        "c_version": C_VERSION,
        "commit_sha": commit_sha,
        "smoke": smoke,
        "n_seeds": len(seeds),
        "arms": list(ARMS),
        "subdomains": list(subdomains),
        "fixture_path": str(fixture_path),
        "wall_time_s": wall,
        "cells": list(cells),
        "retention_by_arm": retention,
        "verdict": verdict,
        "hyperparams": {
            "lr": hp.lr,
            "iters": hp.iters,
            "rank": hp.rank,
            "alpha": hp.alpha,
            "batch_size": hp.batch_size,
            "n_train": n_train,
            "n_eval": n_eval,
        },
    }
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(
        json.dumps(payload, indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
    out_md.write_text(_render_md_report(payload), encoding="utf-8")
    return payload


def main(argv: Optional[list[str]] = None) -> int:
    """CLI entrypoint — parse args, dispatch to :func:`run_pilot`."""
    parser = argparse.ArgumentParser(
        description=(
            "G6-Studio Path A pilot — real LoRA SpikingKiki-V4 × "
            "MMLU CL stream. Studio-only at full scale ; M1 Max "
            "via --smoke."
        ),
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=5,
        help="Number of seeds (Option B = 5, Option A = 10, smoke = 1).",
    )
    parser.add_argument(
        "--smoke",
        action="store_true",
        help=(
            "Smoke path : skip the 58 GB model load, use stub "
            "generator + stub LoRA delta. Runs on M1 Max in seconds."
        ),
    )
    parser.add_argument(
        "--fixture-path",
        type=Path,
        default=DEFAULT_FIXTURE_PATH,
        help="MMLU JSONL fixture (default : synthetic G6 fixture).",
    )
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--iters", type=int, default=50)
    parser.add_argument("--rank", type=int, default=8)
    parser.add_argument("--alpha", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument(
        "--base-path",
        default=DEFAULT_BASE_PATH,
        help=(
            "Path to the SpikingKiki-V4 root (Studio default : "
            f"{DEFAULT_BASE_PATH})."
        ),
    )
    parser.add_argument(
        "--adapter-path",
        type=Path,
        default=DEFAULT_ADAPTER_PATH,
        help=(
            "Path to the V4 LoRA adapter directory (Studio default :"
            f" {DEFAULT_ADAPTER_PATH})."
        ),
    )
    parser.add_argument(
        "--out-json", type=Path, default=DEFAULT_OUT_JSON,
    )
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--registry-db", type=Path, default=DEFAULT_REGISTRY_DB,
    )
    parser.add_argument(
        "--smoke-subdomains",
        action="store_true",
        help=(
            "Restrict the CL stream to a single subdomain (anatomy) "
            "for quick smoke runs. Implied by --smoke."
        ),
    )
    args = parser.parse_args(argv)

    # Metal cache configuration : prevent OOM at 35B bf16 + LoRA on
    # Studio M3 Ultra (512 GB unified memory). Cache limit at 3x
    # model footprint (~210 GB) lets MLX retain allocations across
    # CL stream subdomains without thrashing ; memory limit at
    # 400 GB leaves 100+ GB headroom for OS / other processes.
    if not args.smoke:
        try:
            import mlx.core as mx
            mx.metal.set_memory_limit(400 * 1024**3)
            mx.metal.set_cache_limit(210 * 1024**3)
        except (ImportError, AttributeError):
            pass

    hp = TrainHyperparams(
        lr=args.lr,
        iters=args.iters,
        rank=args.rank,
        alpha=args.alpha,
        batch_size=args.batch_size,
    )
    seeds: tuple[int, ...] = tuple(range(int(args.n_seeds)))
    if not seeds:
        raise SystemExit("--n-seeds must be >= 1")
    use_smoke_subdomains = args.smoke or args.smoke_subdomains
    subdomains = (
        SMOKE_SUBDOMAINS if use_smoke_subdomains else DEFAULT_SUBDOMAINS
    )
    payload = run_pilot(
        fixture_path=args.fixture_path,
        out_json=args.out_json,
        out_md=args.out_md,
        registry_db=args.registry_db,
        seeds=seeds,
        n_train=args.n_train,
        n_eval=args.n_eval,
        hp=hp,
        adapter_keys=("layer_0_lora_B",),
        base_path=args.base_path,
        adapter_path=args.adapter_path,
        subdomains=subdomains,
        smoke=args.smoke,
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Cells : {len(payload['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
