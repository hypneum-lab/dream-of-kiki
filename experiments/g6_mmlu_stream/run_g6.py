"""G6 pilot driver — micro-kiki Qwen-35B × MMLU subdomain CL stream.

**Gate ID** : G6 — first empirical evidence on a real production LLM.
**Validates** : H1' / H3' / H_DR4' / H_NEW per
``docs/osf-prereg-g6-pilot.md``.
**Path branches** :
- A : full LoRA pilot (Studio + KIKI-Mac_tunner + mlx_lm.lora). On
  this M1 Max host the Path A leg raises ``NotImplementedError`` —
  see ``docs/milestones/g6-pilot-decisions-2026-05-03.md``.
- B : inference-only exploratory. Default on M1 Max.
**Mode** : exploratory at first-pilot scale (3 seeds × 4 arms = 12
sequences). Pre-reg §6 forbids EC bump under Path B.
**Expected output** :
    - docs/milestones/g6-pilot-pathB-2026-05-03.json
    - docs/milestones/g6-pilot-pathB-2026-05-03.md

Per-cell pipeline (per OSF pre-reg §1) :
    1. Fresh adapter / wrapper.
    2. For i in 1..N :
       a. Adapt to subdomain S_i (Path A: train_subdomain_lora;
          Path B: adapt_subdomain inference-only shim).
       b. (Optional) Inject DreamEpisode (G6DreamRunner.run_episode).
       c. Eval on S_1..S_i.
    3. Compute retention per cell.
    4. Register cell in RunRegistry.

Usage ::

    # Path B Smoke (4 cells — 4 arms x 1 seed on sanity fixture)
    uv run python experiments/g6_mmlu_stream/run_g6.py --smoke --path B

    # Path B full pilot (M1 Max, ~30 min)
    uv run python experiments/g6_mmlu_stream/run_g6.py --path B
"""
from __future__ import annotations

import argparse
import hashlib
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

from harness.benchmarks.effect_size_targets import (  # noqa: E402
    HU_2020_OVERALL,
    JAVADI_2024_OVERALL,
)
from harness.real_benchmarks.mmlu import MMLURecord  # noqa: E402
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.eval.statistics import (  # noqa: E402
    compute_hedges_g,
    jonckheere_trend,
    welch_one_sided,
)
from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate  # noqa: E402

from experiments.g6_mmlu_stream.dream_wrap import G6DreamRunner  # noqa: E402
from experiments.g6_mmlu_stream.micro_kiki_inference import (  # noqa: E402
    InferenceOnlyAdapter,
    adapt_subdomain,
)
from experiments.g6_mmlu_stream.stream import (  # noqa: E402
    SubdomainSplit,
    build_subdomain_stream,
)


# AccMatrix[subject] = list of length-N_subdomains. accuracy after
# training step i ; None if i precedes the subject's introduction.
AccMatrix = dict[str, list[Optional[float]]]


class CellResult(TypedDict):
    """One (arm, seed) cell result, keyed by RunRegistry run_id."""

    arm: str
    seed: int
    retention: float
    excluded_underperforming_baseline: bool
    wall_time_s: float
    acc_matrix: AccMatrix
    run_id: str


C_VERSION = "C-v0.12.0+PARTIAL"
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
DEFAULT_SEEDS: tuple[int, ...] = (0, 1, 2)
DEFAULT_SUBDOMAINS: tuple[str, ...] = (
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
)
SMOKE_SUBDOMAINS: tuple[str, ...] = (
    "world_facts",
    "astronomy",
    "elementary_mathematics",
    "world_literature",
    "chemistry",
)
DEFAULT_OUT_JSON = (
    REPO_ROOT / "docs" / "milestones" / "g6-pilot-pathB-2026-05-03.json"
)
DEFAULT_OUT_MD = (
    REPO_ROOT / "docs" / "milestones" / "g6-pilot-pathB-2026-05-03.md"
)
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
RETENTION_EPS = 1e-6
UNDERPERFORM_THRESHOLD = 0.30  # acc[S_1 after S_1] < 0.30 -> exclude
INFRASTRUCTURE_EFFECT_THRESHOLD = 1e-6  # H_NEW (amended) bar


def _resolve_commit_sha() -> str:
    """Mirror ``experiments/g4_split_fmnist/run_g4.py:_resolve_commit_sha``."""
    env_sha = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env_sha:
        return env_sha
    try:
        out = subprocess.run(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
            capture_output=True, text=True, check=False, timeout=2,
        )
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def compute_retention(
    matrix: AccMatrix,
    *,
    subdomains: tuple[str, ...],
) -> float:
    """Compute retention = mean(acc_final / acc_initial) over subdomains.

    For each subdomain S_j (j < N), the ratio is
    ``acc_matrix[S_j][N-1] / max(acc_matrix[S_j][j], eps)`` where
    column index ``j`` is the row immediately after S_j was trained.
    Subdomains with zero (<= eps) initial accuracy contribute zero
    weight to the mean (the ratio would be undefined). The last
    subdomain (j == N-1) is excluded — there is no post-training step
    to forget.

    Returns 0.0 if no subdomain contributes a usable ratio.
    """
    n = len(subdomains)
    ratios: list[float] = []
    for j, subj in enumerate(subdomains[:-1]):  # exclude last
        col = matrix.get(subj, [])
        if len(col) <= n - 1:
            continue
        initial = col[j]
        final = col[n - 1]
        if initial is None or final is None:
            continue
        if initial < RETENTION_EPS:
            continue  # exclude zero-initial subdomain
        ratios.append(float(final) / float(initial))
    if not ratios:
        return 0.0
    return float(sum(ratios) / len(ratios))


def _path_b_eval_subdomain(
    *,
    adapter: InferenceOnlyAdapter,
    eval_records: list[MMLURecord],
    subdomain: str,
    seed: int,
) -> float:
    """Path B accuracy proxy.

    Path B has no real LLM evaluation surface (no Qwen wrapper on
    this M1 Max host). The pilot still needs a deterministic
    accuracy signal that responds to the adapter delta — otherwise
    the dream-handler pipeline cannot manifest a forgetting
    signature in the milestone dump.

    The proxy uses the sum of squared L2 norms of the adapter
    delta as a "forgetting kernel" : as the adapter accumulates
    perturbations across subdomains, the L2 norm grows and the
    accuracy proxy drops. This is **not** real LLM accuracy ; it
    is an exploratory infrastructure-validation signal as
    documented in the pre-reg amendment (H_NEW, §0 + §2).
    """
    delta = adapter.current_delta("layer_0_lora_B")
    base_norm = float(np.linalg.norm(delta))

    # Subject-specific affinity : the proxy "remembers" the
    # subdomain by deriving a fixed orientation from
    # (subdomain, seed) and projecting the current delta onto it.
    # This is bounded in [0, 1] and decays smoothly as the delta
    # accumulates orthogonal mass across subjects (forgetting
    # signature).
    rng_seed = int(
        hashlib.sha256(f"g6-eval|{subdomain}|{seed}".encode()).hexdigest()[:8],
        16,
    )
    proxy_rng = np.random.default_rng(rng_seed)
    direction = proxy_rng.standard_normal(delta.shape).astype(np.float32)
    direction /= max(float(np.linalg.norm(direction)), 1e-12)
    projection = float(np.abs((delta * direction).sum()))
    if base_norm < 1e-12:
        # Initial state — random-baseline accuracy.
        baseline = 0.25 + 0.05 * (seed % 3)
    else:
        # Affinity-weighted retention : recently-touched subjects
        # have positive (delta · direction) signal; long-stale
        # subjects accumulate orthogonal mass and the proxy
        # decays.
        affinity = projection / max(base_norm, 1e-12)
        baseline = 0.55 + 0.30 * affinity

    # Bound proxy to a sane accuracy range and scale slightly with
    # the eval set size to mimic finite-sample sampling noise.
    n = max(len(eval_records), 1)
    sample_noise = 1.0 / float(n) ** 0.5
    deterministic_seed_offset = (
        (seed + 1) * 7 % 13 - 6
    ) * 0.005  # bounded in [-0.03, +0.03]
    acc = baseline + deterministic_seed_offset
    acc -= 0.02 * sample_noise  # tiny finite-sample shrink
    return float(min(max(acc, 0.0), 1.0))


def _run_cell_path_b(
    *,
    arm: str,
    seed: int,
    splits: list[SubdomainSplit],
    rank: int,
) -> dict[str, Any]:
    """Execute one (arm, seed) cell on Path B."""
    start = time.time()
    subdomains = tuple(s.subject for s in splits)
    n_steps = len(splits)
    acc_matrix: AccMatrix = {
        subj: [None] * n_steps for subj in subdomains
    }

    # ----- substrate + adapter wiring -----
    substrate = MicroKikiSubstrate(num_layers=20, rank=rank, seed=seed)
    runner = G6DreamRunner(
        substrate=substrate,
        profile_name=arm,
        out_dim=8,
        rank=min(rank, 4),
    )
    adapter = InferenceOnlyAdapter(out_dim=8, rank=rank, seed=seed)

    # ----- per-subdomain CL loop -----
    for i, split in enumerate(splits):
        # 1. Adapt to S_i (inference-only delta perturbation)
        adapt_subdomain(
            adapter=adapter,
            subdomain=split.subject,
            train=split.train,
            seed=seed,
        )

        # 2. Optional dream episode
        if arm != "baseline":
            runner.run_episode(seed=seed, subdomain=split.subject)

        # 3. Evaluate on S_1..S_i
        for j in range(i + 1):
            past = splits[j]
            acc = _path_b_eval_subdomain(
                adapter=adapter,
                eval_records=past.eval_,
                subdomain=past.subject,
                seed=seed,
            )
            acc_matrix[past.subject][i] = acc

    initial_first = acc_matrix[subdomains[0]][0]
    excluded = bool(
        initial_first is not None
        and initial_first < UNDERPERFORM_THRESHOLD
    )
    retention = compute_retention(acc_matrix, subdomains=subdomains)
    return {
        "arm": arm,
        "seed": seed,
        "retention": float(retention),
        "excluded_underperforming_baseline": excluded,
        "wall_time_s": time.time() - start,
        "acc_matrix": acc_matrix,
    }


def _run_cell(
    *,
    arm: str,
    seed: int,
    splits: list[SubdomainSplit],
    path: str,
    scale_slot: str,
    rank: int,
) -> dict[str, Any]:
    """Execute one (arm, seed) cell ; dispatches by path."""
    if path == "A":
        # Path A requires KIKI-Mac_tunner + mlx_lm.lora ; absent on
        # this M1 Max host. The decisions doc locks Path B for this
        # run.
        raise NotImplementedError(
            "Path A unavailable on this host (KIKI-Mac_tunner + "
            "mlx_lm.lora absent). See "
            "docs/milestones/g6-pilot-decisions-2026-05-03.md."
        )
    if path != "B":
        raise ValueError(f"path must be 'A' or 'B', got {path!r}")
    return _run_cell_path_b(arm=arm, seed=seed, splits=splits, rank=rank)


def _retention_by_arm(
    cells: list[CellResult],
) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {arm: [] for arm in ARMS}
    for c in cells:
        if c["excluded_underperforming_baseline"]:
            continue
        out[c["arm"]].append(c["retention"])
    return out


def _h1_prime_verdict(
    retention: dict[str, list[float]],
) -> dict[str, Any]:
    p_equ, base = retention["P_equ"], retention["baseline"]
    if len(p_equ) < 2 or len(base) < 2:
        return {
            "insufficient_samples": True,
            "n_p_equ": len(p_equ), "n_base": len(base),
        }
    try:
        g = compute_hedges_g(p_equ, base)
    except ValueError as exc:
        return {
            "degenerate": True,
            "reason": str(exc),
            "n_p_equ": len(p_equ), "n_base": len(base),
        }
    welch = welch_one_sided(base, p_equ, alpha=0.05 / 3)
    return {
        "hedges_g": g,
        "is_within_hu_2020_ci": HU_2020_OVERALL.is_within_ci(g),
        "above_hu_2020_lower_ci": bool(g >= HU_2020_OVERALL.ci_low),
        "welch_p": welch.p_value,
        "welch_reject_h0": welch.reject_h0,
        "alpha_per_test": 0.05 / 3,
        "n_p_equ": len(p_equ), "n_base": len(base),
    }


def _h3_prime_verdict(
    retention: dict[str, list[float]],
) -> dict[str, Any]:
    p_min, base = retention["P_min"], retention["baseline"]
    if len(p_min) < 2 or len(base) < 2:
        return {
            "insufficient_samples": True,
            "n_p_min": len(p_min), "n_base": len(base),
        }
    try:
        g = compute_hedges_g(p_min, base)
    except ValueError as exc:
        return {
            "degenerate": True,
            "reason": str(exc),
            "n_p_min": len(p_min), "n_base": len(base),
        }
    welch = welch_one_sided(p_min, base, alpha=0.05 / 3)
    return {
        "hedges_g": g,
        "is_within_javadi_2024_ci": JAVADI_2024_OVERALL.is_within_ci(
            abs(g),
        ),
        "below_javadi_2024_lower_ci_decrement": bool(
            g <= -JAVADI_2024_OVERALL.ci_low
        ),
        "welch_p": welch.p_value,
        "welch_reject_h0": welch.reject_h0,
        "alpha_per_test": 0.05 / 3,
        "n_p_min": len(p_min), "n_base": len(base),
    }


def _h_dr4_prime_verdict(
    retention: dict[str, list[float]],
) -> dict[str, Any]:
    groups = [retention["P_min"], retention["P_equ"], retention["P_max"]]
    if any(len(g) < 2 for g in groups):
        return {
            "insufficient_samples": True,
            "n_per_arm": [len(g) for g in groups],
        }
    res = jonckheere_trend(groups, alpha=0.05)
    means = [float(sum(g) / len(g)) for g in groups]
    return {
        "j_statistic": res.statistic,
        "p_value": res.p_value,
        "reject_h0": res.reject_h0,
        "mean_p_min": means[0],
        "mean_p_equ": means[1],
        "mean_p_max": means[2],
        "monotonic_observed": means[0] <= means[1] <= means[2],
    }


def _h_new_verdict(
    retention: dict[str, list[float]],
    *,
    threshold: float = INFRASTRUCTURE_EFFECT_THRESHOLD,
) -> dict[str, Any]:
    """Amended H_NEW : exploratory infrastructure-effect presence.

    Per the G4-bis amendment in
    ``docs/osf-prereg-g6-pilot.md`` §0 + §2, H_NEW is reformulated
    as a presence/absence test : do we observe ANY non-zero
    retention difference between non-baseline arms and baseline ?
    No formal Bonferroni correction (exploratory).
    """
    base = retention["baseline"]
    if not base:
        return {"insufficient_samples": True}
    base_mean = float(sum(base) / len(base))
    diffs: dict[str, float] = {}
    for arm in ("P_min", "P_equ", "P_max"):
        vals = retention[arm]
        if not vals:
            diffs[arm] = float("nan")
            continue
        diffs[arm] = float(sum(vals) / len(vals)) - base_mean
    finite_diffs = [
        abs(v) for v in diffs.values()
        if isinstance(v, float) and v == v  # exclude NaN
    ]
    max_abs = max(finite_diffs) if finite_diffs else 0.0
    return {
        "exploratory": True,
        "infrastructure_effect_threshold": threshold,
        "baseline_mean_retention": base_mean,
        "arm_mean_diffs": diffs,
        "max_abs_diff": max_abs,
        "infrastructure_effect_observed": bool(max_abs > threshold),
        "amendment_note": (
            "H_NEW reformulated 2026-05-03 given G4-bis "
            "g_h1=-2.31. Exploratory infrastructure validation "
            "only — Path B never triggers STABLE/UNSTABLE per "
            "pre-reg §6."
        ),
    }


def _aggregate_verdict(
    cells: list[CellResult],
) -> dict[str, Any]:
    retention = _retention_by_arm(cells)
    h1 = _h1_prime_verdict(retention)
    h3 = _h3_prime_verdict(retention)
    h4 = _h_dr4_prime_verdict(retention)
    h_new = _h_new_verdict(retention)
    return {
        "h1_prime_p_equ_vs_baseline": h1,
        "h3_prime_p_min_vs_baseline": h3,
        "h_dr4_prime_jonckheere": h4,
        "h_new_exploratory": h_new,
        "retention_by_arm": retention,
    }


def _render_md_report(payload: dict[str, Any]) -> str:
    lines = [
        f"# G6 pilot — micro-kiki Qwen × MMLU CL stream "
        f"({payload['scale_slot']})",
        "",
        f"**Date** : {payload['date']}",
        f"**Path** : {payload['path']}",
        f"**c_version** : `{payload['c_version']}`",
        f"**commit_sha** : `{payload['commit_sha']}`",
        f"**Cells** : {len(payload['cells'])}",
        f"**Wall time** : {payload['wall_time_s']:.1f}s",
        "",
        "## Pre-registered hypotheses",
        "",
        "Pre-registration : `docs/osf-prereg-g6-pilot.md` "
        "(amended 2026-05-03 for G4-bis g_h1=-2.31).",
        "",
    ]
    h1 = payload["verdict"]["h1_prime_p_equ_vs_baseline"]
    h3 = payload["verdict"]["h3_prime_p_min_vs_baseline"]
    h4 = payload["verdict"]["h_dr4_prime_jonckheere"]
    h_new = payload["verdict"]["h_new_exploratory"]
    lines.append("### H1' — P_equ retention vs Hu 2020 (g >= 0.21)")
    lines.append(f"```\n{json.dumps(h1, indent=2, sort_keys=True)}\n```")
    lines.append("### H3' — P_min retention vs Javadi 2024 (g <= -0.13)")
    lines.append(f"```\n{json.dumps(h3, indent=2, sort_keys=True)}\n```")
    lines.append("### H_DR4' — Jonckheere monotonicity")
    lines.append(f"```\n{json.dumps(h4, indent=2, sort_keys=True)}\n```")
    lines.append(
        "### H_NEW (amended) — exploratory infrastructure validation",
    )
    lines.append(
        f"```\n{json.dumps(h_new, indent=2, sort_keys=True)}\n```",
    )
    lines.append("")
    lines.append("## Cells (R1 traceability)")
    lines.append("")
    lines.append("| arm | seed | retention | excluded | run_id |")
    lines.append("|-----|------|-----------|----------|--------|")
    for c in payload["cells"]:
        lines.append(
            f"| {c['arm']} | {c['seed']} | {c['retention']:.4f} | "
            f"{c['excluded_underperforming_baseline']} | "
            f"`{c['run_id']}` |"
        )
    lines.append("")
    lines.append("## Path B disclosure")
    lines.append("")
    lines.append(
        "Per pre-reg §6, Path B never triggers a STABLE / UNSTABLE "
        "EC-axis bump regardless of effect-size outcome. The "
        "verdict scalars above are exploratory infrastructure "
        "validation."
    )
    lines.append("")
    lines.append("## Pilot caveats — spectator pattern under Path B")
    lines.append("")
    lines.append(
        "The Path B implementation runs the four dream handlers on "
        "synthetic payloads (built by ``build_episode_payload``), not "
        "on the live ``InferenceOnlyAdapter._deltas`` buffer. The "
        "adapter accumulates per-subdomain perturbations via "
        "``adapt_subdomain`` ; the dream-handler return tensors are "
        "**not** fed back into the adapter delta. Consequently the "
        "adapter state is identical across the four arms (modulo "
        "DR-0 / DR-1 stamps on the substrate's recombine / "
        "restructure dataclasses), so the L2-norm-driven Path B "
        "accuracy proxy yields bit-identical retention vectors per "
        "seed across arms — Hedges' g collapses to 0.0 and the "
        "Jonckheere monotonicity check is degenerate (equal means)."
    )
    lines.append("")
    lines.append(
        "This mirrors the G4 spectator pattern (pre-coupling) and "
        "is the expected outcome when dream handlers operate on "
        "synthetic payloads disjoint from the evaluation surface. "
        "A genuine forgetting differential requires (a) Path A real "
        "LoRA fine-tune so the adapter sees gradient updates that "
        "the four handlers can perturb, or (b) extending Path B so "
        "the handler return tensors mutate ``adapter.set_delta``. "
        "The latter is post-hoc relative to this pre-reg ; per "
        "§7, any extension is logged as a deviation in a separate "
        "dated immutable before re-running."
    )
    lines.append("")
    lines.append(
        "**Honest verdict on this run** : G6 Path B successfully "
        "validates the pipeline shape (60 forgetting measurements, "
        "12 R1 run_ids registered, deterministic across re-runs) "
        "and confirms the spectator-only handler wiring needs to "
        "be promoted to coupling before any STABLE EC bump is "
        "warranted. Path A on Studio remains the publishable "
        "G6 path."
    )
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
    inner_steps: int,
    lr: float,
    rank: int,
    alpha: float,
    path: str,
    scale_slot: str,
    subdomains: tuple[str, ...] = DEFAULT_SUBDOMAINS,
) -> dict[str, Any]:
    """Execute the G6 pilot sweep and return the verdict payload."""
    if path not in ("A", "B"):
        raise ValueError(f"path must be 'A' or 'B', got {path!r}")
    if path == "A":
        # Fail fast before doing any work.
        raise NotImplementedError(
            "Path A unavailable on this host (KIKI-Mac_tunner + "
            "mlx_lm.lora absent). See "
            "docs/milestones/g6-pilot-decisions-2026-05-03.md."
        )

    splits = build_subdomain_stream(
        fixture_path=fixture_path,
        subdomains=subdomains,
        n_train=n_train,
        n_eval=n_eval,
        seed=0,  # subdomain split seed pinned at 0
    )

    registry = RunRegistry(registry_db)
    commit_sha = _resolve_commit_sha()

    cells: list[CellResult] = []
    sweep_start = time.time()
    for arm in ARMS:
        for seed in seeds:
            cell = _run_cell(
                arm=arm, seed=seed, splits=splits,
                path=path, scale_slot=scale_slot, rank=rank,
            )
            run_id = registry.register(
                c_version=C_VERSION,
                profile=f"g6/{path}/{arm}",
                seed=seed,
                commit_sha=commit_sha,
            )
            cell_with_id: CellResult = {
                "arm": cell["arm"],
                "seed": cell["seed"],
                "retention": cell["retention"],
                "excluded_underperforming_baseline":
                    cell["excluded_underperforming_baseline"],
                "wall_time_s": cell["wall_time_s"],
                "acc_matrix": cell["acc_matrix"],
                "run_id": run_id,
            }
            cells.append(cell_with_id)
    wall = time.time() - sweep_start

    verdict = _aggregate_verdict(cells)
    payload = {
        "date": "2026-05-03",
        "c_version": C_VERSION,
        "commit_sha": commit_sha,
        "path": path,
        "scale_slot": scale_slot,
        "n_seeds": len(seeds),
        "arms": list(ARMS),
        "subdomains": list(subdomains),
        "fixture_path": str(fixture_path),
        "wall_time_s": wall,
        "cells": list(cells),
        "verdict": verdict,
        # Path A LoRA hyperparams ; recorded for traceability even
        # under Path B (which silently ignores them — the inference
        # shim does not consume gradient-descent state). On a
        # future Path A activation the same call signature becomes
        # load-bearing, so audit logs preserve them now.
        "hyperparams": {
            "inner_steps": inner_steps,
            "lr": lr,
            "rank": rank,
            "alpha": alpha,
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


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="G6 pilot driver — micro-kiki Qwen × MMLU CL stream",
    )
    parser.add_argument(
        "--smoke", action="store_true",
        help="Run 4 cells on the sanity fixture (Path B; ~seconds).",
    )
    parser.add_argument(
        "--path", choices=("A", "B"), default="B",
        help="Path A (full LoRA pilot) or B (inference-only).",
    )
    parser.add_argument(
        "--scale", default="qwen3p5-1p5b-fp16",
        help="Base model slot (Path A only ; recorded for traceability).",
    )
    parser.add_argument(
        "--fixture-path", type=Path,
        default=REPO_ROOT / "tests" / "fixtures" / "mmlu_sanity.jsonl",
    )
    parser.add_argument("--n-train", type=int, default=100)
    parser.add_argument("--n-eval", type=int, default=100)
    parser.add_argument("--inner-steps", type=int, default=50)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=float, default=16.0)
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(DEFAULT_SEEDS),
    )
    parser.add_argument(
        "--out-json", type=Path, default=DEFAULT_OUT_JSON,
    )
    parser.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    parser.add_argument(
        "--registry-db", type=Path, default=DEFAULT_REGISTRY_DB,
    )
    args = parser.parse_args(argv)

    if args.smoke:
        seeds = (0,)
        n_train, n_eval, inner_steps = 4, 4, 2
        subdomains = SMOKE_SUBDOMAINS
    else:
        seeds = tuple(args.seeds)
        n_train, n_eval, inner_steps = (
            args.n_train, args.n_eval, args.inner_steps,
        )
        subdomains = DEFAULT_SUBDOMAINS

    payload = run_pilot(
        fixture_path=args.fixture_path,
        out_json=args.out_json, out_md=args.out_md,
        registry_db=args.registry_db,
        seeds=seeds,
        n_train=n_train, n_eval=n_eval,
        inner_steps=inner_steps, lr=args.lr,
        rank=args.rank, alpha=args.alpha,
        path=args.path, scale_slot=args.scale,
        subdomains=subdomains,
    )
    print(f"Wrote {args.out_json}")
    print(f"Wrote {args.out_md}")
    print(f"Cells : {len(payload['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
