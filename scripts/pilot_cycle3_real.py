"""Cycle-3 real ablation pilot — C3.8 Phase A (fresh FP16 + real evals).

**Gate ID** : G10 cycle-3 Gate D real pilot
**Validates** : whether the composite MMLU + HellaSwag + mega-v2 score
  on unquantized Qwen2.5-1.5B-Instruct-bf16 responds to the dream-op
  profiles ``{p_min, p_equ, p_max}`` when dream ops mutate genuine
  backprop-capable bf16 weights (not a 4-dim adapter proxy).
**Mode** : empirical claim at small scale — pipeline-validation
  at full launch scale. Phase A establishes the smoke-cell wall-time
  baseline + one-profile signal so Phase B (C3.8 full 3-scale × 60
  seeds × real benchmarks) can commit ~18 h of Studio compute.
**Expected output** : go/no-go verdict JSON under
  ``docs/milestones/pilot-cycle3-real-1p5b.json``.

Per-cell pipeline (per user spec) :

  1. Load a FRESH Qwen bf16 1.5B wrapper (weights must not leak
     between cells — dream mutates them in-place).
  2. Pre-eval : MMLU (100) + HellaSwag (100) + mega-v2 (100) seeded.
     Composite ``pre_score = mean(mmlu_acc, hs_acc, mv2_acc)``.
  3. Dream run : 5 episodes per profile ; dream ops target the
     real bf16 weights directly (not a 4-dim adapter).
  4. Post-eval : same benchmarks, same seeds → ``post_score``.
  5. ``delta = post - pre``.
  6. Register cell in ``RunRegistry`` under the composite
     ``profile_tag = cycle3/qwen3p5-1p5b-fp16/<profile>/mlx_kiki_oniric``.

After all cells : paired t-test pre vs post per profile. GO rule
(cycle-1 bar, family size 3) : H1 rejected in ≥ 2/3 profiles at
α = 0.0125.

Smoke-cell mode (``--smoke-cell``) runs a single MMLU eval on a
small ``n_samples`` so the whole trip (load + pre-eval + dream +
post-eval) validates the wiring in ~2 min. This is the gate Phase A
must pass before Phase B commits the full matrix.

Design decision (user-approved κ.A) : use Qwen bf16 unquantized so
gradient flows natively through the weights. 1.5B × 2 bytes ≈ 3 GB
so any Apple Silicon host fits the wrapper.

Usage ::

    # Tiny smoke cell — 1 profile × 1 seed × 1 benchmark
    uv run python scripts/pilot_cycle3_real.py --smoke-cell \
        --benchmark=mmlu --n-samples=20

    # Full Phase A pilot (30 seeds × 3 profiles × 3 benchmarks)
    uv run python scripts/pilot_cycle3_real.py

Reference :
  docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md §C3.8
  docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md §5
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ablation_cycle3 import (  # noqa: E402
    HARNESS_VERSION,
    PROFILES,
    _resolve_commit_sha,
)

REAL_SCALE_FP16 = "qwen3p5-1p5b-fp16"
REAL_SEEDS_DEFAULT = tuple(range(30))
REAL_BENCHMARKS = ("mmlu", "hellaswag", "mega_v2")
GO_BONFERRONI_ALPHA = 0.0125
GO_PROFILES_REJECTED_MIN = 2

N_EPISODES_PER_PROFILE = 5
DEFAULT_N_SAMPLES = 100


# -------------------------------------------------------------------
# CLI parsing
# -------------------------------------------------------------------


def _parse_cli(argv: list[str]) -> argparse.Namespace:
    """Parse CLI flags : smoke-cell + full-run + benchmark filter."""
    parser = argparse.ArgumentParser(
        description="Cycle-3 C3.8 Phase A real ablation pilot "
        "(Qwen FP16 + MMLU + HellaSwag + mega-v2).",
    )
    parser.add_argument(
        "--smoke-cell",
        action="store_true",
        help="Run exactly 1 cell (p_min, seed 0, one benchmark) "
        "to validate the pipeline end-to-end.",
    )
    parser.add_argument(
        "--benchmark",
        choices=REAL_BENCHMARKS,
        default="mmlu",
        help="Benchmark used in --smoke-cell mode. Ignored by full run.",
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=DEFAULT_N_SAMPLES,
        help="Number of eval records per benchmark.",
    )
    parser.add_argument(
        "--n-seeds",
        type=int,
        default=len(REAL_SEEDS_DEFAULT),
        help="Number of seeds in the full pilot.",
    )
    parser.add_argument(
        "--profile",
        choices=("p_min", "p_equ", "p_max"),
        default="p_min",
        help="Profile used in --smoke-cell mode. Ignored by full run.",
    )
    return parser.parse_args(argv)


# -------------------------------------------------------------------
# Model loading (fresh per cell so weights do not leak)
# -------------------------------------------------------------------


def _load_fresh_fp16_wrapper():
    """Load a FRESH bf16 Qwen 1.5B wrapper.

    Fails fast with a clear message if ``mlx-lm`` is missing or the
    HF cache has no bf16 weights — Phase A cannot proceed without
    a gradient-bearing model.
    """
    from harness.real_models.qwen_mlx_fp16 import load_qwen_fp16

    return load_qwen_fp16(REAL_SCALE_FP16)


def _seed_everything(seed: int) -> None:
    import mlx.core as mx
    import numpy as np

    mx.random.seed(seed)
    np.random.seed(seed)


# -------------------------------------------------------------------
# Per-cell pipeline
# -------------------------------------------------------------------


def _evaluate_benchmarks(
    wrapper,
    *,
    benchmarks: tuple[str, ...],
    n_samples: int,
    seed: int,
) -> dict[str, dict[str, float]]:
    """Run each benchmark and return ``{benchmark -> {accuracy, n}}``."""
    from harness.real_benchmarks.hellaswag import evaluate_hellaswag
    from harness.real_benchmarks.mega_v2_eval import evaluate_mega_v2
    from harness.real_benchmarks.mmlu import evaluate_mmlu

    tokenizer = wrapper.tokenizer
    results: dict[str, dict[str, float]] = {}
    if "mmlu" in benchmarks:
        results["mmlu"] = evaluate_mmlu(
            wrapper, tokenizer, n_samples=n_samples, seed=seed
        )
    if "hellaswag" in benchmarks:
        results["hellaswag"] = evaluate_hellaswag(
            wrapper, tokenizer, n_samples=n_samples, seed=seed
        )
    if "mega_v2" in benchmarks:
        results["mega_v2"] = evaluate_mega_v2(
            wrapper, tokenizer, n_samples=n_samples, seed=seed
        )
    return results


def _composite_score(
    eval_results: dict[str, dict[str, float]]
) -> float:
    """Mean of ``accuracy`` across all evaluated benchmarks."""
    accs = [
        float(v["accuracy"])
        for v in eval_results.values()
        if "accuracy" in v
    ]
    return sum(accs) / max(len(accs), 1)


def _dream_episodes(
    wrapper,
    profile_name: str,
    *,
    seed: int,
    n_episodes: int,
) -> None:
    """Run ``n_episodes`` dream episodes that genuinely mutate Qwen FP16.

    Phase B wiring : the four real-weight ops now target the wrapped
    Qwen ``nn.Module`` directly (no decoupled 4-dim adapter). SGD +
    layer swap operate on the transformer stack so pre/post logits
    genuinely diverge.

    Per-profile sequence :

    - ``p_min`` — 3 episodes × ``[replay_real]``. SGD cross-entropy
      on mega-v2 ``context → expected`` pairs (lr=1e-4, 5 inner
      steps per episode) mutates every Qwen parameter tensor.
    - ``p_equ`` — 2 episodes × ``[replay_real, downscale_real,
      recombine_real]``. ``downscale_real`` no-ops on
      ``TransformerBlock`` (no direct ``.weight`` / ``.bias``) but
      still tags K1 ; its contribution is the SGD + optional
      structural channel activation for profile differentiation.
      ``recombine_real`` runs a local VAE-light over a small
      encoder/decoder pair so its K1 tag + guards fire without
      feeding the Qwen forward pass.
    - ``p_max`` — 2 episodes × ``[replay_real, downscale_real,
      restructure_real, recombine_real]``. ``restructure_real``
      swaps two adjacent transformer blocks
      (``i = seed % (n_layers - 1)``) — a NAS-style perturbation
      that produces measurable logit shifts.

    Reference : ``scripts/pilot_cycle3_real.py`` C3.8 Phase B.
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim
    import numpy as np

    from kiki_oniric.dream.episode import (
        BudgetCap,
        DreamEpisode,
        EpisodeTrigger,
        Operation,
        OutputChannel,
    )
    from kiki_oniric.dream.operations.downscale_real import (
        DownscaleRealState,
        downscale_real_handler,
    )
    from kiki_oniric.dream.operations.recombine_real import (
        RecombineRealState,
        recombine_real_handler,
    )
    from kiki_oniric.dream.operations.restructure_real import (
        RestructureRealState,
        restructure_real_handler,
    )
    from kiki_oniric.dream.runtime import DreamRuntime

    from harness.real_benchmarks.mega_v2_eval import (
        _load_mega_v2_records,
    )

    qwen = wrapper.model
    tokenizer = wrapper.tokenizer
    n_layers = len(qwen.layers)

    # ---------------------------------------------------------------
    # replay_real : custom CE-loss SGD adapter.
    # ``replay_real_handler`` is a generic MSE-on-forward SGD. For a
    # Qwen LM the natural loss is cross-entropy on next-token logits,
    # so we register a custom handler that mirrors the replay_real
    # contract (K1 tag + S1 no-op on empty records) but feeds the
    # Qwen token-ID forward pass. Every ``mx.nn.value_and_grad`` step
    # mutates every Qwen parameter tensor.
    # ---------------------------------------------------------------

    replay_lr = 1e-4
    replay_inner_steps = 5
    replay_optimizer = optim.SGD(learning_rate=replay_lr)

    def _ce_loss(qwen_inner, input_ids, target_ids):
        # input_ids : (1, L) ; target_ids : (1, L) shifted by one.
        logits = qwen_inner(input_ids)
        # Widen to fp32 for stable logsumexp (bf16 saturates easily
        # on long vocab rows). mlx.nn.losses.cross_entropy handles
        # reduction + log-softmax internally.
        logits_fp32 = logits.astype(mx.float32)
        loss = nn.losses.cross_entropy(
            logits_fp32.reshape(-1, logits_fp32.shape[-1]),
            target_ids.reshape(-1),
            reduction="mean",
        )
        return loss

    replay_grad_fn = nn.value_and_grad(qwen, _ce_loss)

    def _replay_qwen_handler(episode: DreamEpisode) -> None:
        records = episode.input_slice.get("beta_records", [])
        if not records:
            return
        for record in records:
            ctx_ids = record["x"]
            cont_ids = record["y"]
            if not ctx_ids or not cont_ids:
                continue
            # Build (input, target) : predict continuation given
            # the (context + continuation[:-1]) prefix. Target is
            # the continuation tokens aligned to each prefix step.
            full_ids = list(ctx_ids) + list(cont_ids)
            if len(full_ids) < 2:
                continue
            input_ids_arr = mx.array([full_ids[:-1]])
            target_ids_arr = mx.array([full_ids[1:]])
            for _ in range(replay_inner_steps):
                _, grads = replay_grad_fn(
                    qwen, input_ids_arr, target_ids_arr
                )
                replay_optimizer.update(qwen, grads)
                mx.eval(qwen.parameters())

    downscale_state = DownscaleRealState()
    restructure_state = RestructureRealState()
    recombine_state = RecombineRealState()

    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, _replay_qwen_handler)
    runtime.register_handler(
        Operation.DOWNSCALE,
        downscale_real_handler(downscale_state, model=qwen),
    )
    if profile_name in ("p_equ", "p_max"):
        # Small local encoder/decoder — recombine_real's latent
        # sample doesn't feed back into Qwen for Phase B ; the op
        # still exercises S2 + K1 and activates the LATENT_SAMPLE
        # channel so p_equ / p_max trajectories diverge from p_min.
        class _Encoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def __call__(self, x):
                h = self.fc(x)
                return h, h * 0.0

        class _Decoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc = nn.Linear(4, 4)

            def __call__(self, x):
                return self.fc(x)

        encoder = _Encoder()
        decoder = _Decoder()
        runtime.register_handler(
            Operation.RECOMBINE,
            recombine_real_handler(
                recombine_state,
                encoder=encoder,
                decoder=decoder,
                seed=seed,
            ),
        )
    if profile_name == "p_max":
        runtime.register_handler(
            Operation.RESTRUCTURE,
            restructure_real_handler(restructure_state, model=qwen),
        )

    # Profile → (ops, channels, n_episodes).
    if profile_name == "p_min":
        ops: tuple[Operation, ...] = (Operation.REPLAY,)
        channels: tuple[OutputChannel, ...] = (
            OutputChannel.WEIGHT_DELTA,
        )
        profile_episodes = 3
    elif profile_name == "p_equ":
        ops = (
            Operation.REPLAY,
            Operation.DOWNSCALE,
            Operation.RECOMBINE,
        )
        channels = (
            OutputChannel.WEIGHT_DELTA,
            OutputChannel.LATENT_SAMPLE,
        )
        profile_episodes = 2
    elif profile_name == "p_max":
        ops = (
            Operation.REPLAY,
            Operation.DOWNSCALE,
            Operation.RESTRUCTURE,
            Operation.RECOMBINE,
        )
        channels = (
            OutputChannel.WEIGHT_DELTA,
            OutputChannel.HIERARCHY_CHG,
            OutputChannel.LATENT_SAMPLE,
        )
        profile_episodes = 2
    else:
        raise ValueError(f"unknown profile : {profile_name!r}")

    # Let the caller-provided n_episodes narrow the schedule (e.g.
    # for a fast smoke cell) while keeping the per-profile cap.
    effective_episodes = min(profile_episodes, n_episodes)

    # Deterministic adjacent-block swap per seed — restructure_real
    # short-circuits below ``n_layers - 1`` so this picks a valid
    # index pair for every Qwen scale slot.
    swap_i = seed % max(n_layers - 1, 1)
    swap_indices = [swap_i, swap_i + 1]

    # Draw mega-v2 records once for the whole dream run — deterministic
    # under ``seed``. Each episode consumes a 4-record batch.
    batch_size = 4
    total_records_needed = batch_size * max(effective_episodes, 1)
    mega_records = _load_mega_v2_records(
        n_samples=total_records_needed, seed=seed
    )

    def _tokenize_record(record) -> tuple[list[int], list[int]]:
        try:
            ctx = tokenizer.encode(record.context)
        except TypeError:
            ctx = tokenizer.encode(
                record.context, add_special_tokens=True
            )
        try:
            cont = tokenizer.encode(record.expected)
        except TypeError:
            cont = tokenizer.encode(
                record.expected, add_special_tokens=False
            )
        if not cont:
            cont = [0]
        return list(ctx), list(cont)

    for ep_idx in range(effective_episodes):
        rng = np.random.default_rng(seed + ep_idx)
        batch_start = ep_idx * batch_size
        batch = mega_records[batch_start : batch_start + batch_size]
        beta_records = []
        for record in batch:
            ctx_ids, cont_ids = _tokenize_record(record)
            beta_records.append({"x": ctx_ids, "y": cont_ids})

        latents = [rng.standard_normal(4).tolist() for _ in range(2)]
        runtime.execute(
            DreamEpisode(
                trigger=EpisodeTrigger.SCHEDULED,
                input_slice={
                    "beta_records": beta_records,
                    "shrink_factor": 0.995,
                    "topo_op": "reroute",
                    "swap_indices": swap_indices,
                    "delta_latents": latents,
                },
                operation_set=ops,
                output_channels=channels,
                budget=BudgetCap(
                    flops=10_000_000_000,
                    wall_time_s=600.0,
                    energy_j=100.0,
                ),
                episode_id=(
                    f"real-{profile_name}-ep{ep_idx}-seed{seed}"
                ),
            )
        )


def _run_cell(
    profile_name: str,
    seed: int,
    *,
    benchmarks: tuple[str, ...],
    n_samples: int,
    n_episodes: int,
) -> dict:
    """Execute one cell end-to-end : load + pre-eval + dream + post-eval.

    Returns ``{profile, seed, pre, post, delta, wall_time_s,
    pre_results, post_results, error}``. A fresh wrapper is loaded
    per-cell so dream mutations do not leak between cells.
    """
    start = time.time()
    _seed_everything(seed)
    cell: dict = {
        "profile": profile_name,
        "seed": seed,
        "benchmarks": list(benchmarks),
        "n_samples": n_samples,
    }
    try:
        wrapper = _load_fresh_fp16_wrapper()
    except Exception as exc:
        cell["error"] = f"{type(exc).__name__}: {exc}"
        cell["wall_time_s"] = time.time() - start
        return cell

    try:
        pre = _evaluate_benchmarks(
            wrapper,
            benchmarks=benchmarks,
            n_samples=n_samples,
            seed=seed,
        )
        pre_score = _composite_score(pre)
        _dream_episodes(
            wrapper, profile_name, seed=seed, n_episodes=n_episodes
        )
        post = _evaluate_benchmarks(
            wrapper,
            benchmarks=benchmarks,
            n_samples=n_samples,
            seed=seed,
        )
        post_score = _composite_score(post)
    except Exception as exc:
        traceback.print_exc()
        cell["error"] = f"{type(exc).__name__}: {exc}"
        cell["wall_time_s"] = time.time() - start
        return cell

    cell.update(
        {
            "pre": pre_score,
            "post": post_score,
            "delta": post_score - pre_score,
            "pre_results": pre,
            "post_results": post,
            "wall_time_s": time.time() - start,
            "model": REAL_SCALE_FP16,
        }
    )
    return cell


# -------------------------------------------------------------------
# H1 test across cells
# -------------------------------------------------------------------


def _h1_test(cells: list[dict]) -> dict:
    """Paired t-test pre vs post per profile. Skips cells with errors."""
    from scipy import stats

    out: dict[str, dict] = {}
    for profile in PROFILES:
        pres = [
            c["pre"]
            for c in cells
            if c.get("profile") == profile and "pre" in c
        ]
        posts = [
            c["post"]
            for c in cells
            if c.get("profile") == profile and "post" in c
        ]
        if len(pres) < 2 or len(posts) < 2:
            out[profile] = {
                "t": None,
                "p": None,
                "reject_h0": False,
                "n": len(pres),
                "note": "insufficient samples",
            }
            continue
        try:
            res = stats.ttest_rel(posts, pres)
            p = float(res.pvalue)
            out[profile] = {
                "t": float(res.statistic),
                "p": p,
                "reject_h0": bool(p < GO_BONFERRONI_ALPHA),
                "n": len(pres),
            }
        except Exception as exc:  # pragma: no cover - defensive
            out[profile] = {
                "t": None,
                "p": None,
                "reject_h0": False,
                "n": len(pres),
                "note": f"{type(exc).__name__}: {exc}",
            }
    return out


# -------------------------------------------------------------------
# Run registry
# -------------------------------------------------------------------


def _register_cell(profile_name: str, seed: int) -> str:
    from harness.storage.run_registry import RunRegistry

    registry = RunRegistry(REPO_ROOT / ".run_registry.sqlite")
    profile_tag = (
        f"cycle3/{REAL_SCALE_FP16}/{profile_name}/mlx_kiki_oniric"
    )
    return registry.register(
        c_version=HARNESS_VERSION,
        profile=profile_tag,
        seed=seed,
        commit_sha=_resolve_commit_sha(),
    )


# -------------------------------------------------------------------
# Main
# -------------------------------------------------------------------


def _print_banner(total_cells: int, n_samples: int) -> None:
    print("=" * 64)
    print("CYCLE-3 C3.8 PHASE A REAL ABLATION PILOT")
    print("=" * 64)
    print(f"harness_version : {HARNESS_VERSION}")
    print(f"scale           : {REAL_SCALE_FP16}")
    print(f"profiles        : {PROFILES}")
    print(f"benchmarks      : {REAL_BENCHMARKS}")
    print(f"n_samples/bench : {n_samples}")
    print(f"cells           : {total_cells}")
    print("-" * 64)


def main(argv: list[str] | None = None) -> int:
    """Phase A entrypoint — smoke-cell or full 90-cell run."""
    args = _parse_cli(list(argv) if argv is not None else sys.argv[1:])
    out_dir = REPO_ROOT / "docs" / "milestones"
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.smoke_cell:
        _print_banner(total_cells=1, n_samples=args.n_samples)
        benchmarks = (args.benchmark,)
        start = time.time()
        cell = _run_cell(
            args.profile,
            0,
            benchmarks=benchmarks,
            n_samples=args.n_samples,
            n_episodes=N_EPISODES_PER_PROFILE,
        )
        wall = time.time() - start
        if "error" in cell:
            print(f"[smoke-cell] FAILED : {cell['error']}")
            smoke_path = out_dir / "pilot-cycle3-real-1p5b-smoke.json"
            with smoke_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "harness_version": HARNESS_VERSION,
                        "mode": "smoke-cell",
                        "benchmark": args.benchmark,
                        "n_samples": args.n_samples,
                        "wall_time_s": wall,
                        "model": REAL_SCALE_FP16,
                        "cell": cell,
                        "status": "FAILED",
                    },
                    fh,
                    indent=2,
                )
            return 1
        try:
            run_id = _register_cell(args.profile, 0)
        except Exception as exc:  # pragma: no cover - defensive
            run_id = f"register-failed:{exc}"
        pre_acc = cell["pre"]
        post_acc = cell["post"]
        print(
            f"[smoke-cell] pre_acc={pre_acc:.4f} "
            f"post_acc={post_acc:.4f} "
            f"delta={cell['delta']:+.4f} "
            f"wall={wall:.2f}s model={REAL_SCALE_FP16} "
            f"benchmark={args.benchmark} run_id={run_id}"
        )
        smoke_path = out_dir / "pilot-cycle3-real-1p5b-smoke.json"
        with smoke_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "harness_version": HARNESS_VERSION,
                    "mode": "smoke-cell",
                    "benchmark": args.benchmark,
                    "n_samples": args.n_samples,
                    "wall_time_s": wall,
                    "model": REAL_SCALE_FP16,
                    "cell": {**cell, "run_id": run_id},
                    "status": "OK",
                },
                fh,
                indent=2,
            )
        print(f"[smoke-cell] dump written to {smoke_path}")
        return 0

    # Full Phase A pilot — 3 profiles × n_seeds × 3 benchmarks.
    seeds = tuple(range(args.n_seeds))
    total_cells = len(PROFILES) * len(seeds)
    _print_banner(total_cells=total_cells, n_samples=args.n_samples)
    cells: list[dict] = []
    failures: list[dict] = []
    run_start = time.time()
    idx = 0
    for profile_name in PROFILES:
        for seed in seeds:
            idx += 1
            cell = _run_cell(
                profile_name,
                seed,
                benchmarks=REAL_BENCHMARKS,
                n_samples=args.n_samples,
                n_episodes=N_EPISODES_PER_PROFILE,
            )
            if "error" in cell:
                failures.append(cell)
                print(
                    f"[cell {idx}/{total_cells}] {profile_name} "
                    f"seed={seed} FAILED : {cell['error']}"
                )
                continue
            try:
                run_id = _register_cell(profile_name, seed)
            except Exception as exc:  # pragma: no cover - defensive
                run_id = f"register-failed:{exc}"
            cells.append({**cell, "run_id": run_id})
            print(
                f"[cell {idx}/{total_cells}] {profile_name} "
                f"seed={seed:02d} pre={cell['pre']:.4f} "
                f"post={cell['post']:.4f} delta={cell['delta']:+.4f} "
                f"wall={cell['wall_time_s']:.1f}s"
            )
    run_wall = time.time() - run_start

    h1 = _h1_test(cells)
    rejected = sum(1 for v in h1.values() if v.get("reject_h0"))
    verdict = "GO" if rejected >= GO_PROFILES_REJECTED_MIN else "NO-GO"
    print("-" * 64)
    for profile, r in h1.items():
        print(
            f"H1 {profile}: p={r.get('p')} "
            f"reject_h0={r.get('reject_h0')} n={r.get('n')}"
        )
    print(f"verdict          : {verdict} ({rejected}/{len(PROFILES)})")
    print(f"wall-clock total : {run_wall:.1f}s")
    print(f"failures         : {len(failures)}")

    dump_path = out_dir / "pilot-cycle3-real-1p5b.json"
    with dump_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "harness_version": HARNESS_VERSION,
                "scale": REAL_SCALE_FP16,
                "profiles": list(PROFILES),
                "seeds": list(seeds),
                "benchmarks": list(REAL_BENCHMARKS),
                "n_samples_per_benchmark": args.n_samples,
                "planned_cell_count": total_cells,
                "completed_cells": len(cells),
                "failed_cells": len(failures),
                "go_rule": {
                    "alpha": GO_BONFERRONI_ALPHA,
                    "profiles_rejected_min": GO_PROFILES_REJECTED_MIN,
                    "profiles_total": len(PROFILES),
                },
                "h1": h1,
                "verdict": verdict,
                "wall_time_s": run_wall,
                "cells": cells,
                "failures": failures,
            },
            fh,
            indent=2,
        )
    print(f"results dumped to {dump_path}")
    return 0 if verdict == "GO" else 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
