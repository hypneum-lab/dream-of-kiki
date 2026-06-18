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
import os
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

# Default scale slot — kept for backwards compatibility with the Phase
# B 1.5B reference run. Callers can override via ``--scale`` to any
# bf16/fp16 slot registered in ``base_model_registry.py`` (e.g.
# ``qwen3p5-7b-fp16`` for the C3.8 scale-law check, or
# ``qwen3p5-35b-fp16`` for the 35B extrapolation slot).
REAL_SCALE_FP16 = "qwen3p5-1p5b-fp16"
REAL_SCALE_CHOICES = (
    "qwen3p5-1p5b-fp16",
    "qwen3p5-7b-fp16",
    "qwen3p5-35b-fp16",
    "qwen3p6-35b-bf16-local",
)
REAL_SEEDS_DEFAULT = tuple(range(30))
REAL_BENCHMARKS = ("mmlu", "hellaswag", "mega_v2")
GO_BONFERRONI_ALPHA = 0.0125
GO_PROFILES_REJECTED_MIN = 2

N_EPISODES_PER_PROFILE = 5
DEFAULT_N_SAMPLES = 100


def _scale_slug(scale: str) -> str:
    """Return the filename-friendly scale slug (e.g. ``1p5b``, ``7b``).

    Strips the ``qwen3p5-`` prefix and the ``-fp16`` suffix so milestone
    JSON dumps stay readable (``pilot-cycle3-real-7b.json`` rather than
    ``pilot-cycle3-real-qwen3p5-7b-fp16.json``). Unknown slot patterns
    fall back to the raw slot name — never silently drop characters.
    """
    slug = scale
    if slug.startswith("qwen3p5-"):
        slug = slug[len("qwen3p5-"):]
    if slug.endswith("-fp16"):
        slug = slug[: -len("-fp16")]
    return slug or scale


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
    parser.add_argument(
        "--report-power",
        action="store_true",
        help="Print the E1 TOST power analysis (required n/arm for "
        "80%% power at SESOI g=0.29) and exit. Zero compute — no "
        "model load.",
    )
    parser.add_argument(
        "--scale",
        choices=REAL_SCALE_CHOICES,
        default=REAL_SCALE_FP16,
        help=(
            "Base-model scale slot — must be a bf16/fp16 pin in the "
            "base-model registry. Default keeps the Phase B 1.5B "
            "reference slot. C3.8 goal (d) scale law uses "
            "``qwen3p5-7b-fp16`` ; 35B extrapolation slot lives at "
            "``qwen3p5-35b-fp16``."
        ),
    )
    return parser.parse_args(argv)


# -------------------------------------------------------------------
# Model loading (fresh per cell so weights do not leak)
# -------------------------------------------------------------------


def _load_fresh_fp16_wrapper(scale: str = REAL_SCALE_FP16):
    """Load a FRESH bf16 Qwen wrapper for ``scale``.

    Fails fast with a clear message if ``mlx-lm`` is missing or the
    HF cache has no bf16 weights — Phase A/B cannot proceed without
    a gradient-bearing model.
    """
    from harness.real_models.qwen_mlx_fp16 import load_qwen_fp16

    return load_qwen_fp16(scale)


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
    """Run each benchmark and return ``{benchmark -> {accuracy, n}}``.

    ``mega_v2`` is evaluated on the HELD-OUT fold — the same fixture
    records never consumed by ``replay_real`` during dream episodes.
    This is the Phase B invariant that makes the mega-v2 delta a
    generalisation signal rather than training-set recall.
    """
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
            wrapper,
            tokenizer,
            n_samples=n_samples,
            seed=seed,
            fold="eval",
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
    # No persistent optimizer: the A3B MoE + hybrid attention emit a
    # gradient tree whose key set varies per replay episode (sparse
    # expert routing / per-seed activation). A reused optimizer's state
    # tree, init'd on an earlier episode's gradients, then lacks keys
    # ('self_attn' / 'linear_attn') present later -> apply_gradients
    # tree_map KeyError. A fresh stateless SGD per step keeps state
    # aligned with the current gradient tree (S2 finite still holds).

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
            # MoE routing indices are stop_gradient'd only in training mode
            # (mlx_lm SwitchGLU); without train() value_and_grad raises
            # "[GatherMM] Cannot calculate VJP with respect to indices" on
            # the A3B expert gather. Restore eval() so post-dream scoring is
            # deterministic inference.
            qwen.train()
            try:
                for _ in range(replay_inner_steps):
                    _, grads = replay_grad_fn(
                        qwen, input_ids_arr, target_ids_arr
                    )
                    # Fresh SGD each step (momentumless => identical
                    # numerics) so the optimizer state tree matches this
                    # step's gradient key set; avoids tree_map KeyError on
                    # the hybrid attention / MoE expert keys.
                    optim.SGD(learning_rate=replay_lr).update(qwen, grads)
                    mx.eval(qwen.parameters())
            finally:
                qwen.eval()

    downscale_state = DownscaleRealState()
    restructure_state = RestructureRealState()
    recombine_state = RecombineRealState()

    # ---------------------------------------------------------------
    # Nested-tree downscale — the stock ``downscale_real_handler``
    # iterates ``model.layers`` looking for direct ``.weight`` /
    # ``.bias`` attributes. Qwen's TransformerBlock nests its weight
    # tensors under sub-modules (``self_attn.q_proj.weight`` etc.),
    # so the generic handler silently no-ops on Qwen. This helper
    # walks the nested parameter dict via ``mx.utils.tree_flatten``,
    # applies the shrink factor to every leaf ``mx.array``, and runs
    # an S2 finite guard over the post-shrink tree.
    # ---------------------------------------------------------------

    def _qwen_downscale_recursive(
        episode: DreamEpisode,
    ) -> None:
        from mlx.utils import tree_flatten, tree_unflatten

        from kiki_oniric.dream.guards.finite import check_finite

        factor = episode.input_slice.get("shrink_factor", 1.0)
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )

        params = qwen.parameters()
        flat = tree_flatten(params)
        # Shrink every leaf tensor in-place at the tree level. MLX
        # arrays are immutable so we build a new leaf list ; the
        # model absorbs it via ``update``.
        shrunk: list[tuple[str, mx.array]] = []
        for key, leaf in flat:
            if isinstance(leaf, mx.array):
                shrunk.append((key, leaf * factor))
            else:
                shrunk.append((key, leaf))
        new_params = tree_unflatten(shrunk)
        qwen.update(new_params)
        # Force evaluation so the next forward pass reads mutated
        # weights rather than a deferred graph that masks divergence.
        mx.eval(qwen.parameters())

        # S2 finite guard over the post-shrink tree.
        weights_by_key: dict[str, np.ndarray] = {}
        for key, leaf in tree_flatten(qwen.parameters()):
            if isinstance(leaf, mx.array):
                weights_by_key[key] = np.asarray(
                    leaf.astype(mx.float32)
                )
        check_finite(weights_by_key)

        downscale_state.compound_factor *= factor
        # K1 tag : ~1 multiply per scalar parameter across the tree.
        total_params = sum(
            int(leaf.size)
            for _, leaf in tree_flatten(qwen.parameters())
            if isinstance(leaf, mx.array)
        )
        downscale_state.last_compute_flops = max(total_params, 1)

    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, _replay_qwen_handler)
    runtime.register_handler(
        Operation.DOWNSCALE, _qwen_downscale_recursive
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
    # ``fold="train"`` so replay_real consumes the TRAIN partition only ;
    # the evaluator reads the disjoint ``fold="eval"`` partition, so
    # the post-dream delta is a generalisation signal rather than
    # training-set recall.
    batch_size = 4
    total_records_needed = batch_size * max(effective_episodes, 1)
    mega_records = _load_mega_v2_records(
        n_samples=total_records_needed, seed=seed, fold="train"
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


def _snapshot_first_weight(wrapper) -> tuple[str, float] | None:
    """Return ``(key, first_scalar)`` for the first Qwen weight leaf.

    Used by the smoke-cell path to verify the recursive downscale
    actually mutates Qwen's nested weight tensors. Returns ``None``
    if no weight leaf is accessible — we refuse to silently pass
    that signal on.
    """
    try:
        import mlx.core as mx
        from mlx.utils import tree_flatten

        params = wrapper.model.parameters()
        for key, leaf in tree_flatten(params):
            if isinstance(leaf, mx.array) and leaf.size > 0:
                scalar = float(
                    leaf.reshape(-1)[0].astype(mx.float32).item()
                )
                return (key, scalar)
    except Exception:
        return None
    return None


def _run_cell(
    profile_name: str,
    seed: int,
    *,
    benchmarks: tuple[str, ...],
    n_samples: int,
    n_episodes: int,
    verify_downscale: bool = False,
    scale: str = REAL_SCALE_FP16,
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
        "scale": scale,
    }
    try:
        wrapper = _load_fresh_fp16_wrapper(scale)
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
        pre_snap = (
            _snapshot_first_weight(wrapper)
            if verify_downscale
            else None
        )
        _dream_episodes(
            wrapper, profile_name, seed=seed, n_episodes=n_episodes
        )
        if verify_downscale:
            post_snap = _snapshot_first_weight(wrapper)
            print(
                f"[verify-downscale] pre_weight={pre_snap} "
                f"post_weight={post_snap} "
                f"mutated={pre_snap != post_snap}"
            )
            cell["weight_mutation_verified"] = bool(
                pre_snap is not None
                and post_snap is not None
                and pre_snap != post_snap
            )
            cell["pre_weight_snapshot"] = pre_snap
            cell["post_weight_snapshot"] = post_snap
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
            "model": scale,
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
# E1 — powered TOST equivalence (paired pre vs post per profile)
# -------------------------------------------------------------------


def _e1_tost(cells: list[dict]) -> dict:
    """Powered standardized TOST (Hedges' g) pre vs post per profile.

    Complements the H1 paired t-test : H1 asks "did dream *improve*?",
    E1's TOST asks "is post *equivalent* to pre within the SESOI
    g=0.29?" — the only way to *support* the coupling NULL (Lakens
    2017). Bounds and α are pre-registered in :mod:`harness.stats.tost`.
    Skips profiles with < 2 paired cells.
    """
    from harness.stats.tost import (
        SESOI_HEDGES_G,
        TOST_ALPHA_BONFERRONI,
        required_n_for_power,
        tost_equivalence_g,
    )

    power = required_n_for_power(design="paired")
    out: dict = {
        "sesoi_hedges_g": SESOI_HEDGES_G,
        "alpha_per_test": TOST_ALPHA_BONFERRONI,
        "power_analysis": {
            "target_power": power.target_power,
            "required_n_pairs": power.required_n,
            "achieved_power_at_required_n": power.achieved_power,
            "available_seeds": len(REAL_SEEDS_DEFAULT),
            "adequately_powered": (
                len(REAL_SEEDS_DEFAULT) >= power.required_n
            ),
        },
        "per_profile": {},
    }
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
            out["per_profile"][profile] = {
                "equivalent": None,
                "n": len(pres),
                "note": "insufficient samples",
            }
            continue
        res = tost_equivalence_g(treatment=posts, control=pres)
        out["per_profile"][profile] = {
            "observed_g": res.observed_g,
            "p_lower": res.p_lower,
            "p_upper": res.p_upper,
            "p_tost": res.p_tost,
            "equivalent": res.equivalent,
            "n": res.n_pairs,
        }
    return out


# -------------------------------------------------------------------
# Run registry
# -------------------------------------------------------------------


def _register_cell(
    profile_name: str, seed: int, scale: str = REAL_SCALE_FP16
) -> str:
    from harness.storage.run_registry import RunRegistry

    registry = RunRegistry(REPO_ROOT / ".run_registry.sqlite")
    profile_tag = (
        f"cycle3/{scale}/{profile_name}/mlx_kiki_oniric"
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


def _print_banner(
    total_cells: int,
    n_samples: int,
    scale: str = REAL_SCALE_FP16,
) -> None:
    print("=" * 64)
    print("CYCLE-3 C3.8 PHASE A REAL ABLATION PILOT")
    print("=" * 64)
    print(f"harness_version : {HARNESS_VERSION}")
    print(f"scale           : {scale}")
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

    scale = args.scale
    slug = _scale_slug(scale)

    if args.report_power:
        from harness.stats.tost import (
            SESOI_HEDGES_G,
            TOST_ALPHA_BONFERRONI,
            required_n_for_power,
            tost_power_at_n,
        )

        paired = required_n_for_power(design="paired")
        two = required_n_for_power(design="two_sample")
        n_seeds = len(REAL_SEEDS_DEFAULT)
        print("E1 TOST power analysis (no compute)")
        print(f"  SESOI (Hedges' g)        : {SESOI_HEDGES_G}")
        print(f"  alpha per one-sided test : {TOST_ALPHA_BONFERRONI:.6f}")
        print(f"  target power             : {paired.target_power}")
        print(
            f"  required n (paired)      : {paired.required_n} pairs "
            f"(power {paired.achieved_power:.3f})"
        )
        print(
            f"  required n (two-sample)  : {two.required_n}/arm "
            f"(power {two.achieved_power:.3f})"
        )
        print(
            f"  available seeds          : {n_seeds} "
            f"(power@{n_seeds}={tost_power_at_n(n_seeds):.3f})"
        )
        print(
            "  adequately powered       : "
            f"{n_seeds >= paired.required_n}"
        )
        return 0

    if args.smoke_cell:
        _print_banner(
            total_cells=1, n_samples=args.n_samples, scale=scale
        )
        benchmarks = (args.benchmark,)
        start = time.time()
        # ``verify_downscale=True`` so the smoke-cell log lands a
        # pre/post weight snapshot — this is the gate that proves
        # Qwen's nested weight tree actually mutates before we
        # commit ~1 h of Phase B compute.
        cell = _run_cell(
            args.profile,
            0,
            benchmarks=benchmarks,
            n_samples=args.n_samples,
            n_episodes=N_EPISODES_PER_PROFILE,
            verify_downscale=True,
            scale=scale,
        )
        wall = time.time() - start
        smoke_path = (
            out_dir / f"pilot-cycle3-real-{slug}-smoke.json"
        )
        if "error" in cell:
            print(f"[smoke-cell] FAILED : {cell['error']}")
            with smoke_path.open("w", encoding="utf-8") as fh:
                json.dump(
                    {
                        "harness_version": HARNESS_VERSION,
                        "mode": "smoke-cell",
                        "benchmark": args.benchmark,
                        "n_samples": args.n_samples,
                        "wall_time_s": wall,
                        "model": scale,
                        "cell": cell,
                        "status": "FAILED",
                    },
                    fh,
                    indent=2,
                )
            return 1
        try:
            run_id = _register_cell(args.profile, 0, scale)
        except Exception as exc:  # pragma: no cover - defensive
            run_id = f"register-failed:{exc}"
        pre_acc = cell["pre"]
        post_acc = cell["post"]
        print(
            f"[smoke-cell] pre_acc={pre_acc:.4f} "
            f"post_acc={post_acc:.4f} "
            f"delta={cell['delta']:+.4f} "
            f"wall={wall:.2f}s model={scale} "
            f"benchmark={args.benchmark} run_id={run_id}"
        )
        with smoke_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "harness_version": HARNESS_VERSION,
                    "mode": "smoke-cell",
                    "benchmark": args.benchmark,
                    "n_samples": args.n_samples,
                    "wall_time_s": wall,
                    "model": scale,
                    "cell": {**cell, "run_id": run_id},
                    "status": "OK",
                },
                fh,
                indent=2,
            )
        print(f"[smoke-cell] dump written to {smoke_path}")
        return 0

    # Full Phase A/B pilot — 3 profiles × n_seeds × 3 benchmarks.
    seeds = tuple(range(args.n_seeds))
    total_cells = len(PROFILES) * len(seeds)
    _print_banner(
        total_cells=total_cells, n_samples=args.n_samples, scale=scale
    )
    # Resumable cell cache (JSONL) : one successful cell per line. Each cell
    # is deterministic under R1 (profile, seed, commit_sha), so skipping
    # already-computed cells on a restart yields an identical final dump —
    # the run can be stopped and relaunched without losing progress. Failed
    # cells are NOT cached, so they retry on the next run (the recombine fix
    # then lets p_max cells that previously KeyError'd complete). Set
    # HL_E1_FRESH=1 to ignore an existing cache and recompute from scratch.
    cache_path = out_dir / f"pilot-cycle3-real-{slug}.cells.jsonl"
    cells: list[dict] = []
    done: set[tuple[str, int]] = set()
    if cache_path.exists() and not os.getenv("HL_E1_FRESH"):
        with cache_path.open(encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                cells.append(rec)
                done.add((rec["profile"], int(rec["seed"])))
        print(f"[resume] {len(done)} cached cells loaded; only the rest will run")
    failures: list[dict] = []
    run_start = time.time()
    idx = 0
    for profile_name in PROFILES:
        for seed in seeds:
            idx += 1
            if (profile_name, seed) in done:
                continue
            cell = _run_cell(
                profile_name,
                seed,
                benchmarks=REAL_BENCHMARKS,
                n_samples=args.n_samples,
                n_episodes=N_EPISODES_PER_PROFILE,
                scale=scale,
                # Snapshot the first Qwen weight pre/post-dream so each cell
                # carries parametric evidence (weight_mutation_verified) that
                # the dream corpus actually moved the model — complementing the
                # behavioural pre->post benchmark delta.
                verify_downscale=True,
            )
            if "error" in cell:
                failures.append(cell)
                print(
                    f"[cell {idx}/{total_cells}] {profile_name} "
                    f"seed={seed} FAILED : {cell['error']}"
                )
                continue
            try:
                run_id = _register_cell(profile_name, seed, scale)
            except Exception as exc:  # pragma: no cover - defensive
                run_id = f"register-failed:{exc}"
            record = {**cell, "run_id": run_id}
            cells.append(record)
            with cache_path.open("a", encoding="utf-8") as fh:
                fh.write(json.dumps(record) + "\n")
            print(
                f"[cell {idx}/{total_cells}] {profile_name} "
                f"seed={seed:02d} pre={cell['pre']:.4f} "
                f"post={cell['post']:.4f} delta={cell['delta']:+.4f} "
                f"wall={cell['wall_time_s']:.1f}s"
            )
    run_wall = time.time() - run_start

    h1 = _h1_test(cells)
    e1_tost = _e1_tost(cells)
    rejected = sum(1 for v in h1.values() if v.get("reject_h0"))
    verdict = "GO" if rejected >= GO_PROFILES_REJECTED_MIN else "NO-GO"
    print("-" * 64)
    for profile, r in h1.items():
        print(
            f"H1 {profile}: p={r.get('p')} "
            f"reject_h0={r.get('reject_h0')} n={r.get('n')}"
        )
    pa = e1_tost["power_analysis"]
    print(
        f"E1 TOST: SESOI g={e1_tost['sesoi_hedges_g']} "
        f"required_n={pa['required_n_pairs']} pairs "
        f"powered={pa['adequately_powered']}"
    )
    for profile, r in e1_tost["per_profile"].items():
        print(
            f"E1 {profile}: equivalent={r.get('equivalent')} "
            f"p_tost={r.get('p_tost')} n={r.get('n')}"
        )
    print(f"verdict          : {verdict} ({rejected}/{len(PROFILES)})")
    print(f"wall-clock total : {run_wall:.1f}s")
    print(f"failures         : {len(failures)}")

    dump_path = out_dir / f"pilot-cycle3-real-{slug}.json"
    with dump_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "harness_version": HARNESS_VERSION,
                "scale": scale,
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
                "e1_tost": e1_tost,
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
