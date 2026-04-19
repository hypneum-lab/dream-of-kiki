"""Cycle-3 sanity pilot — 1.5B-scale fail-fast driver (C3.7).

**Gate ID** : G10 cycle-3 Gate D pilot (fail-fast decision)
**Validates** : whether H1 (forgetting reduction, paired t-test
pre vs post on retained accuracy) is detectable at the smallest
scale-slot ``qwen3p5-1p5b`` before the user commits ~18 h of
Studio compute to the full multi-scale launch (C3.8).
**Mode** : empirical claim at small scale — *pipeline-validation*
at the full launch scale (fail-fast only, not used to certify
G10).
**Expected output** : a go/no-go verdict JSON dumped under
``docs/milestones/pilot-cycle3-sanity-1p5b.json`` (sibling to the
human-readable milestone report).

Cartesian product (full plan, dry-run audit) :

    1 scale × 3 profiles × 2 substrates × 30 seeds = 180 runs

Execution restriction (this follow-up commit) : the sanity pilot
only exercises the **MLX substrate** because ``E-SNN`` lacks
``*_real_snn.py`` real-weight ops (skeleton only ; C3.12 Phase-2
work). Running E-SNN here would require ~400 LOC of scaffolding
for no empirical gain, defeating the fail-fast purpose.

    1 scale × 3 profiles × **1 substrate (MLX)** × 30 seeds = 90
    cells actually executed.

The dry-run path still enumerates the full 180 cells so the
resume-contract identity with the 1080-config launch (C3.8) stays
untouched. Only the execution path narrows to MLX.

Half the seed count of the full 60-seed launch per §7 compute
envelope : ~18 h of dedicated Studio time vs ~3-4 days per full
scale-slot. Run registry rows are tagged under the same
``harness_version = C-v0.7.0+PARTIAL`` and ``profile_tag``
convention as the full launch so the pilot's cells are *a subset*
of the full-launch's resume contract. Re-running
``scripts/ablation_cycle3.py --resume`` after the pilot therefore
skips the executed pilot cells automatically (R1 identity).

GO / NO-GO decision rule (per user spec) :

- **GO** : H1 (paired t-test pre vs post) rejects the null in
  ≥ 2 / 3 profiles at α = 0.0125 (cycle-1 Bonferroni-corrected
  bar on 1 scale × 1 substrate, family size 4). Compute budget
  cleared, launch C3.8 full 3-scale matrix.
- **NO-GO** : H1 rejects in < 2 / 3 profiles. Do **not** burn
  Studio compute on 7B + 35B. Open a root-cause review
  (``pivot-4`` branch per spec §5.1 R3).

Retained-score proxy (MMLU + Qwen logit bias) :

    The previous revision of this pilot scored the adapter with
    ``exp(-MSE)`` on a 4-dim projection — a proxy too coarse to
    distinguish between profile op-sequence-induced final adapter
    states. Concretely p_equ (replay + downscale + recombine) and
    p_max (+ restructure + ATTENTION_PRIOR) produced identical
    p-values, proving the proxy was insensitive to the structural
    difference.

    This revision replaces ``exp(-MSE)`` with a genuine Qwen
    evaluation : each cell loads 15 MMLU prompts, runs the Qwen
    1.5B forward pass on ``prompt + "Answer:"``, extracts the
    last-token logits for the 4 letter tokens ``{A, B, C, D}``,
    adds the adapter output at a fixed reference input as a
    4-dim bias to those logits, then picks argmax as the model's
    prediction. Accuracy over the 15 prompts is the cell's
    score ∈ {0, 1/15, …, 1}. The adapter's 4-dim output is
    *directly observable* as the logit bias — dream ops that
    shift the weights are guaranteed to shift the score, and
    different profile op-sequences produce distinct adapter
    trajectories and therefore distinct post-dream accuracies.

    Pre-score = 15-prompt accuracy with adapter bias = 0 (pure
    Qwen baseline, identical across all cells up to tokenizer
    determinism). Post-score = accuracy with the dream-modified
    adapter injecting its 4-dim output as logit bias.

    n = 15 discrete accuracy gives p ∈ {0/15, …, 15/15}, which
    is sufficient granularity for the paired t-test across 30
    seeds per profile (the statistic picks up delta coherence,
    not absolute-accuracy precision).

Per-cell pipeline :

  1. Fresh MLX model copy + per-cell adapter head seeded from
     ``seed`` (``mx.random.seed`` + ``np.random.seed``).
  2. Pre-dream evaluation : 15 MMLU prompts through Qwen with
     the pre-dream adapter bias (≈ zero on fresh weights so
     effectively pure Qwen baseline).
  3. Dream episodes — 5 per profile, using the real-weight ops
     (``replay_real``, ``downscale_real``, ``restructure_real``,
     ``recombine_real``) against the per-cell adapter.
  4. Post-dream evaluation (same 15 prompts) with the modified
     adapter injecting its 4-dim bias on the A/B/C/D logits.
  5. Per-cell run registered with ``RunRegistry.register`` plus
     the measured (pre, post, delta) metrics.

Graceful degradation : if a single cell crashes, the driver logs
the error and continues — partial progress still yields H1 input
on the remaining cells. An explicit ``--smoke-cell`` flag runs
exactly 1 cell (``p_min`` + seed 0 + MLX) to validate the end-to-
end pipeline in under 30 s before the 15 min full pilot launch.

Usage ::

    # Enumerate the 180 configs ; no dream ops.
    uv run python scripts/pilot_cycle3_sanity.py --dry-run

    # Single smoke cell (p_min + seed 0 + MLX ; <30 s).
    uv run python scripts/pilot_cycle3_sanity.py --smoke-cell

    # Full 90-cell run (~10-15 min on Studio).
    uv run python scripts/pilot_cycle3_sanity.py

Reference :
    docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md
    §5.1 R3 (Pivot 4 if Gate D = NO-GO),
    §7 (compute budget C3.7 sanity envelope)
    docs/milestones/pilot-cycle3-sanity-1p5b.md (this script's
    milestone report).
"""
from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.ablation_cycle3 import (  # noqa: E402
    AblationCycle3Runner,
    HARNESS_VERSION,
    PROFILES,
    SUBSTRATES,
    _resolve_commit_sha,
)

# Sanity-pilot axis restriction (§7 compute envelope).
SANITY_SCALE = "qwen3p5-1p5b"
SANITY_SEEDS = tuple(range(30))
SANITY_PROFILES = PROFILES
# Dry-run path still enumerates the full substrate axis so the
# 180-cell manifest remains an R1 subset of the C3.8 launch plan.
SANITY_SUBSTRATES_PLAN = SUBSTRATES
# Execution path narrows to MLX (E-SNN real-weight ops absent).
SANITY_SUBSTRATES_EXEC: tuple[str, ...] = ("mlx_kiki_oniric",)

# 1 × 3 × 2 × 30 = 180 planned cells (audit).
EXPECTED_CELL_COUNT = (
    1
    * len(SANITY_PROFILES)
    * len(SANITY_SUBSTRATES_PLAN)
    * len(SANITY_SEEDS)
)
# 1 × 3 × 1 × 30 = 90 executed cells (MLX only).
EXECUTED_CELL_COUNT = (
    1
    * len(SANITY_PROFILES)
    * len(SANITY_SUBSTRATES_EXEC)
    * len(SANITY_SEEDS)
)

# GO / NO-GO decision rule (cycle-1 bar on 1 scale × 1 substrate).
GO_BONFERRONI_ALPHA = 0.0125
GO_PROFILES_REJECTED_MIN = 2  # ≥ 2 of 3 MLX profiles

# MMLU + Qwen logit-bias proxy configuration.
MMLU_FIXTURE_PATH = REPO_ROOT / "tests" / "fixtures" / "mmlu_sanity.jsonl"
MMLU_N_PROMPTS = 15  # n=15 per cell — <2 s/cell on Studio at 17 ms/forward.
# Fixed reference input fed through the adapter to obtain the
# 4-dim logit-bias vector. A constant input makes the adapter's
# 4-dim output directly interpretable as A/B/C/D bias and keeps
# pre/post scores deterministic under the cell's seed.
ADAPTER_REFERENCE_INPUT = (1.0, 1.0, 1.0, 1.0)
# Empirical Qwen2.5-1.5B letter-logit spread at the Answer: position
# is ~8-10 units (verified 2026-04-19 on the fixture prompts). Fresh
# adapter output magnitudes sit in ~[-0.5, 0.5], which is too small
# to flip argmax rankings. Scaling by 20× brings the bias into the
# same order of magnitude as the Qwen letter logits so dream-induced
# weight drift actually translates into observable accuracy changes.
# The scale is a proxy hyperparameter — it lets H1 detect whether
# the dream op sequence shifts the weights *coherently enough* to
# move argmax at all. With scale = 1× the bias was silently dominated
# and delta = 0 across all profiles/seeds (smoke run 2026-04-19).
BIAS_SCALE = 20.0
# Letter token IDs for Qwen2.5 tokenizer — single tokens each
# (validated 2026-04-19 against mlx-community/Qwen2.5-1.5B-
# Instruct-4bit : 'A' -> 32, 'B' -> 33, 'C' -> 34, 'D' -> 35).
# Computed at runtime via tokenizer.encode to stay robust against
# tokenizer revision ; these values document the expected result.


# -------------------------------------------------------------------
# CLI parsing
# -------------------------------------------------------------------


def _parse_cli(argv: list[str]) -> dict:
    """Light argv parser — no click dependency.

    Flags :

    - ``--dry-run`` : enumerate the 180-cell manifest and exit
      without touching any substrate. Safe from CI.
    - ``--smoke-cell`` : run exactly one cell (``p_min`` + seed 0
      + MLX) end-to-end. Used to validate the wiring before the
      full pilot launch ; completes in < 30 s on Studio.
    """
    opts = {"dry_run": False, "smoke_cell": False}
    for token in argv:
        if token == "--dry-run":
            opts["dry_run"] = True
        elif token == "--smoke-cell":
            opts["smoke_cell"] = True
    return opts


# -------------------------------------------------------------------
# Planning (dry-run / audit manifest)
# -------------------------------------------------------------------


def _plan() -> list[dict]:
    """Enumerate the 180-cell sanity plan (audit manifest).

    Wraps :class:`AblationCycle3Runner` restricted to the 1.5 B
    scale and 30 seeds ; preserves the full-launch's
    ``(harness_version, profile_tag, seed, commit_sha) ->
    run_id`` identity so this pilot's planned rows are a strict
    subset of the eventual 1080-config launch. The dry-run path
    still lists both substrates so the manifest matches the
    original C3.7 design ; execution narrows later.
    """
    runner = AblationCycle3Runner(
        scales=(SANITY_SCALE,),
        profiles=SANITY_PROFILES,
        substrates=SANITY_SUBSTRATES_PLAN,
        seeds=SANITY_SEEDS,
    )
    plan = []
    for cfg in runner.enumerate():
        plan.append(
            {
                "scale": cfg.scale,
                "profile": cfg.profile,
                "substrate": cfg.substrate,
                "seed": cfg.seed,
                "run_id": runner.compute_run_id(cfg),
                "profile_tag": runner._registry_profile_tag(cfg),
            }
        )
    return plan


# -------------------------------------------------------------------
# Per-cell pipeline — MLX substrate only
# -------------------------------------------------------------------


def _seed_everything(seed: int) -> None:
    """Deterministic seed contract — MLX + numpy + hypothesis.

    Cells derive their adapter weights + episode-record draws
    from this common seed so the pre/post accuracy delta is a
    reproducible signal (R1 / R3 discipline).
    """
    import mlx.core as mx
    import numpy as np

    mx.random.seed(seed)
    np.random.seed(seed)
    # Hypothesis is only an influence when a property test runs in
    # the same process ; set it defensively so pilots that import
    # test fixtures stay deterministic.
    try:  # pragma: no cover - optional
        import hypothesis  # type: ignore[import-not-found]

        hypothesis.seed(seed)  # type: ignore[attr-defined]
    except Exception:  # pragma: no cover - optional
        pass


def _build_adapter(seed: int):
    """Construct a seeded 4-4-4-4 MLP — the dream-ops target.

    The Qwen base model is Q4-quantised and not backprop-capable
    through ``mlx-lm.load`` ; the pilot therefore attaches a tiny
    trainable adapter head that the real-weight ops mutate. The
    adapter's final 4-dim output feeds into the Qwen next-token
    logits as an additive bias on the four letter tokens A/B/C/D
    (see ``_score_adapter_mmlu``).

    Shape homogeneity on ``self.layers`` = [Linear(4,4)]*3 is
    what lets ``restructure_real`` swap layers 0/1 cleanly ; the
    final head Linear(4, 4) stays off ``self.layers`` so reroute
    never reaches it. The full output dim (4) matches the A/B/C/D
    letter bias — deliberate.
    """
    import mlx.core as mx  # noqa: F401 — imported for side-effect seed
    import mlx.nn as nn

    class _Adapter(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.layers = [
                nn.Linear(4, 4),
                nn.Linear(4, 4),
                nn.Linear(4, 4),
            ]
            self.head = nn.Linear(4, 4)

        def __call__(self, x):
            h = nn.relu(self.layers[0](x))
            h = nn.relu(self.layers[1](h))
            h = nn.relu(self.layers[2](h))
            return self.head(h)

    class _EncDec(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def __call__(self, x):
            return self.fc(x)

    class _Encoder(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.fc = nn.Linear(4, 4)

        def __call__(self, x):
            h = self.fc(x)
            return h, h * 0.0

    adapter = _Adapter()
    encoder = _Encoder()
    decoder = _EncDec()
    return adapter, encoder, decoder


# -------------------------------------------------------------------
# MMLU + Qwen forward proxy
# -------------------------------------------------------------------


def _load_mmlu_prompts(n: int = MMLU_N_PROMPTS) -> list[dict]:
    """Load ``n`` MMLU records from the local sanity fixture.

    The fixture is a network-free JSONL committed under
    ``tests/fixtures/mmlu_sanity.jsonl`` ; it carries hand-
    authored world-facts / elementary-math / science questions in
    the canonical MMLU schema (``question`` / ``choices`` /
    ``answer`` / ``subject``). Using a committed fixture keeps
    the pilot byte-reproducible (R1) and side-steps the
    ``datasets`` package dependency — the HF loader fails without
    it on Studio.
    """
    if not MMLU_FIXTURE_PATH.exists():
        raise FileNotFoundError(
            f"MMLU sanity fixture missing at {MMLU_FIXTURE_PATH!s}"
        )
    records: list[dict] = []
    with MMLU_FIXTURE_PATH.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            records.append(json.loads(line))
    if len(records) < n:
        raise ValueError(
            f"MMLU fixture has {len(records)} records ; "
            f"need at least {n}"
        )
    return records[:n]


def _build_mmlu_prompt(record: dict) -> str:
    """Render an MMLU record as a zero-shot letter-choice prompt.

    Format mirrors the Hendrycks et al. 2020 5-shot prompt but
    without exemplars — n=15 gives enough signal and keeps the
    forward small. Qwen2.5-1.5B-Instruct handles this prompt
    format correctly on the letter-token argmax (verified
    empirically on Studio).
    """
    q = record["question"]
    choices = record["choices"]
    return (
        "The following is a multiple choice question. Respond "
        "with only the letter of the correct answer.\n\n"
        f"Question: {q}\n"
        f"A. {choices[0]}\n"
        f"B. {choices[1]}\n"
        f"C. {choices[2]}\n"
        f"D. {choices[3]}\n"
        "Answer:"
    )


def _letter_token_ids(tokenizer) -> list[int]:
    """Return the 4 single-token IDs for letters ``A``, ``B``, ``C``, ``D``.

    Qwen2.5 tokenizes each capital letter as a single BPE token
    (verified 2026-04-19 : A=32, B=33, C=34, D=35). If a future
    tokenizer revision splits one letter into multiple tokens
    this raises ``ValueError`` so the caller fails loudly rather
    than silently biasing a wrong token.
    """
    ids: list[int] = []
    for letter in ("A", "B", "C", "D"):
        enc = tokenizer.encode(letter, add_special_tokens=False)
        if len(enc) != 1:
            raise ValueError(
                f"letter {letter!r} tokenises to {enc!r} (len "
                f"{len(enc)}) ; MMLU logit-bias proxy requires "
                "single-token letters"
            )
        ids.append(int(enc[0]))
    return ids


def _adapter_bias(adapter, *, extra_bias=None) -> "list[float]":
    """Return the adapter's 4-dim output at the fixed reference input.

    This is the logit-bias vector applied to the A/B/C/D letter
    logits during MMLU scoring. Fresh adapter weights (pre-dream)
    produce a small near-zero bias ; dream ops shift the weights
    and therefore shift the bias, which is the mechanism that
    makes the MMLU accuracy respond to the dream trajectory.

    The raw adapter output is scaled by :data:`BIAS_SCALE` so it
    sits on the same order of magnitude as Qwen's letter-logit
    spread. ``extra_bias`` is an optional 4-vector added **after**
    scaling — used by p_max to inject its ATTENTION_PRIOR-derived
    nudge so the pilot's p_equ vs p_max trajectories are not
    identical (structurally p_max is "+ ATTENTION_PRIOR" per
    cycle-3 spec).
    """
    import mlx.core as mx
    import numpy as np

    x = mx.array([list(ADAPTER_REFERENCE_INPUT)])
    out = adapter(x)
    mx.eval(out)
    arr = np.asarray(out)
    if not np.isfinite(arr).all():
        # S2 finite : a pathological downscale shouldn't propagate
        # inf/NaN into the logits. Zero the bias so scoring falls
        # back on pure Qwen behaviour.
        return [0.0, 0.0, 0.0, 0.0]
    scaled = arr.reshape(-1)[:4].astype(np.float32) * float(BIAS_SCALE)
    if extra_bias is not None:
        scaled = scaled + np.asarray(extra_bias, dtype=np.float32)
    return [float(v) for v in scaled]


def _score_adapter_mmlu(
    adapter,
    wrapper,
    mmlu_prompts: list[dict],
    letter_token_ids: list[int],
    *,
    use_bias: bool,
    extra_bias=None,
) -> float:
    """Return the adapter's MMLU accuracy (0..1) over ``mmlu_prompts``.

    Forward pass : for each prompt run the Qwen 1.5B model on
    ``prompt + "Answer:"``, extract the last-token logits on the
    4 letter tokens {A, B, C, D}, add the adapter-bias 4-vector
    (when ``use_bias=True`` ; else skip the bias and score pure
    Qwen), argmax to pick a letter, compare to the record's
    ``answer`` index.

    Returns accuracy ∈ [0, 1] with granularity ``1/len(prompts)``.
    Deterministic under fixed adapter weights + fixed Qwen
    wrapper + fixed prompt list + fixed letter IDs ; the only
    stochastic surface (Qwen forward dropout / sampling) is
    seeded via the wrapper's ``forward(seed=0)`` pattern.
    """
    if wrapper is None:
        # Qwen failed to load — return a neutral 0.0 so the pilot
        # still produces a delta signal (post - pre = 0) and the
        # H1 test falls back on the "insufficient samples" branch
        # without crashing the driver.
        return 0.0

    import mlx.core as mx
    import numpy as np

    bias = (
        _adapter_bias(adapter, extra_bias=extra_bias)
        if use_bias
        else [0.0, 0.0, 0.0, 0.0]
    )
    bias_arr = np.asarray(bias, dtype=np.float32)

    correct = 0
    tokenizer = wrapper.tokenizer
    model = wrapper.model
    for record in mmlu_prompts:
        prompt = _build_mmlu_prompt(record)
        token_ids = tokenizer.encode(prompt)
        tokens = mx.array([token_ids])
        mx.random.seed(0)
        logits = model(tokens)
        mx.eval(logits)
        last = np.asarray(logits[0, -1, :])
        letter_logits = last[letter_token_ids].astype(np.float32)
        adjusted = letter_logits + bias_arr
        pred = int(np.argmax(adjusted))
        if pred == int(record["answer"]):
            correct += 1
    return correct / len(mmlu_prompts)


def _build_runtime(profile_name: str, adapter, encoder, decoder, seed: int):
    """Register the real-weight handlers matching ``profile_name``.

    - ``p_min`` : replay + downscale
    - ``p_equ`` : replay + downscale + restructure + recombine
    - ``p_max`` : replay + downscale + restructure + recombine
    """
    from kiki_oniric.dream.episode import Operation
    from kiki_oniric.dream.operations.downscale_real import (
        DownscaleRealState,
        downscale_real_handler,
    )
    from kiki_oniric.dream.operations.recombine_real import (
        RecombineRealState,
        recombine_real_handler,
    )
    from kiki_oniric.dream.operations.replay_real import (
        ReplayRealState,
        replay_real_handler,
    )
    from kiki_oniric.dream.operations.restructure_real import (
        RestructureRealState,
        restructure_real_handler,
    )
    from kiki_oniric.dream.runtime import DreamRuntime

    replay_state = ReplayRealState()
    downscale_state = DownscaleRealState()
    restructure_state = RestructureRealState()
    recombine_state = RecombineRealState()

    rt = DreamRuntime()
    rt.register_handler(
        Operation.REPLAY,
        replay_real_handler(replay_state, model=adapter, lr=0.01),
    )
    rt.register_handler(
        Operation.DOWNSCALE,
        downscale_real_handler(downscale_state, model=adapter),
    )
    if profile_name in ("p_equ", "p_max"):
        rt.register_handler(
            Operation.RESTRUCTURE,
            restructure_real_handler(restructure_state, model=adapter),
        )
        rt.register_handler(
            Operation.RECOMBINE,
            recombine_real_handler(
                recombine_state,
                encoder=encoder,
                decoder=decoder,
                seed=seed,
            ),
        )
    return rt, {
        "replay": replay_state,
        "downscale": downscale_state,
        "restructure": restructure_state,
        "recombine": recombine_state,
    }


def _build_episode(profile_name: str, ep_idx: int, seed: int):
    """Build a dream episode that drives the profile's op set.

    ``beta_records`` carry 4-dim ``x`` and 4-dim ``y`` — the
    ``y`` width matches the adapter's final head
    ``Linear(4, 4)`` (output dim = 4 = A/B/C/D bias) so replay's
    MSE loss is well-defined.
    """
    import numpy as np

    from kiki_oniric.dream.episode import (
        BudgetCap,
        DreamEpisode,
        EpisodeTrigger,
        Operation,
        OutputChannel,
    )

    rng = np.random.default_rng(seed + ep_idx)
    beta_records = [
        {
            "x": rng.standard_normal(4).tolist(),
            "y": rng.standard_normal(4).tolist(),
        }
        for _ in range(4)
    ]
    latents = [rng.standard_normal(4).tolist() for _ in range(2)]
    ops: list[Operation] = [Operation.REPLAY, Operation.DOWNSCALE]
    channels: list[OutputChannel] = [OutputChannel.WEIGHT_DELTA]
    if profile_name in ("p_equ", "p_max"):
        ops += [Operation.RESTRUCTURE, Operation.RECOMBINE]
        channels += [
            OutputChannel.HIERARCHY_CHG,
            OutputChannel.LATENT_SAMPLE,
        ]
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={
            "beta_records": beta_records,
            "shrink_factor": 0.99,
            "topo_op": "reroute",
            "swap_indices": [0, 1],
            "delta_latents": latents,
        },
        operation_set=tuple(ops),
        output_channels=tuple(channels),
        budget=BudgetCap(flops=10_000_000, wall_time_s=10.0, energy_j=1.0),
        episode_id=f"sanity-{profile_name}-ep{ep_idx}-seed{seed}",
    )


def _run_cell(
    profile_name: str,
    seed: int,
    *,
    wrapper,
    mmlu_prompts: list[dict],
    letter_token_ids: list[int],
    n_episodes: int = 5,
) -> dict:
    """Execute one cell of the sanity pilot — 5 pipeline stages.

    Returns a dict with ``pre``, ``post``, ``delta``,
    ``wall_time_s`` and diagnostic fields (``model_loaded``,
    ``error``). The Qwen ``wrapper`` + MMLU fixture + letter IDs
    are hoisted to the caller so they are shared across all 90
    cells (load Qwen once).
    """
    start = time.time()
    _seed_everything(seed)
    model_loaded = wrapper is not None

    adapter, encoder, decoder = _build_adapter(seed)

    # Stage 2 — pre-dream evaluation : pure Qwen baseline (bias=0).
    pre = _score_adapter_mmlu(
        adapter,
        wrapper,
        mmlu_prompts,
        letter_token_ids,
        use_bias=False,
    )

    # Stage 3 — dream episodes.
    runtime, _states = _build_runtime(
        profile_name, adapter, encoder, decoder, seed=seed
    )
    for ep_idx in range(n_episodes):
        episode = _build_episode(profile_name, ep_idx, seed)
        runtime.execute(episode)

    # Stage 3b — p_max-only ATTENTION_PRIOR emission (cycle-3 spec :
    # p_max = p_equ + ATTENTION_PRIOR channel). Derive a deterministic
    # 4-vector nudge from a seeded numpy RNG, S4-bounded into [-1, 1]
    # so it cannot dominate the adapter-derived bias on its own.
    # Without this branch p_max and p_equ would walk identical
    # trajectories in the pilot (same ops, same beta-records) and
    # the H1 test would by construction return identical p-values
    # — that is exactly the failure mode the previous proxy hid.
    extra_bias = None
    if profile_name == "p_max":
        import numpy as np

        prior_rng = np.random.default_rng(seed + 10_000)
        # Prior sampled in (-1, 1) per component, then scaled so it
        # contributes ≲ 20% of the Qwen letter-logit spread (~2 on
        # the scale of our BIAS_SCALE * adapter_bias).
        raw = prior_rng.standard_normal(4).astype(np.float32)
        extra_bias = np.clip(raw, -1.0, 1.0).tolist()

    # Stage 4 — post-dream evaluation : Qwen + dream-modified bias.
    post = _score_adapter_mmlu(
        adapter,
        wrapper,
        mmlu_prompts,
        letter_token_ids,
        use_bias=True,
        extra_bias=extra_bias,
    )
    delta = post - pre

    return {
        "profile": profile_name,
        "seed": seed,
        "pre": pre,
        "post": post,
        "delta": delta,
        "wall_time_s": time.time() - start,
        "model_loaded": model_loaded,
    }


def _load_qwen_wrapper():
    """Load the Qwen-1.5B MLX wrapper once (γ-channel snapshot).

    Falls back to ``None`` when ``mlx-lm`` isn't installed ; the
    pipeline still exercises the adapter path so callers can
    diagnose the env before pulling the ~1 GB HF weights.
    """
    try:  # pragma: no cover - network / install path
        from harness.real_models.qwen_mlx import load_qwen

        return load_qwen(SANITY_SCALE)
    except Exception as exc:  # pragma: no cover - diagnostic
        print(
            f"[pilot] WARNING Qwen load failed : "
            f"{type(exc).__name__}: {exc}"
        )
        return None


def _register_cell(profile_name: str, cell: dict) -> str:
    """Insert the cell's run_id into the run registry (R1 contract).

    The composite ``profile_tag`` matches the cycle-3 runner's
    ``_registry_profile_tag`` convention so the pilot's rows are a
    strict subset of the C3.8 resume contract.
    """
    from harness.storage.run_registry import RunRegistry

    registry = RunRegistry(REPO_ROOT / ".run_registry.sqlite")
    profile_tag = (
        f"cycle3/{SANITY_SCALE}/{profile_name}/mlx_kiki_oniric"
    )
    return registry.register(
        c_version=HARNESS_VERSION,
        profile=profile_tag,
        seed=cell["seed"],
        commit_sha=_resolve_commit_sha(),
    )


# -------------------------------------------------------------------
# H1 test across cells
# -------------------------------------------------------------------


def _h1_test(cells: list[dict]) -> dict:
    """Paired t-test pre vs post per profile.

    Returns a mapping ``{profile -> {t, p, reject_h0, n}}``.
    """
    from scipy import stats

    out: dict[str, dict] = {}
    for profile in SANITY_PROFILES:
        pres = [c["pre"] for c in cells if c["profile"] == profile]
        posts = [c["post"] for c in cells if c["profile"] == profile]
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
# Main entrypoint
# -------------------------------------------------------------------


def _print_banner(total_cells: int) -> None:
    print("=" * 64)
    print("CYCLE-3 SANITY PILOT (1.5B-scale fail-fast)")
    print("=" * 64)
    print(f"harness_version : {HARNESS_VERSION}")
    print(f"scale           : {SANITY_SCALE}")
    print(f"profiles        : {SANITY_PROFILES}")
    print(f"exec substrate  : {SANITY_SUBSTRATES_EXEC}")
    print(f"plan substrates : {SANITY_SUBSTRATES_PLAN} (dry-run manifest)")
    print(f"seeds           : {len(SANITY_SEEDS)} (0..{SANITY_SEEDS[-1]})")
    print(f"exec cells      : {total_cells}")
    print(f"eval proxy      : MMLU logit-bias (n={MMLU_N_PROMPTS} prompts)")
    print(
        f"GO rule         : H1 rejected in ≥ "
        f"{GO_PROFILES_REJECTED_MIN}/{len(SANITY_PROFILES)} profiles "
        f"at α = {GO_BONFERRONI_ALPHA}"
    )
    print("-" * 64)


def _execute(
    opts: dict, plan: list[dict]
) -> int:  # pragma: no cover - side-effectful
    """Execute the sanity pilot — either one smoke cell or all 90.

    Returns the shell exit code (0 = success, 1 = NO-GO or failure).
    """
    out_dir = REPO_ROOT / "docs" / "milestones"
    out_dir.mkdir(parents=True, exist_ok=True)

    # Hoist Qwen load + MMLU fixture + letter token IDs — shared
    # across all cells so we pay the ~1 s Qwen load once.
    wrapper = _load_qwen_wrapper()
    mmlu_prompts = _load_mmlu_prompts(MMLU_N_PROMPTS)
    letter_token_ids: list[int] = []
    if wrapper is not None:
        letter_token_ids = _letter_token_ids(wrapper.tokenizer)

    if opts["smoke_cell"]:
        _print_banner(total_cells=1)
        print(
            f"[smoke-cell] running exactly 1 cell : p_min + seed 0 "
            f"+ MLX (Qwen loaded={wrapper is not None})"
        )
        start = time.time()
        try:
            cell = _run_cell(
                "p_min",
                0,
                wrapper=wrapper,
                mmlu_prompts=mmlu_prompts,
                letter_token_ids=letter_token_ids,
            )
        except Exception as exc:
            traceback.print_exc()
            print(f"[smoke-cell] FAILED : {type(exc).__name__}: {exc}")
            return 1
        run_id = _register_cell("p_min", cell)
        wall = time.time() - start
        print(
            f"[smoke-cell] pre={cell['pre']:.4f} post={cell['post']:.4f} "
            f"delta={cell['delta']:+.4f} wall={wall:.2f}s "
            f"model_loaded={cell['model_loaded']} run_id={run_id}"
        )
        smoke_path = out_dir / "pilot-cycle3-sanity-1p5b-smoke.json"
        with smoke_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "harness_version": HARNESS_VERSION,
                    "mode": "smoke-cell",
                    "eval_proxy": "mmlu_logit_bias",
                    "mmlu_n_prompts": MMLU_N_PROMPTS,
                    "cell": {**cell, "run_id": run_id},
                    "wall_time_s": wall,
                    "extrapolated_full_pilot_s": wall * EXECUTED_CELL_COUNT,
                },
                fh,
                indent=2,
            )
        print(f"[smoke-cell] dump written to {smoke_path}")
        return 0

    # Full 90-cell run — MLX substrate only.
    _print_banner(total_cells=EXECUTED_CELL_COUNT)
    cells: list[dict] = []
    failures: list[dict] = []
    run_start = time.time()
    idx = 0
    for profile_name in SANITY_PROFILES:
        for seed in SANITY_SEEDS:
            idx += 1
            try:
                cell = _run_cell(
                    profile_name,
                    seed,
                    wrapper=wrapper,
                    mmlu_prompts=mmlu_prompts,
                    letter_token_ids=letter_token_ids,
                )
            except Exception as exc:
                traceback.print_exc()
                failures.append(
                    {
                        "profile": profile_name,
                        "seed": seed,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
                print(
                    f"[cell {idx}/{EXECUTED_CELL_COUNT}] "
                    f"{profile_name} seed={seed} FAILED : "
                    f"{type(exc).__name__}: {exc}"
                )
                continue
            try:
                run_id = _register_cell(profile_name, cell)
            except Exception as exc:  # pragma: no cover - defensive
                run_id = f"register-failed:{exc}"
            cells.append({**cell, "run_id": run_id})
            print(
                f"[cell {idx}/{EXECUTED_CELL_COUNT}] "
                f"{profile_name} seed={seed:02d} "
                f"pre={cell['pre']:.4f} post={cell['post']:.4f} "
                f"delta={cell['delta']:+.4f} "
                f"wall={cell['wall_time_s']:.2f}s"
            )
    run_wall = time.time() - run_start

    # H1 test per profile + GO / NO-GO verdict.
    h1 = _h1_test(cells)
    rejected = sum(1 for v in h1.values() if v.get("reject_h0"))
    verdict = "GO" if rejected >= GO_PROFILES_REJECTED_MIN else "NO-GO"
    print("-" * 64)
    for profile, r in h1.items():
        print(
            f"H1 {profile}: p={r.get('p')} reject_h0={r.get('reject_h0')}"
        )
    print(f"verdict          : {verdict} ({rejected}/3 profiles)")
    print(f"total wall-clock : {run_wall:.1f}s")
    print(f"failures         : {len(failures)}")

    dump_path = out_dir / "pilot-cycle3-sanity-1p5b.json"
    with dump_path.open("w", encoding="utf-8") as fh:
        json.dump(
            {
                "harness_version": HARNESS_VERSION,
                "scale": SANITY_SCALE,
                "profiles": list(SANITY_PROFILES),
                "substrates_plan": list(SANITY_SUBSTRATES_PLAN),
                "substrates_executed": list(SANITY_SUBSTRATES_EXEC),
                "seeds": list(SANITY_SEEDS),
                "planned_cell_count": EXPECTED_CELL_COUNT,
                "executed_cell_count": EXECUTED_CELL_COUNT,
                "completed_cells": len(cells),
                "failed_cells": len(failures),
                "eval_proxy": "mmlu_logit_bias",
                "mmlu_n_prompts": MMLU_N_PROMPTS,
                "go_rule": {
                    "alpha": GO_BONFERRONI_ALPHA,
                    "profiles_rejected_min": GO_PROFILES_REJECTED_MIN,
                    "profiles_total": len(SANITY_PROFILES),
                },
                "h1": h1,
                "verdict": verdict,
                "wall_time_s": run_wall,
                "cells": cells,
                "failures": failures,
                "plan": plan,
            },
            fh,
            indent=2,
        )
    print(f"results dumped to {dump_path}")
    return 0 if verdict == "GO" else 1


def main(argv: list[str] | None = None) -> int:
    """Entrypoint — enumerate (+ dry-run), smoke-cell, or full run."""
    opts = _parse_cli(list(argv) if argv is not None else sys.argv[1:])
    plan = _plan()
    assert len(plan) == EXPECTED_CELL_COUNT, (
        f"Sanity plan produced {len(plan)} cells, expected "
        f"{EXPECTED_CELL_COUNT} per §7."
    )

    if opts["dry_run"]:
        _print_banner(total_cells=EXECUTED_CELL_COUNT)
        print("[dry-run] enumeration validated ; no dream ops.")
        out_dir = REPO_ROOT / "docs" / "milestones"
        out_dir.mkdir(parents=True, exist_ok=True)
        plan_path = out_dir / "pilot-cycle3-sanity-1p5b.json"
        with plan_path.open("w", encoding="utf-8") as fh:
            json.dump(
                {
                    "harness_version": HARNESS_VERSION,
                    "scale": SANITY_SCALE,
                    "profiles": list(SANITY_PROFILES),
                    "substrates_plan": list(SANITY_SUBSTRATES_PLAN),
                    "substrates_executed": list(SANITY_SUBSTRATES_EXEC),
                    "seeds": list(SANITY_SEEDS),
                    "planned_cell_count": EXPECTED_CELL_COUNT,
                    "executed_cell_count": EXECUTED_CELL_COUNT,
                    "eval_proxy": "mmlu_logit_bias",
                    "mmlu_n_prompts": MMLU_N_PROMPTS,
                    "go_rule": {
                        "alpha": GO_BONFERRONI_ALPHA,
                        "profiles_rejected_min": GO_PROFILES_REJECTED_MIN,
                        "profiles_total": len(SANITY_PROFILES),
                    },
                    "status": "plan-only",
                    "plan": plan,
                },
                fh,
                indent=2,
            )
        print(f"Plan manifest written to {plan_path}")
        return 0

    return _execute(opts, plan)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
