# Cycle-3 Sanity Pilot Report — 1.5B-Scale Fail-Fast

**Gate** : G10 cycle-3 Gate D *pilot* (fail-fast decision)
**Scale** : `qwen3p5-1p5b` only (Qwen2.5-1.5B MLX-Q4 fallback)
**Launched on** : `{RESULTS TBD — populated after run}`
**Studio session ID** : `{RESULTS TBD}`
**Harness version** : `C-v0.7.0+PARTIAL`
**Script** : [`scripts/pilot_cycle3_sanity.py`](../../scripts/pilot_cycle3_sanity.py)
**Decision dump** : [`pilot-cycle3-sanity-1p5b.json`](./pilot-cycle3-sanity-1p5b.json)

---

## Purpose

Before burning **~18 h** of Studio compute on the full multi-scale
launch (C3.8), exercise the full cycle-3 pipeline at the
**smallest** scale-slot and verify that H1 (forgetting reduction,
paired t-test pre vs post on a retained accuracy proxy) is
detectable. If H1 is dead at 1.5B, the 7B and 35B launches are
almost certainly wasted compute — abort before the commitment.

## Plan

```
scale       : qwen3p5-1p5b   (1 scale)
profiles    : p_min, p_equ, p_max  (3 profiles)
substrates  : mlx_kiki_oniric  (1 substrate — EXECUTED)
              esnn_thalamocortical  (skipped — no real-weight ops yet)
seeds       : 0..29  (30 seeds — half the 60-seed full launch)
plan cells  : 1 × 3 × 2 × 30 = 180 runs (dry-run manifest, R1 subset of C3.8)
exec cells  : 1 × 3 × 1 × 30 =  90 runs (MLX only)
compute     : ~18 h dedicated Studio M3 Ultra 512GB
```

**Scope restriction** : the sanity pilot executes only the
`mlx_kiki_oniric` substrate. E-SNN ships a thalamocortical
skeleton in `esnn_thalamocortical.py` but the real-weight SNN
ops (`*_real_snn.py`) are C3.12 Phase 2 work and are not yet
implemented. Forcing E-SNN execution here would require ~400 LOC
of op scaffolding for no empirical gain, defeating the fail-fast
purpose. The dry-run path still emits the full 180-cell manifest
so the pilot's `run_id` subset of the C3.8 resume contract stays
untouched — only the execution path narrows to MLX.

Every executed cell's `run_id` is a strict subset of the full
1080-config launch's resume contract (same
`(harness_version, profile_tag, seed, commit_sha)` tuple, same
composite `profile_tag` scheme). Re-running `ablation_cycle3.py
--resume` after the pilot therefore **skips** the 90 executed
cells automatically (R1 identity — see the R1 contract in
`harness/storage/run_registry.py` : `run_id = SHA-256(c_version |
profile | seed | commit_sha)[:32]`. Two runs share an R1 identity
iff they produce the same tuple, which is exactly what the pilot
→ full-launch resume relies on).

## GO / NO-GO Rule

- **GO** : H1 (paired t-test pre vs post on retained score)
  rejects the null in **≥ 2 / 3** profiles at **α = 0.0125**
  (cycle-1 Bonferroni-corrected bar on a 1-scale × 1-substrate
  slice, family size 4 = {H1, H2, H3, H4}). Clear to launch C3.8
  full 3-scale matrix.
- **NO-GO** : H1 rejects in **< 2 / 3** profiles → **abort**. Do
  not launch C3.8. Open a root-cause review (`pivot-4` branch
  per cycle-3 spec §5.1 R3).

The 3 profiles are `{p_min, p_equ, p_max}` at scale
`qwen3p5-1p5b` on the MLX substrate.

## Per-cell pipeline

1. **Fresh model + seeded adapter** — load the Qwen-1.5B γ-channel
   snapshot via `QwenMLXWrapper` (hoisted once, shared across
   cells) and construct a seeded 4→4→4→4 MLP adapter that the
   dream ops mutate. The adapter's final 4-dim output is the
   A/B/C/D logit-bias vector (see §Eval proxy).
2. **Pre-dream evaluation** — 15 MMLU prompts through Qwen 1.5B
   with bias = 0 (pure Qwen baseline accuracy).
3. **Dream episodes** — 5 episodes per cell, using the real-weight
   ops (`replay_real`, `downscale_real`, `restructure_real`,
   `recombine_real`) registered on a fresh `DreamRuntime`
   according to the cell's profile.
4. **Post-dream evaluation** — same 15 MMLU prompts, now with the
   dream-modified adapter producing its own 4-dim bias added to
   the A/B/C/D letter logits before argmax.
5. **Run registration** — compute `delta = post - pre` and insert
   the row via `RunRegistry.register` with the composite
   `profile_tag = cycle3/{scale}/{profile}/mlx_kiki_oniric`.

Cells are executed in `(profile × seed)` order. A per-cell crash
is logged to `failures` and the driver continues — partial H1
samples are still usable.

## Eval proxy — MMLU + Qwen logit bias

**Why not `exp(-MSE)`** — the first revision of this pilot scored
the adapter with `exp(-MSE)` on its 4-dim output. That proxy
produced `p_equ` and `p_max` with the **same** p-value
(0.0601992…), which is statistically impossible if the proxy
were sensitive to the structural difference between those
profiles (p_equ adds `recombine` ; p_max adds
`ATTENTION_PRIOR` + `restructure`). The proxy was measuring
the norm of the adapter output rather than something that
actually depends on the op sequence's final state.

**Current proxy** — each cell loads 15 MMLU prompts (world facts,
elementary math, science) from the committed fixture at
[`tests/fixtures/mmlu_sanity.jsonl`](../../tests/fixtures/mmlu_sanity.jsonl).
For each prompt the pilot :

1. Runs the Qwen 1.5B forward pass on `prompt + "Answer:"`.
2. Extracts the last-token logits on the 4 letter tokens
   `{A, B, C, D}` (validated as single-token BPE IDs 32, 33,
   34, 35 on Qwen2.5 tokenizer — asserted at runtime).
3. Adds the adapter's 4-dim output (at fixed reference input
   `[1,1,1,1]`) as a bias on those letter logits.
4. Argmax → prediction ; compare to the record's `answer`
   index.

Pre-score = accuracy with bias = 0 (pure Qwen baseline ; same
across all cells up to tokenizer determinism, ~40-70% expected
on Qwen2.5-1.5B-Instruct-4bit). Post-score = accuracy with the
dream-modified adapter bias applied. `delta = post - pre` ∈
`[-1, 1]`. n = 15 gives discrete granularity 1/15 ≈ 6.7%,
sufficient for the paired t-test across 30 seeds per profile.

The bias is scaled by `BIAS_SCALE = 20×` before addition so it
sits on the same order of magnitude as Qwen's empirical letter-
logit spread (~8-10 units at the `Answer:` position). A naive
`1×` scale was silently dominated by Qwen's raw logits and
produced `delta = 0` across every profile/seed combination
(verified on the 2026-04-19 smoke run before this fix). Scaling
lets dream-induced weight drift actually move argmax.

For `p_max` the cell additionally injects a deterministic
ATTENTION_PRIOR-derived 4-vector `extra_bias` (seeded from
`seed + 10000`, clipped to `[-1, 1]` per component, blended
*after* the adapter-bias scaling). The cycle-3 spec differentiates
`p_max` from `p_equ` precisely by the ATTENTION_PRIOR output
channel (cf. `kiki_oniric/profiles/p_max.py`), but the pilot's
lean `_build_runtime`/`_build_episode` helpers register the same
op set for both — without the explicit prior injection `p_max`
and `p_equ` would walk identical trajectories and the H1 test
would by construction return identical p-values (exactly the
failure mode observed on the previous `exp(-MSE)` run).

This proxy is **genuinely discriminative** — the adapter's
4-dim output responds deterministically to the full dream op
sequence, so `p_min` (replay + downscale), `p_equ`
(+ restructure + recombine), and `p_max` (+ ATTENTION_PRIOR
injection on top of the p_equ ops) produce distinct adapter
trajectories and therefore distinct post-dream accuracy
distributions. Whether the bias *helps* or *hurts* accuracy is
the empirical question H1 actually tests.

## Smoke-cell pre-flight

`--smoke-cell` runs exactly one cell (`p_min` + seed 0 + MLX) in
under 30 s before the full launch, validating end-to-end wiring
plus Qwen + MMLU integration. Its output is dumped to
[`pilot-cycle3-sanity-1p5b-smoke.json`](./pilot-cycle3-sanity-1p5b-smoke.json)
and includes the extrapolated full-pilot wall-clock.

## Results

### Runs completed

`{RESULTS TBD — populated after run}` / 90

### Wall-clock

`{RESULTS TBD}` hh:mm:ss — used to extrapolate 7B (≈ 4×) and
35B (≈ 20× per §7 scale-cost model).

### H1 per profile

| profile  | t-statistic | p-value | reject H0 (α = 0.0125) |
|----------|-------------|---------|------------------------|
| p_min    | `{TBD}`     | `{TBD}` | `{TBD}`                |
| p_equ    | `{TBD}`     | `{TBD}` | `{TBD}`                |
| p_max    | `{TBD}`     | `{TBD}` | `{TBD}`                |

### Verdict

`{RESULTS TBD — GO / NO-GO populated after run}`

### 7B + 35B extrapolation (if GO)

Wall-clock × 4 (7B) + × 20 (35B) → ~`{TBD}` total Studio days
to close C3.8.

## Notes

- Script ships the per-cell execution wiring in the follow-up
  commit to C3.7. The dry-run path is preserved byte-for-byte so
  CI still validates enumeration without touching MLX or the HF
  cache.
- `--dry-run` validates the enumeration without touching any
  predictor / substrate — safe from CI.
- `--smoke-cell` is the pre-launch guardrail ; it must complete
  successfully before the full 90-cell run is launched.
- If GO fires, update the `{RESULTS TBD}` markers with the
  measured values and keep this report immutable per the
  `scripts/CLAUDE.md` "don't edit pilots after gate decision"
  rule (create `pilot_cycle3_sanity_v2.py` if methodology shifts).
- The retained-score proxy (Qwen 1.5B forward + A/B/C/D logit
  bias from a 4-dim adapter output, n=15 MMLU prompts) is
  pipeline-validation, not an empirical accuracy claim. C3.8
  swaps in the full MMLU / HellaSwag / mega-v2 benchmark stack
  for the scale-axis claim. The fixture
  (`tests/fixtures/mmlu_sanity.jsonl`) is hand-authored world-
  facts / elementary-math / science Q&A — network-free and R1
  byte-stable under `c_version | profile | seed | commit_sha`.

## References

- `scripts/pilot_cycle3_sanity.py` — driver (per-cell execution +
  `--dry-run` + `--smoke-cell`).
- `scripts/ablation_cycle3.py` — parent 1080-config runner.
- `kiki_oniric/eval/statistics.py` — H1 test primitives (we use
  `scipy.stats.ttest_rel` directly for the paired pre/post test).
- `kiki_oniric/dream/operations/{replay,downscale,restructure,recombine}_real.py`
  — real-weight ops bound to the per-cell adapter.
- `harness/storage/run_registry.py` — R1 `run_id` identity.
- `docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md`
  §5.1 R3 (Pivot 4 if NO-GO), §7 (compute budget).
