# OSF Pre-Registration — G6 pilot (micro-kiki Qwen-35B × MMLU subdomain CL stream)

**Project** : dreamOfkiki
**Parent registration** : 10.17605/OSF.IO/Q6JYN (Cycle 1)
**Amendment ancestry** : OSF amendment #1 (Bonferroni family
  restructure for cycle-3) at `10.17605/OSF.IO/TPM5S` ; this G6
  pre-reg is a daughter document of that amendment family, locked
  before any G6 cell registers in `RunRegistry`.
**PI** : Clement Saillant (L'Electron Rare)
**Date drafted** : 2026-05-03
**Lock target** : before any G6 cell is registered in
  `harness/storage/run_registry.RunRegistry`.

## 0. G4-bis amendment notice (binding context for H_NEW)

The original H_NEW formulation in the early G6 plan draft assumed
a **positive** synthetic g_h1 from G4-bis (replay-coupling pilot
on the MLX substrate) so that "real LLM g >= synthetic g - 0.10"
made sense as a non-inferiority margin test.

The G4-bis run that landed
(`docs/milestones/g4-pilot-2026-05-03-bis.json`, commit `96b9dae`)
returned :

- g4_bis_g_h1 = -2.3067 (sign-reversed from the H1 prediction
  direction ; H1 expected `g_h1 >= 0.21`, observed g_h1 was
  strongly negative)
- g4_bis_above_hu_2020_lower_ci = false (distance from Hu 2020 CI
  lower bound : -2.5967)
- g4_bis_h_dr4_observed_monotonic = true but **degenerate** :
  P_min / P_equ / P_max retention vectors were bit-identical
  because RESTRUCTURE + RECOMBINE remained spectator-only on the
  binary MLP (no weight mutation through those handlers ; only
  REPLAY + DOWNSCALE coupled).

H_NEW as originally drafted ("real LLM g >= synthetic G4-bis g
- 0.10 margin") therefore becomes mathematically vacuous : because
g_synthetic is far below zero, *any* observed real-LLM g satisfies
the inequality trivially, and the test would no longer measure
what it was meant to measure (synthetic-to-real generalisation of
a *positive* consolidation signal).

**Reformulation (binding for this pre-reg)** : H_NEW is recast as
exploratory **infrastructure validation** — does running the full
substrate-handler pipeline against a real-LLM evaluation surface
produce *any* non-zero retention/forgetting effect, regardless of
sign ? STABLE EC-axis promotion is conditioned on a future Path A
confirmation (Studio + KIKI-Mac_tunner + mlx_lm.lora real
fine-tune); Path B is documented as exploratory only and never
triggers a STABLE / UNSTABLE bump (see §6).

The chain of scientific review is therefore :

```
G4 spectator  ->  G4-bis coupling  ->  G6 Path B inference-only
(synthetic,    (synthetic, partial   (real LLM eval surface,
 H1/H3 null)    weight mutation,      exploratory only ;
                g_h1 sign-reversed)   Path A future-work)
```

All three pilots are publishable, and all three are honest about
their limitations. This pre-reg locks H_NEW as an
**exploratory** rather than confirmatory test on this run.

## 1. Study design

Within-architecture × within-benchmark sweep on the
`kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate` (Qwen3.6-35B-A3B
+ LoRA, rank 16, alpha 16). Five MMLU subdomains are presented as a
sequential continual-learning stream :

S1 = anatomy → S2 = astronomy → S3 = business_ethics →
S4 = clinical_knowledge → S5 = college_biology

For each (arm, seed) cell the driver :

1. Loads a fresh wrapper (Path A) or fresh adapter buffer (Path B).
2. For i in 1..5 :
   a. Adapt to subdomain S_i (Path A: LoRA fine-tune on 100 train
      examples, lr 5e-5, 50 inner steps; Path B: deterministic
      perturbation of the LoRA delta tensor seeded by (subdomain,
      seed)).
   b. (Optional, profile-dependent) Inject one DreamEpisode whose
      operation set is dictated by the active profile.
   c. Evaluate on S_1..S_i held-out eval splits (n_eval per
      subject, letter-argmax proxy via
      `harness.real_benchmarks.mmlu.evaluate_mmlu`).
3. Compute per-subdomain forgetting :
   `forgetting[S_j] = acc[S_j after S_j] - acc[S_j after S_5]` for j < 5.
4. Compute retention metric :
   `retention = mean(acc[S_j after S_5] / max(acc[S_j after S_j], eps))`
   over j in 1..4.

## 2. Hypotheses

### H1' — P_equ retention floor on real LLM matches Hu 2020 anchor
**Statement** : observed Hedges' g of (P_equ retention vs baseline
retention) on the real Qwen + MMLU stream is >= HU_2020_OVERALL.ci_low
(0.21).
**Operationalisation** :
- g_h1' = compute_hedges_g(retention[P_equ], retention[baseline])
- Reject H0 iff g_h1' >= 0.21
- Welch one-sided (baseline, P_equ) at α = 0.05 / 4 (Bonferroni
  for 4 hypotheses)

### H3' — P_min retention decrement on real LLM matches Javadi 2024
**Statement** : observed |Hedges' g| of (P_min vs baseline) >=
JAVADI_2024_OVERALL.ci_low (0.13), with negative sign (decrement).
**Operationalisation** :
- g_h3' = compute_hedges_g(retention[P_min], retention[baseline])
- Reject H0 iff g_h3' <= -0.13
- Welch one-sided (P_min, baseline) at α = 0.05 / 4

### H_DR4' — DR-4 monotonicity on real LLM
**Statement** : mean retention is monotonically ordered
P_max >= P_equ >= P_min on the real Qwen stream.
**Operationalisation** :
- Jonckheere-Terpstra trend on
  [retention[P_min], retention[P_equ], retention[P_max]]
- α = 0.05 (separate family from Welch tests)

### H_NEW — exploratory infrastructure validation (amended 2026-05-03)

**Original (vacated by G4-bis amendment)** : "real LLM g_h1' >=
synthetic G4-bis g_h1 - 0.10 (one-sided non-inferiority margin)".
This is mathematically vacuous given G4-bis g_h1 = -2.31 :
any real-LLM g_h1' trivially satisfies the inequality. The
hypothesis as drafted no longer measures synthetic-to-real
generalisation of a *positive* consolidation signal.

**Amended (binding for this run)** : Given the G4-bis synthetic
finding (g_h1 = -2.31 sign-reversed; H_DR4 degenerate equal-means
due to spectator-only RESTRUCTURE + RECOMBINE on binary MLP),
H_NEW is reformulated as :

> Do we observe ANY non-zero retention or forgetting effect on a
> real LLM evaluation surface (Qwen 1.5B fallback in Path B,
> Qwen 35B in Path A future), regardless of sign, when the
> dream-handler pipeline is wired through `MicroKikiSubstrate` ?

**Operationalisation** :
- effect_observed_real_llm = (
      max_arm(|mean_retention[arm] - mean_retention[baseline]|)
      > infrastructure_threshold (set to 1e-6)
  )
- This is **infrastructure validation**, not a confirmatory
  inference. STABLE EC-axis promotion is conditioned on a future
  Path A run (real LoRA fine-tune on Studio); Path B never
  triggers STABLE / UNSTABLE per §6 below.
- No formal Bonferroni correction is applied to H_NEW under the
  amended formulation (it is exploratory). The H1' / H3' / H_DR4'
  family-wise correction (α/3) on Path B remains applicable but
  is itself flagged exploratory in the milestone dump.

## 3. Pre-specified analyses

- H1', H3' : `kiki_oniric.eval.statistics.welch_one_sided`
  + `compute_hedges_g`
  + `harness.benchmarks.effect_size_targets.{HU_2020_OVERALL, JAVADI_2024_OVERALL}`
- H_DR4' : `kiki_oniric.eval.statistics.jonckheere_trend`
- H_NEW (amended) : direct comparison of arm means against
  baseline (no inferential statistic ; presence/absence of effect
  only) — implemented in `experiments.g6_mmlu_stream.run_g6._h_new_verdict`.
- Bonferroni family size = 3 (H1', H3', H_DR4'),
  α_per_test = 0.05 / 3 ; H_NEW is exploratory (no correction).

## 4. Sample size / power

- Path A : N = 3 seeds per arm × 4 arms = 12 cell sequences (60
  forgetting measurements).
- Path B : same N (compute budget irrelevant for Path B).
- Power note : N = 3 vs N = 3 is severely underpowered for
  absolute g magnitudes (minimum detectable g at 80 % power
  ≈ 2.4). The pilot is **exploratory** for absolute effects;
  pre-registered floors (Hu, Javadi) serve only to anchor the
  *direction* of the verdict.
- A confirmatory N >= 30 follow-up is scheduled iff this pilot
  returns exploratory positive evidence (g_h1' >= 0.21 with Welch
  p < α/3) on Path A.

## 5. Data exclusion rules

- Cells where MicroKikiSubstrate raises any BLOCKING invariant
  (S1 retained-non-regression, S2 finite weights) are excluded;
  their EpisodeLogEntry.error is recorded as exclusion reason.
- Cells where `acc[S_1 after S_1] < 0.30` (random-baseline + 5%)
  are excluded as underperforming-baseline (Qwen failed to learn
  S_1 at all, retention is meaningless).
- Cells where Path A LoRA fine-tune raises an MLX OOM are
  excluded; the run continues with the remaining seeds. If > 50%
  of cells OOM, the pilot is aborted and the decision log records
  the failure.

## 6. DualVer outcome rules (binding)

| Outcome | EC bump | Rationale |
|---|---|---|
| Path A: H1', H3', H_DR4' all reject H0 in predicted direction AND H_NEW infrastructure-effect observed | PARTIAL → STABLE (scope: G4 + G5 + G6) | Three confirmatory pilots cross §12.3 STABLE bar |
| Path A: H1' confirmed, others mixed | stays PARTIAL | Partial confirmation only |
| Path A: any predicted direction violated (e.g. P_max < P_min mean) | PARTIAL → UNSTABLE | §12.3 falsification rule |
| Path B (any outcome) | stays PARTIAL | Path B is exploratory only — never triggers STABLE/UNSTABLE |

No FC bump in any outcome (no axiom or primitive change).

## 7. Deviations from pre-registration

Any post-hoc deviation logged in
`docs/osf-deviations-g6-<date>.md` (separate dated immutable).

## 8. Data and code availability

- Pilot driver : `experiments/g6_mmlu_stream/run_g6.py`
- Stream loader : `experiments/g6_mmlu_stream/stream.py`
- Train shim (Path A, future-work) :
  `experiments/g6_mmlu_stream/micro_kiki_train.py`
- Inference shim (Path B) :
  `experiments/g6_mmlu_stream/micro_kiki_inference.py`
- Verdict anchors :
  `harness.benchmarks.effect_size_targets.{HU_2020_OVERALL, JAVADI_2024_OVERALL}`
- Run registry : `harness/storage/run_registry.RunRegistry`,
  SQLite at `.run_registry.sqlite`
- Outcome dump : `docs/milestones/g6-pilot-pathB-2026-05-03.{json,md}`
- Decisions log : `docs/milestones/g6-pilot-decisions-2026-05-03.md`
- Parent G4-bis dump :
  `docs/milestones/g4-pilot-2026-05-03-bis.{json,md}`

## 9. Path A vs Path B disclosure

The pilot host availability check (Task 0.5 step 1) determines
which branch executes. The path selection is **locked before any
cell registers** and recorded in the decisions log. Switching
paths post-hoc requires a deviation document + a new dated
pre-reg.

This run on M1 Max / 32 GB selected **Path B** (decisions log
field PATH = B). Path A (Studio + KIKI-Mac_tunner +
mlx_lm.lora) is unavailable on this host and is scheduled as
future work; its modules are stubbed accordingly.

## 10. Contact

Clement Saillant — clement@saillant.cc — L'Electron Rare, France

---

**Lock this document before any G6 cell is registered in the run
registry.**
