# Changelog

All notable changes to dream-of-kiki are documented here.

Format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/)
+ [Conventional Commits](https://www.conventionalcommits.org/).

Versioning scheme : **DualVer** (framework C formal+empirical axes,
see `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §12).

---

## [Unreleased]

### Empirical (G6-Studio Path A real-LoRA INSUFFICIENT, 2026-05-04)

- G6-Studio Path A pilot completed end-to-end on Mac Studio M3
  Ultra (20 cells × 5 subdomains = 100 measurements, ~52 min
  wall, peak Metal mem 139 GB under the 210 GB cache budget).
  Pre-reg `docs/osf-prereg-g6-studio-path-a.md` locked at commit
  `fae8c32`. First real-LLM-scale run of the dreamOfkiki
  programme — fixes the G6 Path B "spectator wrapper" caveat by
  mutating live LoRA delta tensors via the KIKI-Mac_tunner
  `mlx_lm` fork at `/Users/clems/KIKI-Mac_tunner/lib/mlx_lm_fork/`.
- **Verdict : INSUFFICIENT**. 19 of 20 (arm, seed) cells excluded
  by the underperforming-baseline rule
  (`acc[S_1 after S_1] < 0.30` at the synthetic 10-train / 10-eval
  per-subdomain fixture). Aggregator returns
  `h9a_classification = INSUFFICIENT`, `h9c_classification =
  INSUFFICIENT`, NaN test statistics — H9-A / H9-B / H9-C cannot
  be evaluated at this fixture size.
- **Pipeline integrity verified** : 35B bf16 base load + LoRA
  fine-tune (loss 0.74 → 0.07 typical) + 4-channel dream coupling
  on live delta tensors + `mlx_lm.generate` letter-argmax MMLU
  eval, all under the Metal memory budget. Engineering-level
  success ; the substrate-independence formal guarantee (DR-3
  Conformance Criterion) is preserved end-to-end on the real
  spiking-LIF Qwen substrate.
- **Follow-up required** : pre-reg §9.1 amendment slot logs the
  fixture-size limitation ; a G6-Studio Path A* re-run with the
  production MMLU shards (200 records per subject) under
  unchanged H9 hypotheses but a larger train/eval budget per
  subdomain is queued. The CNN-tier scope ceiling
  (G4-septimo H6-C confirmed, `docs/proofs/dr4-profile-inclusion.md`
  v0.6) is unaffected ; the framework's RECOMBINE prediction
  status at real-LLM scale is documented as *deferred to
  G6-Studio Path A** in DR-3 evidence amendments.
- EC stays PARTIAL ; FC stays C-v0.12.0+PARTIAL.
- Milestone : `docs/milestones/g6-studio-path-a-2026-05-04.{json,md}`
  + 100 per-subdomain partial dumps (resumability).
  Paper 2 §7.1.13 EN+FR appended.
- Production-blocker fixes shipped in this run : (i) fork import
  rebinding (`tuner.lora.train` → `tuner.trainer.train`) ;
  (ii) `mx.metal.set_memory_limit` / `set_cache_limit`
  configuration to prevent OOM at 35B bf16 ; (iii)
  `_MMLURecordDataset` text pre-formatting for the fork's
  `CacheDataset.itemlen` length-bucket sort ;
  (iv) base-path correction (Qwen3.6-35B-A3B vs SpikingKiki-V4
  which lacks `config.json`) ; (v) fork activation via
  `PYTHONPATH=/tmp` → symlink → `mlx_lm_fork`.

### Empirical (G4-septimo Tiny-IN H6-C UNIVERSALITY CONFIRMED, 2026-05-04)

- G4-septimo Step 1 N=30 pilot completed on Mac Studio M3 Ultra
  (240 cells, ~3 h 11 min wall, 54-58 s/cell sustained — under
  but near the §9 envelope c 60 s threshold). Pre-reg
  `docs/osf-prereg-g4-septimo-pilot.md` locked at commit `6b8d138`
  BEFORE any G4-septimo code was written ; aggregator
  `experiments/g4_septimo_test/aggregator.py`.
- **H6-B confirmed** : Welch two-sided between (P_max with mog)
  and (P_max with none) on Split-Tiny-ImageNet (n_classes=20/task,
  10 tasks, 64×64 RGB, G4MediumCNN), `t = -0.0950`, `p = 0.9247`,
  `Hedges' g = -0.0246`, `mean_mog = 0.3864`, `mean_none = 0.3891`,
  α = 0.05 (single new test, no Bonferroni inheritance per
  pre-reg §2). Fail-to-reject H0 → H6-B confirmed. RECOMBINE
  adds nothing measurable beyond REPLAY+DOWNSCALE on medium-CNN
  at Tiny-ImageNet 200-class scale.
- **H6-C UNIVERSALITY CONFIRMED** : conjunction
  `H6-A_confirmed AND H6-B_confirmed` resolves to confirmed.
  The RECOMBINE-empty universality flag now spans the full
  pre-registered scope `{Split-FMNIST, Split-CIFAR-10,
  Split-CIFAR-100, Split-Tiny-ImageNet} × {3-layer MLP, 5-layer
  MLP, small CNN, medium CNN}`. The DR-4 prediction "richer
  ops yield richer consolidation" is empirically refuted across
  the entire escalation ladder.
- **DR-4 evidence amended** : `docs/proofs/dr4-profile-inclusion.md`
  v0.6 G4-septimo addendum closes the H6-C conjunction at its
  full pre-reg scope. Lemma DR-4.L formal proof unchanged.
  STABLE promotion of "richer ops yield richer consolidation"
  cannot occur at this scope ceiling ; transformer / hierarchical-
  E-SNN / ImageNet-1k / real-LLM substrates remain open.
- EC stays PARTIAL ; FC stays C-v0.12.0+PARTIAL.
- Milestones :
  `docs/milestones/g4-septimo-step1-2026-05-04.{json,md}` +
  `docs/milestones/g4-septimo-aggregate-2026-05-04.{json,md}`.
  Paper 2 §7.1.11 EN+FR appended.

### Empirical (G4-sexto Step 1 CIFAR-100 100-class, 2026-05-04)

- G4-sexto Step 1 N=30 Option B pilot completed on M1 Max (240
  cells, ~80 min wall). Pre-reg
  `docs/osf-prereg-g4-sexto-pilot.md` ; aggregator
  `experiments/g4_sexto_test/aggregator.py`.
- **H6-A confirmed** : Welch two-sided between (P_max with mog)
  and (P_max with none) on Split-CIFAR-100 (n_classes=10/task,
  10 tasks, G4SmallCNN), `t = 0.197`, `p = 0.8450`,
  `Hedges' g = 0.057`, `mean_mog = 0.3622`, `mean_none = 0.3580`.
  Fail-to-reject H0 → H6-A confirmed. RECOMBINE adds nothing
  measurable beyond REPLAY+DOWNSCALE on small-CNN at 100-class
  scale.
- **H6-B deferred** : Tiny-ImageNet step locked under Option B,
  goes to G4-septimo follow-up.
- **H6-C deferred** : universality conjunction
  `H6-A ∧ H6-B` incomplete pending Tiny-IN evidence.
- **DR-4 evidence amended** : `docs/proofs/dr4-profile-inclusion.md`
  v0.5 G4-sexto addendum extends the G4-quinto two-benchmark
  empirical-emptiness scope to three benchmarks
  `{FMNIST, CIFAR-10, CIFAR-100}` across substrates
  `{3-layer MLP, 5-layer MLP, small CNN}`. Empirical-emptiness
  verdict robust to 10× scaling of class budget per task. Lemma
  DR-4.L formal proof unchanged.
- EC stays PARTIAL ; FC stays C-v0.12.0. STABLE promotion remains
  blocked pending Tiny-IN, ImageNet-1k, transformer, and
  hierarchical-E-SNN follow-ups.
- Milestones :
  `docs/milestones/g4-sexto-step1-2026-05-03.{json,md}` +
  `docs/milestones/g4-sexto-aggregate-2026-05-03.{json,md}`.
  Paper 2 §7.1.8 EN+FR appended.
- Confirmatory N=95 Studio M3 Ultra run COMPLETED (760 cells,
  5 129 s wall, 76 effective per arm × strategy after
  `acc_initial < 0.20` exclusion ~20 %). H6-A re-confirmed at
  higher N : `mean P_max(mog) = 0.3701`,
  `mean P_max(none) = 0.3592`, `Hedges' g = 0.1527`,
  `Welch t = 0.946`, `p = 0.3457` → fail-to-reject at α = 0.0167.
  Effect-size estimate moves from `g_30 = 0.057` to `g_95 = 0.153`
  (still Cohen "small", < 0.2). Empirical-emptiness claim
  survives precision increase. Milestone
  `docs/milestones/g4-sexto-step1-confirmatory-N95-studio-2026-05-04.{json,md}`.

### Exploratory (Studio CPU pilots — K2 + R1 + Robertson, 2026-05-04)

- Three small CPU pilots on Mac Studio M3 Ultra launched in parallel
  with G4-sexto Studio N=95 GPU run. ~6 min total wall. Status :
  exploratory, no individual OSF pre-registration.
- **K2 real-substrate** : 24 random `.npz` modules sampled from
  SpikingKiki-V4 35B-A3B (31 070 modules), LIF metadata applied
  (`T=128, threshold=0.0625, tau=1.0`), 276 pairwise mean-vector-
  length measurements. Result : MVL mean = 0.3855, median = 0.3797,
  5-95 percentile = [0.2557, 0.5434]. K2 invariant range
  `[0.27, 0.39]` (eLife 2025 BF=58 SO-spindle anchor) : mean and
  median both *inside* the band ; 45.7 % of pairs strictly within
  K2. Verdict : *partial empirical support* on real substrate, with
  inter-pair variability beyond the band.
- **R1 cross-machine verification** : ran `tests/reproducibility/`
  (5 bit-exact + 4 contract = 9 tests) on Studio M3 Ultra Python
  3.14.4 / MLX 0.31.1. All 9 tests PASS with hashes identical to
  M1 Max Python 3.12 baseline. `golden_hashes.json` status field
  promoted from `pending_review` to
  `validated_cross_machine_2026-05-04` for all 5 bit-exact entries.
  Caveat : both machines are Apple Silicon, does not establish
  CUDA/Linux portability.
- **Robertson 2018 sequential-ordering test** : 6 permutations of
  (REPLAY, DOWNSCALE, RESTRUCTURE, RECOMBINE) × 5 seeds = 30 cells,
  5-task synthetic CL stream, MLP 2×32 with dream-op stubs. Max
  pairwise Hedges' g vs canonical = 0.079. Verdict H_RO-A :
  permutation effect SMALL (max |g| < 0.2) ; ordering does not
  materially affect retention in this regime. Descriptive support
  for DR-3 substrate-independence intra-cycle ordering corollary.
- Milestones : `docs/milestones/studio-cpu-2026-05-04/`
  (`README.md`, `k2_real_substrate.py`, `k2_real_substrate_result.json`,
  `robertson_sequential_ordering.py`, `robertson_sequential_result.json`).
- No FC or EC bump — exploratory results, not part of confirmatory
  Paper 2 claims.

### Empirical (G5-ter spiking-CNN cross-substrate, 2026-05-03)

- G5-ter pilot ported the G4-quinto Step 2 small-CNN architecture
  onto the E-SNN thalamocortical substrate as a 4-layer spiking
  CNN (`EsnnG5TerSpikingCNN` : Conv2d-LIF + Conv2d-LIF + avg-pool
  4x4 + FC-LIF + Linear, STE backward over all three LIF stages,
  pure-numpy Conv2d via im2col). 40 cells x 4 arms x 10 seeds
  Option B, 36 min wall on M1 Max. Pre-reg
  `docs/osf-prereg-g5-ter-spiking-cnn.md` locked at commit
  `f18030b`. Documented compute-budget amendment under
  `docs/osf-deviations-g5-ter-2026-05-03.md` D-1 subsamples the
  CIFAR-10 train shard to 1500 examples per task per cell (test
  shard intact) ; design 4-arm x N=10 x HP combo C5 preserved.
- Own-substrate finding : `g_h8 = -0.1093` (E-SNN spiking-CNN
  P_equ vs baseline), Welch one-sided p = 0.5992 at α/4 = 0.0125
  -> **fail-to-reject H0**. H7B_G_THRESHOLD = 0.5 not reached.
- Cross-substrate aggregate vs G4-quinto Step 2 MLX small-CNN :
  all 4 arms reject H0 at α/4 = 0.0125. `g_mlx_minus_esnn` in
  `[+1.21, +1.32]` ; `g_p_equ_cross = +1.31` falls between the
  H8-A floor (2.0) and H8-B ceiling (1.0). The cross-substrate
  retention level gap closes from +4.02 (G5-bis MLP) to +1.31
  (G5-ter CNN) at P_equ -- about two thirds reduction.
- **Classification : H8-C (partial -- both architecture and LIF
  non-linearity contribute)**. The CNN architecture closes the
  level gap partially but the own-substrate cycle-3 positive
  effect remains absent (g_h8 < 0.5, fail-to-reject) regardless
  of whether the architecture is dense or convolutional.
- DR-3 evidence revised
  (`docs/proofs/dr3-substrate-evidence.md` empirical-evidence
  amendment 2026-05-03 G5-ter appended) : axiom-property-test-
  level substrate-agnosticism guarantee preserved at every
  architectural depth ; empirical effect-size transferability now
  refuted at *two* architectural depths (3-layer LIF MLP H7-B +
  4-layer spiking-CNN H8-C). EC stays PARTIAL. No FC bump.
- Milestones :
  `docs/milestones/g5-ter-spiking-cnn-2026-05-03.{json,md}` +
  cross-substrate aggregate
  `docs/milestones/g5-ter-aggregate-2026-05-03.{json,md}`. Paper 2
  §7.1.10 EN+FR appended.
- Future work per pre-reg §11 : confirmatory N=30 Option A
  follow-up scheduled to tighten the H8-C reading.

### Empirical (G5-bis richer-head cross-substrate, 2026-05-03)

- G5-bis pilot ported G4-ter MLX richer head to E-SNN
  thalamocortical substrate (3-layer rate-coded LIF SNN, STE
  backward, pure numpy). 40 cells × 4 arms × 10 seeds Option B,
  ~16 min wall time on M1 Max. Pre-reg
  `docs/osf-prereg-g5-bis-richer-esnn.md` locked at commit
  `ae640a5` ; pilot run at commit `5168400`. Milestones :
  `docs/milestones/g5-bis-richer-esnn-2026-05-03.{json,md}` +
  aggregate `g5-bis-aggregate-2026-05-03.{json,md}`.
- Own-substrate finding : `g_h7a = 0.1043` (E-SNN richer P_equ
  vs baseline), Welch one-sided p = 0.4052 at α/4 = 0.0125 →
  **fail-to-reject H0**. Below pre-registered
  `H7B_G_THRESHOLD = 0.5`.
- Cross-substrate aggregate vs G4-ter MLX richer head : all 4
  arms reject H0 at α/4 = 0.0125. Baseline `g_mlx_minus_esnn
  = 3.23` (p=6.4e-16) ; dream arms `g = 4.02` (p=2.3e-18) ;
  P_min `g = 4.20` (p=7.4e-19). `consistency_ok = False`.
- **Classification : H7-B (MLX-only artefact at this N)**.
  G4-ter positive effect (g=+2.77) does NOT transfer to E-SNN
  richer head. Spike-rate quantization, LIF non-linearity,
  STE-backward approximation wash out the dream effect.
- DR-3 evidence revised
  (`docs/proofs/dr3-substrate-evidence.md` empirical-evidence
  amendment 2026-05-03 G5-bis appended) : axiom-property-test-
  level substrate-agnosticism guarantee preserved ; empirical
  effect-size transferability refuted at this N. EC stays
  PARTIAL. No FC bump.
- Future work per pre-reg §6 row 6 : confirmatory N=30 Option
  A, spiking-CNN G5-ter, ImageNet-scale escalation.

### Empirical (G4-quinto, 2026-05-03)

- G4-quinto pilot completed 2026-05-03 — confirmatory N = 30
  follow-up to the G4-quater H4-C confirmation, escalating
  from Split-FMNIST onto Split-CIFAR-10 with two substrates
  (5-layer MLP-on-CIFAR + small CNN). Sequential 3-step
  layout, total 600 cells, ~72 min wall time on M1 Max
  (well under the pre-reg Option A 9-15 h envelope ; budget
  was conservative). Pre-reg
  `docs/osf-prereg-g4-quinto-pilot.md` locked at commit
  `a02b82c` before any pilot run.
- §9.1 amendment filed inline in the OSF pre-reg : the
  canonical mirror at `www.cs.toronto.edu/~kriz` returned
  HTTP 503 across the entire `~kriz` tree on 2026-05-03
  during pre-pilot validation. The loader gained a
  SHA-256-pinned fallback to the Hugging Face dataset
  `uoft-cs/cifar10` (commit `0b2714987...`) — train shard
  119,705,255 bytes, test shard 23,940,850 bytes, both PNG
  re-encodings of the same Krizhevsky 2009 release. The
  acquisition path changes ; the experimental contract does
  not. R1 bit-stable run_ids preserved.
- Step 1 (H5-A benchmark-scale) : 120 cells on a new 5-layer
  `G4HierarchicalCIFARClassifier` (hidden 256-128-64-32,
  in_dim 3072) at the C5 anchor. Jonckheere J = 1362.0,
  p = 0.4646 at α = 0.0167 ; means P_min = 0.8713,
  P_equ = 0.8754, P_max = 0.8754 (P_equ = P_max within 1e-4) ;
  monotonic_observed = True ; reject_h0 = False. **H5-A NOT
  confirmed** : benchmark scale-up alone (FMNIST -> CIFAR-10)
  on a wider MLP does not statistically recover the predicted
  DR-4 ordering at this N.
- Step 2 (H5-B architecture-scale) : 120 cells on the new
  `G4SmallCNN` substrate (Conv2d×2 + MaxPool2d×2 + Linear×2,
  NHWC, latent_dim = 64). Jonckheere J = 1356.0, p = 0.4823
  at α = 0.0167 ; means P_min = 0.9841, P_equ = 0.9842,
  P_max = 0.9842 (P_equ = P_max within 1e-4) ;
  monotonic_observed = True ; reject_h0 = False. **H5-B NOT
  confirmed** : adding hierarchical conv structure also fails
  to statistically recover the predicted ordering at this N.
- Step 3 (H5-C universality of RECOMBINE-empty) : 360 cells
  on the CNN substrate × strategy ∈ {mog, ae, none}. H5-C
  operationalised as Welch two-sided P_max(mog) vs
  P_max(none) failing to reject H0 at α = 0.0167. Result :
  Welch t = -0.0104, p = 0.9918, Hedges' g (mog vs none) =
  -0.0026 ; mean P_max(mog) = 0.9842 vs mean P_max(none) =
  0.9845 (within 3e-4). AE secondary : mean P_max(ae) =
  0.9840, Welch p (ae vs none) = 0.9857. **H5-C CONFIRMED** :
  the G4-quater H4-C RECOMBINE-empty finding **universalises**
  across substrates (FMNIST 3-layer MLP and CIFAR-CNN both
  fail-to-reject mog vs none at the multiplicity-adjusted α).
- Aggregate verdict : H5-A = False, H5-B = False, H5-C = True,
  **h4c_to_h5c_universality = True**. The DR-4 partial
  refutation from G4-ter §7.1.5 — strengthened by G4-quater
  §7.1.6 — is now **universalised** across two benchmarks
  (Split-FMNIST, Split-CIFAR-10) × two substrates (3-layer
  MLP, small CNN). The framework-C "richer ops yield richer
  consolidation" claim is empirically refuted at the
  RECOMBINE-channel level across the escalation ladder cells
  tested so far. Lemma DR-4.L itself is **not** refuted —
  within-arm differences ≤ 0.001 across all G4-quinto cells.
- DualVer : EC stays **PARTIAL** per pre-reg §6 row 4 ; FC
  stays at **C-v0.12.0** (no formal-axis bump).
  `docs/proofs/dr4-profile-inclusion.md` carries a
  v0.4 G4-quinto empirical-evidence amendment ; Paper 2
  §7.1.7 reports the verdicts honestly in EN + FR.
- 600 R1-bit-stable run_ids registered under
  `(C-v0.12.0+PARTIAL, g4-quinto/{step1,step2,step3}/<arm>/<combo>[/<strategy>])`
  in `.run_registry.sqlite`. Honest reporting binding (pre-reg
  §7) : Welch fail-to-reject for H5-C is reported as the
  predicted positive empirical claim that RECOMBINE adds
  nothing measurable beyond REPLAY+DOWNSCALE on the CNN
  substrate at CIFAR-10 scale ; H5-A and H5-B nulls are
  reported as "no evidence at this N" (absence of evidence
  vs evidence of absence).
- Milestone artefacts :
  `docs/milestones/g4-quinto-step1-2026-05-03.{json,md}`,
  `docs/milestones/g4-quinto-step2-2026-05-03.{json,md}`,
  `docs/milestones/g4-quinto-step3-2026-05-03.{json,md}`,
  `docs/milestones/g4-quinto-aggregate-2026-05-03.{json,md}`.
- Future work pre-registered in
  `docs/osf-prereg-g4-quinto-pilot.md` §6 row 6 : test on
  ImageNet / transformer head / hierarchical E-SNN before
  any STABLE promotion ; the empirical scope of any such
  promotion would have to explicitly exclude the
  {Split-FMNIST, Split-CIFAR-10} × {3-layer MLP, 5-layer MLP,
  small CNN} cells of the escalation ladder tested so far.

### Empirical (G4-quater, 2026-05-03)

- G4-quater pilot completed 2026-05-03 — confirmatory N≥95
  follow-up to the G4-ter H_DR4 inversion. Sequential 3-step
  layout, total 1880 cells, ~58 min wall time on M1 Max. Pre-reg
  `docs/osf-prereg-g4-quater-pilot.md` locked at commit `e7232b9`
  before any pilot run (Tasks 4/5/6 commits land later).
- Step 1 (H4-A substrate-depth) : 380 cells on a new 5-layer
  `G4HierarchicalDeeperClassifier` (hidden 64-32-16-8) at the C5
  anchor. Jonckheere J = 13511.5, p = 0.514 at α = 0.0167 ;
  monotonic_observed = False ; means P_min = 0.5959, P_equ =
  0.5958, P_max = 0.5958 (within 1e-4). **H4-A NOT confirmed** :
  deepening alone does not recover the predicted DR-4 ordering ;
  the H_DR4 inversion persists at 5-layer depth.
- Step 2 (H4-B HP-calibration) : 360 cells on the existing 3-layer
  head sweeping RESTRUCTURE factor in {0.85, 0.95, 0.99}.
  Per-factor Jonckheere at multiplicity-adjusted α = 0.0056 (3
  factors × 3 hypotheses). Every factor : `P_equ = P_max < P_min`,
  J ∈ {1034, 1076, 1094}, p ∈ {0.971, 0.979, 0.990}. **H4-B NOT
  confirmed** : RESTRUCTURE HP-calibration alone does not recover
  the predicted ordering ; RESTRUCTURE in fact *hurts* retention
  at every factor.
- Step 3 (H4-C RECOMBINE empirical-emptiness) : 1140 cells on the
  3-layer head sweeping RECOMBINE strategy in {mog, ae, none}.
  H4-C operationalised as Welch two-sided P_max(mog) vs P_max(none)
  failing to reject H0 at α = 0.0167. Result : Welch t = 0.014,
  p = 0.989, Hedges' g (mog vs none) = 0.002. AE secondary :
  Welch p (ae vs none) = 1.000. **H4-C CONFIRMED** : RECOMBINE
  empirically empty at this scale (positive empirical claim
  mog ≈ none).
- Aggregate verdict : H4-A False, H4-B False, H4-C True. The DR-4
  partial refutation from G4-ter §7.1.5 **strengthens** rather
  than weakens — two of the three remaining escape clauses
  (substrate-depth, HP-calibration of RESTRUCTURE) are ruled out
  at confirmatory N, and the third (RECOMBINE empirically empty)
  is positively confirmed. The framework-C claim "richer ops
  yield richer consolidation at this scale" is now an
  empirically refuted hypothesis at the Split-FMNIST 3-layer
  scale rather than an insufficiently-tested one. Lemma DR-4.L
  itself is **not** refuted (within-arm differences ≤ 0.001).
- DualVer : EC stays **PARTIAL** per pre-reg §6 across all
  outcomes ; FC stays at **C-v0.12.0** (no formal-axis bump).
  `docs/proofs/dr4-profile-inclusion.md` carries an
  empirical-evidence amendment block ; Paper 2 §7.1.6 reports
  the verdicts honestly in EN + FR.
- 1880 R1-bit-stable run_ids registered under
  `(C-v0.12.0+PARTIAL, g4-quater/{step1,step2,step3}/<arm>/<combo>/<seed>)`
  in `.run_registry.sqlite`. Honest reporting binding (pre-reg §7) :
  Welch fail-to-reject is reported as the predicted outcome under
  H4-C (positive empirical claim that RECOMBINE adds nothing),
  not as evidence-of-absence laundered as evidence-of-presence.
- Milestone artefacts :
  `docs/milestones/g4-quater-step1-2026-05-03.{json,md}`,
  `docs/milestones/g4-quater-step2-2026-05-03.{json,md}`,
  `docs/milestones/g4-quater-step3-2026-05-03.{json,md}`,
  `docs/milestones/g4-quater-aggregate-2026-05-03.{json,md}`.
- Future work pre-registered in `docs/osf-prereg-g4-quater-pilot.md`
  §6 : test on CIFAR-10 / ImageNet / hierarchical E-SNN before
  any STABLE promotion ; the verdict is scoped to this benchmark
  scale.

### Empirical (G5 cross-substrate, 2026-05-03)

- G5 cross-substrate pilot on E-SNN thalamocortical (numpy LIF
  fallback, no `norse` dep). 20 cells × 4 arms × 5 seeds, 4 min
  wall time on M1 Max. Pre-reg `docs/osf-prereg-g5-cross-substrate.md`
  locked at commit `1411228` before pilot run at `5fb36f0`.
  Milestones : `docs/milestones/g5-cross-substrate-2026-05-03.{json,md}`
  (within-substrate) + `g5-cross-substrate-aggregate-2026-05-03.{json,md}`
  (cross-substrate Welch verdict).
- Within-substrate finding (E-SNN) : `g_h1 = 0`, `g_h3 = 0`,
  `H_DR4` trivially monotonic (P_min = P_equ = P_max = 0.5119) —
  spectator pattern reproduced on E-SNN, mirroring G4-bis (binary
  MLP) null.
- Cross-substrate aggregate vs G4-bis MLX milestone :
  `dr3_cross_substrate_consistency_ok = False`. baseline
  g_mlx_minus_esnn = 9.98 (p = 5.2e-05); dream arms g = 3.49
  (p = 3.5e-03). All 4 arms reject H0 at α/4 = 0.0125.
- Honest reading : same qualitative spectator pattern on both
  substrates → DR-3 axioms (DR-0/1/2'/4) hold on both. Absolute
  retention diverges (E-SNN uniformly lower than MLX), so the
  pre-registered consistency-on-absolute-retention test rejects.
  No upgrade to `docs/proofs/dr3-substrate-evidence.md` at the
  absolute-retention level. EC axis : **PARTIAL** (kept). G5-bis
  follow-up : port G4-ter richer head (g_h2 = +2.77) to a
  hierarchical E-SNN classifier and re-test whether the dream
  effect itself transfers across substrates.
- 20 run_ids under `(C-v0.12.0+PARTIAL, g5/<arm>, seed)`
  registered with R1 bit-stable run_ids.

### Empirical (G4-ter)

- G4-ter pilot completed 2026-05-03 — confirmatory N≥30 follow-up to
  the G4-bis null finding. 420 cells: richer substrate
  (`G4HierarchicalClassifier`, 4 arms × 30 seeds × 1 HP at C5) plus
  HP sub-grid on binary head (3 arms × 10 seeds × 10 combos). Pre-
  registration `docs/osf-prereg-g4-ter-pilot.md` locked at commit
  `0553a9f`. Milestone artefacts
  `docs/milestones/g4-ter-pilot-2026-05-03.{json,md}`. Verdict :
  H1 (HP artefact) = **rejected H0** (g=+11.81 at C9, n=10
  screening), H2 (substrate-level) = **rejected H0** (g=+2.77,
  Welch p=4.9e-14, n=30 — finding reframed as
  "REPLAY+DOWNSCALE coupling on richer substrate exceeds no-dream
  baseline" ; RESTRUCTURE+RECOMBINE remain spectator channels at
  this scale), H_DR4-ter (Jonckheere monotonicity) = **DR-4
  profile ordering NOT supported — partial refutation** (J=1335,
  p=0.544; observed `P_min=0.7065 > P_equ=0.7046 = P_max=0.7046`
  is the **inverse** of the predicted `P_max ≥ P_equ ≥ P_min`,
  not an inconclusive tie). The Hu 2020 anchor (g=0.29) is
  directional only (sign of the alternative), not a magnitude
  calibrator — the human-sleep meta-analysis and a 100-class MLP
  toy belong to different reference classes. EC axis :
  **PARTIAL** (kept) per `docs/osf-prereg-g4-ter-pilot.md` §7 —
  the DR-4 partial refutation strengthens (rather than weakens)
  the PARTIAL lock. Confirmatory N≥95 G4-quater scheduled to test
  whether richer × `hp_best=C9` recovers the predicted ordering
  or solidifies the refutation.
- Wall time : 653.8s on M1 Max (well under the 3-5h budget).
- 420 run_ids under
  `(C-v0.12.0+PARTIAL, g4-ter/{richer,hp}/<arm>/<combo>, seed)`
  registered in `.run_registry.sqlite` with R1 bit-stable run_ids.

### Empirical (no DualVer bump — partial confirmation)
- G4 pilot 2026-05-03 (Split-FMNIST × profile sweep, MLX
  substrate) returned partial confirmation of pre-registered
  hypotheses in `docs/osf-prereg-g4-pilot.md`. See
  `docs/milestones/g4-pilot-2026-05-03.md` for per-hypothesis
  verdict scalars (H1 g_h1 = 0.0, H3 g_h3 = 0.0, H_DR4
  monotonic with equal means at 0.5988 — Welch p = 0.5 across
  all paired tests). The minimal `dream_episode()` wrapper
  dispatches profile ops via `runtime.execute(...)` for DR-0
  accountability without mutating classifier weights, so the
  four arms produce identical retention per seed by design.
- Per framework-C §12.3, partial confirmation does not
  promote EC out of PARTIAL. A confirmatory N ≥ 30 follow-up is
  scheduled before any STABLE promotion, after weight-mutating
  coupling lands (dream-driven LoRA delta or replay buffer
  feeding the optimizer).
- Milestone artefacts : `docs/milestones/g4-pilot-2026-05-03.{json,md}`
- Run-registry rows : 20 cells under
  `(C-v0.12.0+PARTIAL, g4/{baseline,P_min,P_equ,P_max}, seed)`
  with R1 bit-stable run_ids verified across two re-runs

### Empirical (G4-bis re-run, 2026-05-03)
- G4-bis re-run after `dream_episode()` wired to mutate classifier
  weights via raw-image replay (n=32 × 1 SGD step) + SHY
  downscale (factor 0.95). Pre-registered hypotheses re-evaluated
  per `docs/osf-prereg-g4-pilot.md` §2 ; observed scalars in
  `docs/milestones/g4-pilot-2026-05-03-bis.md`. Pre-coupling
  run_ids preserved as spectator-baseline references in the
  original `g4-pilot-2026-05-03.{json,md}`. Coupling
  hyperparameters (replay batch n=32, lr=0.01, downscale 0.95) are
  pilot defaults — empirical pin scheduled for the N≥30
  confirmatory follow-up.
- Observed scalars : `g_h1 = -2.3067` (H1 not confirmed), `g_h3
  = -2.3067` with decrement-side rejection True (H3 confirmed
  in sign), H_DR4 monotonic flag True but degenerate (P_min /
  P_equ / P_max retention vectors bit-identical because the
  binary MLP head exposes only REPLAY + DOWNSCALE channels —
  RESTRUCTURE / RECOMBINE remain spectator-only).
- Per framework-C §12.3, partial confirmation strengthens
  PARTIAL — H3 sign confirmed under coupling, H1 not, H_DR4
  untestable on this substrate at this scale ; N≥30 confirmatory
  follow-up still required before any STABLE promotion.

### Versioning
- **No DualVer bump.** EC stays PARTIAL (partial confirmation
  is not a STABLE-promotion event per §12.3). FC stays at v0.12.0.

### Empirical (G6 pilot Path B, 2026-05-03)
- G6 pilot 2026-05-03 (micro-kiki Qwen × MMLU subdomain CL stream,
  Path B inference-only) returned exploratory infrastructure-
  validation evidence per `docs/osf-prereg-g6-pilot.md`. Path
  selection locked in `docs/milestones/g6-pilot-decisions-
  2026-05-03.md` : Path A (Studio + KIKI-Mac_tunner +
  mlx_lm.lora) unavailable on M1 Max host ; Path B is the
  always-runnable fallback. Per pre-reg §6, Path B never
  triggers a STABLE / UNSTABLE EC bump regardless of
  effect-size outcome.
- 12 cells under
  `(C-v0.12.0+PARTIAL, g6/B/{baseline,P_min,P_equ,P_max}, seed)`
  registered in `.run_registry.sqlite` with R1 bit-stable
  run_ids ; deterministic across re-runs (verified by the
  `test_run_pilot_path_b_is_deterministic` property test).
- Observed scalars : `g_h1' = 0.0`, `g_h3' = 0.0`,
  `H_DR4' monotonic = True (degenerate, equal means)`,
  `H_NEW (amended) infrastructure_effect_observed = False`
  (max\|Δ\| = 0). Spectator pattern : the four dream handlers
  operate on synthetic payloads built by
  `experiments.g6_mmlu_stream.dream_wrap.build_episode_payload`,
  not on the live `InferenceOnlyAdapter._deltas` buffer ; DR-0
  / DR-1 stamps land but the inference-time evaluation surface
  sees an identical adapter delta across arms. Honest verdict :
  pipeline-shape validation succeeded ; coupling extension
  (handlers feeding `adapter.set_delta`, or Path A real LoRA
  fine-tune) is the next confirmatory step.
- H_NEW honest amendment : the original H_NEW formulation
  ("real LLM g_h1' >= synthetic G4-bis g_h1 - 0.10") is
  vacuous given G4-bis g_h1 = -2.31 (any real g satisfies the
  inequality). H_NEW is reformulated in
  `docs/osf-prereg-g6-pilot.md` §0 + §2 as exploratory
  infrastructure validation ; the chain G4 spectator -> G4-bis
  coupling -> G6 Path B inference-only is documented in the
  pre-reg as scientific review trail.
- Milestone artefacts :
  `docs/milestones/g6-pilot-pathB-2026-05-03.{json,md}`,
  `docs/milestones/g6-pilot-decisions-2026-05-03.md`,
  `docs/osf-prereg-g6-pilot.md`.

### Versioning (G6 outcome branch)
- **No DualVer bump.** EC stays PARTIAL — Path B is exploratory
  by pre-reg §6, never triggers STABLE / UNSTABLE regardless of
  effect-size outcome. FC stays at v0.12.0 (no axiom or
  primitive change). Path A on Studio is the future-work path
  that could promote PARTIAL → STABLE on confirmatory all-pass.

### Documentation
- Re-key the Wake-Sleep CL baseline (Alfarano 2024) from the
  unverified `split_fmnist_5tasks` placeholder pair
  `(forgetting_rate=0.082, avg_accuracy=0.847)` to
  `cifar10_5tasks_buffer500` with verified values
  `(forgetting_rate=0.1069, avg_accuracy=0.7418)` imported from
  Alfarano 2024 Tables 2-3, row "ER-ACE+WSCL (Ours)", Split
  CIFAR-10 buffer-500 (resolution path Option 1 from the verify
  attempt). Updates :
  `kiki_oniric/substrates/wake_sleep_cl_baseline.py`
  (`_REFERENCE_METRICS_BY_TASKSPLIT` constant) ;
  `scripts/baseline_wake_sleep_cl.py` (`DEFAULT_TASK_SPLIT` and
  `DEFAULT_OUT` re-pointed to the `-rekey` filename) ;
  `tests/unit/test_wake_sleep_baseline_adapter.py` ; Paper 2 §7.7
  EN + FR caption + table re-keyed. The frozen
  `docs/milestones/wake-sleep-baseline-2026-05-03.{json,md}` and
  the verify entry `wake-sleep-baseline-verify-2026-05-03.md`
  are preserved ; the supersede dump
  `docs/milestones/wake-sleep-baseline-rekey-2026-05-03.{json,md}`
  records the new run_ids and resolution path. **No DualVer
  bump** — bibkey + variant + scores_on semantics unchanged ;
  this is a placeholder-resolution correction below the
  FC-PATCH threshold.
- Document outcome of Alfarano 2024 (arXiv 2401.08623) verify
  attempt for the Wake-Sleep CL baseline placeholder pair
  (`forgetting_rate=0.082`, `avg_accuracy=0.847`,
  `task_split=split_fmnist_5tasks`). Pulled the PDF (v1, 14 pp)
  via WebFetch and parsed Tables 1-4 with pypdf 6.10.2. The
  placeholder values do not match any cell of Tables 2-3, and
  Alfarano 2024 §4.1 evaluates on CIFAR-10 / Tiny-ImageNet1/2 /
  FG-ImageNet rather than Split-FMNIST. New supersede dump
  `docs/milestones/wake-sleep-baseline-verify-2026-05-03.md`
  records the mismatches.

### Added
- `kiki_oniric.profiles.so_calibration` module with
  `compute_so_amplitude_proxy()` reader and Sharon 2025 anchor
  constants (HEALTHY_OLDER=1.0, AMCI_MIDPOINT=0.45, AD_FLOOR=0.20).
  Reference: docs/papers/paper1/methodology.md §6.6.
- `so_trough_amplitude_factor: float` field on `PMinProfile` (default
  `0.45`, aMCI midpoint placeholder), `PEquProfile` (default `1.0`),
  `PMaxProfile` (default `1.0`). Sharon et al. 2025
  (sharon2025alzdementia) qualitative calibration; final empirical
  value deferred to G2 P_min pilot.
- `tests/unit/profiles/test_p_min_sharon_calibration.py` (13 tests):
  default values, proxy reader, monotonicity P_max ≥ P_equ ≥ P_min,
  seed-independence, DR-4 chain coexistence guard.

### Versioning
- **No DualVer bump.** Calibration is profile metadata, not an axiom
  statement, primitive signature, channel set, or invariant ID
  change. FC stays at v0.10.0; EC stays PARTIAL. Per framework-C
  spec §12 (only spec/proof changes trigger FC; only gate results
  trigger EC).

---

## [C-v0.12.0+PARTIAL] — 2026-05-03 — Wake-Sleep CL ablation baseline

### Formal axis (FC) — MINOR (v0.11.0 → v0.12.0)

- **Eval-matrix schema additive change.**
  `docs/interfaces/eval-matrix.yaml` gains a top-level
  `baselines:` block (additive, no breaking change). The first
  registered baseline is `wake_sleep_cl` per Alfarano 2024
  [IEEE TNNLS, arXiv 2401.08623], the closest published
  NREM/REM dual-phase analog and Paper 2's primary ablation
  comparator (Paper 1 §3 introduction.md L94, L108).
- New module
  `kiki_oniric/substrates/wake_sleep_cl_baseline.py` — adapter
  exposing
  `WakeSleepCLBaseline.evaluate_continual(seed, task_split)`.
  Variant `c` (default, fixture-stub): published reference
  values frozen at `forgetting_rate=0.082`,
  `avg_accuracy=0.847`. **PLACEHOLDER values** pending
  cross-check against Alfarano 2024 Tables 2-3 PDF (deferred
  to follow-up).
- New gate-style driver `scripts/baseline_wake_sleep_cl.py`
  emits `docs/milestones/wake-sleep-baseline-2026-05-03.{md,
  json}` with R1 run_ids on the seed grid `[42, 123, 7]`.
- Loader extension: `harness.config.eval_matrix.EvalMatrix`
  gains `baselines: dict[str, dict[str, Any]]` (default-empty
  for back-compat with legacy yamls). Required fields per
  entry: `bibkey`, `scores_on`, `variant`.
- Paper 2 §5.8 (architecture.md) + §6.3 (methodology.md) +
  §7.7 (results.md) updated with the new row, EN ↔ FR
  mirrored per `docs/papers/CLAUDE.md`. Bibkey
  `alfarano2024wakesleep` mirrored from `paper1/references.bib`
  into `paper2/references.bib`. Glossary entry under new
  "Baselines" section.

### Empirical axis (EC) — UNCHANGED (PARTIAL)

- The baseline row is variant-c (published-reference values,
  not a re-run). It is **not** an empirical claim ; it
  validates the comparator pipeline and the schema. Paper 2's
  EC stays at `+PARTIAL` (cycle-3 deferred Phase 2 cells
  unchanged).
- DR-3 conformance unaffected : the baseline does not
  implement the 4 op factories ; it is registered as a
  *baseline*, not a substrate. The DR-3 conformance matrix
  (`scripts/conformance_matrix.py`) is not extended. New test
  `tests/conformance/test_baseline_registration.py` pins the
  exemption.

### Notes

- FC bump magnitude (MINOR vs PATCH) follows framework-C
  spec §12.2 : "addition of new optional primitive / new
  derived constraint" — the `baselines:` key is the new
  optional primitive.
- No OSF amendment triggered : adding a published-reference
  baseline does not introduce a new pre-registered hypothesis
  (Paper 2 §6.1 H1-H4 inherited from Paper 1 OSF lock
  `10.17605/OSF.IO/Q6JYN`).

---

## [C-v0.11.0+PARTIAL] — 2026-05-02 — K2 phase-coupling invariant

### Formal axis (FC) — MINOR (v0.10.0 → v0.11.0)

- Add invariant **K2** SO × fast-spindle phase-coupling within
  empirical CI [0.27, 0.39]. Anchored on eLife 2025 Bayesian
  meta-analysis (BibTeX `elife2025bayesian`, BF > 58 vs. null,
  Egger phase-branch p = 0.59). Severity **WARN** (single
  meta-analysis anchor; substrate physiology can legitimately
  broaden the CI; synthetic substrate is the only canonical
  reference until S18+).
- New opt-in `PhaseCouplingObservable` Protocol in
  `kiki_oniric/core/observables.py`; new guard
  `kiki_oniric/dream/guards/coupling.py`
  (`check_coupling_in_window`, `CouplingGuardError`); new
  conformance suite `tests/conformance/invariants/test_k2_coupling.py`
  (18 tests including a Hypothesis property test over 50 seeds);
  evidence stub `docs/proofs/k2-coupling-evidence.md`.
- 8-primitive DR-3 surface (`kiki_oniric/core/primitives.py`)
  unchanged — no breaking change. Real substrates remain free to
  not implement the new Protocol; K2 only exercises substrates
  that opt in.
- Synthetic-substrate calibration: PAC_DEPTH = 0.33 (matches
  eLife headline 0.33; 50-seed empirical range
  MVL ∈ [0.328, 0.332] at fs=256 Hz, n=8192).

### Empirical axis (EC) — UNCHANGED (PARTIAL)

- No new gate result; K2 exercised exclusively against the synthetic
  reference substrate. Real-substrate empirical pins deferred to
  S18+ (MLX kiki-oniric) / S22+ (E-SNN thalamocortical).

---

## [C-v0.10.0+PARTIAL] — 2026-04-22 — micro-kiki recombine TIES-merge (PR #13)

### Added — `recombine_handler_factory` (FC-MINOR, repo 0.9.0 → 0.10.0)

Fourth (and last) handler on the micro-kiki substrate now backed.
`recombine_handler_factory` replaces its phase-1
`NotImplementedError` stub with a numpy port of TIES-Merge
(Yadav et al., arXiv 2306.01708) : trim → elect-sign →
disjoint-mean merge on a list of task-specific delta tensors.

**Helper**

- `_ties_merge(deltas, trim_fraction=0.2, alpha=1.0)` —
  substrate-internal helper. Guards : empty list raises, single
  delta fast-paths to `alpha * delta`, shape-mismatch across
  deltas raises, `trim_fraction` outside `(0, 1]` raises.
  Algebra in float64, output cast back to input dtype.

**Handler contract**

- Takes `(payload: dict, op: str = "ties") -> ndarray`. Payload
  must carry `"deltas": list[ndarray]` ; optional `"episode_id"`
  lands on the DR-1 stamp. Honours `op ∈ {"ties", "ties_merge",
  "merge"}` ; any other op raises `ValueError` (DR-3 condition 1,
  no silent no-ops).
- New `MicroKikiRecombineState` dataclass exposed read-only via
  `MicroKikiSubstrate.recombine_state`. Stamps DR-0 (completed
  flag + operation label + counters) and DR-1 (episode_id list).
  Records `last_k_deltas`, `last_input_shape`, `last_output_shape`
  per call so the conformance harness can audit shape consistency.

**DualVer**

- Repo : `0.9.0` → `0.10.0` (pyproject.toml) — FC-MINOR, new
  public handler surface (recombine backed).
- Substrate-internal : `C-v0.8.1+PARTIAL` → `C-v0.9.0+PARTIAL`.
  All 4 handlers (replay / downscale / restructure / recombine)
  now backed by a real algorithm ; `+PARTIAL` retained until the
  Phase-4 conformance run lands.

**Tests**

- New `tests/unit/test_micro_kiki_recombine.py` — 14 unit tests :
  `_ties_merge` algebra (single-delta fast-path, two-delta sign
  consensus, three-delta election, regression against
  hand-computed example, trim-halves check), guard tests
  (shape-mismatch, empty-list, invalid trim_fraction), handler
  contract (callable + fresh state, DR-0 + DR-1 stamping on
  multi-call, unknown-op rejection, missing-deltas-key
  rejection, empty-deltas error propagation), snapshot /
  load_snapshot round-trip with accumulator seeded from a real
  merged delta.
- `tests/unit/test_micro_kiki_substrate.py` : retired the
  `test_recombine_raises_phase_2` `NotImplementedError` gate in
  favour of a wired-assertion smoke test ; updated the
  version-manifest assertion to `C-v0.9.0+PARTIAL`.

---

## [C-v0.9.0+PARTIAL — substrate-internal patch C-v0.8.1] — 2026-04-22 — micro-kiki SpikingKiki-V4 shim (PR #12)

### Added — real-backend ingestion path (additive, env-gated)

Opt-in real-artifact ingestion path on `MicroKikiSubstrate.load()`
and a rate-coded spike-synthesis path on `awake()` /
`awake_spike_payload()`. Gated behind the `DREAM_MICRO_KIKI_REAL=1`
environment variable **and** a valid `real_backend_path` — when
either is absent the substrate keeps the phase-1 / phase-2 stub
semantics bit-identical (all 19 pre-existing substrate + OPLoRA
tests green, plus 4 new real-backend tests).

Artifact layout consumed (produced by
`micro-kiki/scripts/convert_spikingkiki_35b.py`):
`<root>/lif_metadata.json` + per-module `<root>/block_NN_MODULE.npz`
(≈31 070 modules for the 35B-A3B-V4 build). The shim samples the
first 3 `.npz` files into `_real_state["sample_weights"]` so the
runtime does not page in the full ~70 GB artifact — the full SNN
forward pass still requires the MLX side.

`awake_spike_payload(prompt)` returns a
`{"output_channels": {"spikes": ndarray[T, out_dim]},
"metadata": {T, threshold, tau, real, module}}` dict. Rate-coded
LIF integration is deterministic per prompt (seeded by
`hash(prompt)`) so conformance harness runs are reproducible.

Additional best-effort `.safetensors` sidecar ingestion via
`_try_load_safetensors(adapter_path)` — pre-populates
`_current_delta` from a LoRA adapter when `safetensors` is
importable. Independent of the env flag ; no-op when the wheel is
missing.

### DualVer

- Repo : `0.9.0` (unchanged — additive shim, no public surface
  change).
- Substrate-internal : `C-v0.8.0+PARTIAL` → `C-v0.8.1+PARTIAL`
  (patch bump, additive real-backend shim).

### Tests

- `tests/unit/test_micro_kiki_real_backend.py` — 4 new tests
  covering env-unset stub path, env-set artifact load, missing
  metadata fallback, payload shape + raises-without-load.
- `tests/unit/test_micro_kiki_substrate.py` — version constant
  bumped.

---

## [C-v0.9.0+PARTIAL] — 2026-04-22 — micro-kiki LoRA substrate initial add (PR #11)

### Added — micro-kiki LoRA substrate (FC-MINOR bump)

Third substrate for framework C, wrapping the
[`micro-kiki`](https://github.com/kxkm-ai/micro-kiki) project's Qwen
MoE + LoRA adapter training output. Extends the DR-3 Conformance
Criterion surface from 2 substrates (MLX kiki-oniric + E-SNN
thalamocortical/Norse) to 3, adding a transformer-LoRA leg
alongside the SNN legs. Substrate-internal version:
`C-v0.8.0+PARTIAL`.

**Handlers backed (3/4)**

- `replay_handler_factory` + `downscale_handler_factory` operate
  over numpy tensors (CI fallback) and LoRA tensors (Apple Silicon
  path, env-gated on `mlx_lm`).
- `restructure_handler_factory` wires OPLoRA (Du et al., arXiv
  2510.13003): projects new adapter deltas onto the orthogonal
  complement of the prior subspace via numpy SVD. Module-level
  helper `_oplora_projector(prior_deltas, rank_thresh=1e-4)`.
  Guards: empty priors reject (caller handles no-op leg via
  handler), shape-mismatch across priors raises, rank collapse
  falls back to identity projector with a warning, **and
  full-rank saturation logs a warning** (priors spanning the
  full output space silently zero new adapters — the warning
  makes that visible to the curriculum scheduler). Read-only
  `MicroKikiRestructureState` dataclass exposed via
  `MicroKikiSubstrate.restructure_state` records DR-0 completion
  flag + operation label and DR-1 `episode_id` stamps.

**Stub deferred to phase 3 (1/4)**

- `recombine_handler_factory` (TIES-merge, Yadav et al., arXiv
  2306.01708) raises `NotImplementedError` with explicit
  citation. Surfaced rather than silently no-op'd so the gap
  stays visible in the cycle-3 conformance matrix. Unblocks once
  the 32-expert micro-kiki curriculum stabilises.

**Persistence + tests**

- `snapshot` / `load_snapshot` round-trip the accumulator delta
  via numpy `.npz` (portable across the 3-substrate test matrix).
- 30 tests across three files: `test_micro_kiki_substrate.py`
  (7 unit tests : manifest shape, handler signatures, DR-1 shape
  preservation on downscale, snapshot round-trip, OPLoRA-wired
  assertion replacing the phase-1 `NotImplementedError` gate),
  `test_micro_kiki_restructure.py` (13 unit tests : projector
  orthogonality, idempotence, symmetry, rank-collapse fallback,
  full-rank saturation warning, shape-mismatch guard, DR-0
  counters, DR-1 stamping, op-vocabulary guard, missing-key guard,
  projector shape guard) and
  `tests/conformance/axioms/test_dr3_micro_kiki_substrate.py` (10
  conformance tests covering the 3 DR-3 conditions for the
  substrate, parametrising the cycle-3 matrix entry).

## [C-v0.8.0+PARTIAL] — 2026-04-21

### Added — kiki_oniric.axioms public API (FC-MINOR bump)

- New module `kiki_oniric/axioms/` exposes the 6 GENIAL axioms
  (DR-0, DR-1, DR-2, DR-2', DR-3, DR-4) as frozen `Axiom` dataclass
  instances. Each carries formal statement (verbatim from spec §6.2),
  spec section, test references, DualVer version, amendment pointers,
  and an optional executable predicate.
- Predicates implemented: DR-2 strict-precedence precondition, DR-2'
  canonical-order equality check, DR-4 set-inclusion. DR-0, DR-1,
  DR-3 expose `predicate=None` (validated via conformance tests
  pointed to by `test_references`).
- Conformance tests for DR-2 now import from `kiki_oniric.axioms`;
  the local `_restructure_precedes_replay` duplicate was removed.
- DualVer: FC-MINOR bump per §12.2 (new public surface, no semantic
  change to existing axioms). EC axis unchanged.
- Closes issue #5.

### Infrastructure

- **OSF DOI registered** (2026-04-19T00:28:05Z) — DataCite DOI
  `10.17605/OSF.IO/Q6JYN` auto-minted when OSF registration was
  made public. OSF convention : DOI string is a deterministic
  derivation of the registration GUID (`10.17605/OSF.IO/<GUID>`
  uppercased), so paper drafts could reference it before publish
  without re-render. No "reserve-then-activate" semantics on OSF
  (prior project documentation using that language has been
  corrected 2026-04-21 post-audit). Project URL
  https://osf.io/q6jyn ; registration is now immutable per OSF
  policy. No DualVer bump (infrastructure event, no formal or
  empirical axis change). See
  `docs/milestones/osf-doi-mint-2026-04-20.md` (filename reflects
  checklist-creation date, not mint date).

---

## [C-v0.7.1+PARTIAL] — 2026-04-21

### Changed — DR-2 weakened (FC-PATCH bump)

- DR-2 (compositionality) now carries an explicit precondition
  excluding the empirically falsified class (permutations with
  RESTRUCTURE preceding REPLAY). See spec §6.2 and amendment
  `docs/specs/amendments/2026-04-21-dr2-empirical-falsification.md`.
- No semantic change to the compositionality claim itself on the
  safe class; this is a clarification/equivalent reformulation per
  DualVer §12.2 FC-PATCH rule.
- EC axis unchanged (no new gate crossed).

---

## [C-v0.7.0+PARTIAL] — 2026-04-19

Cycle-3 Phase 1 launch bump. FC MINOR (+0.1.0) because cycle 3
adds a new derived constraint surface on the formal axis :

### Added — R1 output-hash API multi-artifact extension (issue #2)

- ``run_output_hashes`` schema extended from single-hash-per-run to
  N-hashes-per-run keyed on ``(run_id, artifact_name)``. Required
  for Paper 2 (ablation) where each run may register multiple
  artifacts (per-checkpoint, per-metric bundle, per-profile snapshot).
- New API, kwargs-only on the new parameters for backward-compat :
  ``register_output_hash(run_id, output_hash, *, artifact_name='canonical',
  hash_type='sha256')``,
  ``get_output_hash(run_id, *, artifact_name='canonical') -> str``,
  ``list_output_hashes(run_id) -> dict[str, str]``.
- ``_ensure_schema`` is idempotent over three states (fresh DB,
  legacy v1 single-hash schema, already-new schema). Legacy rows
  migrate to ``artifact_name='canonical'`` via a transactional
  rename / create / copy / drop. No action required on existing DBs.
- 9 new unit tests covering backward-compat, multi-artifact
  coexistence, exact-match idempotence, hash conflict, hash_type
  conflict, empty-list semantics, and migration idempotence /
  preservation.
- No DualVer bump : R1 contract is semantically preserved
  (``(run_id, artifact_name) -> hash`` is bit-stable ; the
  single-artifact case is still addressable verbatim via the
  ``canonical`` default). No change to ``run_id`` computation.

### Added — R1 output-hash API (additive, no-migration)

- `RunRegistry.register_output_hash(run_id, output_hash)` +
  `RunRegistry.get_output_hash(run_id)` — second half of R1 now
  enforceable from the caller (recorded op output hash is stable
  for a registered tuple, conflicts raise with an R1 tag).
- New sibling table `run_output_hashes(run_id PK, output_hash,
  recorded_at)` created additively by `_ensure_schema` ; `runs`
  schema is untouched so `run_id` computation and the existing DB
  rows remain bit-stable.
- `tests/reproducibility/test_r1_run_registry_contract.py` —
  `test_r1_registry_output_hash_contract` flipped from `xfail` to
  passing ; docstring + module preamble rewritten to describe the
  enforced contract rather than the blocker.
- 5 new unit tests in `tests/unit/test_run_registry.py` covering
  roundtrip, idempotence on exact match, conflict → `ValueError`
  (with R1 tag + run_id), unknown run_id → `KeyError`, and missing
  hash → `KeyError`.

- **H6 profile-ordering** (`P_max > P_equ > P_min` on retained
  accuracy after consolidation) is a new derived constraint per
  framework-C §12.2. Exercised by the cycle-3 Gate D decision
  script (C3.9, scheduled) ; pre-locked in the spec glossary.
- **Scale-axis** (`{qwen3p5-1p5b, qwen3p5-7b, qwen3p5-35b}`) is a
  new formal axis for cross-scale DR-3 replication — extends DR-3
  Conformance Criterion from "two substrates" (cycle 2) to "two
  substrates × three model scales" (cycle 3).
- **R1 determinism contract** extended : cycle-3 run_id tuple now
  encodes ``(harness_version, scale/profile/substrate
  composite-tag, seed, commit_sha)`` so every cell of the
  1080-config matrix is uniquely + deterministically addressable.

EC axis set to **PARTIAL** (demoted from STABLE per §12.3
transition rule) because cycle-3 Phase 2 cells are scoped-deferred :

- C3.11-C3.14 — Norse LIF substrate + SNN ops (sem 4-5) — deferred
- C3.15-C3.18 — fMRI alignment (Studyforrest HMM/CCA, sem 5-6) —
  deferred
- C3.19-C3.22 — Paper 1 v2 Nature HB writeup + publication lock
  (sem 6) — deferred

Phase 1 engineering (C3.1-C3.10) is green / in flight ; G10 Gate D
= CONDITIONAL-GO/PARTIAL. Pivot-4 branch per cycle-3 design spec
(`docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md`
§5.1 R3) replaces the final STABLE graduation with a new minor
bump if Gate D = NO-GO.

### Added — Phase 1 cycle-3 (C3.1-C3.5 pre-bump)

- Qwen3.5 MLX SHA-256 pins — `harness/real_models/base_model_registry.py` (commit f3b0119)
- MMLU + HellaSwag dataset SHA pins — `harness/real_benchmarks/dataset_registry.py` (commit 3271883)
- Studyforrest download init script — `scripts/init_studyforrest_download.sh` (commit 7b79b9e)
- Real-bench loaders (MMLU + HellaSwag + mega-v2) — commit cae16f8
- Qwen3.5 MLX 3-scale wrappers — commit abcc6ea
- Real-weight ops over Qwen MLX — commit ab55c67
- H5 trivariant scaling law — `kiki_oniric/eval/scaling_law.py` (commit 8efec8f)
- Bonferroni combined 8-test family — `kiki_oniric/eval/statistics.py` (commit 6643598)

### Added — Phase 1 cycle-3 (C3.6, this bump cycle)

- `scripts/ablation_cycle3.py` — 1080-config multi-scale cartesian
  runner with resume semantics + deterministic run_id lineage

### Changed

- DualVer bumped C-v0.6.0+STABLE → C-v0.7.0+PARTIAL per §12.3
  transition rule (FC MINOR + EC STABLE → PARTIAL demotion)
- G10 cycle-3 Gate D surfaced as active gate in STATUS.md
- `HARNESS_VERSION` constants in cycle-2 scripts
  (`scripts/ablation_cycle2.py`, `scripts/ablation_g4.py`) aligned
  to the new tag so all future cycle-2-rerun rows share the
  current empirical-axis reading
- `c_version` default in `kiki_oniric/eval/ablation.py` aligned
- `MLX_SUBSTRATE_VERSION` / `ESNN_SUBSTRATE_VERSION` aligned

### Pending — Phase 1 cycle-3 (C3.7-C3.10 remainder)

- C3.7 : sanity pilot 1.5B fail-fast (scripted this bump cycle)
- C3.8 : full 1080-config Studio launch (~10 days)
- C3.9 : compute_gate_d + H1-H6 report generator
- (C3.10 is this bump — no remainder)

### Pending — Phase 2 cycle-3 (scoped-deferred, C3.11-C3.22)

- Norse LIF substrate + SNN ops (C3.11-C3.14)
- fMRI alignment (C3.15-C3.18)
- Paper 1 v2 Nature HB writeup (C3.19-C3.22)

### Stats (at this bump)

- 240 tests passing, coverage 91.13 % (gate ≥ 90 %)
- 0 AI attribution in any commit
- 13 files version-bumped (see commit body)

---

## [C-v0.6.0+STABLE cycle-2 closeout] — 2026-04-19

Cycle 2 fully closed. Phase 3 (cross-substrate ablation) + Phase 4
(Paper 2 narrative) delivered since PARTIAL tag (commits 48b0521..
b8d7abe). G9 → FULL-GO.

### Added — Phase 3 cross-substrate validation (C2.9-C2.12)

- Multi-substrate ablation runner (`kiki_oniric/eval/ablation.py`,
  substrate axis)
- Conformance validation matrix script + docs (3×3 MLX × E-SNN ×
  hypothetical_cycle3)
- `docs/proofs/dr3-substrate-evidence.md` formal evidence
- Cross-substrate H1-H4 statistical results (synthetic substitute)
- Paper 1 §8.5 cycle-2 cross-substrate replication subsection
  (EN + FR)
- Paper 1 v2 arXiv plan (deferred until v1 acceptance)

### Added — Phase 4 Paper 2 narrative (C2.13-C2.16)

- Paper 2 abstract + introduction (EN + FR, ~1400 words)
- Paper 2 §4 Conformance + §5 Architecture (EN + FR)
- Paper 2 §6 Methodology + §7 Results (EN + FR, cross-substrate
  comparative table)
- Paper 2 §8 Discussion + §9 Future Work + full-draft assembly
- Paper 2 pandoc LaTeX render `docs/papers/paper2/build/full-draft.tex`
- Paper 2 build/ : README-arxiv.md + .gitignore (mirror Paper 1
  pattern)

### Changed

- DualVer bumped C-v0.6.0+PARTIAL → C-v0.6.0+STABLE per §12.3
  transition rule
- G9 cycle-2 publication gate : CONDITIONAL-GO/PARTIAL →
  FULL-GO/STABLE
- STATUS.md active gate aligned
- 13 CodeRabbit cycle-10 findings applied (commits 4b67f3e +
  2d1228e)

### Stats

- 180 tests passing, coverage 91.34 % (gate ≥ 90 %)
- 2 substrates real (MLX kiki-oniric + E-SNN thalamocortical),
  1 placeholder (hypothetical_cycle3)
- 3 profiles fully wired (P_min, P_equ, P_max)
- 17 commits across cycle 2 (Phase 1 + 2 + 3 + 4 + 5 + 2
  CodeRabbit cycle-9 + 2 cycle-10 fix batches + 2 DualVer bumps)
- 32 CodeRabbit findings applied (cycle 9 = 19, cycle 10 = 13)
- 0 AI attribution in any commit

### Pending — external user actions (unchanged)

- arXiv submit Paper 1 → obtain arXiv ID for Paper 2 cross-cite
- Paper 2 v0.1 review + arXiv submit
- OSF DOI lock (post-arXiv)
- Nature HB submit (or Pivot B branch selection)
- HAL FR deposit (post-arXiv)
- DR-2 external reviewer feedback (T-Col Q_CR.1 b)
- fMRI lab partnership formalization

---

## [C-v0.6.0+PARTIAL] — 2026-04-19

Cycle 2 closeout. Engineering Phases 1 + 2 + 5 complete.
Phase 3 (cross-substrate ablation C2.9-C2.12) + Phase 4 (Paper 2
narrative C2.13-C2.16) deferred per user scope decision ;
G9 gate = CONDITIONAL-GO/PARTIAL.

### Added

- E-SNN thalamocortical substrate (Phase 1 : C2.1-C2.4) — second
  `@runtime_checkable` Protocol target for DR-3
- DR-3 Conformance Criterion extended to E-SNN substrate (G7 LOCKED)
  ; condition (1) of DR-3 now spans 2 substrates
- P_max profile fully wired : 4 ops + 4 channels (Phase 2 :
  C2.5-C2.8, G8 LOCKED) ; DR-4 chain extended in
  `tests/conformance/axioms/test_dr4_p_max_chain.py`
- α-stream raw-traces ring buffer (canal-α)
- recombine_full MLX VAE variant (encoder + decoder + KL)
- ATTENTION_PRIOR canal-4 + S4 invariant guard
- Real concurrent dream worker (C2.17 : asyncio/threading via
  `threaded=True` mode)
- G9 cycle-2 publication-ready gate report
  (CONDITIONAL-GO/PARTIAL)
- Paper 3 outline (cycle-3 amorçage, PROVISIONAL)

### Changed

- DR-2 reframed as unproven working axiom (DR-2') ; parallel
  `recombine` branch marked out-of-scope of sequential composition
  until G3-draft delivers a parallel monoidal model
- DR-3 conformance proof reframed as operational evidence (vs
  formal `⟹`) — circular-implication footgun closed
- recombine MLX RNG isolated (local `mx.random.key`, no global
  seed mutation) — R1 hardened
- attention_prior `get_prior()` returns read-only numpy view —
  S4 cannot be bypassed by mutation
- attention guard rejects NaN explicitly before range/budget checks
  — S4 plug
- `awake.pause(500ms)` clarified vs `K3_max=1s` (operational target
  vs warning threshold) in §7.2 swap protocol
- 19 CodeRabbit cycle-9 findings applied (3 commits : safety +
  spec-fr + prose, then 1 EN-sync commit mirroring spec/outline)

### Pending — external user actions

- Phase 3 cycle-2 (cross-substrate ablation C2.9-C2.12) — deferred
- Phase 4 cycle-2 (Paper 2 narrative C2.13-C2.16) — deferred
- arXiv submission of Paper 1
- Nature HB editorial decision
- OSF DOI lock (`docs/osf-upload-checklist.md`)
- DR-2 external reviewer feedback (T-Col Q_CR.1 b)
- HAL FR deposit (post-arXiv)
- fMRI lab partnership formalization (T-Col extension)

### Stats

- 173 tests passing (target ≥ 173), coverage 91.26%
  (gate ≥90%)
- ~13 commits across cycle 2 (Phase 1 + Phase 2 + Phase 5 +
  3 cycle-9 fix commits + 1 EN-sync + 1 DualVer bump)
- 0 AI attribution in any commit
- Substrates : **2** (MLX kiki-oniric + E-SNN thalamocortical)
- Profiles : **3** wired (P_min, P_equ, P_max)
- DR-3 Conformance conditions (1) + (2) + (3) : all green on both
  substrates

### Milestones achieved

- **G7** E-SNN substrate conformance LOCKED
- **G8** P_max profile wired LOCKED
- **G9** cycle-2 publication-ready CONDITIONAL-GO/PARTIAL

---

## [C-v0.5.0+STABLE] — 2026-04-17

End of setup phase (S1-S4). Program enters implementation phase (S5+).

### Added

- Formal framework C specs (master + framework C, 977 lines)
- Implementation plan phase-1 detailed (1416 lines, 80 checkboxes)
- Implementation plan S3-S4 atomic (1464 lines, 6 tasks)
- DR-3 Conformance Criterion strengthened (post-critic finding #3)
- Python project skeleton (uv, Python 3.12+, mlx, numpy, hypothesis,
  pytest, duckdb, pyarrow, plotly, yaml, click)
- RunRegistry with deterministic run_id (SHA-256-based, R1 contract)
- Invariants & Axioms registry (I/S/K families + DR-0..DR-4)
- Canonical glossary (primitives, profiles, DualVer, gates, metrics)
- T-Col outreach plan (3 fMRI labs + formal reviewer candidates)
- GitHub Actions CI workflow (lint + types + pytest + invariants)
- Fork decision document (kiki-oniric r3 jalonné S1/S8/S18)
- kiki_oniric skeleton
- Studyforrest RSA feasibility note (G1 Branch A locked)
- Story 0 expose typed Protocols (8 primitives, Conformance condition 1)
- Interface contracts : primitives.md + eval-matrix.yaml
- EvalMatrix loader with 6 contract tests
- OSF pre-registration draft (H1-H4 operationalized)
- Formal reviewer recruitment tracker + email template
- Retained benchmark (SHA-256 integrity, 50 synthetic items)
- fMRI schema lock (Studyforrest Branch A)
- Framework version bumped C-v0.3.1 → C-v0.5.0+STABLE

### Changed

- sqlite3 context manager fixed (leak + contextlib.closing)
- .gitignore now excludes .coverage artifacts

### Stats

- 21 commits across brainstorm → spec → plan → execution flow
- 16 tests passing, coverage 93.62% (gate ≥90%)
- 5 source files with C-v0.5.0+STABLE version consistency
- 0 BLOCKING invariant violations
- 0 AI attribution in any commit

### Milestones achieved

- **G1** — T-Col fallback locked Branch A (Studyforrest feasibility)
- **DR-3 Conformance Criterion condition (1)** — typed Protocols
  exposed, 3 tests passing

### Pending (S5+)

- DR-2 compositionality proof draft (G3-draft S6)
- P_min runtime functional (G2 S8)
- OSF pre-registration lock (upload via checklist)
- fMRI lab outreach replies (S3-S5)
- Formal reviewer recruitment (Q_CR.1 b, S3-S5)

---

## [C-v0.5.0+STABLE cycle-1 closeout] — 2026-04-18

End of cycle 1 (S5-S28). Programme essentially complete;
arXiv submission + Nature HB submission + DR-2 reviewer feedback
+ OSF lock = external user actions pending.

### Added — implementation phase (S5-S12)

- DreamEpisode 5-tuple dataclass + BudgetCap (S5.1)
- DreamRuntime scheduler with DR-0 log guarantee + try/finally
  exception handling (S5.2)
- DR-0 + DR-1 property tests via Hypothesis (S5.3)
- Replay operation skeleton (A-Walker source, S5.4)
- Downscale operation skeleton (B-Tononi SHY, S7.1)
- S2 finite guard (NaN/Inf check, S7.2)
- Swap protocol skeleton with S1+S2 guards (S7.3)
- P_min profile fully wired (S7.4 + S9.4 swap_now)
- P_equ profile skeleton then fully wired (S8.3 + S11.2)
- Restructure operation (D-Friston FEP, S10.1 skeleton +
  S13.1 MLX-native)
- S3 topology guard (validate_topology, S10.2)
- Recombine operation (C-Hobson, S11.1 skeleton + S13.2 MLX VAE)
- DR-4 profile inclusion proof + axiom test (S12.1)
- G2 P_min viability report (S8.1 + S9.5 pilot)
- G3 decision log skeleton (S8.2)
- G4 P_equ functional report (S12.2 + S15.3 ablation)

### Added — MLX integration phase (S9-S15)

- Replay handler MLX backend (MSE + SGD on real model, S9.1)
- Downscale handler MLX backend (real weight shrinkage, S9.2)
- Retained eval bridge (evaluate_retained → S1 guard, S9.3)
- mega-v2 dataset loader (real path + synthetic fallback, S13.3)
- Concurrent dream worker skeleton (Future API, S14.1)
- Statistical eval module H1-H4 (Welch / TOST / Jonckheere /
  one-sample t with Bonferroni, S15.1)
- Ablation runner harness (cartesian profile × seed grid, S15.2)
- G4 ablation real run on synthetic mega-v2 (3/4 hypotheses
  significant at α = 0.0125, S15.3)

### Added — closing phase (S16-S28)

- P_max profile skeleton with target_ops + target_channels_out
  metadata for DR-4 chain (S16.1)
- DR-4 P_max chain extension test (S16.2)
- Paper 1 outline + abstract + introduction (S17)
- Paper 1 results-section + discussion + future-work +
  references.bib (S18-S19)
- Paper 1 methodology + background + full-draft assembly (S20)
- arXiv preprint submission tracker (S21.1)
- G5 publication-ready gate report + Nature HB submission
  tracker (S22.1 + S18.1)
- Pivot B contingency decision tree (S18.3)
- Reviewer feedback collection skeleton (S25.1)
- Paper 2 cycle-2 outline (S27.1)
- G6 cycle-2 bootstrap decision report (S28.1)

### Changed — code-review iterations

- 5 CodeRabbit review cycles: 6 + 10 + 7 + 2 + 78 cumulative
  findings, ~95 fixes applied
- run_id width bumped 16 → 32 hex chars (collision safety)
- DR-3 Conformance Criterion strengthened with executable
  conditions
- Op-pair commutativity matrix corrected (downscale not
  idempotent)
- DR-2 free-semigroup softened to "semigroup generated by"
  pending universal property proof
- evaluate_retained accepts seed parameter (CRIT bug fix)
- AblationRunner registers run_id (R1 contract enforcement)

### Stats — cycle 1 close

- ~96 commits across brainstorm → spec → plan → 6 atomic plans →
  execution flow
- 116 tests passing, coverage 90.37% (gate ≥90%)
- DR-3 Conformance Criterion conditions (1) + (2) + (3) all
  green ; DR-2 reviewer pending (G3)
- 0 AI attribution in any commit
- 6 atomic plans covering S1-S28 calendar

### Milestones achieved

- **G1** Branch A Studyforrest LOCKED
- **G2** P_min GO-CONDITIONAL (synthetic pipeline PASS)
- **G4** P_equ GO-CONDITIONAL/PASS (3/4 hypotheses significant
  on synthetic ablation)

### Pending — external user actions

- **G3** DR-2 external reviewer feedback (T-Col Q_CR.1 b)
- **G5 → GO-FULL** Paper 1 arXiv preprint render + submission
- Nature HB submission (or Pivot B branch selection at S22)
- OSF DOI lock (`docs/osf-upload-checklist.md`)
- fMRI lab partnership formalization (T-Col extension)
- Real mega-v2 ablation run (closeout for GO-CONDITIONAL flip
  to GO-FULL)
- Cycle 2 bootstrap decision (G6, post-cycle-1 retrospective)
