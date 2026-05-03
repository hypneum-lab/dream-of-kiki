# G4-sexto pilot pre-registration

**Date:** 2026-05-03
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**Sister pilot:** G4-quinto (H5-A False, H5-B False, H5-C
**CONFIRMED** — Welch p = 0.9918, Hedges' g = -0.0026,
mean P_max(mog) = 0.9842, mean P_max(none) = 0.9845, N = 30,
commit `a02b82c`).
**Substrates:**
- Step 1 — MLX small CNN multi-class head
  (`G4SmallCNN`, 2 Conv2d + 2 MaxPool2d + 2 Linear,
  `latent_dim = 64`, `n_classes = 10` per task — the only
  divergence from G4-quinto Step 3 is the n_classes constant).
- Step 2 — MLX medium CNN
  (`G4MediumCNN`, 3 Conv2d + 3 MaxPool2d + 2 Linear,
  `latent_dim = 128`, `n_classes = 20` per task, 64×64 RGB
  input). **DEFERRED** to G4-septimo follow-up under the locked
  Option B compute path (see §6 row 5 + §9 deviation envelope d).
**Benchmark:**
- Step 1 — Split-CIFAR-100 (10 sequential class-incremental
  tasks of 10 fine classes each, fine labels remapped to
  `{0..9}` per task).
- Step 2 — Split-Tiny-ImageNet (10 sequential 20-class tasks,
  64×64 RGB JPEG decoded from HF parquet shards, deferred under
  Option B).
**Compute option:** **B (CIFAR-100 only, N = 30 seeds/arm × 2
strategies × 4 arms = 240 cells, ~12-20 h overnight on M1 Max).**
Option A (Step 1 + Step 2, 480 cells, 32-52 h M1 Max or 6-10 h
Mac Studio) was considered but ruled out for this pilot — a
Mac Studio session was not available, and a multi-night M1 Max
run would push beyond the OSF lock-to-evidence-fresh window
agreed at the 2026-05-03 G5-bis closeout. Option C smoke (32
cells) is reserved for CI / pipeline validation only and never
produces a milestone artefact under `docs/milestones/`.
**Lock commit:** *(to be filled by the introducing commit hash)*
**Lock timestamp:** 2026-05-03 (pre-Task 6 / Step 1 driver run).

## §1 Background

G4-quinto (commit `a02b82c`, milestones
`docs/milestones/g4-quinto-{aggregate,step1,step2,step3}-2026-05-03.{json,md}`)
reported **H5-A False, H5-B False, H5-C CONFIRMED** on
Split-CIFAR-10 across a 3-step sequential pipeline (600 cells,
~72 min M1 Max). The H5-C confirmed verdict — Welch two-sided
p = 0.9918, Hedges' g = -0.0026, N = 30, P_max(mog) = 0.9842
vs P_max(none) = 0.9845 — establishes that, at the
Split-CIFAR-10 small-CNN scale, the RECOMBINE channel is
empirically empty. Combined with the G4-quater H4-C verdict
on Split-FMNIST (Welch p = 0.989, g = 0.002, N = 95), the
universality flag fires across two benchmarks × two substrates :
{Split-FMNIST, Split-CIFAR-10} × {3-layer MLP, small CNN}.

The G4-quinto pre-reg §6 row 6 explicitly flags
"ImageNet escalation prerequisite for any STABLE promotion"
and notes that the empirical scope of any future framework-C
"richer ops" claim must "explicitly exclude the {Split-FMNIST,
Split-CIFAR-10} × {3-layer MLP, 5-layer MLP, small CNN} cells
of the escalation ladder". G4-sexto fires that follow-up at
**mid-large benchmark + class-count scale**, escalating from
binary 2-class per-task heads (FMNIST, CIFAR-10) to a
**multi-class** 10-class per-task head on Split-CIFAR-100 and,
under Option A, a 20-class per-task head on Split-Tiny-ImageNet
with a 64×64 RGB input + 3-Conv backbone.

The Hu 2020 anchor (g = 0.29, human sleep-dependent memory
consolidation) is used here strictly as a **directional**
reference — sign of the alternative hypothesis — not a magnitude
calibrator. A cross-class comparison between a biological cohort
and a seeded numerical pilot would be a category error.

## §2 Hypotheses (confirmatory)

- **H6-A (mid-large class count)** — on Split-CIFAR-100 with
  the small CNN substrate (`G4SmallCNN`, `latent_dim = 64`,
  `n_classes = 10` per-task head, identical architecture to
  G4-quinto Step 3 modulo the multi-class output linear),
  `retention(P_max with RECOMBINE = mog)` is not statistically
  distinguishable from `retention(P_max with RECOMBINE = none)`.
  Test : Welch two-sided fails to reject H0 at
  α = 0.05 / 3 = 0.0167 (Bonferroni for 3 hypotheses).
  **Failing** to reject **is** the predicted positive empirical
  claim that the H5-C RECOMBINE-empty finding generalises to
  CIFAR-100 at 100-class scale.

- **H6-B (mid-large resolution + class count)** — on
  Split-Tiny-ImageNet with the medium CNN substrate
  (`G4MediumCNN`, `latent_dim = 128`, `n_classes = 20`
  per-task head, 64×64 RGB input, Conv2d×3 + MaxPool2d×3 +
  Linear×2), same Welch two-sided contract at α = 0.0167.
  **DEFERRED** to G4-septimo follow-up under the locked Option
  B compute path. The aggregator emits `h6b_deferred = True`
  for this pilot.

- **H6-C (universality of RECOMBINE-empty across 4 benchmarks
  × 2 substrate families)** — derived conjunction
  `H6-A_confirmed AND H6-B_confirmed`. If both H6-A and H6-B
  confirm, the universality claim spans {Split-FMNIST,
  Split-CIFAR-10, Split-CIFAR-100, Split-Tiny-ImageNet} ×
  {3-layer MLP, 5-layer MLP, small CNN, medium CNN}. If only
  H6-A confirms (Option B locked path), universality is
  **scope-bound** to {FMNIST, CIFAR-10, CIFAR-100} × {3-layer
  MLP, 5-layer MLP, small CNN}, with H6-B deferred — an open
  empirical question rather than a falsification.

No additional Welch test for H6-C ; it is a logical aggregation
(no Bonferroni adjustment needed beyond the per-step α).

## §3 Power analysis

N = 30 seeds per arm at α = 0.0167 detects |g| ≥ 0.74 at 80 %
power (Welch two-sided, equal-variance approximation,
`scipy.stats.power.tt_ind_solve_power`). Effect sizes below
|g| ≈ 0.74 remain exploratory at this N. The lower N (30 vs
G4-quater's 95) is dictated by the compute envelope — Option B
already targets ~12-20 h on M1 Max ; N = 95 would push wall
time beyond a single overnight session. Identical N to G4-quinto
Step 3 — the H6-A verdict is therefore directly comparable to
H5-C in absolute terms.

## §4 Exclusion criteria

- For multi-class heads, the "random chance" baseline is
  `1 / n_classes_head` (= 0.10 for the CIFAR-100 head with
  `n_classes = 10`, = 0.05 for Tiny-IN with `n_classes = 20`).
  A defensive widening over the binary `< 0.5` threshold is
  applied : `acc_initial < 2 × random_chance` (= 0.20 for
  CIFAR-100, = 0.10 for Tiny-IN) — exclude cell. This preserves
  the "seed failed to learn task 1" semantics of the binary
  G4-quinto rule while adapting to the multi-class regime.
- `acc_final` non-finite — exclude cell.
- any seed reproducing run_id collision with prior pilot's
  registry — abort and amend pre-reg.

## §5 Substrate / driver paths

- Step 1 driver : `experiments/g4_sexto_test/run_step1_cifar100.py`
- Step 2 driver : `experiments/g4_sexto_test/run_step2_tiny_imagenet.py`
  (deferred under Option B)
- Substrates :
  - small CNN (multi-class head) :
    `experiments.g4_quinto_test.small_cnn.G4SmallCNN`
    (reused unchanged ; `n_classes = 10`)
  - medium CNN (deferred) :
    `experiments.g4_sexto_test.medium_cnn.G4MediumCNN`
- Loaders :
  - `experiments.g4_sexto_test.cifar100_dataset.load_split_cifar100_10tasks_auto`
  - `experiments.g4_sexto_test.tiny_imagenet_dataset.load_split_tiny_imagenet_10tasks_auto`
    (deferred)
- Sources :
  - CIFAR-100 canonical : https://www.cs.toronto.edu/~kriz/cifar-100-binary.tar.gz
    (~163 MB, SHA-256 pinned in
    `cifar100_dataset.CIFAR100_TAR_SHA256` at first-download
    lock commit). HF parquet fallback :
    `https://huggingface.co/datasets/uoft-cs/cifar100`
    (commit pinned at first download — same §9.1-style amendment
    pattern as G4-quinto).
  - Tiny-ImageNet HF :
    `https://huggingface.co/datasets/zh-plus/tiny-imagenet`
    (commit pinned at first download). Canonical
    https://image-net.org/data/tiny-imagenet-200.zip is fallback
    only ; out of scope for this pilot's loader.

## §6 DualVer outcome rules

| Outcome | EC bump | FC bump |
|---|---|---|
| Row 1 — H6-A and H6-B both confirmed | EC stays PARTIAL ; H6-C confirmed (RECOMBINE-empty universalises across 4 benchmarks × 4 substrates) ; DR-4 partial refutation universalises further ; DR-4 evidence file revised to v0.5. | FC stays C-v0.12.0 |
| Row 2 — H6-A confirmed only | EC stays PARTIAL ; H6-C partial (scope-bound to CIFAR-100 ; Tiny-IN open Q) ; DR-4 evidence v0.5 records partial extension. | FC stays C-v0.12.0 |
| Row 3 — H6-B confirmed only | EC stays PARTIAL ; H6-C partial (scope-bound to Tiny-IN ; CIFAR-100 anomaly) ; DR-4 evidence v0.5 records partial extension. | FC stays C-v0.12.0 |
| Row 4 — both falsified (Welch rejects H0 on at least one) | EC stays PARTIAL ; **H5-C universality breaks at mid-large scale** ; DR-4 partial refutation tagged as low-scale-bound ; DR-4 evidence v0.5 records the boundary. | FC stays C-v0.12.0 |
| Row 5 — **Option B chosen (LOCKED)** ; Step 2 deferred | EC stays PARTIAL ; aggregator reports `h6b_deferred = True` ; H6-C deferred (scope-bound conjunction over {FMNIST, CIFAR-10, CIFAR-100} × {3-layer MLP, 5-layer MLP, small CNN}) ; document deferral in CHANGELOG. | FC stays C-v0.12.0 |
| Row 6 — Option C smoke | no science verdict, milestone artefacts NOT committed (use `--smoke`). | n/a |

EC stays PARTIAL across **all** rows. FC stays at v0.12.0 across
all rows (no formal-axis bump scheduled by this pilot).

**Path A Studio recommended** (verbatim §6 banner from plan
§"Architecture decisions") : the locked Option B path forces
M1-Max-only execution and defers H6-B to G4-septimo. Multi-night
M1 Max sessions for the full 480-cell sweep are explicitly
declined here.

## §7 Reporting commitment

Honest reporting of all observed scalars regardless of outcome ;
if Welch tests do not reject H0, the verdict is "no evidence for
the hypothesis at this N", not "hypothesis falsified" (absence
of evidence vs evidence of absence).

H6-A (and H6-B once it runs) confirmation specifically requires
Welch failing to reject difference between RECOMBINE = mog and
RECOMBINE = none — *"Welch fail-to-reject = absence of evidence
at this N for a difference between mog and none — under
H6-A/H6-B specifically, this **is** the predicted positive
empirical claim that RECOMBINE adds nothing measurable beyond
REPLAY+DOWNSCALE on the substrate at this scale."* (Verbatim
honest-reading clause adapted from G4-quinto §7, embedded in
every G4-sexto milestone MD.)

If H6-A is confirmed (the pre-locked Option B success path),
the partial refutation of DR-4 established by G4-ter and
universalised by G4-quinto is **further extended** to CIFAR-100
at 100-class scale, and the DR-4 empirical evidence file
(`docs/proofs/dr4-profile-inclusion.md`) is amended with a
G4-sexto §"Empirical-evidence amendment" addendum (v0.5).

## §8 Audit trail

Every cell registered via `harness/storage/run_registry.py` with
profile keys `g4-sexto/{step1,step2}/<arm>/<combo>/<strategy>`
and R1 bit-stable run_ids. Milestone artefacts under
`docs/milestones/g4-sexto-step{1,2}-2026-05-03.{json,md}` plus
aggregate `docs/milestones/g4-sexto-aggregate-2026-05-03.{json,md}`.

## §9 Deviations

Any deviation from this pre-reg requires (a) a separate amendment
commit before the affected cell runs, or (b) a post-hoc honest
disclosure in the paper section acknowledging the deviation and
its impact on the confirmatory status of the hypothesis.

Pre-known deviation envelopes (executor open questions) :

a. CIFAR-100 / Tiny-IN download fails (network unavailable /
   SHA-256 mismatch on canonical mirror) — abort the affected
   step and file a §9.1 amendment ; do not proceed with cells.
b. Step 1 / Step 2 `acc_initial < 2 × random_chance` (= 0.20 for
   CIFAR-100, = 0.10 for Tiny-IN) for a majority of seeds —
   raise per-task epochs from 3 to 5, document in the step
   milestone header.
c. Step 2 wall time > 16 h (overnight ceiling) — reduce N from
   30 to 20 for Step 2 only, tag exploratory (Option A only ;
   does not apply under the locked Option B path).
d. **Option B chosen (LOCKED for this pilot)** — Step 2 is
   deferred to a G4-septimo follow-up ; aggregator reports
   `h6b_deferred = True`, no H6-B verdict in this pilot ;
   document deferral in CHANGELOG entry. The H6-C conjunction
   is "deferred" (not "falsified" or "partial").
e. Path A Studio recommended (verbatim §6 banner) — M1-Max-only
   execution implicitly forces Option B unless multi-night
   session accepted. **For this pilot, Option B is locked
   pre-Task 6 ; this deviation envelope is consumed by §6 row
   5 and the executor proceeds without further amendment.**

### §9.1 Amendment — epochs raised from 3 to 8 (filed 2026-05-03)

**Trigger** : pre-pilot probe across 8 seeds with the exact
G4-quinto Step 3 hyperparameters (epochs=3, batch_size=64,
lr=0.01) showed `acc_initial` ranging 0.138-0.184 — uniformly
below the multi-class exclusion threshold
`2 × random_chance = 0.20`. Envelope b authorizes raising
epochs from 3 to 5 ; an intermediate probe at epochs=5 still
produced 2/3 seeds below threshold (acc_initial ∈
{0.212, 0.153, 0.154}). The CIFAR-100 100-class learning
problem at task 0 (10 fine classes, ~5000 training images) is
materially harder than the binary CIFAR-10 task 0
(~10000 images, 2 classes) and needs more epochs to clear the
exclusion floor at majority of seeds.

**Amendment** : the Step 1 driver is run with `epochs=8`
(intermediate beyond envelope b's authorized 3 → 5 step).
A subsequent probe at epochs=8 across 8 seeds produced
acc_initial ∈ {0.188, 0.209, 0.209, 0.270, 0.268, 0.238, 0.247,
0.226} — 7/8 above the exclusion threshold. The exclusion
criterion (§4) is unchanged ; only the per-task training
budget changes.

**Why this preserves confirmatory status** : the H6-A
hypothesis is operationalised on `retention(P_max with mog)`
vs `retention(P_max with none)`. Both arms are trained with
identical epochs and HP, so the change in epochs cannot
asymmetrically advantage one strategy over the other. The
exclusion threshold is also strategy-symmetric. The amendment
adjusts the *training budget* to reach a meaningful baseline ;
the *experimental contract* (Welch placebo at α=0.0167) is
unchanged. R1 bit-stable run_ids are still keyed on
`(c_version, profile, seed, commit_sha)` — the epochs change
is folded into the driver source captured by `commit_sha`.

**Wall-time impact** : per-cell training time ~2s/task × 10
tasks = ~20s + dream episodes overhead ; 240 cells × ~25s ≈
100 minutes total ; well within the Option B 12-20 h budget.
