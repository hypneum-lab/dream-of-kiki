# G4-nono cross-domain RECOMBINE pre-registration

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI).
**Date:** 2026-10-23 (revision 2 after independent review)
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**Sister pre-regs (locked, not modified):**
`docs/osf-prereg-g4-sexto-pilot.md` (H6-A, CIFAR-100),
`docs/osf-prereg-g4-septimo-pilot.md` (H6-B / H6-C, Tiny-ImageNet).
Their power sentences carry an arithmetic slip; see
`docs/osf-prereg-erratum-g4-power-2026-10-23.md` (the locked files
are not edited).
**c_version at drafting:** C-v0.25.0+PARTIAL
**Substrate:** MLX small CNN (`G4SmallCNN`,
`experiments.g4_quinto_test.small_cnn`) with `n_classes = 20`,
`latent_dim = 64`: one shared 20-way head over coarse labels
(class-incremental, no per-task label remap).
**Benchmark:** Split-CIFAR-100-coarse two-domain stream (§5).
**Design:** matched-block (seed = block), paired analysis (§2, §6).
**Compute:** N = 30 seeds × 8 cells per seed = 240 cells (§5).
**Power script:** `scripts/prereg_power_2026_10.py` (§3).
**Lock commit:** *(filled by introducing commit hash at filing)*
**Lock timestamp:** *(filled at filing, pre-driver-run)*

## §1 Background

The G4 escalation ladder (G4-quater → G4-septimo) confirmed the
"RECOMBINE-empty" finding across {Split-FMNIST, Split-CIFAR-10,
Split-CIFAR-100, Split-Tiny-ImageNet} × {3-layer MLP, 5-layer MLP,
small CNN, medium CNN} (H6-C CONFIRMED, G4-septimo, Welch
p = 0.9247, g = −0.0246). DR-4's prediction "richer ops yield richer
consolidation" is recorded as empirically refuted across that
ladder (`docs/proofs/dr4-profile-inclusion.md`, DR-4 evidence v0.6).
Lemma DR-4.L itself is not refuted.

Every ladder stream is **single-domain** class-incremental, and the
scored outcome is retention of earlier tasks. The ladder RECOMBINE
strategies (`mog`, `ae`) sample per-class synthetic latents; they
never mix classes from different domains.

Zahn, Evans & Eagleman (2026), *Discovery by Dreaming: Cross-Domain
Recombination in Artificial Memory* (arXiv:2607.16256), report that
recombinatory replay **across** domains helps both a LoRA pipeline
and a symbolic engine, while **within-domain** rehearsal does not.

G4-nono asks a narrower, operational question: on a two-domain
CIFAR-100-coarse stream, does mixing buffer latents **across** the
two domains improve the network's ability to **discriminate a class
of one domain from a class of the other** (CDPA, §6), beyond no
recombination and beyond matched within-domain mixing?

**Scope of the claim.** CDPA is a two-class discrimination score
computed on one corpus. It does **not** measure cross-domain
knowledge transfer, analogy or "discovery" in the sense of Zahn et
al. A confirmed H12 supports the statement "cross-domain latent
mixing improves cross-domain class discrimination on this stream",
and only that statement can be used to re-scope the DR-4 evidence
record. No transfer endpoint is pre-registered here.

The LLM-tier variant (LoRA math → code) is out of scope: in the CUDA
E1 harness (`scripts/pilot_cycle3_cuda.py`, `_apply_dream_ops`)
`recombine` is a weight-neutral no-op.

## §2 Hypotheses (confirmatory)

Strategies: `cross` (cross-domain latent mixing), `within` (matched
within-domain mixing), `none` (RECOMBINE skipped — the ladder's
H4-C / H5-C / H6-A placebo). All confirmatory contrasts are in the
`P_max` arm.

**Pairing.** Within a seed, every cell uses the same initial weights
(`G4SmallCNN(seed)`), the same per-task data order
(`np.random.default_rng(seed)` inside `train_task`), the same buffer
fill draws (`default_rng(seed + 5000)`) and the same dream-episode
seeds. `cross`, `within` and `none` are therefore **matched**, not
independently randomised, and are bit-identical up to the first
dream episode where they differ (§5). The unit of analysis is the
seed; every contrast is analysed on the per-seed difference
`D_s = Y_s(strategy 1) − Y_s(strategy 2)`.

**Family.** Three hypotheses, Bonferroni α = 0.05 / 3 = **1/60
≈ 0.0167** per hypothesis. For each contrast let `L` and `U` be the
one-sided (1 − α) lower and upper confidence bounds for the mean of
`D` (paired t, df = n − 1), i.e. the limits of the two-sided
(1 − 2α) = 96.67 % interval. Effect sizes:

- **g_av** (primary effect size, comparable to the ladder's
  between-group g): `mean(D) / sqrt((s₁² + s₂²) / 2) × J`, with
  `J = 1 − 3 / (4(n − 1) − 1)` (Lakens 2013 average-SD form).
- **g_z** (reported): `mean(D) / sd(D) × J`.

**Margin.** ε = **0.01** (one percentage point of pairwise accuracy,
raw units, fixed here). Below ε a change in CDPA or WDPA is declared
practically negligible for the DR-4 record. It is a substantive
choice, not derived from data; power against it is tabulated in §3.

- **H12-A (cross-domain gain)** — `CDPA(cross) − CDPA(none) > 0`.
  Paired one-sided t-test (`scipy.stats.ttest_rel(...,
  alternative="greater")`) at α = 1/60.
- **H12-B (cross-domain specificity)** —
  `CDPA(cross) − CDPA(within) > 0`. Same test.
- **H12-C (within-domain null replicates, bounded)** —
  `|WDPA(within) − WDPA(none)| < ε`. Paired TOST at α = 1/60 per
  side: equivalence iff `L > −ε` and `U < ε`.

**Per-hypothesis states** (evaluated in the order listed; the first
that applies is the state; the list is exhaustive):

For H12-A and H12-B:

| State | Rule |
|---|---|
| CONFIRMED | `L > 0` and `g_av ≥ 0.5` (the G4-octavo H7-A / G6 H9-A floor) |
| DETECTED-BELOW-FLOOR | `L > 0` and `g_av < 0.5` |
| BOUNDED-NO-GAIN | `U < ε` (no gain of ε or more; includes negative estimates) |
| INCONCLUSIVE | none of the above |

If `U < 0`, the state is BOUNDED-NO-GAIN and the negative direction
is reported as an **exploratory** harm signal. No harm conclusion is
confirmatory in this family.

For H12-C:

| State | Rule |
|---|---|
| EQUIVALENT | `L > −ε` and `U < ε` |
| DIFFERENT | `L > 0` or `U < 0` (and not EQUIVALENT) |
| INCONCLUSIVE | none of the above |

A non-significant test never establishes a null in this document.
Only BOUNDED-NO-GAIN and EQUIVALENT, which rest on confidence bounds
against ε, support "no effect of practical size".

Secondary contrasts (reported, outside the family, same state rules
at nominal α = 0.05): `CDPA(P_max, cross)` vs `CDPA(P_equ, cross)`,
`CDPA(P_max, cross)` vs `CDPA(P_min)`, and the same three contrasts
on retention and final 20-way accuracy.

## §3 Power analysis

All numbers are reproduced by `scripts/prereg_power_2026_10.py`
(exact noncentral t for test-only MDEs; Monte Carlo, 20 000 or
200 000 draws, for complete rules).

**Test-only MDE at 80 % power, one-sided α = 1/60.** The paired
test needs `d_z = 0.564` (g_z ≈ 0.550) at n = 30 and `d_z = 0.308`
at n = 95. Converted to the g_av scale with
`d_av = d_z × sqrt(2(1 − r))`, where r is the within-seed
correlation between the two strategies' outcomes:

| n | r = 0 | r = 0.3 | r = 0.5 | r = 0.7 | r = 0.9 |
|---|---|---|---|---|---|
| 30 | 0.798 | 0.668 | 0.564 | 0.437 | 0.252 |
| 95 | 0.436 | 0.365 | 0.308 | 0.239 | 0.138 |

For reference, an independent-groups design at n = 30 per group would
need d = 0.782 (g ≈ 0.772); at n = 95, d = 0.434.

**Complete confirmation rule** (paired t rejects **and**
`g_av ≥ 0.5`), power by true d_av and r:

| n = 30 | d_av = 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|
| r = 0 | 0.39 | 0.54 | 0.80 | 0.94 |
| r = 0.3 | 0.46 | 0.64 | 0.89 | 0.98 |
| r = 0.5 | 0.48 | 0.68 | 0.94 | 0.99 |
| r = 0.7 | 0.48 | 0.72 | 0.97 | 1.00 |
| r = 0.9 | 0.47 | 0.81 | 1.00 | 1.00 |

| n = 95 | d_av = 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|
| r = 0 | 0.49 | 0.74 | 0.98 | 1.00 |
| r = 0.5 | 0.49 | 0.82 | 1.00 | 1.00 |
| r = 0.9 | 0.48 | 0.95 | 1.00 | 1.00 |

A true effect exactly at the floor (d_av = 0.5) is confirmed at most
about half the time at any n; this is a property of an observed-
effect floor, stated here so it is not read as a design flaw later.

**H12-C TOST** (true difference 0, ε = 0.01), power by SD of the
paired WDPA difference:

| n | SD_D = 0.005 | 0.0075 | 0.01 | 0.015 | 0.02 |
|---|---|---|---|---|---|
| 30 | 1.00 | 1.00 | 1.00 | 0.83 | 0.39 |
| 95 | 1.00 | 1.00 | 1.00 | 1.00 | 0.99 |

**r and SD_D are unknown.** No ladder milestone stores per-seed
paired differences for CDPA or WDPA (these metrics are new). The
within-seed correlation is plausibly high (shared initialisation,
data order and the first five tasks), but this is an assumption.
It is recorded as PI open decision §10.1 together with N. The
pre-filing smoke (§9, seeds 900-902, outside the confirmatory list)
records r and SD_D; those three seeds are not used for inference.

## §4 Exclusions, replacement and retained-N inference

- **Learning manipulation check (seed level).** Right after task 1,
  before any dream episode, accuracy on the 1000 task-1 test images
  is scored **restricted to the two task-1 logits** (2-way). The seed
  is excluded from every contrast if this accuracy is below **0.60**.
  Rationale: after task 1 the other 18 output units are untrained,
  so the meaningful chance level is the 2-way 0.50, not the 20-way
  0.05 inherited from the ladder's "2 × chance" rule. With 1000 test
  images the binomial SD at chance is ≈ 0.016, so 0.60 is ≈ 6 SD
  above chance; it checks that learning happened, nothing more. This
  check precedes every treatment difference, so it is identical for
  all eight cells of a seed and cannot bias a contrast.
- **Cell-level exclusions.** A cell is excluded if CDPA, WDPA or
  final accuracy is non-finite, or if a BLOCKING invariant (I1, I2,
  S1, S2, S3, K1, K4) is violated
  (`docs/osf-preregistration-draft.md` §4). A contrast uses the seeds
  for which **both** of its cells are retained (pairwise deletion).
- **No replacement.** The seed list is fixed (§5). Excluded seeds are
  not replaced. Inference uses the retained n; the achieved n and
  every excluded seed with its reason are reported.
- **Differential-exclusion guard.** If a confirmatory contrast loses
  more than 3 of its 30 seeds (more than 9 of 95) to **cell-level**
  exclusions, that hypothesis's state is still reported but is
  labelled exploratory: cell-level exclusions can depend on the
  treatment.
- **Abort.** If more than 50 % of seeds fail the manipulation check,
  or a run_id collides with a prior registry entry, the run is
  aborted and amended (raise epochs per G4-sexto §9.1). On abort,
  **all artifacts are preserved**: every completed cell JSON, the
  registry rows, driver logs and the partial aggregate are written to
  `docs/milestones/g4-nono-ABORTED-<date>.{json,md}`, labelled
  ABORTED and non-confirmatory, and the amendment cites that file.
  No confirmatory milestone is written for an aborted run.

## §5 Stream, substrate and driver (executable specification)

- **Seeds.** Confirmatory seeds `0..29` (N = 30) or `0..94` if the PI
  picks N = 95 (§10.1). Smoke seeds `900..902` (never analysed).
- **Domains** (fixed; CIFAR-100 coarse label id in brackets):
  - Domain A (animals): aquatic_mammals [0], fish [1], insects [7],
    large_carnivores [8], large_omnivores_and_herbivores [11],
    medium_mammals [12], non-insect_invertebrates [13], people [14],
    reptiles [15], small_mammals [16].
  - Domain B (plants, objects, scenes): flowers [2],
    food_containers [3], fruit_and_vegetables [4],
    household_electrical_devices [5], household_furniture [6],
    large_man-made_outdoor_things [9],
    large_natural_outdoor_scenes [10], trees [17], vehicles_1 [18],
    vehicles_2 [19].
- **Stream.** 10 tasks of 2 superclasses, consecutive in id order
  within each domain: tasks 1-5 = A {0,1}, {7,8}, {11,12},
  {13,14}, {15,16}; tasks 6-10 = B {2,3}, {4,5}, {6,9}, {10,17},
  {18,19}. Each task uses all 5000 training images (2 × 2500) and
  all 1000 test images (2 × 500) of its two superclasses. Labels are
  the coarse ids (0-19) on the 20-way head. No per-seed reshuffle of
  task composition.
- **Training per task.** `train_task(epochs = 3, batch_size = 64,
  lr = 0.01)`, plain SGD, cross-entropy, no augmentation, input
  scaling as in `experiments.g4_sexto_test.cifar100_dataset` (G4-sexto
  defaults). The last partial batch is kept.
- **Dream episodes (ladder schedule).** One DE after each of tasks
  1-9, i.e. before training tasks 2-10, with DE seed `seed + k` for
  the DE that precedes task k + 1 (k = 1..9). No DE after task 10.
  Operations per arm as in the G4-sexto driver: `P_min` = REPLAY +
  DOWNSCALE; `P_equ` and `P_max` = REPLAY + DOWNSCALE + RESTRUCTURE +
  RECOMBINE; `baseline` = no DE. Constants: replay HP combo C5
  (`representative_combo()`: downscale 0.95, replay batch 32, 5
  steps, lr 0.01), `RESTRUCTURE_FACTOR = 0.05`,
  `RECOMBINE_N_SYNTHETIC = 16`, `RECOMBINE_LR = 0.01`.
- **β buffer.** Capacity 320, never evicts in this stream. After each
  task, 32 training records are pushed, **16 per superclass**, drawn
  without replacement with `default_rng(seed + 5000)`. Each record
  stores the image and its 64-d post-ReLU `fc1` latent computed by
  the network **at push time**. Latents are **never refreshed**
  (ladder policy); domain-A latents mixed at later DEs were computed
  by an earlier network. This is part of what is tested.
- **RECOMBINE strategies** (extension of
  `experiments/g4_quater_test/recombine_strategies.py`; module
  `experiments/g4_nono_test/recombine_cross.py`, to be written and
  reviewed against the checklist below **before filing**). At a DE
  with recombine seed `seed + k + 30000`, one `default_rng` stream
  is used in this order:
  1. draw λ₁..λ₁₆ ~ Uniform(0.2, 0.8) (so `cross` and `within` use
     identical λ values within a seed);
  2. draw the 16 class pairs;
  3. draw one buffer record per class in each pair, uniformly among
     that class's records.
  - `cross`: pair j = (a, b) with a uniform over domain-A classes
    present in the buffer and b uniform over domain-B classes
    present. Synthetic latent `z = λ_j z_a + (1 − λ_j) z_b`, soft
    target `λ_j e_a + (1 − λ_j) e_b` (20-d). Active only once both
    domains are in the buffer, i.e. at the DEs before tasks 7-10
    (k = 6..9). At k = 1..5, `cross` is a no-op, identical to `none`.
  - `within`: pairs 1-8 are two **distinct** domain-A classes, pairs
    9-16 two distinct domain-B classes (drawn without replacement
    within the pair). Same λ, same target construction, same DEs
    (k = 6..9), so `cross` and `within` have identical sample counts
    and SGD steps.
  - `none`: RECOMBINE skipped.
  - Update for `cross` / `within`: one SGD step (lr 0.01) on the
    head `fc2` only, mean soft-target cross-entropy
    `−Σ_c t_c log softmax(fc2(z))_c` over the 16 samples.
- **Cells per seed.** `baseline` and `P_min` do not run RECOMBINE, so
  they are run once per seed. `P_equ` and `P_max` are run with each
  of the three strategies. 2 + 2 × 3 = **8 cells per seed**; 240
  cells at N = 30, 760 at N = 95.
- **Driver / aggregator.** `experiments/g4_nono_test/run_cross_domain.py`
  and `experiments/g4_nono_test/aggregator.py` (to be written before
  filing). The aggregator implements §2 exactly and calls
  `scipy.stats.ttest_rel(..., alternative="greater")` for H12-A/B.
- **Data pin.** CIFAR-100 archive and SHA-256 pins from
  `experiments.g4_sexto_test.cifar100_dataset`. That loader drops the
  coarse-label byte; the driver adds a coarse-label decode path from
  the same pinned files (no new download).

**Implementation fidelity checklist** (all items ticked in the filing
commit; each has a unit test under `tests/`):

1. `recombine_cross.py` reuses the `SyntheticLatentBatch` interface
   but returns soft targets; the ladder `mog` / `ae` code paths are
   untouched.
2. λ is drawn before classes and records (shared λ across `cross`
   and `within` for a seed) — test: same seed ⇒ same λ vector.
3. `within` never pairs a class with itself; `cross` never pairs two
   classes of one domain — test over 10⁴ draws.
4. `cross` and `within` are no-ops at k ≤ 5 — test: weights after the
   DE before task 6 are bit-identical to `none`.
5. Coarse-label decode matches the HF parquet `coarse_label` column
   on the pinned files — test on the first 1000 rows.
6. Buffer push is 16 + 16 per task, latents frozen at push — test.
7. CDPA / WDPA tie rule (§6) — test with constructed equal logits.
8. The aggregator reproduces a hand-computed paired example
   (L, U, g_av, g_z, state) — test.

## §6 Outcome metrics

All metrics are computed after task 10 training (end of stream).

- **CDPA (primary).** For every pair (a, b) with a ∈ A and b ∈ B
  (100 pairs): binary accuracy over the 1000 test images of a and b,
  predicting the class with the larger of the two logits {a, b}. An
  exact logit tie counts 0.5. CDPA is the unweighted mean over the
  100 pairs.
- **WDPA (H12-C).** Same computation over the 45 + 45 within-domain
  pairs; unweighted mean over the 90 pairs.
- **Retention (secondary, ladder-comparable).** Accuracy on domain-A
  test images after task 10, restricted to the 10 domain-A logits.
- **Final 20-way accuracy (secondary).**
- Effect sizes: g_av and g_z as defined in §2 (paired). The ladder's
  independent-cell Hedges' g is not used for confirmatory contrasts.

## §7 DualVer outcome rules

Rows are evaluated top to bottom; the **first matching row** applies.
The table is exhaustive: every combination of states reaches a row.

| # | Condition | Reading | EC | FC |
|---|---|---|---|---|
| 0 | Abort condition of §4 | Abort and amend; artifacts preserved as ABORTED. No reading. | n/a | n/a |
| 1 | H12-C = DIFFERENT | The ladder's within-domain null does not replicate on this stream. H12-A/B states are reported but read as exploratory until the discrepancy is explained. | stays PARTIAL | stays |
| 2 | H12-A = CONFIRMED ∧ H12-B = CONFIRMED ∧ H12-C = EQUIVALENT | **DR-4 refutation re-scoped (operationally).** Cross-domain latent mixing improves cross-domain class discrimination beyond no mixing and beyond within-domain mixing, while within-domain mixing changes WDPA by less than ε. DR-4 evidence gets a G4-nono addendum restating the refutation as "refuted for within-domain retention; not refuted for cross-domain discrimination on one CIFAR stream". | stays PARTIAL | stays |
| 3 | H12-A = CONFIRMED ∧ H12-B = CONFIRMED ∧ H12-C = INCONCLUSIVE | Cross-domain-specific CDPA gain, but the within-domain null is not bounded. Addendum records the gain; the re-scoping of row 2 is **not** made. | stays PARTIAL | stays |
| 4 | H12-A = CONFIRMED ∧ H12-B = BOUNDED-NO-GAIN | CDPA gain is generic synthetic-SGD: `cross` beats `none` but not `within` by ε or more. DR-4 refutation stands; no cross-domain specificity. | stays PARTIAL | stays |
| 5 | H12-A = CONFIRMED ∧ H12-B ∈ {DETECTED-BELOW-FLOOR, INCONCLUSIVE} | Gain over `none`; specificity unresolved. No re-scoping. | stays PARTIAL | stays |
| 6 | H12-A = BOUNDED-NO-GAIN | No CDPA gain of ε or more from cross-domain mixing. The DR-4 refutation is **extended** to this two-domain stream for the CDPA endpoint. | stays PARTIAL | stays |
| 7 | H12-A = DETECTED-BELOW-FLOOR | A positive CDPA difference below the g_av floor. Reported; no re-scoping, no extension. | stays PARTIAL | stays |
| 8 | H12-A = INCONCLUSIVE | Inconclusive at this N: neither a gain ≥ floor nor a bound below ε. DR-4 record unchanged; no extension is claimed. | stays PARTIAL | stays |

No row promotes EC to STABLE. A single two-domain CNN stream cannot
carry a universality claim in either direction.

## §8 Reporting commitment and stopping rule

- **Fixed N, no optional stopping.** All cells of all confirmatory
  seeds run. No hypothesis test is computed before the last cell is
  registered.
- Honest-reading clause: *"A paired test that does not reject is
  absence of evidence at this N. Absence of a practically relevant
  effect is claimed only through the ε-bounded states
  (BOUNDED-NO-GAIN, EQUIVALENT)."*
- All observed scalars are reported regardless of outcome, including
  `baseline`, `P_min`, `P_equ` cells, every secondary metric, the
  observed r and SD_D, and per-seed exclusions.

## §9 Audit trail and deviations

- Cells registered via `harness/storage/run_registry.py` with profile
  keys `g4-nono/<arm>/<strategy>` (`<strategy>` = `na` for `baseline`
  and `P_min`) and R1 bit-stable run_ids.
- Milestones: `docs/milestones/g4-nono-cross-domain-<date>.{json,md}`.
- **Pre-filing smoke:** seeds 900-902, all 8 cells, on the filing
  hardware. It verifies the driver, the checklist tests and per-cell
  wall time, and records r and SD_D. Its numbers are reported but
  never pooled with confirmatory seeds.
- Known deviation envelopes:
  a. CIFAR-100 SHA-256 mismatch: abort and amend with the new pin.
  b. Manipulation-check failure on most seeds: abort per §4, raise
     epochs per G4-sexto §9.1 in the amendment.
  c. Per-cell wall > 60 s sustained: escalate to the PI before
     running further cells. Do not reduce N silently.
- Any other deviation needs an amendment commit **before** the
  affected cells run, or a post-hoc disclosure in Paper 2 that marks
  the affected hypotheses exploratory (per
  `docs/osf-preregistration-draft.md` §5).

## §10 Open decisions for the PI (resolve before filing)

Everything above is the complete default protocol. The items below
are genuine choices; changing one requires editing the named section
before the lock commit.

1. **N and the r assumption (§3, §5).** N = 30 (default,
   ladder-comparable) or N = 95. If the smoke shows r < 0.5 or
   SD_D(WDPA) > 0.015, N = 95 is recommended: at N = 30 the complete
   rule then has < 0.70 power at d_av = 0.6 and the H12-C TOST
   < 0.83.
2. **Margin ε (§2).** 0.01 pairwise-accuracy points (default) or
   another fixed value chosen before any confirmatory data.
3. **Domain split (§5).** Animals / non-animals (default) or a
   stronger shift (second corpus; needs a new loader and pin).
4. **Hardware.** M1 Max or Studio. The protocol does not change; R1
   run_ids are per machine.
