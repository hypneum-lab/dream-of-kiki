# G4-nono cross-domain RECOMBINE pre-registration

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI).
**Date:** 2026-10-23 (revision 4: PI decisions of 2026-10-23 on N
and ε applied; ε reverted to 0.01 and N = 108 tier added; revision 5,
2026-10-24: pre-filing code status, no protocol change; revision 6,
2026-10-24: PI decision removes the N = 30 line of the N-selection
rule, so N is 95 or 108)
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
**Compute:** N = 95 seeds × 8 cells per seed = 760 cells, or
N = 108 (864 cells) under the pre-stated N-selection rule (§3).
**Power script:** `scripts/prereg_power_2026_10.py` (§3).
**Lock commit:** *(filled by introducing commit hash at filing)*
**Lock timestamp:** *(filled at filing, pre-driver-run)*

## §1 Background

The G4 escalation ladder (G4-quater → G4-septimo) confirmed the
"RECOMBINE-empty" finding (here and below, "confirmed" and "refuted"
are the locked decision labels; those non-rejections did not
establish equivalence) across {Split-FMNIST, Split-CIFAR-10,
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
Recombination in Artificial Memory* (arXiv:2607.16256, **version 2,
21 July 2026**; v1 was 28 June 2026), report that recombinatory
replay **across** domains helps both a LoRA pipeline and a symbolic
engine, while **within-domain** rehearsal does not. G4-nono is framed
as a **vision generality test** of that claim (class-incremental
Split-CIFAR-100-coarse, small CNN), not as a new hypothesis: the
cross- versus within-domain idea is published for LLM and symbolic
systems, the vision test is not.

**What v2 retains, and what it does not.** The v2 full text retracts
several numbers that the v1 abstract carried, and the abstract text
still shown on the arXiv listing page predates that revision. Only
the following v2 evidence is relied on here, as read in the v2 full
text (table "Provenance audit", §3.3-§3.5, §5.1, §5.3):
- Neural arm, Llama-3.1-8B, LoRA rank 256, matched conditions:
  +5.64 ± 2.31 pp (5 of 5 seeds positive, paired t p = 0.0055,
  bootstrap 95 % CI [+3.56, +6.81] pp). The authors state the gain is
  averaged over a task mix only partly cross-domain; on subtasks
  requiring cross-domain transfer (e.g. unseen GSM8K reasoning) the
  gain is reported as +14.5 pp.
- At rank 128 the neural arm is null across three base models, which
  the authors attribute to capacity under-saturation; the
  within-domain null they cite (−1.8 ± 4.4 pp, n = 3 × 150,
  p > 0.30) is a rank-128 matched-condition result. At 70B/72B scale
  the results are inconclusive within a ±5 pp floor.
- Symbolic arm: replay-surfaced bridges lie farther apart in
  embedding space than within-domain pairs (Spearman ρ = 0.54,
  n = 32, bootstrap 95 % CI [0.24, 0.77]) and survive a matched
  scramble control (5 of 5 true domain-pairs versus 0-1 of 5
  re-paired; Fisher exact p ≈ 0.05). The authors describe the
  external OpenAlex placement as consistent with, not proof of, the
  mechanism.
- Weights, not prompt: prepending cross-domain material in-context to
  a ~671B model gave −9.2 pp (5 seeds). This is a different control
  from this study's.
- **Retracted in v2 and not used here:** the LLM-judged symbolic
  accuracy headline (85.7 %, +21 pp over baseline; the authors state
  judged connection rates are ceilinged near 100 % in all
  conditions), the single-seed +26.8 % / +19.5 % neural gains, the
  57.5 → 74.4 % sequential figure, and the magnitude of the +7.2 %
  component-synergy figure.

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

**Domain-shortcut confound (Zhang et al., arXiv:2607.22994).**
Zhang, Fan, Liang et al., *Breaking the Synthetic-Real Domain
Shortcut for Training-Free Generative Replay-based Class Incremental
Learning*, show that mixing synthetic old-class data with real
new-class data lets a model rely on domain-discriminative features
instead of class cues. CDPA is an A-versus-B discrimination score,
so a `cross` gain could reflect a domain shortcut or head
recalibration (the `fc2` soft-target step is a logit-rebalancing step
between domains, which may counteract recency bias toward domain B)
rather than recombination. A confirmed H12-A does not exclude these
mechanisms; the domain-only probe of §6 is the check.

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

**Degenerate case.** If every retained D is identical (`sd(D) = 0`),
the t statistic is undefined; then `L = U = mean(D)` and the state
follows from these bounds. An undefined g_z or g_av (zero denominator) is
reported as undefined and does **not** meet the `g_av ≥ 0.5` floor.

**Margin.** ε = **0.01** (one percentage point of pairwise accuracy,
raw units; PI decision of 2026-10-23, which first halved it to 0.005
and then reverted it to 0.01 after seeing the power numbers of §3).
Below ε a change in CDPA or WDPA is declared
practically negligible for the DR-4 record. It is a substantive
choice, not derived from data; power against it is tabulated in §3.

- **H12-A (cross-domain gain)** — `CDPA(cross) − CDPA(none) > 0`.
  Paired one-sided t-test (`scipy.stats.ttest_rel(...,
  alternative="greater")`) at α = 1/60.
- **H12-B (cross-domain specificity)** —
  `CDPA(cross) − CDPA(within) > 0`. Same test.
- **H12-C (new within-domain equivalence check, bounded)** — tests
  WDPA under latent mixing, not the ladder's retention endpoint —
  `|WDPA(within) − WDPA(none)| < ε`. Paired TOST at α = 1/60 per
  side: equivalence iff `L > −ε` and `U < ε`.

**Per-hypothesis states** (evaluated in the order listed; the first
that applies is the state; the list is exhaustive):

For H12-A and H12-B:

| State | Rule |
|---|---|
| CONFIRMED | `L > 0`, observed `g_av ≥ 0.5` (the G4-octavo H7-A / G6 H9-A floor) **and** observed `mean(D) ≥ ε` |
| POSITIVE-NEGLIGIBLE | `L > 0` and `U < ε` (positive but bounded below the practical margin) |
| DETECTED-BELOW-FLOOR | `L > 0` and neither of the above |
| BOUNDED-NO-GAIN | `U < ε` (no gain of ε or more; includes negative estimates) |
| INCONCLUSIVE | none of the above |

**What CONFIRMED means.** `L > 0` establishes a positive population
difference at level α; the observed `g_av ≥ 0.5` and `mean(D) ≥ ε`
are magnitude filters on the **point estimate** (standardised and
raw), so a tiny but very precise difference cannot be CONFIRMED. CONFIRMED is therefore read as "a
positive effect whose estimate reaches the floor", never as "the
population effect is at least 0.5". The same applies to every
observed-floor rule in this document.

If `U < 0`, the state is BOUNDED-NO-GAIN and the negative direction
is reported as an **exploratory** harm signal. No harm conclusion is
confirmatory in this family.

For H12-C:

| State | Rule | Status |
|---|---|---|
| EQUIVALENT | `L > −ε` and `U < ε` | confirmatory |
| MEANINGFULLY-DIFFERENT | `L > ε` or `U < −ε` | exploratory |
| INCONCLUSIVE | none of the above (includes a difference from 0 whose interval still overlaps ±ε) | — |

**Error accounting.** Each hypothesis spends α once. For H12-A/B the
confirmatory conclusions are CONFIRMED (false only if the true
difference μ ≤ 0), BOUNDED-NO-GAIN (false only if μ ≥ ε) and
POSITIVE-NEGLIGIBLE (their conjunction). Because
ε > 0, at most one of them can be false for any μ, so the
probability of a false confirmatory conclusion is ≤ α per
hypothesis. For H12-C only EQUIVALENT is confirmatory (TOST, α);
MEANINGFULLY-DIFFERENT is exploratory and spends no α. The family
bound is therefore 3α = 0.05.

A non-significant test never establishes a null in this document.
Only BOUNDED-NO-GAIN and EQUIVALENT, which rest on confidence bounds
against ε, support "no effect of practical size". A difference from
zero that is smaller than ε does not count as non-replication.

Secondary contrasts (reported, outside the family, same state rules
at nominal α = 0.05): `CDPA(P_max, cross)` vs `CDPA(P_equ, cross)`,
`CDPA(P_max, cross)` vs `CDPA(P_min)`, `CDPA(P_max, within)` vs
`CDPA(P_max, none)` (needed before any "generic synthetic-SGD"
reading), and the same contrasts on retention and final 20-way
accuracy.

## §3 Power analysis

All numbers are reproduced by `scripts/prereg_power_2026_10.py`
(exact noncentral t for test-only MDEs; Monte Carlo, 20 000 or
200 000 draws, for complete rules).

**Test-only MDE at 80 % power, one-sided α = 1/60.** The paired
test needs `d_z = 0.308` (g_z ≈ 0.306) at n = 95 and `d_z = 0.289`
(g_z ≈ 0.287) at n = 108. Converted to the g_av scale with
`d_av = d_z × sqrt(2(1 − r))`, where r is the within-seed
correlation between the two strategies' outcomes:

| n (d_av MDE) | r = 0 | r = 0.3 | r = 0.5 | r = 0.7 | r = 0.9 |
|---|---|---|---|---|---|
| 95 | 0.436 | 0.365 | 0.308 | 0.239 | 0.138 |
| 108 | 0.408 | 0.342 | 0.289 | 0.224 | 0.129 |

For reference, an independent-groups design would need d = 0.434 at
n = 95 per group and d = 0.406 at n = 108.

**Complete confirmation rule** (paired t rejects **and**
`g_av ≥ 0.5`), power by true d_av and r:

| n = 95 | d_av = 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|
| r = 0 | 0.49 | 0.74 | 0.98 | 1.00 |
| r = 0.3 | 0.49 | 0.78 | 0.99 | 1.00 |
| r = 0.5 | 0.49 | 0.82 | 1.00 | 1.00 |
| r = 0.7 | 0.49 | 0.87 | 1.00 | 1.00 |
| r = 0.9 | 0.48 | 0.95 | 1.00 | 1.00 |

| n = 108 | d_av = 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|
| r = 0 | 0.49 | 0.76 | 0.98 | 1.00 |
| r = 0.3 | 0.50 | 0.79 | 0.99 | 1.00 |
| r = 0.5 | 0.49 | 0.83 | 1.00 | 1.00 |
| r = 0.7 | 0.49 | 0.89 | 1.00 | 1.00 |
| r = 0.9 | 0.49 | 0.97 | 1.00 | 1.00 |

A true effect exactly at the floor (d_av = 0.5) is confirmed at most
about half the time at any n; this is a property of an observed-
effect floor, stated here so it is not read as a design flaw later.

The raw floor `mean(D) ≥ ε` (ε = 0.01) only bites when arm SDs are
small. With r = 0.5, n = 95 and power shown by true d_av:

| arm SD | d_av = 0.5 | 0.6 | 0.8 | 1.0 |
|---|---|---|---|---|
| 0.01 | 0.00 | 0.00 | 0.03 | 0.50 |
| 0.02 | 0.45 | 0.80 | 1.00 | 1.00 |
| 0.04 | 0.49 | 0.82 | 1.00 | 1.00 |

When the seed SD of CDPA is ≈ 0.01, a standardised effect of 0.5-0.8
is only 0.005-0.008 accuracy points, i.e. below ε, and is by design
not confirmable. From an arm SD of 0.02 upwards the raw floor barely
changes the complete-rule power.

**"No practical effect" rules at ε = 0.01** (true difference 0):
H12-C EQUIVALENT (paired TOST, Monte Carlo) and H12-A/B
BOUNDED-NO-GAIN (`U < ε`, exact noncentral t), power by SD of the
paired difference SD_D:

| rule, N | SD_D = 0.0075 | 0.01 | 0.015 | 0.02 | 0.025 | 0.03 | 0.04 |
|---|---|---|---|---|---|---|---|
| EQUIVALENT, 95 | 1.00 | 1.00 | 1.00 | 0.99 | 0.92 | 0.72 | 0.22 |
| EQUIVALENT, 108 | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 0.81 | 0.34 |
| BOUNDED-NO-GAIN, 95 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.86 | 0.61 |
| BOUNDED-NO-GAIN, 108 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.90 | 0.67 |

Largest SD_D with 80 % power (EQUIVALENT / BOUNDED-NO-GAIN): 0.0282 /
0.0324 at N = 95, 0.0301 / 0.0346 at N = 108. Seeds needed for 80 %: 29 / 23 at SD_D = 0.015, 49 / 38 at
0.02, 75 / 58 at 0.025, 108 / 82 at 0.03, 189 / 144 at 0.04.

**N-selection rule (PI decisions of 2026-10-23 and 2026-10-24).** The
pre-filing smoke (§9, seeds 900-902, outside the confirmatory list,
never used for inference) computes from its three seeds SD̂_D, the
sd (ddof = 1) of the three per-seed differences
`WDPA(within) − WDPA(none)`. N is then fixed by the first matching
line:

1. SD̂_D ≥ 0.015 → **N = 108** (seeds 0..107);
2. otherwise → **N = 95** (seeds 0..94).

The rule is applied once, before any confirmatory cell runs; its
inputs and outcome are written into the filed version, and N is
fixed from then on. With three seeds SD̂_D is very imprecise: the
rule is a pre-stated deterministic trigger, not an estimate of the
population SD_D. No ladder milestone stores per-seed paired CDPA or
WDPA differences (the metrics are new), so there is no prior
estimate; the planning value for SD_D is the threshold, 0.015. The
smoke also reports r̂, the Pearson correlation across the smoke seeds
between the two cells of each confirmatory contrast (CDPA
`cross`/`none`, CDPA `cross`/`within`, WDPA `within`/`none`; an
undefined r̂ is reported as such). r̂ is descriptive only: it no
longer selects N.

**Why the former N = 30 line was removed (PI decision,
2026-10-24).** The revision-4 rule had a third line (N = 30 when
SD̂_D < 0.015 and every r̂ ≥ 0.5). A three-seed sd often falls below
the threshold when the true SD_D is above it: with normal D,
2 SD̂_D² / SD_D² follows χ²₂, so P(SD̂_D ≥ 0.015) = exp(−(0.015 /
SD_D)²), which is only 0.37 at SD_D = 0.015 and 0.57 at 0.02. When
the r̂ line did not trigger, the old rule therefore stayed at N = 30
too often, and its unconditional EQUIVALENT power was 0.89 / 0.74 /
0.69 / 0.63 at SD_D = 0.015 / 0.02 / 0.025 / 0.03.

**Unconditional power of the registered rule.** Method: the smoke
seeds are disjoint from the confirmatory seeds, so the rule's power
at a true SD_D is P(N = 95) × power(95) + P(N = 108) × power(108),
with P(N = 108) = exp(−(0.015 / SD_D)²) from the χ²₂ law above and
power(n) from the "no practical effect" table (TOST Monte Carlo for
EQUIVALENT, exact noncentral t for BOUNDED-NO-GAIN). True difference
0, normal D, ε = 0.01, α = 1/60 (`scripts/prereg_power_2026_10.py`,
section 5b):

| SD_D | 0.0075 | 0.01 | 0.015 | 0.02 | 0.025 | 0.03 | 0.04 |
|---|---|---|---|---|---|---|---|
| P(N = 108) | 0.02 | 0.11 | 0.37 | 0.57 | 0.70 | 0.78 | 0.87 |
| EQUIVALENT | 1.00 | 1.00 | 1.00 | 0.995 | 0.94 | 0.79 | 0.33 |
| BOUNDED-NO-GAIN | 1.00 | 1.00 | 1.00 | 0.998 | 0.97 | 0.89 | 0.66 |

**What the rule can and cannot establish (stated plainly).** With
the 95-seed floor, both "no practical effect" rules keep at least
80 % power for SD_D up to 0.025 (0.94 / 0.97), whichever line the
smoke selects; at 0.03 they reach 0.79 / 0.89. At larger SD_D
the design is under-powered: at 0.04 EQUIVALENT power is 0.33 and
189 seeds would be needed (144 for BOUNDED-NO-GAIN). The complete
H12-A/B confirmation rule has about 0.5 power at a true d_av exactly
at the 0.5 floor at either N (table above); this is a property of the
observed-effect floor, not of N.

Under-power makes INCONCLUSIVE more likely; it does not change the
error rates of §2.

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
  The estimand is therefore the mean paired difference over seeds
  whose two cells both complete, and the error rates of §2 are
  conditional on that set; the guard below limits, but does not
  remove, selection by treatment-dependent failures.
  A cell whose run raises (for example a BLOCKING invariant raised by
  the dream runtime) writes no outcome record; the aggregator counts
  the missing record as a cell-level exclusion (reason
  `missing_record`) and the driver log is preserved.
- **No replacement.** The seed list is fixed (§5). Excluded seeds are
  not replaced. Inference uses the retained n; the achieved n and
  every excluded seed with its reason are reported.
- **Differential-exclusion guard.** If a confirmatory contrast loses
  more than 9 of its 95 seeds (more than 10 of 108) to
  **cell-level**
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

- **Seeds.** Confirmatory seeds `0..N − 1`, with N ∈ {95, 108}
  fixed by the §3 N-selection rule. Smoke seeds `900..902` (never
  analysed).
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
  `experiments/g4_nono_test/recombine_cross.py`, written and
  reviewed against the checklist below in PR #44). At a DE
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
  of the three strategies. 2 + 2 × 3 = **8 cells per seed**; 760
  cells at N = 95, 864 at N = 108.
- **Driver / aggregator.** `experiments/g4_nono_test/run_cross_domain.py`
  (one cell per call, PR #44) and `experiments/g4_nono_test/aggregator.py`
  (PR #41). The aggregator implements §2, §4 and §7 exactly; its
  bounds come from the paired helper
  `kiki_oniric.eval.statistics.paired_tost` (paired SE, df = n − 1),
  whose one-sided bound `L > 0` is the same decision as
  `scipy.stats.ttest_rel(..., alternative="greater")` at α. Its
  `--select-n` mode applies the §3 N-selection rule to the smoke
  records.
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

Tests per item (status 2026-10-24): 1 `test_ladder_module_untouched`,
`test_shapes_and_soft_targets`; 2
`test_same_seed_same_lambda_across_strategies`; 3
`test_distinct_class_and_domain_constraints_1e4_draws`; 4
`test_cross_and_within_noop_before_task6`; 5
`test_pinned_bin_coarse_matches_parquet_first_1000` (skipped unless
both pinned files are present, so it must be run on the filing host;
see §9); 6 `test_buffer_push_16_plus_16_latents_frozen`; 7
`test_tie_on_a_single_pair_counts_half_per_image`,
`test_all_equal_logits_give_exactly_half`; 8
`test_paired_summary_hand_example`
(`tests/unit/test_g4_nono_{recombine_cross,driver,loader_coarse,metrics,aggregator}.py`).

## §6 Outcome metrics

All metrics are computed after task 10 training (end of stream).

- **CDPA (primary).** For every pair (a, b) with a ∈ A and b ∈ B
  (100 pairs): binary accuracy over the 1000 test images of a and b,
  predicting the class with the larger of the two logits {a, b}. An
  exact logit tie counts 0.5. CDPA is the unweighted mean over the
  100 pairs.
- **WDPA (H12-C).** Same computation over the 45 + 45 within-domain
  pairs; unweighted mean over the 90 pairs.
- **Domain-only probe, DOA (secondary, exploratory;
  `experiments.g4_nono_test.metrics.compute_doa`, cell field `doa`).**
  For each of the 20000 test images of the 20 classes (1000 per class), predict
  the *domain* by comparing the maximum logit over the 10 domain-A
  classes with the maximum over the 10 domain-B classes (tie counts
  0.5); DOA is the accuracy against the true domain. It uses no
  within-pair class information. The paired contrasts `cross − none`
  and `cross − within` on DOA are reported with the same state rules
  at nominal α = 0.05, outside the confirmatory family, spending no
  α. Reading rule (descriptive): if the `cross` gain in DOA is of
  the same size as the gain in CDPA, the CDPA gain cannot be
  separated from a domain-shortcut or head-recalibration mechanism
  (Zhang et al., arXiv:2607.22994) and H12-A / H12-B are reported with
  that qualification. Nothing in §2 or §7 changes.
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
| 1 | H12-C = MEANINGFULLY-DIFFERENT | Exploratory signal that within-domain mixing changes WDPA by more than ε on this stream (the ladder's within-domain null may not carry over). H12-A/B states are reported but read as exploratory until this is explained. | stays PARTIAL | stays |
| 2 | H12-A = CONFIRMED ∧ H12-B = CONFIRMED ∧ H12-C = EQUIVALENT | **DR-4 refutation re-scoped (operationally).** Cross-domain latent mixing gives a positive CDPA difference over no mixing and over within-domain mixing (point estimates at or above the floor), while within-domain mixing changes WDPA by less than ε. DR-4 evidence gets a G4-nono addendum restating the refutation as "refuted for within-domain retention; not refuted for cross-domain discrimination on one CIFAR stream". | stays PARTIAL | stays |
| 3 | H12-A = CONFIRMED ∧ H12-B = CONFIRMED ∧ H12-C = INCONCLUSIVE | Cross-domain-specific CDPA gain, but the within-domain null is not bounded. Addendum records the gain; the re-scoping of row 2 is **not** made. | stays PARTIAL | stays |
| 4 | H12-A = CONFIRMED ∧ H12-B ∈ {BOUNDED-NO-GAIN, POSITIVE-NEGLIGIBLE} | Gain over `none`; the cross-domain-**specific** advantage over `within` is bounded below ε. No cross-domain specificity is claimed; the DR-4 refutation stands for this endpoint. Whether the gain is generic synthetic SGD is read only from the secondary `within` vs `none` CDPA contrast, not concluded here. | stays PARTIAL | stays |
| 5 | H12-A = CONFIRMED ∧ H12-B ∈ {DETECTED-BELOW-FLOOR, INCONCLUSIVE} | Gain over `none`; specificity unresolved. No re-scoping. | stays PARTIAL | stays |
| 6 | H12-A ∈ {BOUNDED-NO-GAIN, POSITIVE-NEGLIGIBLE} | No CDPA gain of ε or more from cross-domain mixing. The DR-4 refutation is **extended** to this two-domain stream for the CDPA endpoint. | stays PARTIAL | stays |
| 7 | H12-A = DETECTED-BELOW-FLOOR | A positive CDPA difference whose estimate misses a floor and is not bounded below ε. Reported; no re-scoping, no extension. | stays PARTIAL | stays |
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
- **Filing gate.** This draft is not filed until (a) the driver,
  `recombine_cross.py` and the aggregator exist at a pinned commit,
  (b) every item of the §5 fidelity checklist has a passing test,
  (c) the aggregator has been run on synthetic paired data covering
  every §2 state and every §7 row, and (d) the smoke results below
  (wall time, SD̂_D, r̂ (descriptive), manipulation-check pass
  rate) and the N fixed by the §3 N-selection rule are written into
  the filed version.
  The lock commit is the commit that contains all four.
  **Status 2026-10-24:** (a) done: driver and `recombine_cross.py`
  (PR #44, three review rounds, APPROVE) and the aggregator (this PR),
  to be pinned by the lock commit; (b) done for items 1-4 and 6-8 on
  synthetic data; item 5 needs the pinned CIFAR-100 binary and parquet
  files on the filing host; (c) done:
  `tests/unit/test_g4_nono_aggregator.py` reaches every §2 state and
  every §7 row end to end on synthetic paired records; (d) open: the
  smoke has not run.
- **Pre-filing smoke:** seeds 900-902, all 8 cells, on the filing
  hardware. It verifies the driver, the checklist tests and per-cell
  wall time, and records SD̂_D for the §3 N-selection rule (and r̂,
descriptive). Its
  numbers are reported but never pooled with confirmatory seeds.
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

Decided by the PI on 2026-10-23 and no longer open: ε = 0.01; N
by the §3 N-selection rule (95 or 108; the N = 30 line was removed
by the PI on 2026-10-24).

**Within-domain equivalence margin (check, no change).** The
pre-registered within-domain margin already exists: H12-C is a
paired TOST on WDPA with ε = 0.01 (§2), consistent with the G4
decision; a non-significant within-domain result is never read as
"no effect" (§2, §8). No new margin is introduced. The
domain × recombination interaction is read from H12-B (`cross` vs
`within`), and retention versus cross-domain discrimination are
reported as separate endpoints (§6).

1. **Domain split (§5).** Animals / non-animals (default) or a
   stronger shift (second corpus; needs a new loader and pin).
2. **Un-mixed control arm (not registered).** An exploratory arm that
   replays the same domain-A and domain-B latents with hard labels
   and the same number of `fc2` steps would separate recombination
   from logit rebalancing (a design suggestion from the literature
   check, not a result in the cited paper). It adds cells and compute
   beyond §3, so it is left to the PI.
3. **Hardware.** M1 Max or Studio. The protocol does not change; R1
   run_ids are per machine.
