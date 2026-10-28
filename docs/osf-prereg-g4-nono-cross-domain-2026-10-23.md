# G4-nono cross-domain RECOMBINE pre-registration

**Split:** 2026-10-28, moved unchanged from branch `docs/osf-prereg-2026-10` (PR #41) so it can be filed without waiting for E1.
**Status:** DRAFT — not yet filed on OSF (to be filed by the PI).
**Date:** 2026-10-23 (revision 4: PI decisions of 2026-10-23 on N
and ε applied; ε reverted to 0.01 and N = 108 tier added; revision 5,
2026-10-24: pre-filing code status, no protocol change; revision 6,
2026-10-24: PI decision removes the N = 30 line of the N-selection
rule, so N is 95 or 108; revision 7, 2026-10-24: PI amendment after
the first smoke — epochs 3 → 10, acc20 and DOA descriptive only, and
P_max runs recombine_full so that P_equ and P_max differ on the CNN;
revision 9, 2026-10-25: third-smoke N selection re-checked with the
current aggregator, no protocol change)
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
**Benchmark:** two-domain stream, CIFAR-10 (grayscale) then Quick,
Draw! (§5; PI decision 2026-10-24).
**Design:** matched-block (seed = block), paired analysis (§2, §6).
**Compute:** N = 95 seeds × 9 cells per seed (8 confirmatory + 1
exploratory un-mixed control) = 855 cells, or N = 108 (972 cells)
under the pre-stated N-selection rule (§3); 760 / 864 confirmatory
cells. Per-cell time at 10 epochs on macM1 under N10-C load: about
41 s on the new corpus (≈ 10 h for 855 cells; third smoke, §9).
**Domain corpus:** CIFAR-10 (domain A, grayscale) + Quick, Draw!
(domain B), pinned (§5).
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
reading), and the same contrasts on retention. Final 20-way
accuracy (acc20) and the domain-only probe (DOA) are **descriptive
only** (PI decision 2026-10-24): mean, SD and a two-sided 95 % paired
interval, no state (see §6 for why).

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

**"No practical effect" rules at ε = 0.01** (true difference 0; all
powers in this section assume the full planned N is retained):
H12-C EQUIVALENT (paired TOST, Monte Carlo) and, for H12-A/B, two
events. `U < ε` (exact noncentral t) is the union of
BOUNDED-NO-GAIN and POSITIVE-NEGLIGIBLE under the first-match order
of §2; the exact BOUNDED-NO-GAIN state is `L ≤ 0 ∧ U < ε` (Monte
Carlo), which at a true difference of 0 is capped near 1 − α = 0.983
because L > 0 then sends the outcome to POSITIVE-NEGLIGIBLE. Power by
SD of the paired difference SD_D:

| rule, N | SD_D = 0.0075 | 0.01 | 0.015 | 0.02 | 0.025 | 0.03 | 0.04 |
|---|---|---|---|---|---|---|---|
| EQUIVALENT, 95 | 1.00 | 1.00 | 1.00 | 0.99 | 0.92 | 0.72 | 0.22 |
| EQUIVALENT, 108 | 1.00 | 1.00 | 1.00 | 1.00 | 0.95 | 0.81 | 0.34 |
| U < ε (BNG ∪ PN), 95 | 1.00 | 1.00 | 1.00 | 1.00 | 0.96 | 0.86 | 0.61 |
| U < ε (BNG ∪ PN), 108 | 1.00 | 1.00 | 1.00 | 1.00 | 0.98 | 0.90 | 0.67 |
| BOUNDED-NO-GAIN (exact), 95 | 0.98 | 0.98 | 0.98 | 0.98 | 0.96 | 0.86 | 0.61 |
| BOUNDED-NO-GAIN (exact), 108 | 0.98 | 0.98 | 0.98 | 0.98 | 0.97 | 0.90 | 0.67 |

Largest SD_D with 80 % power (EQUIVALENT / `U < ε`): 0.0282 /
0.0324 at N = 95, 0.0301 / 0.0346 at N = 108. Seeds needed for 80 %:
29 / 23 at SD_D = 0.015, 49 / 38 at 0.02, 75 / 58 at 0.025, 108 / 82
at 0.03, 189 / 144 at 0.04.

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
fixed from then on. **Smoke validity.** N is selected only from a
complete smoke: all 24 cells (8 per seed) present, with status `ok`
and finite CDPA / WDPA. The registered abort conditions of §4 apply
to the smoke: a run_id collision, or a manipulation-check failure on
2 or 3 of the 3 smoke seeds, refuses the selection (the cause is
investigated and the amendment path of §4 applies). A single seed
failing the manipulation check is dropped and SD̂_D is computed from
the other two (ddof = 1); this is reported. All 24 cells are
validated first, including those of a dropped seed. Any other missing or
unusable cell (no record, non-finite metric, excluded cell) also
refuses the selection: the cause is fixed and the full smoke is
rerun in a fresh output directory and registry, and both smoke runs
are reported; a driver change made for this is part of the lock
commit. With three seeds SD̂_D is very imprecise: the
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

**Unconditional power of the registered rule.** Method: the
selection is driven by σ_sel, the SD of `WDPA(within) − WDPA(none)`;
P(N = 108) = exp(−(0.015 / σ_sel)²) from the χ²₂ law above. The smoke
seeds are disjoint from the confirmatory seeds, so a rule's power is
P(N = 95) × power(95) + P(N = 108) × power(108), with power(n)
evaluated at the SD of that contrast, σ_con. True difference 0,
normal D, ε = 0.01, α = 1/60, full planned N retained
(`scripts/prereg_power_2026_10.py`, section 5b).

*H12-C.* Its contrast is the selector difference, so σ_con = σ_sel =
SD_D:

| SD_D | 0.0075 | 0.01 | 0.015 | 0.02 | 0.025 | 0.03 | 0.04 |
|---|---|---|---|---|---|---|---|
| P(N = 108) | 0.02 | 0.11 | 0.37 | 0.57 | 0.70 | 0.78 | 0.87 |
| EQUIVALENT | 1.00 | 1.00 | 1.00 | 0.995 | 0.94 | 0.79 | 0.33 |

*H12-A/B.* The CDPA contrast SD σ_con is a different quantity from
σ_sel, so power is given for separate scenarios (rows σ_sel, columns
σ_con), for the exact BOUNDED-NO-GAIN state:

| σ_sel \ σ_con | 0.01 | 0.015 | 0.02 | 0.025 | 0.03 | 0.04 |
|---|---|---|---|---|---|---|
| 0.0075 | 0.98 | 0.98 | 0.98 | 0.96 | 0.86 | 0.61 |
| 0.015 | 0.98 | 0.98 | 0.98 | 0.96 | 0.88 | 0.63 |
| 0.03 | 0.98 | 0.98 | 0.98 | 0.97 | 0.89 | 0.66 |

(`U < ε`, i.e. BOUNDED-NO-GAIN ∪ POSITIVE-NEGLIGIBLE, is the same
except 1.00 in place of 0.98 for σ_con ≤ 0.02.)

**What the rule can and cannot establish (stated plainly).** With
the 95-seed floor, H12-C EQUIVALENT keeps at least 80 % power for
SD_D up to 0.025 (0.94) and reaches 0.79 at 0.03; H12-A/B
BOUNDED-NO-GAIN keeps at least 80 % for a CDPA contrast SD up to
0.03 (0.86-0.89 depending on σ_sel). At larger SDs the design is
under-powered: at 0.04 EQUIVALENT power is 0.33 (189 seeds needed)
and BOUNDED-NO-GAIN 0.61-0.66 (144 needed). The complete H12-A/B
confirmation rule has about 0.5 power at a true d_av exactly at the
0.5 floor at either N (table above); this is a property of the
observed-effect floor, not of N. Retained n below the planned N
(exclusions, §4) lowers every power in this section.

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
  treatment. The §7 row is still computed from the raw states, but
  its reading is **exploratory** whenever a hypothesis that the
  row's first-match condition reads is guarded (row 1 reads H12-C;
  rows 2-5 read H12-A, H12-B and H12-C; rows 6-8 read H12-A and
  H12-C). An exploratory row is reported with its number and never
  supports a re-scoping (row 2), an addendum (rows 3-5) or an
  extension (row 6). The aggregator writes `row_confirmatory` and
  `row_guarded_by`.
- **Worst-case imputation sensitivity (PI decision, 2026-10-24;
  reported only).** For each H12 contrast, every cell lost at cell
  level among the retained seeds (excluded cell or missing record;
  seed-level manipulation exclusions are not imputed) is imputed
  against the treatment: the treatment cell (`P_max/cross` for
  H12-A/B, `P_max/within` for H12-C) takes the lowest value observed
  for that cell over the retained seeds, and the comparator cell the
  highest. The §2 states and the §7 row are recomputed on the imputed
  data and reported next to the primary ones. The observed-extrema
  imputation is a directional sensitivity analysis, not an absolute
  worst-case bound (the true lost values could lie outside the
  observed range, and for the H12-C equivalence claim the
  unfavourable direction is not unique). This analysis never
  changes a decided state, the row or its confirmatory status; a
  disagreement with the primary reading is reported in the
  milestone and in Paper 2.
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
- **Domain corpus (PI decisions 2026-10-24; revision 8).** The
  CIFAR-100 animals / non-animals split is replaced by two corpora with
  a much stronger shift: **domain A = CIFAR-10** (natural photographs)
  and **domain B = Quick, Draw!** (hand-drawn doodles). Module
  `experiments/g4_nono_test/qd_cifar10.py`, build script
  `scripts/g4_build_qd_cifar10.py`.
  - Domain A: the 10 CIFAR-10 classes, HF `uoft-cs/cifar10` at
    revision `0b2714987fa478483af9968de7c934580d0bb9a2`; parquet
    SHA-256 train `8428b53a…d2ae883`, test `841389e6…645ce84de` (the
    G4-quinto pins). CIFAR-10 states no licence; its page asks for a
    citation (Krizhevsky 2009), the same exposure as CIFAR-100 before.
  - Domain B: Quick, Draw! `numpy_bitmap` (28 × 28 grayscale, Google
    Creative Lab, **CC BY 4.0**; attribution: "Quick, Draw! dataset,
    Google, CC BY 4.0"), categories apple, banana, clock, cup, house,
    key, ladder, moon, tree, umbrella (semantically disjoint from
    CIFAR-10; no vehicle or animal category). No official SHA-256
    exists: our SHA-256 of each `.npy`, its byte size and the GCS
    md5 / crc32c are pinned in `QD_PINS` (downloaded 2026-10-24; the
    md5s match GCS).
  - **Colour shortcut neutralised.** CIFAR-10 images are converted to
    grayscale (ITU-R BT.601 luma, rounded) and replicated to 3
    channels; drawings are zero-padded 28 → 32 (2 px each side) and
    replicated to 3 channels. Every image of both domains therefore
    has three equal channels, so the domain cannot be read from
    colour alone (tested). Other low-level cues remain (binary
    strokes on a black background vs grey-level photographs); they
    are part of the stronger shift the PI chose. **Scope of the
    claim.** On this corpus G4-nono is an operational CDPA test: it
    can show whether cross-domain mixing changes cross-domain pairwise
    discrimination, but it cannot establish semantic integration
    across domains, nor exclude that a gain comes from recalibrating
    on domain cues.
  - **Subsampling.** Each class is cut to exactly **2 500 train / 500
    test** images with `default_rng([20261024, corpus, class])`
    (CIFAR-10: from its train and test splits; Quick, Draw!: disjoint
    draws from the full file). The index lists are committed
    (`experiments/g4_nono_test/qd_cifar10_indices.json`, SHA-256
    `f4092ff2…eb11ae7`); the built corpus `qd-cifar10-v1.npz`
    (60 000 images, SHA-256 `4f7fdf1c…bf381b`) is verified by the
    loader and rebuilds bit-identically. Raw `.npy` files are deleted
    after the build (about 1.2 GB downloaded once; 185 MB kept).
    Per-task image counts equal the CIFAR-100 design (5 000 train /
    1 000 test), so per-cell compute is unchanged.
  - **Label map.** The coarse-id space is kept so the stream, metrics
    and RECOMBINE code are unchanged: CIFAR-10 class k → id
    `DOMAIN_A[k]` (airplane 0, automobile 1, bird 7, cat 8, deer 11,
    dog 12, frog 13, horse 14, ship 15, truck 16); Quick, Draw! class
    k → `DOMAIN_B[k]` (apple 2, banana 3, clock 4, cup 5, house 6,
    key 9, ladder 10, moon 17, tree 18, umbrella 19).
- **Stream.** 10 tasks of 2 classes, consecutive in id order within
  each domain: tasks 1-5 = A {0,1}, {7,8}, {11,12}, {13,14},
  {15,16} (airplane/automobile, bird/cat, deer/dog, frog/horse,
  ship/truck); tasks 6-10 = B {2,3}, {4,5}, {6,9}, {10,17}, {18,19}
  (apple/banana, clock/cup, house/key, ladder/moon, tree/umbrella).
  Each task uses its 5 000 training and 1 000 test images. Labels are
  the ids above on the 20-way head. No per-seed reshuffle of task
  composition. The previous CIFAR-100 split remains selectable
  (`--corpus cifar100`) only to reproduce the first two smokes.
- **Training per task.** `train_task(epochs = 10, batch_size = 64,
  lr = 0.01)`, plain SGD, cross-entropy, no augmentation, input
  scaling as in `experiments.g4_sexto_test.cifar100_dataset` (G4-sexto
  defaults). The last partial batch is kept. **Amendment (PI
  decision 2026-10-24): epochs 3 → 10.** The first smoke (§9) hit the
  §4 manipulation-check abort at 3 epochs (2 of 3 seeds below 0.60).
  Short reruns showed genuine under-fitting of the first pair (train
  and test 2-way accuracy equal, loss near ln 2) and task-1 2-way
  accuracy 0.615 / 0.639 at 10 epochs on seeds 900 / 901 (0.609 /
  0.500 at 3 epochs, 0.601 / 0.613 at 8). Precedent: G4-sexto §9.1
  raised epochs (3 → 8) for the same substrate and loader. The change
  applies to every arm and strategy alike, so it cannot favour one
  side of any contrast. Driver default `EPOCHS = 10`.
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
  - **Profile variant (PI decision 2026-10-24).** Framework C §3.1
    gives `P_equ` the op recombine_light and `P_max` recombine_full;
    the op sets otherwise coincide. On the CNN path `P_equ` uses the
    interpolant above (light). `P_max` (full) draws, after the record
    draws of the same stream, ε ~ N(0, I) (16 × 64) and uses
    `max(0, z + σ ⊙ ε)`: a reparameterised Gaussian sample around the
    interpolant, with σ the per-dimension SD of all buffer latents
    (a moment-matched encoder) and the `fc1` ReLU as decoder, so the
    sample stays in the post-ReLU latent space. λ, pairs, records and
    soft targets are identical across the two variants. Before this
    change the two profiles differed only in runtime handlers acting
    on the dummy DE input slice, so every CNN-side update and every
    metric of `P_equ` and `P_max` was identical; the same holds for
    the G4-quinto results (retention 0.8754 for both), which are
    therefore an implementation artefact and are not edited. The
    primary H12 contrasts are within `P_max`, so their treatment cells
    now use recombine_full. `P_equ` and `P_max` still coincide in the
    `none` cells and at the DEs where `cross` / `within` are no-ops
    (k = 1..5), because RECOMBINE is the only op that differs.
    `baseline`, `P_min`, every `P_equ` cell and `P_max/none` are
    bit-identical to the code before the change (regression test on
    macM1; the golden hashes cover a 1-epoch synthetic fixture, not
    the default 10-epoch CIFAR outputs). The buffer-SD Gaussian
    sample is this study's documented CNN operationalisation of
    recombine_full; framework C §3.1 names the op but does not
    prescribe this form. The α input and channel 2 of the framework's `P_max` are
    not modelled on the CNN path.
  - Update for `cross` / `within`: one SGD step (lr 0.01) on the
    head `fc2` only, mean soft-target cross-entropy
    `−Σ_c t_c log softmax(fc2(z))_c` over the 16 samples.
- **Cells per seed.** `baseline` and `P_min` do not run RECOMBINE, so
  they are run once per seed. `P_equ` and `P_max` are run with each
  of the three strategies. 2 + 2 × 3 = **8 confirmatory cells per
  seed** (760 at N = 95, 864 at N = 108), plus the exploratory
  `P_max/unmixed` cell (below): **9 cells per seed**, 855 / 972.
- **Exploratory un-mixed control (PI decision 2026-10-24; reported
  only).** Cell `P_max/unmixed`: at each DE it makes exactly the
  `cross` draws (same stream, λ, pairs and records) but replays the
  two records of each pair **unmixed**, with hard one-hot labels (32
  latents, one `fc2` step at lr 0.01, same mean cross-entropy), and
  with P_max's recombine_full noise drawn for the 32 latents. It
  separates recombination from logit rebalancing. Contrasts
  `CDPA(P_max, cross) − CDPA(P_max, unmixed)` and
  `CDPA(P_max, unmixed) − CDPA(P_max, none)` are reported with the
  §2 state rules at nominal α = 0.05 in an `exploratory_arms` block.
  They are never confirmatory, are outside the family, the guard,
  the §7 row and the completeness check of the 8 registered cells,
  and never change a decided state. Exploratory records never enter
  the abort conditions, the seed-level exclusions or the smoke N
  selection; their own aborted records and manipulation flags are
  reported in the exploratory block.
- **Driver / aggregator.** `experiments/g4_nono_test/run_cross_domain.py`
  (one cell per call, PR #44) and `experiments/g4_nono_test/aggregator.py`
  (PR #41). The aggregator implements §2, §4 and §7 exactly; its
  bounds come from the paired helper
  `kiki_oniric.eval.statistics.paired_tost` (paired SE, df = n − 1),
  whose one-sided bound `L > 0` is the same decision as
  `scipy.stats.ttest_rel(..., alternative="greater")` at α. Its
  `--select-n` mode applies the §3 N-selection rule to the smoke
  records.
- **Data pin.** `qd-cifar10-v1.npz` SHA-256 `4f7fdf1c…bf381b`,
  built from the pinned CIFAR-10 parquet files and Quick, Draw!
  `.npy` files (§5 domain corpus); the driver's default is
  `--corpus qd-cifar10`.

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
5. The corpus file matches its pinned SHA-256 and holds 2 500 /
   500 images per class with three equal channels, and the index
   file matches its pin — `test_checklist_item5_real_corpus` (passes
   on macM1, where the corpus is built; skipped elsewhere). (The
   former item 5, CIFAR-100 coarse-label decode, applies only to the
   superseded split.)
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
- **Domain-only probe, DOA (descriptive only, PI decision
  2026-10-24;
  `experiments.g4_nono_test.metrics.compute_doa`, cell field `doa`).**
  For each of the 20000 test images of the 20 classes (1000 per class), predict
  the *domain* by comparing the maximum logit over the 10 domain-A
  classes with the maximum over the 10 domain-B classes (tie counts
  0.5); DOA is the accuracy against the true domain. It uses no
  within-pair class information. The paired contrasts `cross − none`
  and `cross − within` on DOA are reported descriptively (mean, SD,
  two-sided 95 % paired interval, no state). With one shared
  class-incremental head the network predicts almost only the last
  task's classes (domain B), so DOA is pinned near 0.5 by
  construction (exactly 0.5 in every cell of the first smoke). Reading rule (descriptive): if the `cross` gain in DOA is of
  the same size as the gain in CDPA, the CDPA gain cannot be
  separated from a domain-shortcut or head-recalibration mechanism
  (Zhang et al., arXiv:2607.22994) and H12-A / H12-B are reported with
  that qualification. Nothing in §2 or §7 changes.
- **Retention (secondary, ladder-comparable).** Accuracy on domain-A
  test images after task 10, restricted to the 10 domain-A logits.
- **Final 20-way accuracy, acc20 (descriptive only, PI decision
  2026-10-24).** Reported like DOA, no state. With the collapse onto
  the last task, acc20 is at most 0.1 × the last pair's 2-way accuracy
  (0.05 in every cell of the first smoke), so it cannot carry a
  contrast.
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
  synthetic data; item 5 (new corpus) done on macM1, where the pinned
  corpus is built (it must be rebuilt and pass on the filing host);
  (c) done:
  `tests/unit/test_g4_nono_aggregator.py` reaches every §2 state and
  every §7 row end to end on synthetic paired records; (d) three
  smokes ran on 2026-10-24 and all are reported (§3): the first, at
  3 epochs on CIFAR-100, hit the registered abort condition; the
  second, after the PI amendment (10 epochs, P_max recombine_full),
  passed and selected N = 95 on CIFAR-100; the third, on the new
  CIFAR-10 + Quick, Draw! corpus, passed and selected **N = 95**, the
  value to write into the filed version.
  **First smoke, 2026-10-24 (PI decision: pilot on macM1 now,
  campaign later).** Code at `28e18c5` (`DREAMOFKIKI_COMMIT_SHA`),
  macM1 (M1, 32 GB), CIFAR-100 from the pinned parquet mirror, one
  process at `nice 19`, run **concurrently with the N10-C
  campaign** (Metal near full load, swap 6.6-7.5 of 8 GB), so wall
  times may be inflated. 24 of 24 cells completed (`status: ok`, no
  cell-level exclusion, no non-finite metric); N10-C stayed alive
  throughout and the memory guard never triggered.
  - Wall time: driver-reported 7.7-16.3 s per cell (median 13.8 s;
    289 s for 24 cells); 20-31 s per cell including process start
    and data loading; 613 s for the whole smoke.
  - Manipulation check (2-way task-1 accuracy, threshold 0.60): seed
    900 0.609 (pass), 901 0.500 (fail), 902 0.526 (fail). **2 of 3
    smoke seeds fail, which is the §4 abort condition**; the
    aggregator refuses the N selection (`n: null`).
  - Descriptive only (not used, because the selection is refused):
    SD̂_D of `WDPA(within) − WDPA(none)` = 0.0016 over the three
    seeds (below 0.015, so the rule would have given 95); r̂ = 0.93
    (CDPA cross/none), 0.99 (CDPA cross/within), 0.98 (WDPA
    within/none).
  - Final 20-way accuracy (`acc20`) is 0.05 in every `baseline` and
    `P_max` cell, DOA exactly 0.5, and CDPA / WDPA sit at 0.50-0.52.
    **Diagnosis (2026-10-24, short CPU reruns on macM1): not a
    pipeline bug.** The 10 000 test images are balanced (500 per
    coarse class) and labelled with the coarse ids used by the
    20-way head. After the stream, the shared head predicts almost
    only the last task's classes (seed 901: all 10 000 predictions
    on class 18; seed 900: 9 978 on class 19), the usual
    class-incremental collapse onto the most recent task. acc20 is
    then at most 0.1 × the last task's 2-way accuracy, and that pair
    (18 vs 19) stays at 0.50-0.503 2-way accuracy even at 10 epochs,
    so acc20 = 500 / 10 000 = 0.05. DOA is exactly 0.5 because every
    prediction falls in domain B (the last domain). At 10 epochs
    acc20 is 0.0503: more epochs do not move it. acc20 and DOA are
    therefore near-degenerate on this stream by construction (they
    are secondary endpoints). The manipulation-check failures are
    genuine under-fitting of a hard first pair (aquatic mammals vs
    fish, small CNN, plain SGD): test and train 2-way accuracy are
    equal (0.50-0.63), the cross-entropy ends near ln 2, and seed
    901 predicts a single class after 3 epochs. 2-way task-1
    accuracy by epochs (seed 900 / 901): 3 epochs 0.609 / 0.500,
    8 epochs 0.601 / 0.613, 10 epochs 0.615 / 0.639; lr 0.05 at 3
    epochs 0.586 / 0.500.
  - Per the deviation envelope (b) below, this needs an amendment
    before any confirmatory cell runs (for example more epochs per
    G4-sexto §9.1); the choice is left to the PI. The smoke records
    are kept (macM1 `~/scratch/code/g4-pilot-out/`, with the runner
    log and the refused `n-selection.json`).
  - Checklist item 5 was not run: no host has both pinned CIFAR-100
    files (macM1 and Studio hold only the parquet mirror;
    electron-server holds neither).
  **Second smoke, 2026-10-24 (after the PI amendment).** Code at
  `c64bcee` (10 epochs, P_max recombine_full), same host, data,
  `nice 19` and guard, a fresh output directory and registry, again
  **concurrently with N10-C** (swap 7.0-7.3 of 8 GB), so wall times
  may be inflated. 24 of 24 cells `ok`, no exclusion, no non-finite
  metric; N10-C stayed alive and the guard never triggered.
  - Wall time: driver-reported 22-48 s per cell (median 25 s; 803 s
    for 24 cells); 40-61 s per cell including process start and data
    loading; 1 164 s for the whole smoke. Campaign estimate on macM1
    at this load: about 48 s per cell, so ≈ 11.4 h for 855 cells
    (N = 95, 9 cells per seed; ≈ 10.2 h for the 760 confirmatory
    cells), ≈ 13 h for 972 cells (N = 108); with the driver time
    alone (median 25 s) about half of that.
  - Manipulation check: 0.615 / 0.639 / 0.607 on seeds 900 / 901 /
    902; all pass (margins over 0.60 are thin).
  - **N selection: SD̂_D = 0.0022 < 0.015, so N = 95** (line 2).
    r̂ (descriptive): 0.999 (CDPA cross/none), 0.996 (CDPA
    cross/within), 0.992 (WDPA within/none).
  - **P_equ and P_max now differ** in the `cross` and `within` cells
    (for example CDPA 0.5465 vs 0.5426 at seed 900 `cross`, 0.5203 vs
    0.5246 at seed 901 `within`) and are bit-identical in the `none`
    cells, as designed (§5).
  - CDPA / WDPA are 0.51-0.55; acc20 0.050-0.061 and DOA 0.5 in every
    cell (descriptive only, §6).
  - Exploratory `P_max/unmixed` (run separately at `0b2b48e`, 3
    cells): CDPA 0.5474 / 0.5257 / 0.5269, driver wall 24-46 s.
  - Records: macM1 `~/scratch/code/g4-pilot2-out/` (with
    `n-selection.json`) and `~/scratch/code/g4-pilot2-unmixed/`.
  **Third smoke, 2026-10-24 (new corpus, PI decision; this is the
  smoke whose N is filed).** Code at `6f11823` (CIFAR-10 grayscale +
  Quick, Draw!, 10 epochs, P_max recombine_full), corpus
  `qd-cifar10-v1.npz` (SHA-256 `4f7fdf1c…`), same host, `nice 19` and
  guard, fresh output directory and registry, again **concurrently
  with N10-C** (swap free ≥ 938 MB), so wall times may be inflated.
  27 of 27 cells `ok` (24 confirmatory + 3 exploratory `unmixed`), no
  exclusion, no non-finite metric; N10-C stayed alive.
  - Wall time: driver-reported 23-48 s per cell (median 44 s; 957 s
    for 27 cells); 1 094 s for the whole smoke, about 41 s per cell
    with start-up. Campaign estimate at this load: ≈ 10 h for 855
    cells (N = 95), ≈ 11 h for 972 (N = 108).
  - Manipulation check: 0.771 / 0.603 / 0.609 on seeds 900 / 901 /
    902; all pass (901 and 902 only just).
  - **N selection: SD̂_D = 0.0015 < 0.015, so N = 95** (line 2).
    r̂ (descriptive): 1.000 (CDPA cross/none), 1.000 (CDPA
    cross/within), 0.729 (WDPA within/none).
  - CDPA rises to 0.62-0.82 (baseline 0.62-0.66, dream arms
    0.67-0.82), WDPA 0.60-0.64: the cross-domain pairs are far easier
    to separate on this corpus (low-level cues other than colour
    remain, §5). acc20 0.09-0.12 and DOA ≈ 0.5 except one baseline
    cell (0.64), descriptive only.
  - `P_equ` and `P_max` differ in `cross` / `within` and match in
    `none`, as designed. Exploratory `unmixed` CDPA 0.795 / 0.687 /
    0.810.
  - Checklist item 5 passes on this host for the new corpus.
  - Records: macM1 `~/scratch/code/g4-pilot3-out/` (with
    `n-selection.json`).
  - **Re-check, 2026-10-25 (no rerun).** The aggregator changed after
    this smoke (`0705440`: exploratory cells split from the
    confirmatory ones before the abort, exclusion and completeness
    checks). `--select-n` at `d81652a` on the same 27 cell records
    (all `status: ok`, `commit_sha` `6f11823`) gives an output
    identical to the stored `n-selection.json` (24 confirmatory
    cells, no abort, line 2, **N = 95**). Recomputed by hand: the
    per-seed `WDPA(P_max/within) − WDPA(P_max/none)` are 0.00374,
    0.00067 and 0.00191 (seeds 900 / 901 / 902), sd (ddof = 1)
    0.00155. The smoke was not rerun and N is not re-selected.
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

1. ~~Domain corpus~~ — decided by the PI on 2026-10-24: CIFAR-10
   (grayscale) + Quick, Draw! (§5).
2. ~~Un-mixed control arm~~ — decided by the PI on 2026-10-24:
   registered as an **exploratory** arm (§5), reported only.
3. **Hardware.** Partly decided by the PI on 2026-10-24: the smoke
   runs on macM1 now (done, see §9); the hardware for the
   confirmatory campaign is chosen after the N10-C campaign. The
   protocol does not change; R1 run_ids are per machine.
4. ~~Amendment after the smoke~~ — decided by the PI on 2026-10-24:
   epochs 3 → 10 (§5), acc20 and DOA descriptive only (§2, §6), and
   `P_max` runs recombine_full (§5). The smoke is rerun (§9).
