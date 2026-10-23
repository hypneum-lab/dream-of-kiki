# G4-nono cross-domain RECOMBINE pre-registration

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI).
**Date:** 2026-10-23
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**Sister pre-regs (locked, not modified):**
`docs/osf-prereg-g4-sexto-pilot.md` (H6-A, CIFAR-100),
`docs/osf-prereg-g4-septimo-pilot.md` (H6-B / H6-C, Tiny-ImageNet).
**c_version at drafting:** C-v0.25.0+PARTIAL
**Substrate:** MLX small CNN (`G4SmallCNN`, reused from
`experiments.g4_quinto_test.small_cnn`), single shared 20-way
classification head (class-incremental, no per-task label remap).
**Benchmark:** Split-CIFAR-100-coarse two-domain stream (20
superclasses, 32×32 RGB), defined in §5.
**Compute:** N = 30 seeds × 4 arms × 3 RECOMBINE strategies = 360
cells. Per-cell wall is expected in the G4-sexto range (≈ 20 s on
M1 Max for a 10-task CIFAR stream) ; total ≈ 2-3 h. Hardware is an
open decision (§10).
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

Every ladder stream is **single-domain** class-incremental: all
tasks are drawn from one image distribution, and the scored outcome
is retention of earlier tasks. The ladder RECOMBINE strategies
(`mog`, `ae`) sample per-class synthetic latents; they never mix
classes from different domains.

Zahn, Evans & Eagleman (2026), *Discovery by Dreaming: Cross-Domain
Recombination in Artificial Memory* (arXiv:2607.16256), report that
recombinatory replay **across** domains helps both a LoRA pipeline
and a symbolic engine, while **within-domain** rehearsal does not.
This predicts exactly the ladder's null in single-domain streams and
names the regime where `recombine` should show up: cross-domain
probes, not within-domain retention.

G4-nono therefore asks whether the DR-4 refutation is **scoped to
within-domain streams**. It does not re-test the ladder; it adds one
two-domain stream with a cross-domain outcome.

The LLM-tier variant (LoRA math → code on the 1.5B / 35B substrate)
is **out of scope** here: in the CUDA E1 harness
(`scripts/pilot_cycle3_cuda.py`, `_apply_dream_ops`) `recombine` is a
weight-neutral no-op, so it cannot test this hypothesis without a new
op implementation.

## §2 Hypotheses (confirmatory)

Definitions (§6): CDPA = cross-domain pairwise accuracy, WDPA =
within-domain pairwise accuracy, both measured after the full
stream. Strategies: `cross` (cross-domain latent recombination),
`within` (matched within-domain recombination), `none` (placebo,
RECOMBINE skipped — the ladder's H4-C / H5-C / H6-A control).

- **H12-A (cross-domain gain)** — in the `P_max` arm,
  `CDPA(cross) > CDPA(none)`. Test: Welch one-sided at
  α = 0.05 / 3 = **0.0167** (Bonferroni over H12-A, H12-B, H12-C).
  Confirmed iff H0 is rejected with positive sign **and**
  Hedges' g ≥ 0.5 (the G4-octavo H7-A / G6 H9-A effect floor).

- **H12-B (cross-domain specificity)** — in the `P_max` arm,
  `CDPA(cross) > CDPA(within)`. Same test, α = 0.0167, same
  g ≥ 0.5 floor. H12-B separates "recombining across domains" from
  "any extra synthetic SGD pass".

- **H12-C (within-domain null replicates)** — in the `P_max` arm,
  `WDPA(within)` is not statistically distinguishable from
  `WDPA(none)`. Test: Welch two-sided fails to reject H0 at
  α = 0.0167, the ladder's H6-A / H6-B contract. Failing to reject
  is the predicted outcome. A TOST at SESOI g = 0.29
  (`kiki_oniric.eval.statistics.tost_equivalence`) is reported as a
  secondary analysis. It is not expected to be powered at N = 30
  (§3).

- **H12-D (scoped DR-4, derived)** — logical conjunction
  `H12-A ∧ H12-B ∧ H12-C`. No additional test.

Secondary contrast (reported, not part of the Bonferroni family):
`CDPA(P_max, cross)` vs `CDPA(P_equ, cross)` and vs
`CDPA(P_min, none)`. These link the result to the DR-4 profile chain.

## §3 Power analysis

N = 30 seeds per arm × strategy. At α = 0.0167 one-sided, Welch
detects g ≥ 0.78 at 80 % power. (At the ladder's α = 0.05
two-sided, N = 30 detects g ≥ 0.74, the figure quoted in the G4
pre-regs.) The g ≥ 0.5 confirmation floor is below the 80 %-power
detection floor. Effects with 0.5 ≤ g < 0.78 can therefore be
missed and are reported as exploratory if they do not reject.
N = 30 meets the multi-seed-first-class rule (≥ 5 seeds). Raising N
to 95 (the G4-sexto confirmatory size) is an open decision (§10).

## §4 Exclusion criteria

- Floor: a cell is excluded if accuracy on task-1 test images,
  scored on the full 20-way head right after task 1, is below
  `2 × random_chance = 2 × 1/20 = 0.10` (ladder "2 × chance" rule).
- `CDPA`, `WDPA` or final accuracy non-finite: exclude cell.
- Any BLOCKING invariant violation (I1, I2, S1, S2, S3, K1, K4):
  exclude and log, as in `docs/osf-preregistration-draft.md` §4.
- run_id collision with a prior pilot's registry: abort and amend.
- If more than 50 % of cells in any arm × strategy are excluded,
  abort and amend (raise epochs per the G4-sexto §9.1 pattern). Do
  not commit a milestone.

## §5 Stream, substrate and driver

- **Domains.** The 20 CIFAR-100 superclasses are split once, before
  any run, into two fixed domains:
  - Domain A (animals): aquatic_mammals, fish, insects,
    large_carnivores, large_omnivores_and_herbivores, medium_mammals,
    non-insect_invertebrates, people, reptiles, small_mammals.
  - Domain B (plants, objects, scenes): flowers, food_containers,
    fruit_and_vegetables, household_electrical_devices,
    household_furniture, large_man-made_outdoor_things,
    large_natural_outdoor_scenes, trees, vehicles_1, vehicles_2.
- **Stream.** 10 sequential tasks of 2 superclasses each. Tasks 1-5
  cover domain A, tasks 6-10 cover domain B. The pairing of
  superclasses into tasks is fixed by alphabetical order within each
  domain (no per-seed reshuffle), so seed varies only weights,
  data order and dream sampling.
- **Head.** One 20-way linear head over coarse labels (class-
  incremental). No per-task label remap.
- **Dream episodes.** One DE after each task, as in the ladder.
  β buffer: capacity 320, fill 32 records per task (all records
  retained over the stream). Ladder constants kept:
  `RECOMBINE_N_SYNTHETIC = 16`, `RECOMBINE_LR = 0.01`,
  `RESTRUCTURE_FACTOR = 0.05`, representative HP combo from
  `experiments.g4_ter_hp_sweep.hp_grid.representative_combo`.
- **RECOMBINE strategies** (new module, to be written before
  filing; it extends `experiments/g4_quater_test/recombine_strategies.py`):
  - `cross`: draw 16 pairs (a, b) with a ∈ domain A and b ∈ domain B
    among classes present in the buffer. Build a synthetic latent
    `z = λ·z_a + (1 − λ)·z_b`, where z_a, z_b are buffer latents and
    λ ~ Uniform(0.2, 0.8) (seeded). Use soft target
    `λ·e_a + (1 − λ)·e_b`. One SGD pass on the head.
    Only defined once the buffer holds both domains (DEs after tasks
    6-10). Before that, `cross` is a no-op, identical to `none`.
  - `within`: identical procedure, with both a and b drawn from the
    same domain (half of the pairs A-A, half B-B). It runs at the
    same DEs as `cross` (after tasks 6-10), so `cross` and `within`
    have the same synthetic-sample count and SGD steps.
  - `none`: RECOMBINE skipped (ladder placebo).
- **Arms:** `baseline` (no DE), `P_min`, `P_equ`, `P_max`. The
  strategy switch is threaded through the arms that run RECOMBINE.
- **Driver:** `experiments/g4_nono_test/run_cross_domain.py` (to be
  written). Aggregator: `experiments/g4_nono_test/aggregator.py`.
- **Data pin:** CIFAR-100 archive and SHA-256 pin reused from
  `experiments.g4_sexto_test.cifar100_dataset`. That loader drops
  the coarse-label byte today, so the new driver adds a coarse-label
  decode path from the same pinned archive (no new download).

## §6 Outcome metrics

- **CDPA (primary).** For every pair (a, b) with a ∈ A and b ∈ B
  (100 pairs), binary accuracy on the test images of a and b, scored
  using only the logits {a, b}. CDPA is the mean over the 100 pairs.
  It measures cross-domain discrimination and excludes within-domain
  retention.
- **WDPA (H12-C).** Same computation over the 45 + 45 within-domain
  pairs (A-A and B-B).
- **Retention (secondary, ladder-comparable).** Accuracy on domain-A
  test images after task 10, restricted to domain-A logits.
- **Final 20-way accuracy (secondary).**
- **Effect-size convention:** Hedges' g
  (`kiki_oniric.eval.statistics.compute_hedges_g`), independent cells
  across seeds.

## §7 DualVer outcome rules — reopen vs confirm DR-4

| Outcome | Reading | EC | FC |
|---|---|---|---|
| Row 1 — H12-A ∧ H12-B ∧ H12-C (H12-D holds) | **DR-4 reopened in scope.** RECOMBINE adds cross-domain discrimination that within-domain recombination does not. The ladder refutation is restated as "refuted in within-domain class-incremental regimes". DR-4 evidence gets a G4-nono addendum. | stays PARTIAL | stays (no formal change) |
| Row 2 — H12-A only (cross > none, cross ≯ within) | Gain is generic synthetic-SGD, not cross-domain recombination. **DR-4 refutation stands.** The "within-domain scope" reading is not supported. | stays PARTIAL | stays |
| Row 3 — H12-A fails (fail-to-reject, or g < 0.5) | **DR-4 refutation confirmed and extended** to a two-domain stream with a cross-domain outcome. The Zahn et al. prior does not transfer to this substrate. | stays PARTIAL | stays |
| Row 4 — H12-A rejects with negative sign | Cross-domain recombination **hurts**. DR-4 is further weakened. Record the negative-direction boundary. | stays PARTIAL | stays |
| Row 5 — H12-C rejects (within ≠ none) | The ladder's within-domain null does not replicate on this stream. Report it, then read H12-A/B as exploratory until it is explained. | stays PARTIAL | stays |
| Row 6 — exclusion > 50 % in any arm × strategy | Abort and amend (§4). No milestone. | n/a | n/a |

No row promotes EC to STABLE. A single two-domain CNN stream cannot
carry a universality claim in either direction.

## §8 Reporting commitment and stopping rule

- **Fixed N, no optional stopping.** All 360 cells run. No interim
  hypothesis test is computed before the last cell is registered.
- Honest-reading clause (adapted from G4-sexto §7): *"Welch
  fail-to-reject is absence of evidence at this N, not evidence of
  absence. Under H12-C specifically, it is the predicted outcome,
  and the TOST result is reported next to it."*
- All observed scalars are reported regardless of outcome, including
  `baseline`, `P_min`, `P_equ` cells and every secondary metric.

## §9 Audit trail and deviations

- Cells registered via `harness/storage/run_registry.py` with profile
  keys `g4-nono/<arm>/<strategy>` and R1 bit-stable run_ids.
- Milestones: `docs/milestones/g4-nono-cross-domain-<date>.{json,md}`.
- Known deviation envelopes:
  a. CIFAR-100 SHA-256 mismatch: abort and amend with the new pin.
  b. Floor exclusion on most seeds: raise epochs per G4-sexto §9.1.
     Document this in the milestone header.
  c. Per-cell wall > 60 s sustained: escalate to the PI before
     running further cells. Do not reduce N silently.
- Any other deviation needs an amendment commit **before** the
  affected cells run, or a post-hoc disclosure in Paper 2 that marks
  the affected hypotheses exploratory (per
  `docs/osf-preregistration-draft.md` §5).

## §10 Open decisions for the PI (resolve before filing)

1. **N:** 30 (default, ladder-comparable) or 95 (G4-sexto
   confirmatory size; detects g ≥ 0.43 at α = 0.0167 one-sided).
2. **Domain split:** the animals / non-animals split in §5, or a
   stronger domain shift (e.g. a second image corpus). A stronger
   shift needs a new loader and SHA-256 pin.
3. **Mixing law:** λ ~ Uniform(0.2, 0.8) as above, or Beta(α, α)
   mixup-style sampling.
4. **Hardware:** M1 Max or Studio. This does not change the
   protocol; R1 run_ids are per machine.
