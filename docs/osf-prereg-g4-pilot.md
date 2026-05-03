# OSF Pre-Registration — G4 pilot (MLX × Split-FMNIST × profile sweep)

**Project** : dreamOfkiki
**Parent registration** : 10.17605/OSF.IO/Q6JYN (Cycle 1)
**Amendment** : G4 pilot — first empirical evidence on real
  continual-learning benchmark
**PI** : Clement Saillant (L'Electron Rare)
**Date drafted** : 2026-05-03
**Lock target** : before any G4 run is registered in
  `harness/storage/run_registry.RunRegistry`

## 1. Study design

Within-architecture × within-benchmark sweep on the MLX
`kiki_oniric` substrate (`kiki_oniric.substrates.mlx_kiki_oniric`).
A small MLP image classifier is trained on Split-FMNIST organised
as 5 sequential tasks (2 classes per task), with one
`DreamEpisode` injected between consecutive tasks. The episode's
operation set is dictated by the active profile (`P_min`,
`P_equ`, `P_max`). A `baseline` arm (no dream episode) is run for
direct comparison.

For each `(arm, seed)` cell the driver records :

- `acc_task1_initial` : test accuracy on task 1 immediately after
  training task 1 (before any subsequent task).
- `acc_task1_final` : test accuracy on task 1 after training all
  5 tasks sequentially.
- `retention = acc_task1_final / acc_task1_initial` — fraction of
  initial task-1 performance that survives 4 subsequent tasks.

## 2. Hypotheses

### H1 — P_equ retention floor matches Hu 2020 anchor

**Statement** : observed Hedges' g of `(P_equ retention vs
baseline retention)` is greater than or equal to the lower 95 %
CI bound of HU_2020_OVERALL (g = 0.21).

**Operationalization** :
- `g_h1 = compute_hedges_g(retention[P_equ], retention[baseline])`
- Reject H0 (no consolidation gain) iff `g_h1 >= 0.21`
- Statistical test for inference : Welch's one-sided t-test
  `(retention[baseline], retention[P_equ])` at α = 0.05 / 3
  (Bonferroni for 3 profiles)

### H3 — P_min retention decrement matches Javadi 2024 anchor

**Statement** : observed |Hedges' g| of `(P_min retention vs
baseline retention)` is greater than or equal to the lower 95 %
CI bound of JAVADI_2024_OVERALL (g = 0.13), with P_min showing
a *decrement* (negative g).

**Operationalization** :
- `g_h3 = compute_hedges_g(retention[P_min], retention[baseline])`
- Reject H0 (no decrement) iff `g_h3 <= -0.13`
- Statistical test : Welch's one-sided t-test
  `(retention[P_min], retention[baseline])` at α = 0.05 / 3

### H_DR4 — DR-4 monotonicity ordering across profiles

**Statement** : mean retention is monotonically ordered
`P_max >= P_equ >= P_min` (per framework C DR-4 derived
constraint).

**Operationalization** :
- `mean_retention[P_max] >= mean_retention[P_equ] >= mean_retention[P_min]`
- Statistical test : Jonckheere-Terpstra trend on the three
  groups, ascending order `[P_min, P_equ, P_max]`, α = 0.05

## 3. Pre-specified analyses

- H1 : `kiki_oniric.eval.statistics.welch_one_sided` + Hedges' g
  via `compute_hedges_g` + verdict via
  `harness.benchmarks.effect_size_targets.HU_2020_OVERALL.is_within_ci`
  / `distance_from_target`.
- H3 : Welch's one-sided t + Hedges' g + verdict via
  `JAVADI_2024_OVERALL.is_within_ci` / `distance_from_target`.
- H_DR4 : `kiki_oniric.eval.statistics.jonckheere_trend` on the
  three retention groups in `[P_min, P_equ, P_max]` order.
- Multiple-comparison correction : Bonferroni at family size 3,
  α_per_test = 0.0167. The Jonckheere test is *separate* from
  the Welch family (different question — ordering vs effect floor).

## 4. Sample size / power

- N = 5 seeds per arm × 4 arms = 20 cells.
- This pilot is **exploratory** for absolute g magnitudes : with
  N=5 vs N=5 the minimum detectable g at 80% power, α=0.05
  one-sided, is ~1.4 (very large). Detecting Hu 2020's overall
  g=0.29 floor at 80% power requires N≈95 seeds.
- **Pre-specified outcome interpretation** :
  - If observed g ≥ Hu/Javadi lower CI **and** Welch test
    rejects at α/3 : confirmatory evidence within this pilot's
    statistical power.
  - If observed g ≥ Hu/Javadi lower CI **and** Welch test does
    not reject : exploratory positive evidence — schedule a
    confirmatory N≥30 follow-up.
  - If observed g < Hu/Javadi lower CI : exploratory
    null-or-decrement — schedule a confirmatory follow-up before
    declaring G4 falsified.

## 5. Data exclusion rules

- Cells where MLX substrate raises any BLOCKING invariant (S1
  `retained non-regression`, S2 `finite weights`) are **excluded
  from H1/H3/H_DR4** and logged with `excluded=true` in the JSON
  dump. These cells still register a `run_id` (R1 contract) but
  carry their `EpisodeLogEntry.error` as exclusion reason.
- Cells with `acc_task1_initial < 0.5` are excluded as
  underperforming-baseline (the classifier did not learn task 1
  at all — retention is meaningless).
- Cells where `dream_episode()` exits with NotImplementedError
  (handler missing) are excluded and surface as a *plan* failure,
  not a data issue.

## 6. DualVer outcome rules (binding)

Observed effect sizes feed back into a DualVer bump per
framework-C §12 :

| Outcome | EC bump | Rationale |
|---------|---------|-----------|
| **All three (H1, H3, H_DR4) rejected H0 in the predicted direction** | PARTIAL → STABLE | Empirical confirmation crosses the §12.3 STABLE bar for the G4 scope |
| **H1 confirmed but H3 or H_DR4 inconclusive** | stays PARTIAL | Partial confirmation, not falsification |
| **Any pre-registered hypothesis falsified in the wrong direction (e.g. P_max retention < P_min retention)** | PARTIAL → UNSTABLE | §12.3 transition rule on falsification |

No FC bump in any outcome (no axiom or primitive change).

## 7. Deviations from pre-registration

Any post-hoc deviation will be documented in
`docs/osf-deviations-g4-<date>.md` (separate file, dated
immutable). Deviations include : seed-count change, statistical
test substitution, exclusion-rule relaxation.

## 8. Data and code availability

- Pilot driver : `experiments/g4_split_fmnist/run_g4.py`
- Effect-size helper : `kiki_oniric.eval.statistics.compute_hedges_g`
- Verdict helpers : `harness.benchmarks.effect_size_targets.{HU_2020_OVERALL, JAVADI_2024_OVERALL}`
- Run registry : `harness/storage/run_registry.RunRegistry`,
  SQLite at `.run_registry.sqlite`
- Outcome dump : `docs/milestones/g4-pilot-2026-05-03.{json,md}`

## 9. Contact

Clement Saillant — clement@saillant.cc — L'Electron Rare, France

---

**Lock this document before any G4 cell is registered in the run
registry.**
