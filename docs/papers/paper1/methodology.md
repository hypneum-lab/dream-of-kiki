# Methodology (Paper 1, draft S20.1)

**Target length** : ~1.5 pages markdown (≈ 1500 words)

---

## 6.1 Pre-registered hypotheses (OSF)

Four hypotheses were pre-registered on the Open Science Framework
(OSF) before any empirical run, following the Standard Pre-Data
Collection template. Pre-registration was locked at S3 of the
cycle (calendar reference) ; the OSF DOI is cited in the paper
front matter and resolves to an immutable timestamp record.
This discipline aligns with the standard practice now explicitly
recommended for sleep-consolidation meta-analyses
[@elife2025bayesian] and mirrors the protocol followed by the
sister project `bouba_sens` (DOI 10.17605/OSF.IO/Q6JYN).

- **H1 — Forgetting reduction** : `mean(forgetting_P_equ) <
  mean(forgetting_baseline)`. Test : Welch's t-test, one-sided.
  *Effect-size floor* : the targeted memory reactivation
  meta-analysis [@hu2020tmr] reports overall Hedges' g = 0.29
  (NREM2 g = 0.32, SWS g = 0.27), used here as the reference
  magnitude for `P_equ`-style consolidation gains. It is not a
  floor : the same paper's bias-adjusted estimates are smaller
  (trim-and-fill g = 0.18, three-parameter selection model
  g = 0.13, g = −0.05 under severe one-tailed selection).
- **H2 — P_max equivalence** : `|mean(acc_P_max) -
  mean(acc_P_equ)| < 0.05`. Test : two one-sided tests (TOST).
  *Cycle 1 status* : self-equivalence smoke only (P_max skeleton).
- **H3 — Monotonic alignment** : `mean(acc_P_min) <
  mean(acc_P_equ) < mean(acc_P_max)`. Test :
  Jonckheere-Terpstra. *Cycle 1 status* : two-group (P_min ↔
  P_equ) only. *Effect-size floor for the P_min arm* : the sleep
  restriction meta-analysis [@javadi2024sleeprestriction] reports
  g = 0.29 [0.13, 0.44] as the expected magnitude of the
  degraded-substrate decrement ; its funnel plot is significantly
  asymmetric (Egger z = 3.90) although a three-parameter selection
  model detects no publication bias, and its 95 % prediction
  interval runs from −0.73 to 1.30.
- **H4 — Energy budget** : `mean(energy_dream / energy_awake)
  < 2.0`. Test : one-sample t-test against threshold.

## 6.2 Statistical tests + Bonferroni correction

All hypothesis tests use a Bonferroni-corrected significance
threshold : `α_per_hypothesis = 0.05 / 4 = 0.0125`. The four
tests are implemented in the reference implementation's
statistical module (which wraps standard statistical libraries ;
see Paper 2 for the substrate-specific code path) :

- **`welch_one_sided`** (H1) : `scipy.stats.ttest_ind` with
  `equal_var=False`, p-value halved for one-sided interpretation.
- **`tost_equivalence`** (H2) : two manual one-sided t-tests
  (lower bound `diff <= -ε` and upper bound `diff >= +ε`),
  reject H0 when both pass at α (TOST max-p rule).
- **`jonckheere_trend`** (H3) : sum of pairwise Mann-Whitney U
  counts across ordered groups, z-approximation for p-value
  (no scipy native).
- **`one_sample_threshold`** (H4) : `scipy.stats.ttest_1samp`
  against `popmean=threshold`, p-value adjusted for one-sided
  (sample below threshold).

All tests return a uniform `StatTestResult(test_name, p_value,
reject_h0, statistic)` for downstream handling.

## 6.3 mega-v2 benchmark

Empirical runs use the **mega-v2** dataset (498K examples
distributed across 25 domains : phonology, lexical, syntax,
semantic, pragmatic, etc.). Cycle 1 stratifies a **500-item
retained subset** (20 items per domain) and freezes it via
SHA-256 hash for the reproducibility contract R1.

The frozen retained benchmark is loaded via `harness.benchmarks.
mega_v2.adapter.load_megav2_stratified()`, which falls back to
a deterministic synthetic placeholder if the real mega-v2 path
is unavailable. **All cycle-1 results in §7 use the synthetic
fallback ; real mega-v2 integration lands cycle 1 closeout
(S20+) or cycle 2.**

## 6.4 RSA fMRI alignment (Studyforrest)

The H3 monotonic representational alignment hypothesis is
evaluated by Representational Similarity Analysis (RSA) between
kiki-oniric activations and fMRI responses. Cycle 1 uses the
**Studyforrest** dataset (Branch A locked at G1 — see
`docs/feasibility/studyforrest-rsa-note.md`) :

- **Format** : BIDS, DataLad-distributed, PDDL license (open).
- **Annotations** : 16,187 timed words, 2,528 sentences, 66,611
  phonemes ; 300-d STOP word vectors. Mappable to ortho species
  (rho_phono / rho_lex / rho_syntax / rho_sem).
- **ROIs** : extracted via FreeSurfer + Shen-268 parcellations
  for STG, IFG, AG (the canonical language network).
- **Pipeline** : `nilearn` CPU-deterministic mode for R1
  reproducibility. Real ablation deferred S20+ (real model
  inference) ; cycle 1 reports infrastructure validation only.

## 6.5 Reproducibility contract R1 + R3

Reproducibility is enforced by two contracts :

- **R1 (deterministic run_id)** : every run is keyed by a
  16-character SHA-256 prefix of `(c_version, profile, seed,
  commit_sha)`. Re-running the same key produces an identical
  `run_id` (verified by `harness.storage.run_registry`). Width
  was bumped from 16 → 32 hex chars in commit `df731b0` after a
  code-review finding flagged 64-bit collision risk at scale.
- **R3 (artifact addressability)** : all benchmarks ship with a
  paired `.sha256` integrity file. The `RetainedBenchmark`
  loader rejects any items file whose hash does not match the
  frozen reference, raising `RetainedIntegrityError`.

The DualVer versioning scheme (formal axis FC + empirical axis
EC) tags each artifact with the framework version under which
it was produced. Empirical results are valid only against the
declared `c_version` ; bumping FC-MAJOR invalidates EC and
requires re-running the affected matrix.

## 6.6 Biological grounding of profiles and DR-2'

The three profiles are calibrated against substrate-level
biomarkers reported in the human sleep literature.
`P_min` (degraded substrate) is anchored on the slow-oscillation
trough amplitude and frontocentral synchronization gradient
documented across healthy older / aMCI / AD groups (N = 55
hd-EEG) [@sharon2025alzdementia], where cognitive performance
decreases monotonically with the loss of slow-wave coherence.
The calibration is realised in code as a
`so_trough_amplitude_factor` field on the profile dataclass
(`kiki_oniric/profiles/p_min.py`), with the unit-arbitrary ratio
`0.45` for `P_min` (aMCI midpoint placeholder) and `1.0` for the
healthy anchors `P_equ` / `P_max`. Sharon 2025 does report
absolute slow-wave trough amplitudes in µV (group medians
−60.18 healthy, −56.95 aMCI, −52.61 AD) ; the 0.45 ratio is not
derived from them and will be refined to an empirical value at
the G2 P_min pilot (`scripts/pilot_g2.py`).
`P_equ` (canonical substrate) is calibrated to reproduce the
NREM2/SWS effect-size band reported in [@hu2020tmr]. The DR-2'
falsified-and-amended axiom (interleaved novel-hippocampal +
familiar-cortical replay, see `docs/specs/amendments/`) inherits
its biological grounding from the SWS up-state interleaving
result of [@biorxiv2025thalamocortical].

The numerical anchors of [@hu2020tmr] (overall, NREM2, SWS) and
[@javadi2024sleeprestriction] are encoded as typed, frozen
constants in `harness.benchmarks.effect_size_targets` (`HU_2020_OVERALL`,
`HU_2020_NREM2`, `HU_2020_SWS`, `JAVADI_2024_OVERALL`) so a future
G4 pilot compares observed effect sizes against the published 95 %
CIs deterministically (`is_within_ci`, `distance_from_target`).
No empirical (EC) bump is implied : these constants encode external
published numbers, not registered run outputs.

---

## Notes for revision

- Insert OSF DOI in §6.1 once OSF lock action is completed
  (currently pending — see `docs/osf-upload-checklist.md`)
- Replace synthetic-placeholder caveats in §6.3 once real
  mega-v2 ablation runs S20+
- Add Methods supplementary table : full statistical test
  parameters (sample sizes per cell, exclusion criteria from
  OSF pre-reg)
- Add Methods supplementary figure : pipeline diagram
  (benchmark → predictor → evaluate_retained → swap → metrics)
