# DR-3 global relational probe pre-registration

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI).
**Date:** 2026-10-23
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**c_version at drafting:** C-v0.25.0+PARTIAL
**Claim under test:** the DR-3 anchoring sentence in Paper 1 §8.4
("DR-3 is therefore anchored to local, not global, representational
convergence"), and the mutual-kNN figure it rests on (Paper 1 §7.4:
mutual-kNN = 0.174 at k = 10, N = 1024, 3 seeds, 18.8× a
label-matched random baseline; `nerve-wml` v1.9.0).
**Primary substrate pair:** `MlpWML` vs `LifWML` codes from
`nerve-wml`, pinned to tag `v1.9.0` (the tag Paper 1 cites).
**Secondary substrate pair (exploratory):** MLX MLP vs E-SNN
thalamocortical classifiers from the G5 cross-substrate pilot
(`experiments/g5_cross_substrate/`), penultimate-layer
representations on Split-FMNIST test inputs.
**Compute:** CPU only. N = 10 seeds × (trained pair + capacity-
matched random-init pair) × 2 preprocessing pipelines. Minutes per
seed.
**Lock commit:** *(filled by introducing commit hash at filing)*
**Lock timestamp:** *(filled at filing, pre-probe-run)*

## §1 Background

Paper 1 folds the 2026 PRH critiques into DR-3 by anchoring
substrate-agnosticism to the **local** form of the Platonic
Representation Hypothesis. That form is null-calibrated mutual-kNN
overlap, following Gröger et al., *Revisiting the Platonic
Representation Hypothesis: An Aristotelian View* (arXiv:2602.14486).

Two later results bear on that wording:

- You, Jang, Mo & Jung (2026), *What Converges in the Platonic
  Representation Hypothesis? Structure over Geometry*
  (arXiv:2609.27252). After calibration, **relational structure**
  converges at both local and global scales, and **metric geometry**
  is what fails. They introduce an H₀ "skeleton overlap" as a global
  analogue of mutual-kNN. They argue that the local-vs-global split
  confounds scale with structure-vs-geometry.
- Bangachev, Bresler & Polyanskiy (2026), *Representation Alignment
  Rests on Linear Structure* (arXiv:2605.28870). Centering and
  normalisation consistently increase measured alignment, so an
  alignment probe must **fix its preprocessing in advance**.

This probe tests which reading the DR-3 anchor should carry, "local,
not global" or "relational structure, not metric geometry". It
reports a 2 × 2 of {structure, geometry} × {local, global} under
one pre-fixed preprocessing pipeline. It changes no axiom: DR-3's
statement and the Conformance Criterion C+ are untouched. Only the
§8.4 prose anchor is in scope.

## §2 Hypotheses (confirmatory)

All four statistics (§5) are computed on the same N_in = 1024 probe
inputs per seed. Each is compared against the capacity-matched
random-init null (§6). Family: 4 tests, Bonferroni
α = 0.05 / 4 = **0.0125**. Unit of replication = seed (n = 10 per
condition). Test = Welch one-sided (trained > null),
`scipy.stats.ttest_ind(equal_var=False)`.

- **H13-A (local structure survives)** — mutual-kNN overlap (k = 10)
  of the trained pair exceeds the null. Confirmed iff Welch rejects
  at α = 0.0125 **and** mean ratio trained / null ≥ 2.0. This is a
  replication of the existing §7.4 claim under pinned preprocessing.
- **H13-B (global structure survives)** — H₀ skeleton overlap (§5)
  of the trained pair exceeds the null. Same rule (α = 0.0125,
  ratio ≥ 2.0). This is the test of You et al.'s claim.
- **H13-C (local geometry fails)** — local distance correlation (§5)
  of the trained pair does **not** exceed the null by the H13-A
  rule. "Fails" means Welch does not reject at α = 0.0125, **or**
  the ratio is < 1.5.
- **H13-D (global geometry fails)** — global distance correlation
  (§5), same "fails" rule as H13-C.

The ratio thresholds (2.0 to survive, 1.5 to fail) are fixed here.
A statistic with 1.5 ≤ ratio < 2.0 that rejects is reported as
**indeterminate**, neither surviving nor failing.

## §3 Power analysis

n = 10 seeds per condition. At α = 0.0125 one-sided, Welch detects
g ≥ 1.48 at 80 % power. The existing §7.4 mutual-kNN effect (18.8×
the baseline) is far above that floor. The test is powered for large
structure effects only; small effects are reported as exploratory.
10 seeds meets the multi-seed-first-class rule (≥ 5). The 3 seeds of
the existing §7.4 figure do not, so new seeds are required.

## §4 Exclusion criteria

- A trained `MlpWML` / `LifWML` seed is excluded if its task accuracy
  is below the gate used in `nerve-wml` v1.9.0 for the §7.4
  experiment (copied verbatim into the milestone before the run).
- Representations with any non-finite entry, or with a zero-variance
  feature that survives centering for every input, are excluded.
- If more than 3 of 10 seeds are excluded, abort and amend.

## §5 Statistics and pinned preprocessing

**Preprocessing (primary, pinned).** For each representation matrix
X ∈ ℝ^{1024 × d}:
1. center each feature over the 1024 probe inputs;
2. L2-normalise each row;
3. compute Euclidean distances on the result (equivalent to cosine
   distance on centered features).

**Preprocessing (sensitivity, reported only).** Raw features, no
centering, no normalisation. Results under this pipeline never
change a confirmatory verdict. They are reported to show how
sensitive the probe is, per Bangachev et al.

**Statistics:**
- *Local structure*: mutual-kNN overlap, k = 10. This is the
  mean over inputs of |kNN_X(i) ∩ kNN_Y(i)| / k.
- *Global structure*: H₀ skeleton overlap, operationalised as the
  Jaccard overlap of the minimum-spanning-tree edge sets of the two
  distance graphs. The MST is the 0-dimensional persistence skeleton
  of the Vietoris–Rips filtration. This operationalisation must be
  checked against You et al.'s reference definition before filing
  (§10).
- *Local geometry*: Pearson correlation between the two distance
  matrices, restricted to pairs (i, j) in the union of both k = 10
  neighbour graphs.
- *Global geometry*: Pearson correlation between the full upper
  triangles of the two distance matrices (Mantel-style). Linear CKA
  is reported as a secondary global-geometry statistic.

Sensitivity sweep (reported, not tested): k ∈ {5, 20}.

## §6 Null calibration

- **Primary null: capacity-matched random-init pair.** The same two
  architectures at the same widths, untrained, initialised with the
  same seed family. Representations are computed on the same 1024
  inputs and the same statistic is applied. This follows the null
  that Gröger et al. argue for.
- **Secondary null: permutation.** Shuffle the row correspondence of
  Y (1000 permutations, seeded). Reported as a per-seed p-value. The
  verdicts use the primary null only.
- The label-matched random baseline of the existing §7.4 figure is
  also reported, for continuity with Paper 1.

## §7 Outcome rules — DR-3 anchor wording

| Outcome | DR-3 §8.4 anchor | FC / EC |
|---|---|---|
| Row 1 — H13-A ∧ H13-B ∧ H13-C ∧ H13-D | Re-anchor to **"null-calibrated relational structure (local and global), not metric geometry"**. Prose change in Paper 1 §8.4 (EN + FR, same PR). | FC unchanged (no axiom change), EC unchanged |
| Row 2 — H13-A ∧ ¬H13-B | The current **"local, not global"** anchor stands. | unchanged |
| Row 3 — H13-A ∧ H13-B, with geometry not failing (H13-C or H13-D fails) | Structure and geometry both survive on this pair. Use the structure wording; the geometry result is reported, not built on. | unchanged |
| Row 4 — ¬H13-A | The existing §7.4 local claim does **not** replicate under pinned preprocessing. File a dated correction note in `docs/milestones/` and qualify Paper 1 §7.4 / §8.4. | unchanged |
| Row 5 — exclusion > 3 / 10 seeds | Abort and amend (§4). | n/a |

The secondary G5 pair (MLX MLP vs E-SNN) is reported with the same
2 × 2. It is **exploratory** and cannot change the anchor on its own.

## §8 Reporting commitment and stopping rule

- **Fixed N = 10 seeds, no optional stopping.** No statistic is
  computed on the trained pair before all 10 seeds and their nulls
  are dumped.
- All four statistics, both preprocessing pipelines, both nulls and
  the k sweep are reported regardless of outcome.

## §9 Audit trail and deviations

- New script `scripts/dr3_relational_probe.py` (to be written). It
  consumes representation dumps exported from `nerve-wml` at tag
  `v1.9.0` and from the G5 drivers. Inputs are hashed (`.sha256`
  next to each dump).
- Milestones: `docs/milestones/dr3-relational-probe-<date>.{json,md}`.
- Results are reflected in `docs/proofs/dr3-substrate-evidence.md` as
  a new dated section (append-only).
- Deviations: an amendment commit before the run, or post-hoc
  disclosure marking the affected hypothesis exploratory (per
  `docs/osf-preregistration-draft.md` §5).

## §10 Open decisions for the PI (resolve before filing)

1. **Skeleton-overlap definition:** confirm the MST-edge Jaccard
   operationalisation against You et al.'s code or definition. If
   theirs differs, adopt theirs verbatim and update §5 before
   filing.
2. **Pin:** keep `nerve-wml` `v1.9.0` (matches Paper 1), or move to
   a later tag that also exposes the pre-VQ continuous codes.
3. **Which code layer:** pre-VQ continuous codes (default; this
   avoids the ≈ 10× VQ amplification Paper 1 §7.4 flags) or
   post-VQ codes.
4. **Ratio thresholds:** 2.0 (survive) and 1.5 (fail) as drafted, or
   other values chosen before filing.
