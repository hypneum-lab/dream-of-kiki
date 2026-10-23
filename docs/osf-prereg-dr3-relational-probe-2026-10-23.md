# DR-3 global relational probe pre-registration

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI).
**Date:** 2026-10-23 (revision 3: PI decisions of 2026-10-23 on δ
and n applied)
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**c_version at drafting:** C-v0.25.0+PARTIAL
**Claim under test:** the DR-3 anchoring sentence in Paper 1 §8.4
("DR-3 is therefore anchored to local, not global, representational
convergence"), and the mutual-kNN figure it rests on (Paper 1 §7.4:
mutual-kNN = 0.174 at k = 10, N = 1024, 3 seeds, 18.8× a random
baseline; `nerve-wml` v1.9.0).
**Primary substrate pair:** `MlpWML` vs `LifWML` from `nerve-wml`
tag `v1.9.0` (commit `95ac6b78477c3ce999b31bca775769041e082b46`;
the tag is on the lab Gitea mirror `git.saillant.cc/hypneum-lab/
nerve-wml` but **not yet on the GitHub repository**; the filing gate
(§9) requires the tag, or an archive of that commit, to be public
next to the filed registration), reproduced through
`scripts/platonic_rh_alignment.py` at that commit.
**Secondary substrate pair (exploratory):** MLX MLP vs E-SNN from
`experiments/g5_cross_substrate/` (§7, last paragraph).
**Design:** matched trained / untrained pairs per seed, paired
analysis on calibrated differences.
**Compute:** CPU only; 20 seeds × (trained pair + untrained pair).
Minutes per seed (800 training steps per substrate, d = 16).
**Power script:** `scripts/prereg_power_2026_10.py` (§3).
**Lock commit:** *(filled by introducing commit hash at filing)*
**Lock timestamp:** *(filled at filing, pre-probe-run)*

## §1 Background

Paper 1 folds the 2026 PRH critiques into DR-3 by anchoring
substrate-agnosticism to the **local** form of the Platonic
Representation Hypothesis: null-calibrated mutual-kNN overlap,
following Gröger et al., *Revisiting the Platonic Representation
Hypothesis: An Aristotelian View* (arXiv:2602.14486).

Two later results bear on that wording:

- You, Jang, Mo & Jung (2026), *What Converges in the Platonic
  Representation Hypothesis? Structure over Geometry*
  (arXiv:2609.27252). After calibration, **relational structure**
  converges at both local scale (mKNN, their Eq. 1) and global scale
  (H₀ skeleton overlap, their Eq. 2), while **metric geometry**
  (their distance-aware mKNN and H₀, Eqs. 3-5) is what fails.
- Bangachev, Bresler & Polyanskiy (2026), *Representation Alignment
  Rests on Linear Structure* (arXiv:2605.28870). Centering and
  normalisation change measured alignment, so preprocessing must be
  fixed in advance.

This probe asks which wording the DR-3 anchor should carry: "local,
not global" or "relational structure, not metric geometry". It
changes no axiom; DR-3's statement and the Conformance Criterion
are untouched. Only the §8.4 prose anchor is in scope.

**Direct replication vs robustness test.** The v1.9.0 §7.4 number
used cosine mKNN against two i.i.d. Gaussian matrices (the script's
`random` baseline in `scripts/platonic_rh_alignment.py`; Paper 1
calls it "label-matched", a wording to reconcile separately). H13-A below uses
the same representations and the same mKNN, but a capacity-matched
untrained null and new seeds. It is therefore a **robustness test**
of the §7.4 claim, not a direct replication. The direct replication
(v1.9.0 seeds 0-2, Gaussian baseline, script unchanged) is run and
reported descriptively (§6) and cannot change a verdict.

## §2 Hypotheses (confirmatory)

**Statistics** (§5): S_mKNN (local structure), S_H0 (global
structure), S_mKNN^τ (local geometry), S_H0^τ (global geometry).

**Pairing and calibrated difference.** For seed s, the null pair is
the same two architectures built with the same constructor calls and
seeds as the trained pair, with zero training steps, evaluated on
the same 1024 inputs. Trained and null pairs are therefore matched
within seed. For each statistic S the analysed quantity is the
**calibrated difference**

    D_s = S(trained pair, seed s) − S(null pair, seed s).

Ratios trained / null are not used: S_H0^τ and S_mKNN^τ can be near
zero under the null, which makes a ratio unstable. D is defined for
every value.

**Family.** Four statistics, Bonferroni α = 0.05 / 4 = **0.0125**.
For each statistic, `L` and `U` are the one-sided (1 − α) lower and
upper confidence bounds for mean(D) (paired t, df = n − 1).
Primary test call: `scipy.stats.ttest_1samp(D, 0,
alternative="greater")` (equivalently `ttest_rel(trained, null,
alternative="greater")`). **Degenerate case.** If every retained D is identical (`sd(D) = 0`),
the t statistic is undefined; then `L = U = mean(D)` and the state
follows from these bounds.

**SESOI.** δ = **0.025** in score units for all four statistics (PI
decision of 2026-10-23: half the absolute floor `0.05` that the
v1.9.0 script uses for a "signal", `max(3 × random, 0.05)`, so that
FAILS bounds the calibrated gain more tightly). All four scores lie
in [0, 1] and share the denominators k (mKNN) or n − 1 (H₀); at
k = 10, δ = 0.025 is a quarter of one shared neighbour per input on
average, and for H₀ it is ≈ 26 of the 1023 MST edges. The same δ for
structure and geometry keeps the 2 × 2 comparable.

**Per-statistic states** (exhaustive, non-overlapping):

| State | Rule |
|---|---|
| SURVIVES | `L > 0` and `mean(D) ≥ δ` |
| FAILS | `U < δ` (calibrated difference bounded below δ) |
| INDETERMINATE | neither |

SURVIVES requires `mean(D) ≥ δ`, FAILS implies `mean(D) < U < δ`, so
the two cannot co-occur.

**What SURVIVES means.** `L > 0` establishes a positive calibrated
difference at level α; `mean(D) ≥ δ` is a filter on the **point
estimate**. SURVIVES reads "positive, with an estimate at or above
δ", not "the population difference is at least δ". FAILS, by
contrast, does bound the population difference below δ.

**Error accounting.** SURVIVES can only be false if the true mean
difference μ ≤ 0; FAILS can only be false if μ ≥ δ. As δ > 0, at
most one of the two is false for any μ, so each statistic spends α
once whichever state is reached, and the four-statistic family is
bounded by 4α = 0.05.

- **H13-A (local structure survives)** — S_mKNN state = SURVIVES.
- **H13-B (global structure survives)** — S_H0 state = SURVIVES.
- **H13-C (local distance-weighted overlap bounded)** — S_mKNN^τ
  state = FAILS.
- **H13-D (global distance-weighted overlap bounded)** — S_H0^τ
  state = FAILS.

"Fails" therefore means only: **the calibrated gain in
distance-weighted overlap is bounded below δ**, never a mere
non-rejection, and never "metric geometry is absent". The τ-weighted
scores multiply shared-support membership by distance agreement, so
a small gain can come from limited structural overlap as well as
from poor distance agreement. The conditional quantity that
isolates distance agreement, the mean weight over shared supports
(`S^τ / S`, per seed, trained minus null), is reported as an
exploratory secondary statistic only; it is not powered and carries
no confirmatory weight.

## §3 Power analysis

n = **20** seeds (PI decision of 2026-10-23). Reproduced by
`scripts/prereg_power_2026_10.py` (section 6).

- Test-only MDE at 80 % power, paired one-sided α = 0.0125:
  **d_z = 0.739** (g_z ≈ 0.710). (An independent-groups design at
  n = 20 per group would need d = 1.009, Hedges' g ≈ 0.989.)
- Complete rules, in units of SD(D) (SURVIVES: Monte Carlo; FAILS at
  μ = 0: exact noncentral t):

| δ / SD(D) | SURVIVES, μ/SD = 0.5 | 1.0 | 1.5 | 2.0 | 3.0 | FAILS, μ = 0 |
|---|---|---|---|---|---|---|
| 0.25 | 0.44 | 0.97 | 1.00 | 1.00 | 1.00 | 0.12 |
| 0.5 | 0.40 | 0.97 | 1.00 | 1.00 | 1.00 | 0.44 |
| 1.0 | 0.01 | 0.50 | 0.99 | 1.00 | 1.00 | 0.97 |
| 1.5 | 0.00 | 0.01 | 0.50 | 0.99 | 1.00 | 1.00 |

SURVIVES has ≥ 0.97 power when the true calibrated difference is at
least 1 SD(D) and exceeds δ by at least 0.5 SD(D); at μ = δ it is
0.50 by construction (observed-mean floor).

**Planning SD and the FAILS rule ("no practical effect").** SD(D) is
unknown. The v1.9.0 trained mKNN values over 3 seeds (0.174, 0.170,
0.203) have SD ≈ 0.018; if the untrained null varies comparably and
independently, SD(D) ≈ 0.018 × √2 ≈ **0.025**, the planning value.
This is an assumption, not evidence: the 18.8× figure is a ratio
against a different (Gaussian) baseline and carries no seed variance.
At δ = 0.025 and μ = 0, FAILS has power:

| SD(D) | 0.0125 | 0.018 | 0.025 | 0.035 | 0.05 |
|---|---|---|---|---|---|
| power at n = 20 | 1.00 | 1.00 | 0.97 | 0.77 | 0.44 |
| n for 80 % | 6 | 8 | 13 | 22 | 41 |

At the planning SD, n = 20 is sufficient (13 seeds would give 80 %).
n = 20 reaches 80 % only if SD(D) ≤ 0.034; **if SD(D) is larger, the
registered n is insufficient** for an 80 %-powered FAILS and a
bounded statistic will more often come out INDETERMINATE (§7,
row 5). The same SD(D) applies to the structure statistics, whose
FAILS state matters for rows 3-4.

## §4 Exclusions, replacement and retained-N inference

- **Seeds.** Confirmatory seeds `100..119` (disjoint from the v1.9.0
  seeds 0-2, which are reused only for the descriptive direct
  replication).
- **Learning gate (seed level).** Each trained substrate is scored
  on 1024 held-out examples of **its own task**: build
  `task = HardFlowProxyTask(dim=16, n_classes=12, seed=s)` (same
  centres and XOR hyperplanes as training), then replace only its
  sampling stream, `task._gen = torch.Generator().manual_seed(s +
  5000)`, and draw `task.sample(batch=1024)`. The task parameters are
  unchanged; only the examples are new. Readout
  (`MlpWML`: argmax of `emit_head_pi(core(x))[:, :12]`; `LifWML`:
  argmax of `emit_head_pi(spike(input_proj(encoder(x))))[:, :12]`).
  The seed is excluded if either accuracy is below **0.25** (3× the
  12-class chance of 0.083). v1.9.0 has no accuracy gate, so this
  gate is new and fixed here.
- **Representation check.** A seed is excluded if any representation
  (trained or null) has a non-finite entry, a row with zero norm
  (L2 normalisation undefined), or no positive pairwise distance
  after normalisation (all rows coincide, so the geometry quantile
  of §5 is undefined).
- Exclusions are seed-level and apply to all four statistics. No
  replacement seeds. Inference uses the retained n; all exclusions
  are reported with reasons.
- **Abort.** If more than 6 of 20 seeds are excluded, abort and
  amend. All dumps, hashes and logs are preserved in
  `docs/milestones/dr3-relational-probe-ABORTED-<date>.{json,md}`.

## §5 Measurement protocol (pinned)

**Training and export** (verbatim from
`scripts/platonic_rh_alignment.py` at v1.9.0):
- MLP: `torch.manual_seed(s)`; `MockNerve(n_wmls=2, k=1, seed=s)`
  with gamma phase active; `HardFlowProxyTask(dim=16, n_classes=12,
  seed=s)`; `MlpWML(id=0, d_hidden=16, seed=s)`;
  `train_wml_on_task(steps=800, lr=1e-2)` (Adam, batch 64).
- LIF: `torch.manual_seed(s)`; same task; `LifWML(id=0,
  n_neurons=16, seed=s + 10)`; `input_encoder = Linear(16, 16)`;
  Adam lr 1e-2, 800 steps, batch 64, surrogate-spike readout.
- Null pair: identical constructor sequence and seeds, **0** steps.
- **Layer:** pre-VQ continuous codes. MLP `mlp.core(x)`; LIF
  `lif.input_proj(lif_encoder(x))` (d = 16 each). Post-VQ codes are
  not analysed (Paper 1 §7.4 flags their ≈ 10× amplification).
- **Inputs:** `HardFlowProxyTask(dim=16, n_classes=12,
  seed=s).sample(batch=1024)` on a fresh task object, as in v1.9.0.
  The same 1024 inputs feed the trained and null pairs of seed s.
- Exported matrices (float32, 1024 × 16) are saved with a `.sha256`
  next to each dump before any statistic is computed.

**Preprocessing (primary, fixed).** L2-normalise each row; use
Euclidean distance on the result. No centering. This matches the
v1.9.0 cosine mKNN (Euclidean on unit rows ranks neighbours exactly
as cosine similarity) and You et al.'s L2-normalised supports.
Sensitivity pipelines (reported only, never change a verdict):
(i) centre each feature over the 1024 inputs, then L2-normalise
(Bangachev et al.); (ii) raw features.

**Statistics** (n = 1024 inputs, k = 10):
- *S_mKNN* (You et al. Eq. 1; identical to the v1.9.0 kernel):
  `(1/n) Σ_i |N_k^X(i) ∩ N_k^Y(i)| / k`, self excluded. Neighbour
  ties are broken by lower input index (stable sort on distance).
- *S_H0* (You et al. Eq. 2): `|T_X ∩ T_Y| / (n − 1)`, where T is the
  minimum-spanning-tree edge set (= the H₀ death-edge set of the
  Vietoris-Rips filtration) of the Euclidean distance matrix,
  computed with Prim's algorithm from input 0, updating a parent only
  on a strictly smaller weight and, among vertices with equal
  minimum key, adding the lowest index next (You et al.'s
  deterministic tie rule, their Appendix C.3).
  Edges are unordered pairs. Normalised by n − 1, not Jaccard.
- *Geometry weights* (You et al. Eq. 3): per representation,
  `D̃ = D / Q_0.9({D_ij : D_ij > 0})`;
  `w_ij = exp(−|log D̃_X(i,j) − log D̃_Y(i,j)| / τ)`.
  Zero distances (duplicate rows after normalisation): `w_ij = 1` if
  `D_X(i,j) = D_Y(i,j) = 0`, and `w_ij = 0` if exactly one is zero.
  The number of zero-distance pairs per matrix is reported.
- *S_mKNN^τ* (Eq. 4): `(1/n) Σ_i (1/k) Σ_{j ∈ N_k^X(i) ∩ N_k^Y(i)}
  w_ij`.
- *S_H0^τ* (Eq. 5): `(1/(n − 1)) Σ_{(i,j) ∈ T_X ∩ T_Y} w_ij`.
- **τ = 0.1** for the confirmatory geometry statistics (fixed here,
  the middle of You et al.'s grid). The full grid
  τ ∈ {1, 0.1, 0.01, 0.001, 0.0001} is reported as a sensitivity
  sweep. As τ → ∞ the geometry scores recover the structure scores.
- Sensitivity (reported, not tested): k ∈ {5, 20}; linear CKA and
  the Mantel-style Pearson correlation of full distance matrices.

**Departures from You et al., and why.**
1. *Null.* You et al. calibrate with a permutation null over sample
   correspondences and a rescaled score
   `max{(T_obs − c_α)/(1 − c_α), 0}`. The claim under test (Paper 1
   §8.4) is about convergence beyond what architecture alone
   induces, which is Gröger et al.'s capacity-matched untrained null.
   That null is primary here. Their permutation calibration
   (K = 500, same statistic, their T_cal) is computed and reported as
   a secondary analysis.
2. *Layer selection.* They take a maximum over layer pairs. Here
   there is one pre-registered layer per substrate, so no maximum.
3. *Clipping.* Their 95th-percentile clipping targets outlier
   activations in transformer features; it is not applied to 16-d
   codes (reported as sensitivity only if any |feature| exceeds 10×
   its 95th percentile).
4. *Riemannian variant.* Not used.

## §6 Secondary analyses (reported, not tested)

- Permutation null and You et al.'s T_cal (§5, departure 1).
- Direct v1.9.0 replication: script unchanged, seeds 0-2, Gaussian
  baseline, reported next to Paper 1's 0.174 / 18.8×.
- Sensitivity pipelines, k sweep, τ sweep, CKA, Mantel.

## §7 Outcome rules — DR-3 anchor wording

Rows are evaluated top to bottom; the **first matching row**
applies. Every combination of states reaches a row.

| # | Condition | DR-3 §8.4 anchor | FC / EC |
|---|---|---|---|
| 0 | Abort (§4) | No reading; amend. | n/a |
| 1 | A = SURVIVES ∧ B = SURVIVES ∧ C = FAILS ∧ D = FAILS | Re-anchor to **"null-calibrated relational structure (local and global)"**, adding that the calibrated gain in distance-weighted overlap is bounded below δ at both scales (wording: "structure survives calibration; distance-weighted agreement adds less than δ"). The stronger phrase "not metric geometry" is not used unless the exploratory `S^τ / S` analysis agrees, and then only as a hedged reading. Prose change in Paper 1 §8.4 (EN + FR, same PR). | unchanged |
| 2 | A = SURVIVES ∧ B = SURVIVES ∧ (C = SURVIVES ∨ D = SURVIVES) | Structure survives at both scales and some distance-weighted overlap gain also survives. Re-anchor to **"null-calibrated relational structure (local and global)"** with no statement about geometry. | unchanged |
| 3 | A = SURVIVES ∧ B = FAILS | Local calibrated gain detected; global calibrated gain bounded below δ = 0.025. The current "local, not global" anchor is kept, qualified as "global calibrated gain below 0.025". No local-versus-global difference is tested. | unchanged |
| 4 | A = FAILS | Local calibrated gain bounded below δ = 0.025 under this protocol: the §7.4 local claim is not supported at that margin. This does not disprove a positive alignment or the historical Gaussian-baseline figure. File a dated note in `docs/milestones/` and qualify Paper 1 §7.4 / §8.4 accordingly. | unchanged |
| 5 | Any other combination (some needed state is INDETERMINATE) | Inconclusive at n = 20. Anchor wording unchanged; Paper 1 §8.4 gains one sentence citing the probe as inconclusive, with the four states. | unchanged |

The secondary G5 pair (MLX MLP vs E-SNN, penultimate layer,
Split-FMNIST test inputs, same statistics and preprocessing) is
**exploratory** and cannot change the anchor on its own.

## §8 Reporting commitment and stopping rule

- **Fixed n = 20 seeds, no optional stopping.** No statistic is
  computed on any trained pair before all 20 seeds and their nulls
  are dumped and hashed.
- All four statistics, all states, L / U, both nulls, all
  sensitivity pipelines and sweeps are reported regardless of
  outcome.

## §9 Audit trail and deviations

- **Filing gate.** This draft is not filed until the export helper
  and `scripts/dr3_relational_probe.py` exist at a pinned commit with
  passing tests (below), the learning gate has been run on one smoke
  seed (seed 900, not analysed) to confirm trained substrates pass
  it and untrained ones are unaffected, and the decision code has
  been run on synthetic D vectors covering every §2 state and §7 row.
- New script `scripts/dr3_relational_probe.py` (to be written and
  unit-tested before filing: mKNN on a hand-built example, Prim tie
  rule, H₀ normalisation, τ → ∞ limit equals the structure score).
  It consumes dumps exported from `nerve-wml` at v1.9.0 by an export
  helper that imports the v1.9.0 training functions unchanged.
- Milestones: `docs/milestones/dr3-relational-probe-<date>.{json,md}`.
- Results are appended to `docs/proofs/dr3-substrate-evidence.md` as
  a new dated section (append-only).
- Deviations: an amendment commit before the run, or post-hoc
  disclosure marking the affected hypothesis exploratory (per
  `docs/osf-preregistration-draft.md` §5).

## §10 Open decisions for the PI (resolve before filing)

The protocol above is complete with these defaults; changing one
requires editing the named section before the lock commit.

Decided by the PI on 2026-10-23 and no longer open: δ = 0.025;
n = 20 seeds.

1. **τ (§5).** 0.1 (default) or another single value from You et
   al.'s grid.
