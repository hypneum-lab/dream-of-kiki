# DR-3 global relational probe pre-registration

**Status:** DRAFT — not yet filed on OSF (to be filed by the PI).
**Date:** 2026-10-23 (revision 3: PI decisions of 2026-10-23 on δ
and n applied; revision 4, 2026-10-24: pre-filing code status, no
protocol change; revision 5, 2026-10-24: export helper added per the
PI decision of 2026-10-24, no protocol change; revision 6,
2026-10-24: PI decisions on τ and on c_α of the permutation
secondary, decision-eligibility rules and the "global" wording
guard after an independent review)
**Parent OSF:** 10.17605/OSF.IO/Q6JYN
**c_version at drafting:** C-v0.25.0+PARTIAL
**Claim under test:** the DR-3 anchoring sentence in Paper 1 §8.4
("DR-3 is therefore anchored to local, not global, representational
convergence"), and the mutual-kNN figure it rests on (Paper 1 §7.4:
mutual-kNN = 0.174 at k = 10, N = 1024, 3 seeds, 18.8× a random
baseline; `nerve-wml` v1.9.0).
**Primary substrate pair:** `MlpWML` vs `LifWML` from `nerve-wml`
tag `v1.9.0` (commit `95ac6b78477c3ce999b31bca775769041e082b46`;
the annotated tag is public on `github.com/hypneum-lab/nerve-wml`
and resolves to that commit, verified 2026-10-24), reproduced through
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

Three further results bound what the probe can claim:

- Shi, Han, Li, Thompson & Hernández-Lobato (2026), *From Plato to
  Aristotle: Toward a General Theory of Representation Hypotheses*
  (arXiv:2610.20286). Places neighbour overlap, spanning-tree overlap
  and distance-based criteria on one lattice, and states that You et
  al.'s spanning-tree overlap mostly repeats one-nearest-neighbour
  overlap. A spanning-tree score such as S_H0 therefore cannot, on
  its own, establish a *global* structure that is independent of
  local 1-NN structure (see the comparator in §5 and the wording
  rule in §7). The same paper argues that the loss of global
  convergence reported elsewhere can be an artefact of calibrating a
  maximum over layers; this probe uses one pre-registered layer
  (§5, departure 2).
- Koepke, Zverev, Ginosar & Efros, *Back into Plato's Cave*
  (arXiv:2604.18572). Mutual-kNN on a small sample captures only
  coarse structure (k must grow with n) and bijective input pairing
  inflates alignment. See the scope limits in §5.
- Gröger et al. (arXiv:2602.14486) calibrate with a permutation of
  sample correspondences; see the attribution note in §5.

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
- **Export helper (PI decision, 2026-10-24).**
  `scripts/dr3_export_nerve_wml.py` in this repository is a
  standalone script run with the Python environment of a nerve-wml
  checkout at `v1.9.0`; dream-of-kiki gains no dependency (no torch,
  no nerve-wml). It refuses to run unless the checkout's `HEAD` is
  `95ac6b78477c3ce999b31bca775769041e082b46` with no modified tracked
  file, checks that the imported modules come from that checkout, and
  calls the v1.9.0 `_train_mlp` / `_train_lif` of
  `scripts/platonic_rh_alignment.py` unchanged (steps 800 for the
  trained pair, 0 for the null pair). Invocation:

  ```
  git clone https://github.com/hypneum-lab/nerve-wml.git nwml-v1.9.0
  cd nwml-v1.9.0 && git checkout --detach v1.9.0
  git rev-parse HEAD  # 95ac6b78477c3ce999b31bca775769041e082b46
  uv sync --frozen
  uv run python <dream-of-kiki>/scripts/dr3_export_nerve_wml.py \
      --seeds 100-119 --out <dumps>
  uv run python <dream-of-kiki>/scripts/dr3_export_nerve_wml.py \
      --seeds 900 --out <smoke-dumps>   # §9 smoke, never analysed
  ```

  Outputs (never overwritten): one `dr3-seed<s>.npz` per seed (`seed`,
  `a_trained`, `b_trained`, `a_null`, `b_null`, the four learning-gate
  accuracies of §4 and `gate_pass`), a `dr3-seed<s>.npz.sha256`
  sidecar (`<hex>  <file>`), and `export-manifest.json` with its own
  `.sha256`: tag, commit, clean-tree flag, Python / numpy / torch
  versions, protocol constants and, per seed, the file SHA-256, the
  SHA-256 of each array's float32 bytes, the accuracies and the gate
  outcome. The probe is then run on the dumps with
  `--require-hash --manifest <dumps>/export-manifest.json --decide`
  (plus `--permutation-null` for the §6 secondary); it recomputes the
  gate from the accuracies and excludes failing seeds, so gate
  failures count towards the §4 abort, and it applies the
  eligibility rules of §8.

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
- **τ = 0.1** for the confirmatory geometry statistics (decided by
  the PI on 2026-10-24; the middle of You et al.'s grid). The full grid
  τ ∈ {1, 0.1, 0.01, 0.001, 0.0001} is reported as a sensitivity
  sweep. As τ → ∞ the geometry scores recover the structure scores.
- Sensitivity (reported, not tested): k ∈ {5, 20}; linear CKA and
  the Mantel-style Pearson correlation of full distance matrices.

**1-NN comparator for S_H0 (exploratory, descriptive).** Motivated
by Shi et al. (arXiv:2610.20286). Computed on the same dumps, with
the same preprocessing, the same trained / null pairing and the same
calibrated difference D_s = trained − null as above:
- *S_mKNN(k=1)*: `(1/n) Σ_i 1[nn_X(i) = nn_Y(i)]`, with the same
  tie rule (lower index).
- *S_H0^res* (S_H0 beyond 1-NN edges): let `E_NN(R)` be the set of
  unordered pairs `{i, nn_R(i)}` for representation R. Then
  `S_H0^res = |(T_X ∩ T_Y) \ (E_NN(X) ∪ E_NN(Y))| / (n − 1)`, i.e.
  the shared MST edges that are 1-NN edges in neither representation.
These two scores are **not** members of the Bonferroni family of §2,
spend no α, carry no SURVIVES / FAILS state, and cannot change any
state in §2. They are reported as the mean of D with a two-sided 95 %
paired-t interval, as description only. They add no confirmatory
test, so no power computation is changed.

**Scope limits (Koepke et al., arXiv:2604.18572).** n = 1024 inputs
and k = 10 capture coarse neighbourhood structure only, and the
pairing is bijective by construction (both substrates see the same
inputs), which inflates absolute alignment. Results describe coarse
neighbourhood structure at n = 1024; the k ∈ {5, 20} sensitivity
above is the only check on neighbourhood size, and gallery density
is not varied. The bijective pairing is inherent to a
same-input comparison and is named here as a scope limit, not
corrected.

**Departures from You et al., and why.**
1. *Null.* You et al. calibrate with a permutation null over sample
   correspondences and a rescaled score
   `max{(T_obs − c_α)/(1 − c_α), 0}`. Gröger et al.
   (arXiv:2602.14486) also use a permutation null over sample
   correspondences (plus synthetic random matrices); neither paper
   uses an untrained network as the null. The matched **untrained-pair
   null is this study's own choice**, made because the claim under
   test (Paper 1 §8.4) is about convergence beyond what architecture
   alone induces; it is not attributed to Gröger et al. or You et
   al. That null is primary here. The permutation calibration
   (K = 500, same statistic, You et al.'s T_cal; the
   Gröger/You-comparable form) is computed and reported as a
   secondary analysis. **Specification (PI decision 2026-10-24).**
   Per seed and per pair (trained, and null for reference), each of
   K = 500 draws permutes the rows of substrate B's codes (the sample
   correspondence; substrate A is never permuted) and recomputes the
   same statistic under the primary protocol (l2, k = 10, τ = 0.1).
   c_α is the (1 − α) quantile, **α = 0.05 (95th percentile)**, of the
   pool {T_obs, T_1, …, T_K}, taken as the order statistic of rank
   ⌈(1 − α)(K + 1)⌉ (rank 476 of 501); T_cal = max{(T_obs − c_α) /
   (1 − c_α), 0} (0 if c_α = 1). The add-one permutation p-value is
   reported too. Fixed stream `default_rng([seed, 2609])`
   (`[seed + 1, 2609]` for the null pair). Reported only; it never
   changes a §2 state, the §7 row or the wording rule.
   **Check against You et al. (arXiv:2609.27252, v1 HTML, read
   2026-10-24).** §3.4, Eqs. 7-8 (verbatim): "We define cα as the
   (1−α) quantile of {Tobs,T(1),…,T(K)}", and the calibrated score is
   max{(T_obs − c_α)/(1 − c_α), 0}; the sample correspondences of the
   second model are permuted. §4.1 (verbatim): "calibrated using 500
   correspondence permutations at α=0.05". So c_α at α = 0.05 (the
   95th percentile of the pool) and K = 500 match the paper. Their
   appendix B.1 reportedly states the empirical right-tail quantile
   at rank ⌈(1 − α)(K + 1)⌉; that wording was seen only in a summary
   of the HTML and must be checked against the PDF / TeX source
   before it is quoted. Implementation:
   `scripts/dr3_relational_probe.py --permutation-null` (`calibrate`,
   `permutation_null`).
2. *Layer selection.* They take a maximum over layer pairs. Here
   there is one pre-registered layer per substrate, so no maximum.
3. *Clipping.* Their 95th-percentile clipping targets outlier
   activations in transformer features; it is not applied to 16-d
   codes (reported as sensitivity only if any |feature| exceeds 10×
   its 95th percentile).
4. *Riemannian variant.* Not used.

## §6 Secondary analyses (reported, not tested)

- Permutation null and You et al.'s T_cal (§5, departure 1).
- 1-NN comparator for S_H0 (S_mKNN(k=1), S_H0^res; §5), descriptive.
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

**Wording rule for "global" (applies to rows 1 and 2).** H13-B
SURVIVES establishes a positive calibrated S_H0 gain; S_H0 shares
most of its edges with 1-NN overlap (Shi et al.,
arXiv:2610.20286). The words "global" or "(local and global)" are
therefore used in the new anchor **only if** the descriptive S_H0^res
of §5 shows a point estimate of mean(D) ≥ δ and a 95 % interval that
excludes 0. Otherwise rows 1 and 2 are written with "relational
structure, in which the global (spanning-tree) score is not
separable from one-nearest-neighbour overlap" in place of "(local
and global)". This is a wording rule applied to prose only; it is
not a fifth confirmatory test and changes no state. The decision
code allows "global" only when the run is eligible (§8), not
aborted, the row is 1 or 2, and the S_H0^res criterion holds
(`global_wording_allowed`); every other row never uses it.

The secondary G5 pair (MLX MLP vs E-SNN, penultimate layer,
Split-FMNIST test inputs, same statistics and preprocessing) is
**exploratory** and cannot change the anchor on its own.

## §8 Reporting commitment and stopping rule

- **Decision eligibility.** A confirmatory §2 state and §7 row are
  written only when all of the following hold; otherwise the same
  reading is reported under `exploratory` and the confirmatory
  fields are empty: (i) the dumps are exactly the unique seeds
  100..119; (ii) every dump has a verified `.sha256` sidecar
  (`--require-hash`) and matches the export manifest, whose own hash
  is verified (`--manifest`); (iii) the manifest and every dump come
  from the pinned v1.9.0 checkout (`confirmatory_eligible`, `pinned`;
  an `--allow-unpinned` export is never eligible); (iv) the primary
  protocol (k = 10, τ = 0.1, l2; `--decide` refuses other k or τ);
  (v) every array has the registered shape 1024 × 16; (vi) every
  seed carries finite trained-substrate accuracies. The learning gate
  is recomputed from those accuracies; a supplied `gate_pass` is not
  trusted, and a disagreement is flagged.
- **Fixed n = 20 seeds, no optional stopping.** No statistic is
  computed on any trained pair before all 20 seeds and their nulls
  are dumped and hashed.
- All four statistics, all states, L / U, both nulls, all
  sensitivity pipelines and sweeps are reported regardless of
  outcome.

## §9 Audit trail and deviations

- **Filing gate.** This draft is not filed until the export helper
  (`scripts/dr3_export_nerve_wml.py`, §5)
  and `scripts/dr3_relational_probe.py` exist at a pinned commit with
  passing tests (below), the learning gate has been run on one smoke
  seed (seed 900, not analysed) to confirm trained substrates pass
  it and untrained ones are unaffected, and the decision code has
  been run on synthetic D vectors covering every §2 state and §7 row.
- New script `scripts/dr3_relational_probe.py` (to be written and
  unit-tested before filing: mKNN on a hand-built example, Prim tie
  rule, H₀ normalisation, τ → ∞ limit equals the structure score,
  S_mKNN(k=1) and S_H0^res on a hand-built example).
  It consumes dumps exported from `nerve-wml` at v1.9.0 by the export
  helper `scripts/dr3_export_nerve_wml.py`, which imports the v1.9.0
  training functions unchanged.
- **Status 2026-10-24.** Done: `kiki_oniric.eval.relational_probe`
  and `scripts/dr3_relational_probe.py` (PR #43, review APPROVE) with
  the listed unit tests (hand-built mKNN, Prim tie rule, H₀
  normalisation, τ → ∞ limit); the 1-NN comparator
  (`nn_comparator`: S_mKNN(k=1), S_H0^res, hand-built tests) and the
  decision code (`--decide`: §2 states, §7 row, "global" wording
  rule; tests reach every state and row on synthetic D vectors), both
  in this PR; the export helper `scripts/dr3_export_nerve_wml.py`
  (§5; PI decision 2026-10-24: standalone, run inside a v1.9.0
  checkout, no new dependency here), with tests in
  `tests/unit/scripts/test_dr3_export_nerve_wml.py` that replace
  nerve-wml by a numpy fake (dump contract read back by the probe,
  same inputs and seeds for the trained and null pairs, gate at 0.25
  on the trained substrates only and its exclusion in the probe,
  SHA-256 sidecars and manifest, refusal of an unpinned or modified
  checkout, no overwrite). Integration check (not the filing smoke):
  the helper was run once on seed 900 in a clean v1.9.0 clone on
  macM1 with torch 2.11.0 (the version in the v1.9.0 `uv.lock`) from
  an existing environment rather than `uv sync --frozen`; the trained
  substrates passed the gate (MLP 0.519, LIF 0.545), the untrained
  ones stayed near the 0.083 chance level (0.104, 0.087), and a rerun
  gave identical array hashes. Also done (2026-10-24): the
  permutation-null secondary (§5, departure 1; tests of the order
  statistic, rescaling, clipping and the fixed stream), the
  decision-eligibility checks of §8 with negative tests (missing
  manifest or gate, incomplete seeds, unpinned export, non-primary
  k / τ, wrong shape, mismatched hashes; end to end from the export
  helper), the recomputed gate, and the restricted "global" wording
  rule (tests for abort, ineligible runs and every row). Open: the
  filing smoke on seed 900 with the documented invocation on the
  filing host. The v1.9.0 tag is public on GitHub (verified
  2026-10-24).
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
n = 20 seeds. Decided on 2026-10-24: the export helper is a
standalone script run inside a nerve-wml v1.9.0 checkout, with no
new dependency in this repository (§5).

1. ~~τ (§5)~~ — decided by the PI on 2026-10-24: τ = 0.1.
   (Permutation secondary c_α: decided on 2026-10-24, α = 0.05,
   95th percentile; see §5, departure 1.)
