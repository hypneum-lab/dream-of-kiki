# Pre-registration (L0, draft) — DOWNSCALE dose-response: quasi-symmetry (C1) and temperature (C2)

**Project** : dreamOfkiki (Hypneum Lab)
**Parents** : E1-small controls, `docs/osf-addendum-e1-small-controls-2026-11-06.md`
  section 10 (results of 2026-11-08) ; roadmap v2.2, second round of
  2026-11-10 (`hypneum-lab/.github` PR #45), candidates C1 and C2
**PI** : Clément Saillant — decision of 2026-11-10 13:45 : C1 goes
  first on the LLM dream axis ; no replay of H14 at a larger scale
  before C1 has shown at which dose DOWNSCALE acts
**Level** : **L0** : local draft, not frozen, not deposited. No tag.
  Nothing below has been run.
**Revision** : 2 (2026-11-11), after the critique of revision 1 asked
  by the PI (section 8). PI ruling of 2026-11-11 11:09 : the reading
  gate does not apply to this one-run inference test.
**Date drafted** : 2026-11-11
**Catalogue** : C1 and C2 are to be entered from NH-114 (not done yet)

## 1. Question

In the E1-small controls (564 cells, Qwen3.5-4B), replay with
DOWNSCALE (all floating-point parameters multiplied by 0.99) and
replay alone give the same accuracy (0.3357 to 0.3991 in both arms).
A descriptive read of the same cells on 2026-11-10 shows that
DOWNSCALE moves the weights about 24 times more than replay does
(`norm_change` 8.36 against 0.35) while the divergence of the answers
is the same (`kl_ref` 0.134 against 0.137), and that a random
perturbation of the same norm changes nothing either. Two readings
fit: (a) at f = 0.99 every perturbation is below the sensitivity of
the model, whatever its direction ; (b) a uniform rescaling is close
to a symmetry of a normalised network, at any dose. Reading (b) would
make a replay of H14 on a larger model uninformative by construction.

**Statement C1 (EN, to lock, revision 2)** : On the base Qwen3.5-4B
model, with no replay and no training, the largest relative dose of
DOWNSCALE that leaves multiple-choice accuracy equivalent to the
unperturbed model is at least two grid steps (a factor 4 to 5) above
the largest tolerated dose of a random perturbation that has the same
norm in every tensor.

**Statement C2 (EN, to lock, revision 2)** : At its largest tolerated
dose, the effect of DOWNSCALE on the option distribution is that of a
single temperature T > 1 applied to the unperturbed scores.

## 2. What is already known (reading note of 2026-11-11, to complete before any freeze)

Read in full on 2026-11-11 (library, `references/notes/arxiv-2407.01800.md`,
depth stated in the note) : Lyle, Zheng, Khetarpal, Martens, van Hasselt,
Pascanu, Dabney 2024, « Normalization and effective learning rates in
reinforcement learning », arXiv:2407.01800 v1. What that text establishes,
with its pages :

- A normalisation layer makes the sub-network that feeds it invariant to
  the scale of its weights : f(c theta, x) = f(theta, x) for every c > 0,
  hence grad f(c theta) = grad f(theta) / c (section 2.2, p. 3). Rescaling
  those weights to their initial norm, direction unchanged, « does not
  affect the output » of the scale-invariant model (p. 5) ; the per-layer
  statement is proved for a network that ends with a normalisation
  (eq. 19, p. 18).
- The architecture for which this holds is built for it : a normalisation
  before every non-linearity, bias terms removed (p. 6), scale and offset
  parameters handled apart (pp. 5-6 ; A.8 ; D, p. 25) ; for attention the
  authors add an RMS normalisation on the query and key projections
  (eq. 13, p. 17). For non-homogeneous activations the joint
  scale-offset projection « will not hold » (p. 25).
- The quantity a rescaling does change is the effective learning rate of
  the training that follows : eta / ||theta||^2 for gradient descent,
  eta / ||theta|| for fixed-norm steps (definition 1, p. 3). That is the
  variable of candidate A1, not of this test.
- All experiments train from scratch with the projection on (synthetic
  non-stationarity ; Atari ; a 400 M transformer on C4, table 3, p. 25).
  No rescaling of an already trained model is tested. The paper therefore
  does not predict the result of this test ; the sentence of the watch of
  2026-11-11, « the null of E1-small on DOWNSCALE was predictable from
  published scale invariance », was too strong and is withdrawn here.

What follows for Qwen3.5-4B, a pre-LN transformer with RMSNorm (reader's
inferences, to check against the model configuration and not to assume) :

- A uniform rescaling of **all** parameters by f = 1 - eps is not an exact
  symmetry. Each RMSNorm removes the scale of its input, so what survives
  are the factors carried by the parameters that sit after a
  normalisation : the normalisation gains, the linear maps and the output
  head. With a gain followed by a linear map, each block input is
  multiplied by f^2 ; the output logits by f^2 as well (final gain times
  head). If f^2 on the logits were the only effect, the option
  distribution of D(eps) would be exactly the base distribution at
  temperature T = 1 / f^2 : 1.02 at eps = 0.01, 1.11 at 0.05, 1.23 at
  0.1, 1.56 at 0.2, 4 at 0.5. **C2 therefore has a quantitative
  expectation**, written here before the run : the fitted T is reported
  against 1 / f^2 at every dose ; a T that tracks 1 / f^2 means the head
  dominates, a T that departs from it means the attention or the MLP
  blocks contribute.
- **What the snapshot actually is** (read on 2026-11-11 from
  `config.json` and the safetensors headers of snapshot `851bf6e8`,
  kxkm-ai) : `Qwen3_5ForConditionalGeneration`, a **hybrid** text
  model of 32 layers, 24 of them `linear_attention` (Gated DeltaNet :
  `in_proj_qkv`, `in_proj_z`, gate projections `in_proj_a`, `in_proj_b`,
  a depthwise `conv1d` kernel of width 4, decay parameters `A_log`
  (float32, 32 per layer) and `dt_bias` (32 per layer), an RMS `norm`
  in float32) and 8 `full_attention` with per-head `q_norm` and
  `k_norm` and no bias ; tied embeddings (248 320 x 2 560) ; 4.21 G
  parameters in the language model, plus a **visual tower of 0.33 G**
  (`model.visual.*`, with biases) and a multi-token-prediction head
  (`mtp.*`, 0.13 G) in the checkpoint. 738 tensors, 690 in bf16 and 48
  in float32.
- Consequences for the symmetry argument. In the 8 full-attention
  layers the logits scale as f^2 (gain of `q_norm` times gain of
  `k_norm`), so attention flattens with the dose : not a temperature on
  the output, first candidate for a C2 refutation at large doses. In
  the 24 linear-attention layers the rescaling is **not a scale at
  all** for three groups of parameters : the decays (`A_log` enters as
  -exp(A_log), so f changes every time constant by the factor
  exp((f - 1) A_log) ; `dt_bias` shifts a softplus), the gates
  (`in_proj_a`, `in_proj_b` feed sigmoids and softplus, whose sharpness
  changes with f) and the convolution followed by SiLU. The MLP output
  scales as f^3 against an embedding scaled by f. Every one of these
  effects grows with eps ; none is a symmetry. This is the mechanism by
  which C1 can be refuted : it is written here so that a refutation is
  readable, not a surprise.
- The registered backend applies `downscale` and the noise to every
  floating-point tensor of `model.parameters()`. **To check before a
  freeze** : whether the loaded model carries the visual tower and the
  MTP head ; if it does, they receive the operation without any effect
  on the text scores, and the global noise arm `Rg` spends part of its
  norm budget on them. The per-tensor arm `Rt` is immune by
  construction ; the norm report of section 3 is split by tensor group
  (language matrices, gains, decays and gates, embedding, visual, MTP)
  so that the share is visible.
- Hence DOWNSCALE is not expected to be harmless at every dose ; the
  claim of C1 is only that it is tolerated at much larger doses than a
  perturbation of equal norm and random direction, and section 4 reads
  it as a ratio of tolerance doses.

**Tautology check.** C1 is a control of the instrument, not a discovery
(PI, 2026-11-10) : what it establishes is the dose at which the
registered harness stops seeing DOWNSCALE as inert, against the dose at
which it stops seeing noise as inert ; neither dose is in the
literature for this model. C2 « supported » is expected for the head
alone ; the informative outcome is whether one scalar explains the
whole change of the option distribution (section 4, 80 % of the KL).

The reading gate does not apply to this test (PI, 2026-11-11). The two
references the previous version of this section cited without reading
them were **read in full on 2026-11-11 (evening)**, notes in the library :

- van Laarhoven 2017 (arXiv:1706.05350 v1, `references/notes/arxiv-1706.05350.md`).
  Fact : the invariance y(X ; a w, gamma, beta) = y(X ; w, gamma, beta) is
  stated for batch, weight and layer normalisation, for the incoming
  weights w of a unit whose output is normalised, gains and offsets held
  fixed (p. 2) ; it is read off the formulas, not proved as a theorem.
  Fact : effective learning rate eta / ||w||^2 for SGD and momentum
  (pp. 3-4), eta / ||w|| for RMSProp and Adam (p. 5, table 1). Nothing
  on rescaling a trained network ; gains, offsets, biases, head and
  embeddings not treated (gains said inert under ReLU except in the
  last layer, p. 8). Confirms the sentence « the invariance holds for the
  weights that feed a normalisation », with that scope only.
- Arora, Li, Lyu 2018 (arXiv:1812.03981 v1, ICLR 2019,
  `references/notes/arxiv-1812.03981.md`). Fact : invariance is defined
  **parameter by parameter, the others held fixed** (def. 2.1, p. 3) ;
  the rows of a matrix feeding a batch normalisation are invariant
  (p. 4) ; gains and biases are scale-variant for a general activation
  (p. 4, p. 20) ; the network analysed has no parameter after its last
  normalisation. Fact : eta / ||w||^2 is proved for GD and SGD
  (lemma 2.4, thm 2.5, p. 6) and the norm of an invariant parameter grows
  monotonically, ||w_{t+1}||^2 = ||w_t||^2 + eta^2 ||grad||^2, without
  momentum or weight decay. Fact : theorems 3.1 and 4.2 give convergence
  at the best asymptotic rate for **any** fixed rate of the invariant
  parameters once the rate of the non-invariant ones is tuned ; a change
  of norm only changes constants. No rescaling of a trained model
  anywhere (closest : projected SGD, appendix D, p. 20). Reader's
  inference : in the sense of this text a uniform rescaling of **all**
  parameters is not a symmetry, which is what section 2 above says of
  Qwen3.5-4B ; and for A1 the theory predicts a transient effect of a
  norm reduction, not a lasting gain.

**Tautology check, revised after these readings.** Neither text predicts
C1 or C2 on a trained pre-LN hybrid model ; both make the head factor
f^2 and the per-head attention factor f^2 (q_norm times k_norm)
expected, and both leave the linear-attention decays and gates outside
any invariance. The C2 expectation T = 1 / f^2 stands as written.

## 3. Design

- **Model** : `Qwen/Qwen3.5-4B`, the snapshot and the instrument of
  E1-small (`scripts/e1_harness.py`, registered torch backend,
  determinism settings unchanged). Base weights restored from the RAM
  snapshot before each arm ; the restore is checked by re-scoring 50
  items. No replay, no optimiser step.
- **Items** : the CONFIRMATORY items of the FROZEN E1 corpus, 1 098
  items : `mmlu_cf` 380, `realtimeqa` 380, `daily_oracle` 338.
  **Primary set : `mmlu_cf` + `realtimeqa`, 760 items.**
  `daily_oracle` is reported apart (base accuracy 0.179 with the
  question, under the 0.25 chance level, addendum section 10.6) : an
  exclusion made after seeing data, declared as such ; the 1 098-item
  table is reported next to the primary.
- **Doses** : relative norm eps = 1 - f on a log grid
  {0.01, 0.02, 0.05, 0.1, 0.2, 0.5}.
- **Arms at each dose** :
  - `D(eps)` : `Backend.downscale(1 - eps)`, deterministic, one pass ;
  - `Rt(eps, s)` : **per-tensor** random perturbation, each
    floating-point tensor t receives Gaussian noise of norm
    `eps * ||theta_t||`, seeds s = 0, 1, 2 (new variant, to write) ;
  - `Rg(eps)` : the global variant `perturb_same_norm(rho, "R", 0)` of
    the E1 controls, at eps = 0.01, 0.05, 0.2 only, one seed, for
    continuity ; descriptive ;
  - `base` : unperturbed model, one pass.
  28 scoring passes in total.
- **Measures per pass** : per-item option scores (kept) ; accuracy ;
  mean confidence ; `gap = mean confidence - accuracy` ; `choice_kl`
  to the base pass ; the norm actually applied, in total and per
  tensor group.
- **Seeds** : noise seeds 0, 1, 2 ; item bootstrap RNG 20261111,
  10 000 draws, items resampled, the same indices for every arm. For
  `Rt`, per-item correctness is averaged over the three seeds before
  the bootstrap ; per-seed curves are reported.

## 4. Decision rule (to fix before any run)

`acc(X)` is accuracy on the primary set. Two definitions :

- **Equivalent** at a dose : the 99 % paired bootstrap interval of
  `acc(arm) - acc(base)` lies inside [-0.03, +0.03].
- **Tolerance dose** of an arm : the largest grid dose such that the
  arm is equivalent at this dose and at every smaller one ; rank 0 if
  it is not equivalent at 0.01.
- **Broken** at a dose : `acc(base) - acc(arm)` is at least half of
  `acc(base) - 0.25` (the arm lost half of what the model has above
  chance), interval entirely above 0.

0. **Measurability gate** : `Rt` is broken at some grid dose and
   equivalent at 0.01. If `Rt` is never broken, or already not
   equivalent at 0.01, the grid does not bracket the sensitivity of
   the model : verdict `non_test` ; a new grid needs an amendment
   written before the new run.
1. **C1 supported** iff the tolerance dose of `D` is at least two grid
   steps above the tolerance dose of `Rt`.
2. **C1 refuted** iff the tolerance dose of `D` is equal to or below
   that of `Rt`.
3. One grid step apart : `inconclusive`.
4. **C2**, read at the tolerance dose of `D` (not applicable if its
   rank is 0 or if `choice_kl(base, D) < 0.005`, nothing to explain) :
   fit one temperature T on a log grid of 61 points in [0.25, 4] that
   minimises the mean KL between the option distribution of `D` and
   `softmax(scores_base / T)`. **Supported** iff the 99 % bootstrap
   interval of T is above 1 and the temperature removes at least 80 %
   of the KL (lower 99 % bound at least 60 %). **Refuted** iff the
   interval of T contains 1 or the share removed is under 50 %.
   Otherwise `inconclusive`. Mean confidence and `gap` are reported,
   not decided on.

The whole dose-response table is reported whatever the verdict.

## 5. What each verdict changes

- C1 supported : DOWNSCALE as implemented is close to a symmetry.
  H14 is not replayed at a larger scale ; DR-0 / DR-1 need an
  operation that is not a uniform rescaling.
- C1 refuted with both arms hurting alike : DOWNSCALE is a
  perturbation like any other ; H14 may be replayed only at a dose
  where the same-norm control is itself above the sensitivity
  threshold.
- `non_test` : the instrument cannot see perturbations of this size ;
  the dose grid is redesigned by amendment.
- C2 supported : the link to the calibration line (Zuse-2 `gap`) is
  worth its own pre-registration.

## 6. Cost, host, reading

Inference only : 28 passes over 1 098 four-option items on the 4B
model, kxkm-ai (RTX 4090) [duration to measure on the base pass, read
for its duration only]. No number is read before the 28 passes are
complete ; the verdict is computed by a script committed before the
run.

## 6 bis. Hopfield arm (S5) : a control where the answer is known

Added on 2026-11-11 by decision of the PI. Implemented and run the same
day ; see 6 ter for the outcome, which contradicts expectations 2 and 4
below. The text of this section is left as first written.

**Why.** On the language model, C1 and C2 are open questions. In a
Hopfield network they are textbook identities : the stationary law of
Glauber dynamics depends on the couplings J and the inverse
temperature beta only through the product beta * J. Multiplying J by
(1 - eps) at temperature T is therefore the same model as leaving J
alone at temperature T / (1 - eps) ; at T = 0 the update
`sign(J s)` does not see the scale at all. The arm does not test a
hypothesis about the lab's models. It tests **the decision rule of
section 4** on a system where the right verdict is known in advance.

**Model.** N = 1 000 binary units, Hebbian couplings
J = (1/N) * sum_mu xi_mu xi_mu^T with zero diagonal, random patterns,
loads alpha in {0.02, 0.05, 0.10}. Asynchronous Glauber dynamics,
P(s_i = +1) = sigmoid(2 * h_i / T), 50 sweeps from a cue at overlap
0.8 ; T in {0, 0.2, 0.4}. Measure : final overlap with the cued
pattern, 200 cues per cell. A self-contained numpy script in this
repository, independent of the NH-01 kernel of nerve-wml.

**Arms**, on the dose grid of section 3 :

- `D(eps)` : J multiplied by (1 - eps), T unchanged ;
- `Tq(eps)` : J unchanged, T divided by (1 - eps), same random stream
  as `D` ;
- `Rt(eps, s)` : J plus a symmetric Gaussian matrix of Frobenius norm
  eps * ||J||, zero diagonal, seeds 0, 1, 2 ;
- `base`.

**Expected, stated before any run.**

1. Identity check : with common random numbers, `D(eps)` and
   `Tq(eps)` give the same trajectories at every dose, up to
   floating-point tolerance. A failure is a bug, not a result.
2. At T = 0, `D` is equivalent to `base` at every dose of the grid
   and `Rt` is broken at some dose : the rule of section 4, with
   overlap in place of accuracy and the same +/- 0.03 margin, must
   return **C1 supported**.
3. At T = 0.2 and T = 0.4, the overlap under `D(eps)` must follow the
   base network read at T / (1 - eps) : the C2 analogue must return
   **supported**. Where T / (1 - eps) crosses the retrieval boundary
   of the load, `D` must be reported as broken, not as equivalent :
   in this model DOWNSCALE is harmless only while the rescaled
   temperature stays inside the retrieval phase.
4. Negative control : the `Rt` data relabelled as `D` must return
   **C1 refuted**.

**What each outcome changes.** If 2, 3 and 4 come out as stated, the
decision rule is shown to detect an exact symmetry and to reject a
non-symmetry at this grid and margin ; the language-model verdict can
then be read with that calibration. If the rule fails on 2, 3 or 4,
section 4 is amended **before** the language-model run, and the
amendment says so.

**Limits.** The arm says nothing about whether a transformer behaves
like this ; a known identity cannot support C1 or C2 on the language
model. The mapping from "option distribution" to "overlap" for the C2
analogue is a choice of this draft. Cost : CPU, minutes to an hour
[to measure].

## 6 ter. Control run of 2026-11-11 : two expectations of 6 bis failed

`scripts/hopfield_scale_control.py` was written and run once on
2026-11-11 (defaults of 6 bis, about 4 minutes of CPU, deterministic
seeds ; output sha256 `c8bc9a466a59`, not committed, reproducible by
`python -m scripts.hopfield_scale_control`). Read in full. Nine cells
(three loads, three temperatures).

| expectation of 6 bis | outcome |
|---|---|
| 1. `D(eps)` and `Tq(eps)` identical with common random numbers | **holds** : maximum absolute difference 0 in the nine cells |
| 2. at T = 0 the rule returns C1 supported | **fails** : `non_test` in the nine cells |
| 3. the C2 analogue returns supported at T > 0 | **mixed** : supported in 3 cells, refuted in 3 |
| 4. relabelled noise returns C1 refuted | **fails** : `non_test` (same gate) |

**Why 2 and 4 fail.** The gate of section 4 needs the noise arm to
break the model inside the dose grid. It never does : with Gaussian
noise of relative norm 0.5 on the couplings, retrieval at T = 0 stays
at 1.000, 1.000 and 0.989 for loads 0.02, 0.05 and 0.10. A mean-field
estimate made after the run says why : such noise acts like a load
multiplied by (1 + eps^2), far from capacity on this grid. The
expectation was wrong, not the identity : at T = 0, `D` is exactly
equal to `base` at every dose. With a grid capped at 0.5 and a noise
arm this weak, the rule cannot return "supported" on this model
whatever the truth.

**Why 3 is mixed.** The three refuted cells (load 0.05 at T = 0.4,
load 0.10 at T = 0.2 and 0.4) sit at or beyond the retrieval boundary,
where two independent random streams give overlaps that differ by
more than the +/- 0.03 margin although the two models are the same.
The cell at load 0.10 and T = 0.4 does not retrieve at baseline
(overlap 0.22) and should not have been in the grid.

**An observation not anticipated in 6 bis.** At T > 0 the rescaling
is tolerated **less** than noise of the same norm : tolerance rank of
`D` 4 or 5 against 5 or 6 for `Rt`, and `D` is broken at dose 0.5 in
four of the six cells (overlap 0.14, 0.09, 0.22, 0.08). In this model
a uniform reduction is a heating, and heating is what destroys
retrieval ; unstructured noise is averaged out. The direction is the
opposite of statement C1.

**Consequences, for the PI to decide.** Nothing is changed in
sections 3 and 4 by this note.

- The Hopfield arm validates the temperature identity only. It is not
  a positive control for the C1 rule.
- Options for a positive control of C1 : (a) a small network that is
  scale-invariant by construction (normalisation before every
  non-linearity, no bias) and that noise does break inside the grid ;
  (b) a Hopfield noise arm on a wider dose grid ; (c) no positive
  control, stated as a limit.
- The gate itself held : it refused to give a verdict where the noise
  arm did not bracket the sensitivity of the model. On the language
  model the same gate decides whether C1 is testable at all.
- If the C2 analogue is kept, it should use the common-random-number
  comparison or cells inside the retrieval phase only.

## 6 quater. Positive control on a small network, and a wider Hopfield noise arm

Decided by the PI on 2026-11-11 after 6 ter : both controls. Written
and run the same day, read in full.

**Small scale-invariant network** (`scripts/scale_invariant_control.py`,
numpy, no training loop : six random hidden layers, each followed by
an RMS normalisation and without bias, and a ridge readout ; 4
classes, 760 test items ; base accuracy 0.786). The verdict is
computed by the unmodified `analyse` of `scripts/e1_dose_response.py`.

| variant | what DOWNSCALE is, by construction | C1 | C2 | noise relabelled as D |
|---|---|---|---|---|
| no other parameter | exact no-op | **supported** (ranks 6 against 3) | not applicable (KL = 0) | C1 **refuted** |
| one scalar output gain | exact temperature 1 / (1 - eps) | **supported** (ranks 6 against 3) | **supported**, T = 2.0 at dose 0.5, all of the KL removed | C1 **refuted** |

The rule of section 4 therefore returns the right verdict on a known
symmetry, on a known temperature, and on a known non-symmetry, at this
grid and margin. Output sha256 `1f80f4687ddb`, reproducible by
`python -m scripts.scale_invariant_control`.

Two things were set by hand and are not outcomes. (i) The depth : at
depth 2 the per-tensor noise never breaks the network inside the grid
(accuracy 0.74 at dose 0.5) and the gate returns `non_test` ; depth 6
was kept after looking at five configurations, because the control
needs a noise arm that breaks the model. (ii) The gain variant is a
temperature because the gain is the only parameter outside a
normalisation ; a real language model has many such parameters.

**What this does and does not show.** It calibrates the rule. It says
nothing about the language model. It does show, twice now, that the
gate depends on depth and architecture : whether per-tensor noise
breaks Qwen3.5-4B inside the grid is an empirical question that the
first passes of the run will answer.

**Wider Hopfield noise arm** (`WIDE_NOISE_DOSES` = 1, 2, 4 in
`scripts/hopfield_scale_control.py`, noise arm only since DOWNSCALE is
undefined at eps >= 1 ; descriptive, output sha256 `23b3f22efc77`).
At T = 0, retrieval under noise of relative norm 1, 2 and 4 is 1.000,
0.999, 0.853 at load 0.02 ; 0.998, 0.896, 0.502 at load 0.05 ; 0.845,
0.502, 0.351 at load 0.10, where it is broken at dose 4 only. DOWNSCALE
at T = 0 leaves retrieval unchanged at every dose. So at T = 0 the
direction of C1 holds in the Hopfield model, but the noise must be two
to four times larger than the couplings themselves to break retrieval,
outside any grid shared with DOWNSCALE : the rank rule of section 4
cannot be applied to this model. At T > 0 the result of 6 ter stands :
the rescaling breaks retrieval at dose 0.5 while noise of the same
norm does not.

**A weakness of the rule made visible by these controls** (not
changed here). The tolerance of DOWNSCALE is censored at the top of
the grid (0.5, and 1 at most). If the noise arm is tolerated up to
the top of the grid as well, the two ranks are equal and the rule
would return "refuted" for an exact symmetry, were it not for the
gate. The gate is what prevents that false verdict ; it must not be
relaxed.

## 7. Not done, required before a freeze

- Written on 2026-11-11, tested on the tiny CPU model only, never run
  on the 4B model : the `dose-response` mode of `scripts/e1_harness.py`
  (28 passes, restore check on 50 items before each, one JSON line per
  pass, no score printed) ; `TorchBackend.perturb_per_tensor` ;
  `scripts/e1_dose_response.py`, which refuses an incomplete run and
  prints the verdict of section 4, tested on constructed tables (pure
  symmetry, same tolerance, one step apart, gate, temperature,
  non-temperature).
- Outside item sets, decided by the PI on 2026-11-11 (evening), all
  three rejected by the Zuse-2 dataset review and therefore outside its
  confirmatory sample : SciQ test (1 000 items, 4 options, CC-BY-NC-3.0),
  OpenBookQA test (500 items, 4 options, licence `unknown` on the Hub),
  CommonsenseQA validation (1 221 items, **5 options**, MIT ; chance level
  0.20, the harness and the « broken » rule of section 4 must take the
  number of options from the set). Each set is reported apart from the
  primary E1 set and is not part of the C1 verdict unless an amendment
  says so. Not pinned (revision, SHA-256), not implemented.
- Section 2 rewritten on 2026-11-11 from the reading note of Lyle et
  al. 2024 ; its two remaining references (van Laarhoven 2017, Arora, Li,
  Lyu 2018) **resolved and read the same evening** (section 2) ; the
  attention and head factors of section 2 checked against `config.json`
  of the snapshot (done, section 2 : q_norm and k_norm present, tied
  embeddings, 24 linear-attention layers).
- bf16 rounding, **decided by the PI on 2026-11-11 (evening) after a
  one-hour test** : bf16 is kept (instrument of E1-small unchanged) ;
  the table of section 3 reports, per dose and per arm, the rounding
  norm ||bf16(f theta) - f theta|| as a share of the nominal step, and
  **every dose whose share exceeds 10 % is flagged** in the report (it
  stays in the grid and in the gate). Measured on the real snapshot
  (`scripts/bf16_rounding_probe.py`, CPU of kxkm-ai, nice 19, 281 s,
  738 tensors, no forward pass, result in
  `docs/results/2026-11-11-bf16-rounding-probe-qwen35-4b.json`) :
  share of rounding in the step of `D` = 18.3 % at eps = 0.01, 7.6 % at
  0.02, 3.0 % at 0.05, 1.5 % at 0.1, 0.7 % at 0.2, 0 at 0.5 (times 0.5
  is exact in binary) ; for one seed of `Rt` : 16.1 %, 8.1 %, 3.2 %,
  1.6 %, 0.8 %, 0.4 %, with 16.7 % of the elements left unchanged at
  0.01 and 8.5 % at 0.02. The applied norm equals the nominal one
  within 1.2 % at every dose, in every tensor group. **Only the dose
  0.01 is flagged**, for both arms. Exploratory measurement, no item
  scored, not part of the confirmatory set.

## 8. Critique of revision 1 and what changed

1. **The tolerance of 0.02 could not be established.** With 760 items
   and a base accuracy near 0.40, the standard error of an accuracy is
   about 0.018 ; a point estimate "within 0.02" is one standard error
   and would pass or fail by chance. Revision 2 uses an equivalence
   interval (99 % paired interval inside +/- 0.03), reachable only if
   fewer than about 10 % of items change side, which is the regime the
   claim is about. An arm that reshuffles many items without changing
   the mean is counted as not tolerated : conservative.
2. **The drop of 0.10 ignored the headroom.** The model has about 0.15
   above chance on the primary set, so 0.10 was two thirds of it, a
   fixed number unrelated to the instrument. "Broken" is now half of
   the measured margin above chance.
3. **The single dose f = 0.8 was arbitrary and probably too strong.**
   A uniform rescaling is not an exact symmetry of a transformer
   (section 2) ; at 0.8 DOWNSCALE may degrade the model and C1 would
   have been refuted although DOWNSCALE is far better tolerated than
   noise. The claim is now a ratio of tolerance doses on a log grid.
   The old grid also had no point between 0.01 and 0.1, where the
   transition of the noise arm is most likely.
4. **The control was not matched.** Global noise of a given norm
   spreads the same absolute noise over every tensor, so it hits
   small tensors (normalisation gains, biases) far harder than a
   rescaling does. A difference between DOWNSCALE and that noise could
   be an allocation effect, not a direction effect. The primary
   control is now per-tensor noise of matched norm ; the global
   variant is kept at three doses for continuity with the E1 controls.
5. **The instrument has little headroom.** The E1 corpus was built to
   be learned, so the base model is low on it by design (0.49 on
   `mmlu_cf`, 0.32 on `realtimeqa`). It is kept because the instrument
   is registered and the comparison with E1-small is direct ; an
   outside set where the base model is well above chance would
   separate the arms better. Left to the PI.
6. **The gate could not fail.** "Noise at f = 0.5 hurts" is certain.
   The gate now requires the noise arm to be harmless at the smallest
   dose and broken at some dose, so that the grid brackets the
   sensitivity threshold.
7. **C2 tested a direction, not a temperature.** "Confidence falls by
   0.02" is passed by any perturbation that blurs the output. The
   claim is now that one scalar temperature accounts for the change of
   the option distribution.
8. **What the critique does not fix.** The thresholds that remain
   (0.03, half the margin, two grid steps, 80 %) are reasoned but not
   derived from a power simulation ; three noise seeds are few if the
   per-seed curves differ ; one model, one corpus.
