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

## 2. What is already known (reading note, to complete before any freeze)

Scale invariance of the weights that feed a normalisation layer is
established for batch and weight normalisation (van Laarhoven 2017 ;
Arora, Li, Lyu 2019) [identifiers to resolve by API, sources to read
in full]. What is not settled by that literature, and is the only
open question here: a pre-norm transformer has learned normalisation
gains, embeddings, biases and an output head that are also multiplied
by f, so a uniform rescaling of *all* parameters is not an exact
symmetry. The test measures how far from a symmetry it is, against a
perturbation of equal norm and random direction.

Expected from the architecture, to be checked and not assumed: with
every matrix multiplied by f, attention logits shrink roughly as a
power of f (queries and keys are both rescaled), so attention flattens
as the dose grows. DOWNSCALE is therefore not expected to be harmless
at every dose ; the claim is that it is tolerated at much larger doses
than noise.

The reading gate does not apply (PI, 2026-11-11). The two references
above are still to resolve and read before a freeze.

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

## 7. Not done, required before a freeze

- A `dose-response` mode in `scripts/e1_harness.py` (restore, perturb,
  score, record per-item option scores), the per-tensor noise variant,
  and their tests ; none of it exists at this commit.
- The analysis script that prints the verdict of section 4, tested on
  constructed tables (pure symmetry, pure noise, pure temperature).
- The two references of section 2, resolved and read.
- PI decisions still open : the primary item set (760 items, or
  `mmlu_cf` alone, or an outside set with more headroom, see 8.5) ;
  bf16 rounding kept as is or lifted to float32.

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
