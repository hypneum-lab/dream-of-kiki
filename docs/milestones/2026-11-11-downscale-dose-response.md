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

**Statement C1 (EN, to lock)** : On the base Qwen3.5-4B model, with
no replay and no training, multiplying all floating-point parameters
by f = 0.8 leaves multiple-choice accuracy within 0.02 of the
unperturbed model, while a random perturbation of the same Frobenius
norm lowers it by at least 0.10.

**Statement C2 (EN, to lock)** : At the same dose, DOWNSCALE lowers
the mean confidence of the model (maximum of the softmax over the
option scores) by at least 0.02 while accuracy stays within 0.02 :
it acts as a temperature, not as a change of what the model knows.

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

Open before a freeze: one watch round on scale invariance in
transformers and on synaptic downscaling in LLMs ; the blind
contradictor if the PI rules that this one-hour inference test falls
under the reading gate (roadmap, principle 6 names any GPU campaign).

## 3. Design

- **Model** : `Qwen/Qwen3.5-4B`, the snapshot and the instrument of
  E1-small (`scripts/e1_harness.py`, registered torch backend,
  determinism settings unchanged). Base weights, reloaded before each
  arm. No replay, no optimiser step.
- **Items** : the CONFIRMATORY items of the FROZEN E1 corpus, 1 098
  items : `mmlu_cf` 380, `realtimeqa` 380, `daily_oracle` 338.
  **Primary set : `mmlu_cf` + `realtimeqa`, 760 items.**
  `daily_oracle` is reported apart. This exclusion is a choice made
  after seeing data (base accuracy 0.179 with the question, under the
  0.25 chance level, addendum section 10.6) ; it is declared here as
  such, and the 1 098-item result is reported next to the primary.
- **Doses** : f in {0.99, 0.9, 0.8, 0.5}.
- **Arms at each dose** :
  - `D(f)` : `Backend.downscale(f)`, deterministic, one pass ;
  - `R(f, s)` : `Backend.perturb_same_norm(rho(f), "R", s)` with
    `rho(f) = Backend.downscale_norm(f)`, seeds s = 0, 1, 2 ;
  - `base` : unperturbed model, one pass.
  17 scoring passes in total. The `U` variant (shrink of constant
  magnitude along the sign of each weight) is not run : it is a
  rescaling direction itself and would blur the contrast.
- **Measures per pass** : accuracy (argmax of the option scores) ;
  mean confidence (maximum of the softmax over the option scores) ;
  `gap = mean confidence - accuracy` ; `choice_kl` to the base
  pass ; the norm actually applied.
- **Seeds** : noise seeds 0, 1, 2 ; item bootstrap RNG 20261111,
  10 000 draws, items resampled, the same indices for both arms.

## 4. Decision rule (to fix before any run)

Let `acc(X)` be accuracy on the primary set and `acc(R(f))` the mean
over the three seeds.

0. **Measurability gate** : `acc(base) - acc(R(0.5)) >= 0.10`. If the
   random perturbation at the largest dose does not hurt, the doses
   are too small to test anything : verdict `non_test`, and no dose is
   added without an amendment written before the new run.
1. **C1 supported** iff, at f = 0.8 : `acc(base) - acc(D) <= 0.02`
   and `acc(base) - acc(R) >= 0.10` and the 99 % paired bootstrap
   interval of `acc(D) - acc(R)` is entirely above 0.
2. **C1 refuted** iff the 99 % interval of `acc(D) - acc(R)` contains
   0 at every dose where either arm is more than 0.02 from the base,
   or `acc(base) - acc(D(0.8)) >= 0.10`.
3. Otherwise `inconclusive`, reported with the four dose points.
4. **C2 supported** iff C1's first condition holds at f = 0.8 and
   `conf(base) - conf(D(0.8)) >= 0.02` with a 99 % paired bootstrap
   interval entirely above 0. **Refuted** iff that interval contains
   0 or lies below it. If accuracy moves by more than 0.02, C2 is
   `not_applicable` at this dose and is read at the largest tested
   dose where it does not.

The thresholds 0.02, 0.10 and the dose 0.8 are proposals of the
drafting agent, not derived from any run ; the PI fixes them before
the freeze. The whole dose-response table is reported whatever the
verdict.

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

Inference only : 17 passes over 1 098 four-option items on the 4B
model, kxkm-ai (RTX 4090), estimated under one hour [to verify with a
timing of the base pass, read for its duration only]. No number is
read before the 17 passes are complete ; the verdict is computed by a
script committed before the run.

## 7. Not done, required before a freeze

- A `dose-response` mode in `scripts/e1_harness.py` (reload, perturb,
  score, record per-item option scores) and its tests ; none of it
  exists at this commit.
- The analysis script that prints the verdict of section 4.
- The reading note of section 2 and the PI's ruling on the reading
  gate.
- PI decisions : thresholds and dose of section 4 ; primary item set ;
  bf16 rounding (at f = 0.99 the step is applied in bf16, as in
  `downscale_norm`) kept as is or lifted to float32.
