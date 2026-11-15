# Pre-registration (L0, draft) — A1-r: does a tolerated rescaling change what is learned next, and what is kept?

**Status:** L0 draft written on 2026-11-11 (evening) at the request of
the PI, as the reformulation of candidate A1 after the full reading of
Ash and Adams 2020 (library note `references/notes/arxiv-1910.08475.md`).
Nothing is frozen, nothing has been run, no catalogue number is
assigned (PI: none before Oh et al. 2025, arXiv:2507.04683, is read).
Drafted by an agent. Coupled to the dose–response control C1/C2
(`docs/milestones/2026-11-11-downscale-dose-response.md`, PR #100):
A1-r runs only after C1 has a verdict and takes its doses from it.

## 1. Question, in plain words

C1 asks what a global rescaling of the weights (DOWNSCALE, theta ->
(1 - eps) theta) does to the function the model computes **now**. A1-r
asks what it does to the **next** block of learning and to what was
learned **before**: at a dose where C1 says the function is preserved,
does learning a second block of facts go better after the rescaling
than after nothing, and than after a random perturbation of the same
norm, and at what cost for the first block?

The original A1 (« a rescaling between two learning blocks improves what
is learned in the second, at equal budget ») is **already decided** on
the benchmark of Ash and Adams 2020: on their shrink × noise grid
(fig. 8, ResNet-18, CIFAR-10 arriving online) shrink alone recovers the
accuracy of a fresh initialisation (76.8 % against 76.0 %, warm start
63.7 %) while noise alone plateaus (65.3 %). A1-r keeps what that
benchmark leaves open (section 2) and adds the retention measure that
no one has taken.

## 2. What is known (reading notes of 2026-11-11, library)

- **Ash, Adams 2020** (arXiv:1910.08475 v3, read in full). Facts: the
  warm-start generalisation gap on images (p. 2-4); shrink-and-perturb
  theta <- lambda theta + p xi with lambda = 0.6, p = 0.01 chosen on a
  grid (p. 6); **shrink alone and noise alone are both tested** (fig. 8,
  p. 6; table 4, p. 12: Gaussian noise alone 54.4 % at best against
  55.2 % for a fresh initialisation, at 165 min against 34). The
  mechanism is conjectured (« gradient balancing », p. 6), not derived;
  proposition 1 (p. 12) proves only that theta -> lambda theta
  preserves the arg max for a ReLU network without biases or
  normalisation. Not covered: language models, pre-trained models,
  matched per-tensor noise (their noise is a random initialisation
  times a scalar), equal budget (every arm trained to convergence,
  shrink costing 3 to 4 times the warm-start time), doses softer than
  lambda = 0.9, and **the retention of the first block, never
  measured**.
- **Lyle et al. 2024** (arXiv:2407.01800, read in full). Facts: for the
  weights that feed a normalisation, a rescaling leaves the function
  unchanged and multiplies the effective learning rate of the training
  that follows by 1 / (1 - eps)^2 for gradient descent, 1 / (1 - eps)
  for fixed-norm steps (definition 1, p. 3); a constant effective rate
  is not always better than a decaying one (fig. 3, p. 7); with the
  norm controlled, shrink-and-perturb and plain L2 become « nearly
  indistinguishable » on a MLP (p. 8).
- **van Laarhoven 2017** (arXiv:1706.05350) and **Arora, Li, Lyu 2018**
  (arXiv:1812.03981), read in full: the effective-rate formulas and
  their scope (invariant parameters only; gains, biases, head outside);
  convergence at the best asymptotic rate for any fixed rate of the
  invariant parameters, so a norm change is predicted to have a
  **transient** effect, not a lasting gain.
- **Hernandez-Garcia 2026** and **Han 2026** (read in full on
  2026-11-11, morning): on transformers the link between weight norm and
  plasticity is loose; stronger weight decay in pre-training helps later
  fine-tuning; no post-hoc rescaling tested.
- **Not read:** Oh et al. 2025 (arXiv:2507.04683, soft per-layer
  rescaling, vision only); Dohare et al. 2023 (arXiv:2306.13812). Both
  in the library.

**Tautology check.** Three things are known before any run and are not
claimed as results. (a) Shrink beats noise on an image benchmark at
lambda = 0.6 (Ash, Adams). (b) The mechanism available in the theory is
a change of effective learning rate by 1 / (1 - eps)^2, that is +11 % at
eps = 0.05 and +56 % at eps = 0.2: **small at the doses where C1 is
expected to find DOWNSCALE tolerated, large only where C1 is expected to
find the function broken.** (c) The effect is predicted transient. What
is open: whether, on a pre-trained hybrid language model, there is a
dose at which the function is kept (C1) **and** the next block learns
differently (A1-r), and what that dose costs to the first block. A1-r is
therefore built around the dose that C1 returns, and its first
comparison is against a **learning-rate-matched arm** (same factor on
the learning rate, no rescaling): if that arm reproduces the effect,
A1-r is a replication of the effective-rate mechanism, nothing more.

## 3. Design

- **Model and instrument.** Qwen3.5-4B, snapshot and harness of E1-small
  (`scripts/e1_harness.py`, registered torch backend, bf16 kept as in
  C1), frozen E1 fact corpus (§5.1 of the E1-small preregistration).
- **Two blocks.** For each seed s: block 1 = the 40 replay items of seed
  s (as in E1-small, LR 3e-4, m = 2, the selection (i) of the controls
  addendum of 2026-11-06); block 2 = 40 further items of the
  confirmatory partition, disjoint from block 1 and from the evaluation
  items, drawn by a seeded rule written before the run. Evaluation on
  the held-out items of the seed, as in E1-small.
- **Operation between the blocks**, applied once after block 1:
  - `none` ;
  - `D(eps)` : DOWNSCALE, `Backend.downscale(1 - eps)` ;
  - `Rt(eps, s)` : per-tensor Gaussian noise of norm eps ||theta_t||
    (the C1 control), one seed per cell ;
  - `LR(eps)` : nothing on the weights, learning rate of block 2
    multiplied by 1 / (1 - eps)^2 (the effective-rate mechanism alone).
- **Doses.** Taken from the C1 verdict, written in an amendment before
  this run: the tolerance dose of `D` in C1 (eps_tol) and the dose one
  grid step above it. If C1 returns `non_test` or a tolerance rank 0,
  A1-r is not run.
- **Budget.** Identical number of updates on block 2 in every arm (the
  E1 replay schedule); nothing is trained to convergence.
- **Measures per cell.** `learn` = accuracy on the evaluation items of
  block 2 after block 2 ; `keep` = accuracy on the evaluation items of
  block 1 after block 2 ; both before and after, per-item correctness
  kept ; norm actually applied ; peak memory.
- **Seeds.** A subset of the 141 E1-small seeds, size fixed after the
  one-hour test below (cost per cell); the subset rule is written before
  the run.

## 4. Decision rule (to fix before any run)

Paired differences over seeds, 99 % bootstrap intervals, 10 000 draws,
margin delta = 0.02 of accuracy [reasoned from the E1-small learning
gate, +0.063 on 141 seeds with a 95 % lower bound at +0.059 ; not
derived from a power simulation ; to revisit after the one-hour test].

0. **Gate (positive control of the instrument, CPU).** Before the GPU
   run, the three operations are applied on a small normalised MLP in a
   two-block version of the Ash–Adams setting (CIFAR-10 or a synthetic
   task, online arrival, lambda = 0.6 and the C1 doses). The gate passes
   iff shrink alone beats noise alone on block 2 by more than the
   margin, as in their fig. 8, and the LR-matched arm is reported. A
   simulator that does not reproduce the published ordering is not
   used.
1. **A1-r supported** iff, at eps_tol, `learn(D) - learn(none)` lies
   above +delta **and** `learn(D) - learn(Rt)` lies above +delta **and**
   `learn(D) - learn(LR)` lies above +delta (the effect is not the
   effective-rate change alone).
2. **A1-r is the effective-rate mechanism** iff the first two hold and
   `learn(D) - learn(LR)` lies inside [-delta, +delta] : reported as a
   replication of Lyle 2024 on a language model, not as a new effect.
3. **A1-r refuted** iff `learn(D) - learn(Rt)` lies inside or below
   [-delta, +delta] at both doses.
4. Otherwise `inconclusive`.
5. **Retention, reported and decided on separately:** `keep(D) -
   keep(none)` below -delta at eps_tol is read as « the rescaling costs
   the first block », whatever the verdict on `learn` ; `keep(Rt)` and
   `keep(LR)` are reported next to it.

## 5. What each verdict changes

- **Supported:** the dream axis on language models has one operation
  that preserves the function (C1) and changes what is learned next by
  more than a learning-rate change ; H14 may be reopened on that dose,
  after M3, under the gate.
- **Effective-rate mechanism:** the rescaling is a learning-rate knob
  with a cost on retention or not ; stated as such, H14 stays closed.
- **Refuted:** on a pre-trained language model the Ash–Adams result does
  not carry over at tolerated doses ; the axis stops at C1.
- **Retention cost:** whatever the rest, the operation is not free, and
  the sleep analogy of the lab loses its « harmless » clause.

## 6. Known weaknesses of this draft

- The dose comes from C1 ; a C1 `non_test` blocks A1-r.
- The effect predicted by the theory at eps_tol is small (+11 % of
  learning rate at 0.05) and transient over 40 items ; the design may
  be a null by construction at that dose, which is why the dose one
  step above is kept and why the LR arm is there.
- One model, one corpus, replay of 40 items : a regime where the weight
  norm barely moves (E1-small : rho = 8.36 for DOWNSCALE against 0.35
  for replay alone, descriptive reading of 2026-11-10).
- The gate replicates a published ordering on a toy ; it tests the code,
  not the transfer to the language model.
- Margins reasoned, not derived ; seed subset not yet sized.

## 7. Cost, and what is not done

GPU : per cell, one replay of 40 items plus two evaluations, as one
E1-small cell ; four arms x two doses x the seed subset. **One-hour test
first** (rule of 2026-11-10) : one seed, four arms, one dose, to
measure the cell time and set the subset. Expected 1 to 3 h of GPU
after C1 (PI, 2026-11-11). Not done : the code (block-2 split,
`LR(eps)` arm, `keep` measure), the CPU gate, the reading of Oh et al.
2025 and Dohare et al. 2023, the C1 verdict, the seed subset rule, a
catalogue number.
