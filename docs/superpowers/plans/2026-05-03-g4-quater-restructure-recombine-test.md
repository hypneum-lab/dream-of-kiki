# G4-quater RESTRUCTURE+RECOMBINE Empirical Test Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Empirically resolve the H_DR4 inversion observed in G4-ter (commit `98b1305`, milestone `g4-ter-pilot-2026-05-03.md`) where `P_min retention 0.7065 > P_equ 0.7046 = P_max 0.7046` — the inverse of the predicted `P_max ≥ P_equ ≥ P_min`. Distinguish between three sub-hypotheses : H4-A (substrate too simple), H4-B (HP miscalibrated), H4-C (RESTRUCTURE+RECOMBINE theoretically empty).

**Architecture:** Sequential 3-step pilot, all on M1 Max with MLX. Step 1 tests H4-A by deepening the substrate to 5 layers. Step 2 tests H4-B by sweeping RESTRUCTURE strength on the existing 3-layer head. Step 3 tests H4-C by adding a placebo-control RECOMBINE strategy (none) alongside the existing MoG and a new autoencoder variant. Each step ships its own milestone with R1 run_ids ; the aggregator emits a verdict over the three.

**Tech Stack:** Python 3.12+, MLX 0.31.1, numpy 2.4.4, scipy 1.17.1, hypothesis 6.152.1 with `derandomize=True`, pytest 9.0.3. No new dependencies. Conventional-commit subjects ≤ 50 chars, scope ≥ 3 chars, EN, no Co-Authored-By, body ≤ 72 chars.

---

## Background

G4-ter (Plan `2026-05-03-g4-ter-hp-sweep-richer-substrate.md`) added `experiments.g4_ter_hp_sweep.dream_wrap_hier.G4HierarchicalClassifier` (3-layer : input → 32 → 16 → 2) and the four dream operations REPLAY / DOWNSCALE / RESTRUCTURE / RECOMBINE wired into `dream_episode_hier`. The pilot reported g_h2 = +2.77 (Welch p = 4.9e-14 at N = 30) but the profile-ordering mean retentions came out as `P_min = 0.7065`, `P_equ = 0.7046`, `P_max = 0.7046` — the inverse of the DR-4 prediction. Critic review post-session (commit `751370d` reframe) reframed the finding as "REPLAY+DOWNSCALE coupling on richer substrate exceeds no-dream baseline ; RESTRUCTURE+RECOMBINE remain spectator channels at this scale", a partial refutation of the framework-C profile-ordering claim.

Three competing explanations remain :

- **H4-A (substrate-depth)** : a 3-layer head is too simple for RESTRUCTURE / RECOMBINE to express benefit ; deeper hierarchy will recover the predicted ordering.
- **H4-B (HP-calibration)** : RESTRUCTURE factor `0.95 × σ × N(0,1)` is too aggressive on the small `_l2` weights (16×32 = 512 params), and the Gaussian-MoG RECOMBINE samples are poorly calibrated ; tuning HPs will recover the ordering.
- **H4-C (theoretical-emptiness)** : at this benchmark scale (Split-FMNIST 5 binary tasks, MLP head), RESTRUCTURE and RECOMBINE channels are structurally empty — the dream effect lives entirely in REPLAY+DOWNSCALE regardless of substrate depth or HP. This is a partial refutation of the framework-C claim that "richer ops yield richer consolidation" at this scale.

This plan ships a confirmatory N ≥ 95 sequential pilot that produces a verdict for each.

## Files touched

| File | Action | Approx LOC |
|------|--------|---|
| `experiments/g4_quater_test/__init__.py` | create | 5 |
| `experiments/g4_quater_test/deeper_classifier.py` | create | 80 |
| `experiments/g4_quater_test/recombine_strategies.py` | create | 90 |
| `experiments/g4_quater_test/run_step1_deeper.py` | create | 120 |
| `experiments/g4_quater_test/run_step2_restructure_sweep.py` | create | 110 |
| `experiments/g4_quater_test/run_step3_recombine_strategies.py` | create | 130 |
| `experiments/g4_quater_test/aggregator.py` | create | 80 |
| `tests/unit/experiments/test_g4_quater_*.py` (4 files) | create | ~150 total |
| `tests/conformance/axioms/test_dr3_g4_deeper_substrate.py` | create | 50 |
| `docs/osf-prereg-g4-quater-pilot.md` | create | ~150 |
| `docs/milestones/g4-quater-{step1,step2,step3,aggregate}-2026-05-03.{json,md}` | create | dump artefacts |
| `docs/papers/paper2/results.md` | edit (add §7.1.6) | +60 |
| `docs/papers/paper2-fr/results.md` | edit (add §7.1.6) | +60 |
| `CHANGELOG.md` | edit ([Unreleased] block) | +30 |
| `STATUS.md` | edit (G4-quater row) | +1 |
| `docs/proofs/dr4-profile-inclusion.md` | edit (revise evidence) | +20 |

No mutation of `kiki_oniric/profiles/`, `kiki_oniric/substrates/`, `kiki_oniric/dream/`, or any prior pilot's experiment package (`g4_split_fmnist`, `g4_ter_hp_sweep`, `g5_cross_substrate`, `g6_mmlu_stream`).

## Compute budget

- Step 1 (H4-A, 5-layer × 4 arms × N=95) : 380 cells. Per-cell ~3.0s on M1 Max (deeper than G4-ter's ~1.5s). **~19 min.**
- Step 2 (H4-B, 3-layer × RESTRUCTURE factor ∈ {0.85, 0.95, 0.99} × 4 arms × N=30) : 360 cells. Per-cell ~1.5s. **~9 min.**
- Step 3 (H4-C, 3-layer × RECOMBINE ∈ {mog, ae, none} × 4 arms × N=95) : 1140 cells. Per-cell ~1.7s. **~32 min.**
- **Total ~60 min on M1 Max** (well under the 5-8h budget).

The aggregator and paper updates add < 5 min.

## Pre-registration discipline

OSF pre-reg (Task 1) is locked **before** any pilot run. Hypotheses H4-A / H4-B / H4-C are confirmatory ; effect-size targets and exclusion criteria pre-specified ; no post-hoc reformulation allowed without an amendment.

The pre-reg explicitly cites G4-ter as the exploratory baseline (`g_h2 = +2.77`, `H_DR4 inverted with P_min=0.7065 > P_equ=0.7046`) so the chain G4 → G4-bis → G4-ter → G4-quater is auditable.

---

### Task 0 — Investigate

**Files:** read-only.

- [ ] **Step 1: Read G4-ter substrate.**
  ```bash
  uv run python -c "
  from experiments.g4_ter_hp_sweep import dream_wrap_hier as h
  print('class:', h.G4HierarchicalClassifier.__doc__.split(chr(10))[0])
  print('hidden:', '32 → 16 → 2')
  print('ops:', list(h.PROFILE_OPS.keys()) if hasattr(h, 'PROFILE_OPS') else 'see dream_episode_hier')
  "
  ```
- [ ] **Step 2: Read G4-ter pre-reg + milestone.**
  - `docs/osf-prereg-g4-ter-pilot.md` §2-3 (hypotheses + HP grid)
  - `docs/milestones/g4-ter-pilot-2026-05-03.md` (observed scalars + caveats block)
- [ ] **Step 3: Sanity-check existing tests.**
  ```bash
  uv run pytest tests/unit/experiments/test_g4_ter_*.py --no-cov -q
  ```
  Expected : 22 passed.

No commit.

---

### Task 0.5 — Decision lock

**Files:** documentation comment in `experiments/g4_quater_test/__init__.py` (created in Task 2).

Sequential 3-step layout is locked per the `superpowers:writing-plans` review : factorial 3,420-cell sweep would exceed the M1 Max overnight budget unnecessarily ; sequential lets early-stop if H4-A is confirmed at Step 1 (deeper substrate is enough). The locked recipe :

- Step 1 — H4-A test, 5-layer deeper substrate
- Step 2 — H4-B test, RESTRUCTURE factor sweep on the existing 3-layer head
- Step 3 — H4-C test, RECOMBINE strategy {mog, ae, none}, with **none** as the placebo control isolating REPLAY+DOWNSCALE contribution

If at the end of Step 3 `retention(P_max with RECOMBINE=none) ≈ retention(P_max with RECOMBINE=mog)` (Welch fails to reject), H4-C is confirmed : RECOMBINE channel is theoretically empty at this scale.

No commit.

---

### Task 1 — OSF pre-registration draft

**Files:**
- Create: `docs/osf-prereg-g4-quater-pilot.md`

- [ ] **Step 1: Write pre-reg with binding sections.**

  ```markdown
  # G4-quater pilot pre-registration

  **Date:** 2026-05-03
  **Parent OSF:** 10.17605/OSF.IO/Q6JYN
  **Sister pilot:** G4-ter (g_h2 = +2.77, H_DR4 inverted)
  **Substrate:** MLX hierarchical (G4-ter 3-layer + new 5-layer)
  **Benchmark:** Split-FMNIST 5 binary tasks
  **Lock commit:** [filled at commit time]
  **Lock timestamp:** [filled at commit time]

  ## §1 Background

  G4-ter (commit `98b1305`) reported g_h2 = +2.77 (Welch
  p = 4.9e-14, N = 30) under the hierarchical substrate, but
  observed retention `P_min = 0.7065, P_equ = 0.7046,
  P_max = 0.7046` is the inverse of the DR-4 prediction
  `P_max ≥ P_equ ≥ P_min`. Reframe (commit `751370d`)
  treats this as a partial refutation of the framework-C
  claim that richer op-sets yield richer consolidation.

  ## §2 Hypotheses (confirmatory)

  - **H4-A (substrate-depth)** : on a deeper hierarchical
    head (5 layers, hidden 64-32-16-8), the predicted
    profile ordering `P_max ≥ P_equ ≥ P_min` recovers.
    Test : Jonckheere trend on retention across the three
    profiles, one-sided alpha = 0.05 / 3 = 0.0167
    (Bonferroni for 3 hypotheses).

  - **H4-B (HP-calibration)** : on the existing 3-layer
    head, sweeping RESTRUCTURE factor over {0.85, 0.95,
    0.99} reveals at least one factor where the predicted
    ordering recovers. Test : Jonckheere on retention per
    factor cell, multiplicity-adjusted alpha = 0.05 / 9
    (3 factors × 3 hypotheses) = 0.0056.

  - **H4-C (theoretical-emptiness)** : RECOMBINE channel
    is structurally empty at this scale. Operationalised
    as : `retention(P_max with RECOMBINE=none)` is not
    statistically distinguishable from `retention(P_max
    with RECOMBINE=mog)` (Welch two-sided fails to reject
    H0 at alpha = 0.05 / 3 = 0.0167).

  ## §3 Power analysis

  N = 95 seeds per arm at alpha = 0.0167 detects g ≥ 0.40
  at 80 % power (Welch one-sided, equal variance
  approximation). Effect sizes below 0.40 remain
  exploratory.

  ## §4 Exclusion criteria

  - `acc_initial < 0.5` (random chance) — exclude cell
  - `acc_final` non-finite — exclude cell
  - any seed reproducing run_id collision with prior
    pilot's registry — abort and amend pre-reg

  ## §5 Substrate / driver paths

  - Step 1 : `experiments/g4_quater_test/run_step1_deeper.py`
  - Step 2 : `experiments/g4_quater_test/run_step2_restructure_sweep.py`
  - Step 3 : `experiments/g4_quater_test/run_step3_recombine_strategies.py`
  - Substrates :
    - 5-layer : `experiments.g4_quater_test.deeper_classifier.G4HierarchicalDeeperClassifier`
    - 3-layer : `experiments.g4_ter_hp_sweep.dream_wrap_hier.G4HierarchicalClassifier`

  ## §6 DualVer outcome rules

  | Outcome | EC bump |
  |---|---|
  | All H4-* rejected H0 (DR-4 ordering recovers under any condition) | EC stays PARTIAL ; FC unchanged ; G4-quater scope STABLE row in STATUS |
  | H4-A confirmed only | EC stays PARTIAL ; flag substrate-depth scope-bound STABLE for 5-layer ; document open Q for shallower substrates |
  | H4-B confirmed only | EC stays PARTIAL ; flag HP-window scope-bound STABLE ; document HP-pin amendment |
  | H4-C confirmed (RECOMBINE empty at this scale) | EC stays PARTIAL ; **DR-4 partial refutation strengthens** ; framework C claim "richer ops yield richer consolidation at this scale" formally refuted ; DR-4 evidence file revised |
  | All three confirmed | EC stays PARTIAL ; full empirical revision of DR-4 + paper section ; future work : test deeper benchmarks (CIFAR-10, ImageNet) before any STABLE promotion |

  ## §7 Reporting commitment

  Honest reporting of all observed scalars regardless of
  outcome ; if Welch tests do not reject H0, the verdict
  is "no evidence for the hypothesis at this N", not
  "hypothesis falsified" (absence of evidence vs evidence
  of absence). H4-C confirmation specifically requires
  Welch failing to reject difference between RECOMBINE=mog
  and RECOMBINE=none — a positive empirical claim that
  RECOMBINE adds nothing.

  ## §8 Audit trail

  Every cell registered via `harness/storage/run_registry.py`
  with profile keys
  `g4-quater/{step1,step2,step3}/<arm>/<combo>/<seed>` and
  R1 bit-stable run_ids. Milestone artefacts under
  `docs/milestones/g4-quater-step{1,2,3}-2026-05-03.{json,md}`
  + aggregate.

  ## §9 Deviations

  Any deviation from this pre-reg requires (a) a separate
  amendment commit before the affected cell runs, or (b)
  a post-hoc honest disclosure in the paper section
  acknowledging the deviation and its impact on the
  confirmatory status of the hypothesis.
  ```

- [ ] **Step 2: Commit.**
  ```bash
  cd /Users/electron/hypneum-lab/dream-of-kiki && \
    git add docs/osf-prereg-g4-quater-pilot.md && \
    git commit -m "$(cat <<'EOC'
  docs(osf): G4-quater pilot pre-reg draft

  Confirmatory N>=95 pilot pre-registration locking three
  hypotheses to resolve the G4-ter H_DR4 inversion :
  H4-A (substrate-depth), H4-B (HP-calibration), H4-C
  (RECOMBINE theoretical-emptiness with none-placebo).
  Cites G4-ter findings as exploratory baseline ; binds
  EC PARTIAL across all outcomes.
  EOC
  )"
  ```

---

### Task 2 — `G4HierarchicalDeeperClassifier` (5-layer, H4-A test substrate)

**Files:**
- Create: `experiments/g4_quater_test/__init__.py` (5 lines, import re-exports)
- Create: `experiments/g4_quater_test/deeper_classifier.py`
- Create: `tests/unit/experiments/test_g4_quater_deeper.py`

The 5-layer head is `Linear(in, 64) → ReLU → Linear(64, 32) → ReLU → Linear(32, 16) → ReLU → Linear(16, 8) → ReLU → Linear(8, n_classes)`. RESTRUCTURE perturbs the **middle** layer's weight (the third Linear : `Linear(32, 16)`), preserving the input projection (`_l1`) and the output classifier (`_l5`). RECOMBINE samples synthetic latents from a Gaussian-MoG fitted on the activations after the third ReLU.

- [ ] **Step 1: Write failing test.**
  ```python
  # tests/unit/experiments/test_g4_quater_deeper.py
  """Unit tests for the 5-layer deeper hierarchical classifier."""
  from __future__ import annotations

  import numpy as np
  import pytest

  from experiments.g4_quater_test.deeper_classifier import (
      G4HierarchicalDeeperClassifier,
  )


  def test_deeper_classifier_shape() -> None:
      m = G4HierarchicalDeeperClassifier(
          in_dim=784,
          hidden=(64, 32, 16, 8),
          n_classes=2,
          seed=0,
      )
      logits = m.predict_logits(np.zeros((3, 784), dtype=np.float32))
      assert logits.shape == (3, 2)


  def test_deeper_classifier_seeded_determinism() -> None:
      x = np.random.RandomState(0).randn(5, 784).astype(np.float32)
      a = G4HierarchicalDeeperClassifier(
          in_dim=784, hidden=(64, 32, 16, 8), n_classes=2, seed=42
      )
      b = G4HierarchicalDeeperClassifier(
          in_dim=784, hidden=(64, 32, 16, 8), n_classes=2, seed=42
      )
      np.testing.assert_array_equal(
          a.predict_logits(x), b.predict_logits(x)
      )
  ```
- [ ] **Step 2: Run, expect ImportError.**
  ```bash
  uv run pytest tests/unit/experiments/test_g4_quater_deeper.py --no-cov -q
  ```
- [ ] **Step 3: Implement.**

  Mirror `G4HierarchicalClassifier` from `dream_wrap_hier.py:48-90` with five Linear layers, three ReLU hidden activations, and a `latent()` method that returns hidden_3 activations (after the third ReLU, dimension 16) for the RECOMBINE sampler. Methods : `__post_init__` seeds MLX + numpy, `predict_logits(x: np.ndarray) -> np.ndarray`, `train_task(x, y, epochs, lr)`, `eval_accuracy(x, y) -> float`, `latent(x: np.ndarray) -> np.ndarray`.

- [ ] **Step 4: Run, expect 2/2 pass.**
- [ ] **Step 5: Lint + type.**
  ```bash
  uv run ruff check experiments/g4_quater_test/ tests/unit/experiments/test_g4_quater_deeper.py && \
    uv run mypy --explicit-package-bases experiments/g4_quater_test/deeper_classifier.py
  ```
- [ ] **Step 6: Commit.**
  ```bash
  git commit -m "feat(g4-quater): add 5-layer deeper classifier"
  ```

---

### Task 3 — RECOMBINE strategy variants

**Files:**
- Create: `experiments/g4_quater_test/recombine_strategies.py`
- Create: `tests/unit/experiments/test_g4_quater_recombine.py`

Three strategies, all called via a `RecombineStrategy = Literal["mog", "ae", "none"]` switch :

- **mog** : Gaussian-MoG per-class on β-buffer latents (port from G4-ter `_recombine_step`).
- **ae** : deterministic single-layer autoencoder on latents (`encoder = Linear(latent_dim, latent_dim/2)`, `decoder = Linear(latent_dim/2, latent_dim)`, MSE-trained over the buffer for one pass) ; reconstructs the buffer's latents to produce synthetic samples.
- **none** : placebo — return an empty list, dream_episode then skips the SGD pass on `_l3`. **This is the H4-C control.**

- [ ] **Step 1: Write failing tests.**
  ```python
  # tests/unit/experiments/test_g4_quater_recombine.py
  from __future__ import annotations
  import numpy as np
  from experiments.g4_quater_test.recombine_strategies import (
      sample_synthetic_latents,
  )


  def test_mog_returns_n_synthetic() -> None:
      latents = np.random.RandomState(0).randn(20, 16).astype(np.float32)
      labels = np.array([0]*10 + [1]*10)
      out = sample_synthetic_latents(
          strategy="mog", latents=latents, labels=labels,
          n_synthetic=5, seed=0,
      )
      assert out is not None
      assert out["x"].shape == (5, 16)
      assert out["y"].shape == (5,)


  def test_ae_returns_reconstructed_latents() -> None:
      latents = np.random.RandomState(0).randn(20, 16).astype(np.float32)
      labels = np.array([0]*10 + [1]*10)
      out = sample_synthetic_latents(
          strategy="ae", latents=latents, labels=labels,
          n_synthetic=5, seed=0,
      )
      assert out is not None
      assert out["x"].shape == (5, 16)


  def test_none_returns_none() -> None:
      latents = np.random.RandomState(0).randn(20, 16).astype(np.float32)
      labels = np.array([0]*10 + [1]*10)
      out = sample_synthetic_latents(
          strategy="none", latents=latents, labels=labels,
          n_synthetic=5, seed=0,
      )
      assert out is None
  ```
- [ ] **Step 2: Run, expect ImportError.**
- [ ] **Step 3: Implement** `sample_synthetic_latents(strategy, latents, labels, n_synthetic, seed) -> dict[str, np.ndarray] | None`. The `mog` branch ports G4-ter logic ; `ae` does one-pass MSE on a tiny `nn.Linear(16, 8) → ReLU → Linear(8, 16)` autoencoder seeded by `seed + 40_000` ; `none` returns `None`.
- [ ] **Step 4: Run, expect 3/3 pass.**
- [ ] **Step 5: Lint + type.**
- [ ] **Step 6: Commit.**
  ```
  feat(g4-quater): add 3 RECOMBINE strategies
  ```

---

### Task 4 — Step 1 driver (H4-A : 5-layer deeper substrate)

**Files:**
- Create: `experiments/g4_quater_test/run_step1_deeper.py`
- Create: `tests/unit/experiments/test_g4_quater_step1.py` (smoke test only)

Driver structure mirrors `experiments/g4_ter_hp_sweep/run_g4_ter.py` but uses `G4HierarchicalDeeperClassifier` and runs only the richer-substrate sweep (no HP grid). 4 arms × 95 seeds = 380 cells. Profile keys : `g4-quater/step1/<arm>/<seed>`.

- [ ] **Step 1: Implement driver.** Reuse the cell-runner / retention / aggregator from `run_g4_ter._run_cell_richer` modulo the substrate swap. Output goes to `docs/milestones/g4-quater-step1-2026-05-03.{json,md}`.
- [ ] **Step 2: Smoke test (1 seed × 4 arms).**
  ```bash
  uv run python experiments/g4_quater_test/run_step1_deeper.py --smoke
  ```
  Expected : 4 cells in JSON, all `excluded=False`, wall_time < 30 s.
- [ ] **Step 3: Production run (380 cells).**
  ```bash
  uv run python experiments/g4_quater_test/run_step1_deeper.py
  ```
  Expected wall time ≈ 19 min on M1 Max.
- [ ] **Step 4: Lint + type.**
- [ ] **Step 5: Commit driver + milestone.**
  ```
  feat(g4-quater): step1 deeper substrate pilot
  ```

---

### Task 5 — Step 2 driver (H4-B : RESTRUCTURE factor sweep)

**Files:**
- Create: `experiments/g4_quater_test/run_step2_restructure_sweep.py`
- Create: `tests/unit/experiments/test_g4_quater_step2.py`

Reuses the existing 3-layer `G4HierarchicalClassifier` from G4-ter. Sweeps RESTRUCTURE factor ∈ {0.85, 0.95, 0.99}. 3 factors × 4 arms × 30 seeds = 360 cells. Profile keys : `g4-quater/step2/<arm>/<factor>/<seed>`.

- [ ] **Step 1: Implement driver** with an outer loop over `factor in (0.85, 0.95, 0.99)` calling `dream_episode_hier(..., restructure_factor=factor)`. Dump to `docs/milestones/g4-quater-step2-2026-05-03.{json,md}`.
- [ ] **Step 2: Smoke (1 factor × 1 seed × 4 arms).**
- [ ] **Step 3: Production (360 cells, ≈ 9 min).**
- [ ] **Step 4: Commit.**
  ```
  feat(g4-quater): step2 RESTRUCTURE sweep pilot
  ```

---

### Task 6 — Step 3 driver (H4-C : RECOMBINE strategy + placebo)

**Files:**
- Create: `experiments/g4_quater_test/run_step3_recombine_strategies.py`
- Create: `tests/unit/experiments/test_g4_quater_step3.py`

Reuses 3-layer head. Sweeps `RecombineStrategy ∈ {"mog", "ae", "none"}`. 3 strategies × 4 arms × 95 seeds = 1140 cells. Profile keys : `g4-quater/step3/<arm>/<strategy>/<seed>`.

- [ ] **Step 1: Implement driver** with strategy switch threaded through `dream_episode_hier` to `_recombine_step` (or its replacement using `sample_synthetic_latents` from Task 3). Dump to `docs/milestones/g4-quater-step3-2026-05-03.{json,md}`.
- [ ] **Step 2: Smoke (1 strategy × 1 seed × 4 arms).**
- [ ] **Step 3: Production (1140 cells, ≈ 32 min).**
- [ ] **Step 4: Commit.**
  ```
  feat(g4-quater): step3 RECOMBINE strategy pilot
  ```

---

### Task 7 — Aggregator + verdict

**Files:**
- Create: `experiments/g4_quater_test/aggregator.py`
- Create: `docs/milestones/g4-quater-aggregate-2026-05-03.{json,md}`

The aggregator loads the three step milestones and renders the H4-A / H4-B / H4-C verdicts per pre-reg §2-3.

- [ ] **Step 1: Implement** `aggregate_g4_quater_verdict(step1_path, step2_path, step3_path) -> dict` :
  - **H4-A** : Jonckheere on `step1` retention by arm, alpha = 0.0167.
  - **H4-B** : per-factor Jonckheere on `step2` retention, multiplicity alpha = 0.0056.
  - **H4-C** : Welch two-sided between `step3` strategies "mog" vs "none" arm-by-arm at alpha = 0.0167. **Failure to reject** → H4-C confirmed (RECOMBINE empty).
- [ ] **Step 2: Render markdown** with per-hypothesis verdict blocks + caveats.
- [ ] **Step 3: Run aggregator** on the three step dumps, write JSON + MD.
- [ ] **Step 4: Commit.**
  ```
  feat(g4-quater): aggregate verdict + milestone
  ```

---

### Task 8 — Paper 2 §7.1.6 EN + FR

**Files:**
- Edit: `docs/papers/paper2/results.md` (append §7.1.6)
- Edit: `docs/papers/paper2-fr/results.md` (append §7.1.6 mirror)

Insert §7.1.6 after §7.1.5. The text reports verdicts honestly :

```
## 7.1.6 G4-quater pilot (RESTRUCTURE+RECOMBINE empirical test — 2026-05-03)

Following the G4-ter §7.1.5 finding that RESTRUCTURE+RECOMBINE
remained spectator channels at the 3-layer scale, G4-quater
ran a sequential 3-step confirmatory pilot to distinguish
between three sub-hypotheses pre-registered in
[`docs/osf-prereg-g4-quater-pilot.md`](../../osf-prereg-g4-quater-pilot.md) :
H4-A substrate-depth, H4-B HP-calibration, H4-C theoretical-
emptiness.

[verdict tables filled at execution time from
docs/milestones/g4-quater-aggregate-2026-05-03.json]

Bottom line : [filled honestly per verdict — refutation,
recovery, or partial recovery — without spin].
```

The literal verdict text is filled at Task 8 execution time from the aggregate JSON.

- [ ] **Step 1: Edit paper2/results.md** with §7.1.6 EN.
- [ ] **Step 2: Edit paper2-fr/results.md** with §7.1.6 FR mirror.
- [ ] **Step 3: Verify heading parity** : both files should now have §7.1.5 and §7.1.6.
- [ ] **Step 4: Commit (EN+FR same commit).**
  ```
  docs(paper2): G4-quater 7.1.6 EN+FR
  ```

---

### Task 9 — DR-4 evidence revision

**Files:**
- Edit: `docs/proofs/dr4-profile-inclusion.md`

If H4-C confirmed (RECOMBINE empty), the proof needs an empirical addendum stating that the channel-inclusion claim `channels(P_max) ⊃ channels(P_equ)` is formally true but **empirically vacuous at this scale** for RECOMBINE. The Lemma DR-4.L (capacity-monotone metrics non-decrease) is **not refuted** because retention difference < 1e-3 across profiles is a tie not a regression — but the prediction "richer profile retains more" loses empirical support.

- [ ] **Step 1: Edit `dr4-profile-inclusion.md`** to add an evidence-amendment block citing the G4-quater milestone.
- [ ] **Step 2: Commit.**
  ```
  docs(proofs): DR-4 evidence revised post-G4-quater
  ```

---

### Task 10 — CHANGELOG + STATUS + EC bump (conditional)

**Files:**
- Edit: `CHANGELOG.md` ([Unreleased] empirical block)
- Edit: `STATUS.md` (G4 row)

- [ ] **Step 1: Add [Unreleased] / Empirical (G4-quater) block** to CHANGELOG with the three step verdicts.
- [ ] **Step 2: Update STATUS.md G4 row** :
  - If all three confirmed : `🔶 G4-quater PARTIAL — DR-4 ordering not recovered ; refutation strengthens PARTIAL lock`
  - If H4-A confirmed only : `🔶 G4-quater PARTIAL — depth-bound recovery ; STABLE only at 5-layer scope`
  - If H4-B confirmed only : `🔶 G4-quater PARTIAL — HP-window recovery ; STABLE only at factor=X scope`
  - Per pre-reg §6, EC stays PARTIAL across all outcomes ; FC stays at v0.12.0.
- [ ] **Step 3: Commit.**
  ```
  docs(g4-quater): CHANGELOG + STATUS verdict
  ```

---

### Task 11 — Self-review

**Files:** none modified.

- [ ] **Step 1: Spec coverage.** Every task in this plan landed (0–10 + this 11) ; every pre-reg hypothesis (H4-A / H4-B / H4-C) has a verdict in the aggregator.
- [ ] **Step 2: Placeholder scan.** `grep -rn "TBD\|TODO\|FIXME" experiments/g4_quater_test/` returns no hits introduced by this PR.
- [ ] **Step 3: Type consistency.** `RecombineStrategy = Literal["mog", "ae", "none"]` is the single source of truth ; threaded through Task 3 / Task 6 / Task 7 verbatim.
- [ ] **Step 4: Pre-reg fidelity.** OSF pre-reg lock (Task 1 commit) precedes all three step pilot commits (Tasks 4, 5, 6) ; verifiable via `git log --reverse`.
- [ ] **Step 5: DualVer.** No FC bump committed. EC stayed PARTIAL across all outcomes per pre-reg §6 (verified in `STATUS.md` and `CHANGELOG.md`). `pyproject.toml` version not modified.
- [ ] **Step 6: R1 contract.** `golden_hashes.json` either unchanged or commit-pointer-only-bumped (per project pattern). All step JSON dumps include 380 / 360 / 1140 R1-bit-stable run_ids.
- [ ] **Step 7: Honest reporting.** No verdict softened in paper §7.1.6 vs the aggregator JSON ; Welch failure-to-reject explicitly reported as "no evidence at this N" not as "hypothesis falsified".

---

## Out of scope

- **G5-bis (port G4-ter / G4-quater richer head to E-SNN)** — separate plan ; G4-quater stays MLX-only.
- **G6 Path A (Qwen-35B real LoRA fine-tune)** — separate plan ; Studio + KIKI-Mac_tunner required.
- **Deeper benchmarks (CIFAR-10, ImageNet)** — listed in pre-reg §6 as future work post-G4-quater verdict ; not in scope here.
- **DR-4 proof text revision** — only the empirical-evidence amendment block is in scope ; the Lemma DR-4.L formal statement is not edited.

## Open questions for the executor

1. **AE complexity.** If the Task 3 single-layer autoencoder fails to converge at one MSE pass, deviation : drop the `ae` strategy from Step 3 and run only `{mog, none}` ; document in the milestone caveats. The H4-C verdict still holds at reduced power.
2. **Step 1 5-layer convergence on N = 100 / task.** If `acc_initial` falls below 0.5 (random chance), increase per-task epochs from 3 to 5 in `train_task` ; document in step1 milestone.
3. **Compute overrun.** If total Step 3 wall time exceeds 60 min, fallback : reduce N from 95 to 60 for Step 3 only ; document deviation ; H4-C verdict tagged exploratory at reduced N.
4. **Cross-pollination from concurrent G5-bis (if executed in parallel).** None expected ; G4-quater touches only `experiments/g4_quater_test/` and the shared paper2 / status files.

---

## Self-review (pre-execution)

- **Header format** : `# G4-quater RESTRUCTURE+RECOMBINE Empirical Test Implementation Plan` + REQUIRED SUB-SKILL block + Goal / Architecture / Tech Stack — all present.
- **Bite-sized steps** : every task that produces code follows TDD red→green→commit ; each step is 2-5 minutes.
- **No placeholders** : verdict text in paper2 §7.1.6 is explicitly resolved at Task 8 execution time from the aggregator JSON ; not a "TBD" placeholder.
- **Type consistency** : `G4HierarchicalDeeperClassifier`, `sample_synthetic_latents`, `RecombineStrategy`, `aggregate_g4_quater_verdict` — all signatures defined once and used identically downstream.
- **Compute budget honest** : ≈ 60 min total on M1 Max, well under the 5-8h ceiling.
- **Pre-reg discipline** : Task 1 commits the pre-reg before any pilot run ; Tasks 4, 5, 6 production commits land later.
- **Honest reporting binding** : §7 of pre-reg explicitly forbids softening Welch failure-to-reject into "falsified" ; absence-of-evidence vs evidence-of-absence distinction is preserved.
- **Bilingual mirror** : Task 8 edits both EN and FR in the same commit per `docs/papers/CLAUDE.md`.
- **Append-only milestones** : G4-quater step dumps are new dated files ; no existing milestone is modified.
