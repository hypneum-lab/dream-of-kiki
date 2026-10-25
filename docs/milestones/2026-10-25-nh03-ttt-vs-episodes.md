# Pre-registration (L1) — NH-03 : TTT continuous consolidation vs discrete dream episodes at equal write budget

**Project** : dreamOfkiki (Hypneum Lab)
**Hypothesis** : NH-03 (catalogue `hypneum-lab/.github`
  `docs/research/2026-10-24-revue-hypotheses/synthese/02-catalogue-nouvelles-hypotheses.md`,
  rank 13) ; issue hypneum-lab/dream-of-kiki#51
**PI** : Clément Saillant — decision "go NH-03" on 2026-10-25 19:00
**Level** : L1 (git-locked pre-registration : annotated tag
  `prereg-nh03-20261025` on the commit that adds this file, pushed
  before any confirmatory run ; not an OSF deposit)
**Date drafted** : 2026-10-25
**c_version** : `C-v0.25.0+PARTIAL`

## 1. Question

Review report 2026-10-24 (§3 row 1, §7) proposes reformulation n° 1
of DR-0 : DR-0 should be an explicit operational *budget* (writes,
traffic, duration) that selects the operator, not a bounded
off-line *slot* ; a test-time-training (TTT) module would then be a
conforming substrate. NH-03 asks whether a dream episode (DE) is
defined by its **budget** or by its **clock** : at an equal number
of consolidation writes, does spreading the writes continuously
over wake time into TTT fast weights (Titans / TTT-Linear style)
retain more than concentrating them in bounded discrete episodes ?

Catalogue prediction used as the decision anchor : *if continuous
beats bounded by > 3 points, DR-0 is rewritten as a write budget
and TTT becomes the 5th conforming substrate.*

**Statement (EN, locked)** : On the G4-ter Split-FMNIST benchmark,
at an equal budget of 24 consolidation writes per run, continuous
TTT fast-weight consolidation (arm b) yields task-0 retention more
than 3 points higher than discrete replay + downscale episodes
(arm a), with a 95 % bootstrap CI over seeds excluding 0.

## 2. Substrate under test

`kiki_oniric/substrates/ttt_fast_weights.py` (`ttt_fast_weights`,
MLX). A fast-weight matrix `W_f` (32 × 32) sits after the first
hidden layer as a residual map `h1 -> h1 + h1 W_fᵀ`, is initialised
to 0 (so the network is exactly the G4-ter
`G4HierarchicalClassifier` at start) and is **excluded from the wake
optimiser** ; it is written only by the substrate handlers :
`replay` = one inner gradient step (TTT descent) on `W_f` under the
cross-entropy of a beta mini-batch, `downscale` = forgetting gate
`W_f <- f W_f`, `restructure` = row topology edit, `recombine` =
seeded latent mix (no write). Every write passes the S2 finite guard
(abort-on-violation) and is metered by a `WriteLedger` with a hard
cap (DR-0). C+ (`conforms(S)` = typed ∧ axiom tests ∧ invariants
S1/S2/S3 enforceable) is checked by
`tests/conformance/axioms/test_dr3_ttt_substrate.py` (C1/C2/C3
sections). Deviation from TTT-Linear : the inner objective is
*supervised* on beta records (labels present in beta) instead of a
self-supervised reconstruction loss, so that arms (a) and (b) read
exactly the same information ; the "test-time" character is that
`W_f` is updated during the wake stream by inner descent, not by
the outer optimiser.

## 3. Benchmark and arms

Benchmark = G4-ter richer substrate, unchanged : Split-FMNIST
5 binary tasks (classes {0,1}…{8,9} remapped to {0,1}), MLP
784-32-16-2, wake SGD 3 epochs / batch 64 / lr 0.01, beta FIFO
(capacity 256, 32 exemplars pushed after each task, `seed + 5000`
fill RNG), retention = `acc_final / acc_initial` on task-0 test set,
consolidation HPs = G4-ter C5 (replay batch 32 sampled once per
interval with `seed + k`, 5 replay steps, replay lr 0.01, rescale
factor 0.95). This is the benchmark on which replay + downscale
(P_min ≈ P_equ, retention 0.7065 vs baseline 0.5869) gave
`g_h2 = +2.77` (`docs/milestones/g4-ter-pilot-2026-05-03.md`).
FMNIST IDX sha256 : train-images `3aede38d…a84`, train-labels
`a04f1713…845`, t10k-images `346e55b9…073`, t10k-labels
`67da17c7…dd5` (canonical Zalando files).

Consolidation happens in the 4 intervals k = 1..4 (during / before
training task k). Arms :

| arm | role | writes per interval (×4) | target | clock |
|---|---|---|---|---|
| `baseline` | reference | 0 | — | — |
| `A_discrete` | **(a)** | 5 replay steps + 1 downscale, one DE (REPLAY, DOWNSCALE) before task k | slow weights | bounded |
| `B_ttt_continuous` | **(b)** | 5 TTT-descent steps + 1 forgetting gate, each a single-op micro-DE | fast weights `W_f` | continuous : write j at wake step ⌊j·S/6⌋, S = wake steps of task k |
| `C_sgd_matched` | **(c)** | 6 SGD steps on one 32-sample mini-batch of task k (no beta access, lr 0.01) | slow weights | at (a)'s slot |
| `A_continuous` | exploratory | (a)'s writes | slow weights | (b)'s clock |
| `B_ttt_chunked` | exploratory | (b)'s writes | fast weights | (a)'s clock (one DE) |

Arms (a) and (b) consume the *same* beta mini-batch per interval
(same records, same order) : they differ only by target parameters
and clock. Task-0 training is identical across arms (consolidation
starts after it), so `acc_initial` and the exclusion rule are
arm-free.

## 4. Equal budget — primary definition

**Primary budget = number of consolidation writes**, where one write
is one application of an update to consolidation-eligible
parameters : one gradient step on one mini-batch (≤ 32 records), or
one multiplicative rescale. Budget per run = **24 writes** (6 per
interval × 4), identical for (a), (b), (c) and the exploratory arms,
enforced by the ledger cap and asserted per cell (a run that does
not spend exactly 24 writes aborts).

Justification : (i) it is the unit named by the DR-0 reformulation
under test ("budget d'écriture") and by DR-0's accounting reading
(every output delta traced to a bounded DE) ; (ii) it is
substrate-agnostic — FLOPs would penalise the larger slow-network
replay step and make fast-weight writes look nearly free, so a
FLOP-matched design would hand (b) several times more writes ;
(iii) it is exactly countable and identical under both clocks,
whereas wall-time is host-dependent (and excluded as a criterion).
Secondary meters (reported, not equalised) : sample-gradients
(records × steps ; equal for a and b = 640), parameter elements
written, estimated FLOPs (b uses fewer FLOPs per write than a, so an
advantage of b would hold a fortiori at equal FLOPs).

## 5. TTT inner learning rate (calibrated before lock)

The only free parameter of (b) is the inner lr `η`. It was chosen on
pilot seeds **1000–1004** (disjoint from the confirmatory seeds) by
the rule : argmax of mean `B_ttt_continuous` retention over the grid
{0.01, 0.03, 0.1, 0.3, 1.0}, ties to the smaller value, any S2 abort
disqualifies. The argmax fell on the lower edge (0.01), so the grid
was extended once downward with {0.001, 0.003} before locking (same
rule, same seeds) (`run_nh03.py --mode calibrate`, electron-server CPU,
MLX 0.31.1). Pilot result :

| η | mean retention (b), seeds 1000–1004 | per-seed |
|---|---|---|
| 0.001 | 0.601 | 0.655, 0.584, 0.536, 0.654, 0.573 |
| 0.003 | 0.642 | 0.698, 0.616, 0.614, 0.697, 0.587 |
| **0.01** | **0.767** | 0.824, 0.731, 0.812, 0.812, 0.659 |
| 0.03 | 0.762 | 0.938, 0.530, 0.958, 0.603, 0.780 |
| 0.1 | 0.591 | 0.514, 0.637, 0.582, 0.524, 0.696 |
| 0.3 | 0.673 | 0.515, 0.864, 0.492, 0.579, 0.915 |
| 1.0 | 0.667 | 0.513, 0.777, 0.646, 0.523, 0.876 |

No S2 abort. For reference, arm (a) on the same pilot seeds : mean
retention 0.693 (electron-server CPU ; not a confirmatory number).
Raw pilot JSON is kept off-repo (`~/scratch/nh03/out/calibration*.json`
on electron-server) and summarised here.

**Locked : η = 0.01** for (b) and `B_ttt_chunked`. Arm (a)
keeps the G4-ter C5 HPs (themselves selected by the G4-ter HP sweep).
Choosing η to maximise (b) favours the alternative hypothesis ; this
is deliberate (a fair shot for continuous consolidation) and is
disclosed here.

## 6. Seeds, statistic, decision rule

- Seeds : **0–29 (N = 30)**, the G4-ter seed set ; ≥ 5 required.
- Exclusion : seed dropped from all contrasts iff `acc_initial < 0.5`
  (G4 rule ; arm-free, see §3).
- Statistic : per seed `d_s = 100 · (retention_b − retention_a)`
  (retention points) ; estimate = mean of `d_s` ; 95 % percentile
  bootstrap CI over seeds (10 000 resamples, RNG seed 20261025).
  Also reported : SD, paired Hedges `g_z`, number of seeds with
  `d_s > 3`.
- **Decision (primary contrast b − a)** :
  - mean > 3 points **and** CI lower bound > 0 → **"continuous > bounded"** ;
  - |mean| < 1 point **and** CI ⊂ [−3, +3] → **"equivalent at this scale"** ;
  - otherwise → **"inconclusive"**. If the CI lies entirely below 0
    the result stays "inconclusive" under this rule, with the
    descriptive qualifier "bounded > continuous".
- Secondary contrasts (same statistic, descriptive, no extra
  decision) : a − c, b − c, a − baseline, b − baseline, c − baseline.
- Exploratory (clock vs substrate decomposition) : A_continuous − a
  (clock, slow substrate), b − B_ttt_chunked (clock, TTT substrate),
  B_ttt_chunked − a (substrate, bounded clock).
- Catalogue cross-reference (reported, not decisional) : "continuous
  beats bounded by > 3 points in ≥ 4/5 seeds" → fraction of seeds
  with `d_s > 3`.

## 7. What the outcomes mean

- **continuous > bounded** : at this scale the episode is better
  defined by its budget than by its clock ; supports rewriting DR-0
  as a write budget and admitting `ttt_fast_weights` as 5th
  conforming substrate (formal-axis bump to be proposed separately,
  with the DR-0 amendment issue).
- **equivalent** : the clock does not matter at equal writes
  (budget-defined episode, but no reason to prefer TTT) ; DR-0 as a
  budget is sufficient, the slot reading is not needed.
- **inconclusive / bounded > continuous** : no support for the
  continuous-consolidation threat on this benchmark ; the
  exploratory decomposition says whether clock or substrate carries
  the gap.

Scope : one toy benchmark (G4-ter MLP, Split-FMNIST, synthetic
scale), one TTT variant, N = 30. No claim beyond this scope.

## 8. Compute and host guards

Development and tests : electron-server (CPU, `nice 19`, MLX
0.31.1 with `MLX_DISABLE_COMPILE=1`). Confirmatory run : macM1 GPU
(`~/scratch/nh03/`), launched only if : no N10-C campaign running
(`pgrep -f run_campaign` empty and no file under
`~/scratch/n10c-run/campaign/` modified in the last 10 min) — N10-C
has absolute priority and our run is stopped cleanly if it
restarts ; other sessions' Python jobs are never touched ; shared GPU
→ `nice 19`, ≤ 6 GB ; free memory < 4 GB → wait (10-min checks, up
to 3 h) or fall back to electron-server CPU (reported). Never
GrosMac, never KXKM. Expected cost : 6 arms × 30 seeds = 180 cells
(≈ 2 s/cell on M1 Max GPU, ≈ 16 s/cell on electron-server CPU) —
well under the 8 h ceiling. Wall-time is never a criterion.
Numbers from the two hosts are not mixed (R1).

## 9. Deviations from the catalogue / issue #51

- Benchmark : G4-ter MLP (the benchmark of the `g = +2.77`
  replay + downscale effect) instead of "CNN G4" ; host macM1 instead
  of KXKM-GPU (PI decision 2026-10-25).
- Budget : writes (primary) instead of "FLOPs égaux" (justified §4 ;
  FLOPs reported).
- The 1-pass TOST `g < 0.29` and 5-epoch sub-predictions of the
  catalogue are not tested here ; the decision rule is the one of §6
  (PI brief 2026-10-25).
- TTT inner objective supervised on beta (§2).

## 10. Provenance

- Substrate : `kiki_oniric/substrates/ttt_fast_weights.py`
- Classifier : `experiments/nh03_ttt_vs_episodes/ttt_classifier.py`
- Driver : `experiments/nh03_ttt_vs_episodes/run_nh03.py`
  (`--mode confirmatory --ttt-lr 0.01`)
- Analysis : `experiments/nh03_ttt_vs_episodes/analysis.py`
- Tests : `tests/conformance/axioms/test_dr3_ttt_substrate.py`,
  `tests/unit/experiments/test_nh03_driver.py`
- Results (after the run) : `docs/results/2026-10-25-nh03-ttt.md`
  + JSON sibling.
