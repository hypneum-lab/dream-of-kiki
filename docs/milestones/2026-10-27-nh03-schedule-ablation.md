# Pre-registration (L1) — NH-03 schedule ablation : recency vs spreading of consolidation writes

**Project** : dreamOfkiki (Hypneum Lab)
**Parent** : NH-03 pre-registration `docs/milestones/2026-10-25-nh03-ttt-vs-episodes.md`
  (tag `prereg-nh03-20261025`) and its result
  `docs/results/2026-10-25-nh03-ttt.md` (PR #69, issue #51)
**PI** : Clément Saillant — decision "lance tout" on 2026-10-27 16:41
**Level** : L1 (annotated git tag `prereg-nh03-sched-20261027` on the
  commit adding this file, pushed before the confirmatory run)
**Date drafted** : 2026-10-27
**c_version** : `C-v0.25.0+PARTIAL`

## 1. Question

NH-03 found that TTT continuous consolidation beats discrete episodes
by +4.70 points (95 % CI [+3.29, +6.08]) at an equal budget of 24
writes, and that spreading the writes over wake time helps both
substrates (+15.7 on slow weights, +6.9 on TTT fast weights). In the
"continuous" (uniform) schedule the writes are both **spread**
(interleaved with wake learning) and partly **recent** (the last ones
fall close to the end of each task, hence close to the final
evaluation after task 4). Which of the two carries the gain ?

**Statement (EN, locked)** : At an equal budget of 24 writes on the
G4-ter benchmark, the retention gain of continuous consolidation is
carried by the recency of the writes (late in each task) rather than
by their spreading over wake time.

## 2. Design — 2 × 2 schedule factorial + uniform reference

Benchmark, substrate, consolidation content and budget are those of
NH-03, unchanged : G4-ter Split-FMNIST, MLP 784-32-16-2, 3 epochs,
batch 64, lr 0.01, beta FIFO 256 / 32 per task, G4-ter C5 (one beta
mini-batch of 32 sampled with `seed + k` at the start of interval k,
5 replay-type writes + 1 rescale per interval), TTT fast weights
32 × 32 with inner lr **η = 0.01** (locked in NH-03, not
re-calibrated). Each of the 4 intervals k = 1..4 holds 6 writes ;
budget = **24 writes per run**, ledger-capped and asserted.

Let S be the number of wake steps of task k (563) and T = ⌊S/3⌋.
Schedules (write j = 0..5 is applied before wake step p_j ; p = S
means after the last wake step of task k) :

| schedule | position | form | p_j |
|---|---|---|---|
| `early_block` | early | block (one DE) | 0 (= NH-03 discrete) |
| `early_spread` | early | spread (micro-DEs) | ⌊j·T/6⌋, first third |
| `late_block` | late | block (one DE) | S, end of task |
| `late_spread` | late | spread (micro-DEs) | S − T + ⌊(j+1)·T/6⌋, last third |
| `uniform` | — | spread | ⌊j·S/6⌋ (= NH-03 continuous, reference) |

Crossed with 2 substrates : `ttt_*` (writes into the TTT fast weights,
**primary**) and `slow_*` (G4 replay + downscale on slow weights,
**secondary**) → 10 arms. The legacy NH-03 arm names map bit-exactly
onto `slow_early_block`, `slow_uniform`, `ttt_uniform`,
`ttt_early_block` (unit test). Content and order of writes are
identical across schedules ; only their timing changes.

## 3. Seeds

**Seeds 100–129 (N = 30), new**, disjoint from NH-03 confirmatory
seeds 0–29, NH-03 pilot seeds 1000–1004 and the timing-probe seed
2000. Exclusion : seed dropped iff `acc_initial < 0.5` (arm-free).

## 4. Statistic

Per substrate and per seed, with retention points EB, ES, LB, LS :

- recency main effect **R = (LB + LS − EB − ES) / 2**
- spreading main effect **S = (ES + LS − EB − LB) / 2**
- **D = R − S = LB − ES** (late block vs early spread : recent-but-
  bunched against spread-but-early, the two arms named in the PI
  question)

Estimate = mean over seeds ; 95 % percentile bootstrap CI over seeds
(10 000 resamples, RNG seed 20261027). Also reported : the
interaction (LB − EB) − (LS − ES), arm means, and uniform minus each
cell.

## 5. Decision rule (primary = TTT substrate ; same rule reported for slow)

- **"recency"** : R > 3 points, CI_R lower bound > 0, and CI_D lower
  bound > 0 ;
- **"spreading"** : S > 3 points, CI_S lower bound > 0, and CI_D upper
  bound < 0 ;
- **"both"** : R > 3 and S > 3 with both CI lower bounds > 0, and
  CI_D containing 0 ;
- **"inconclusive"** otherwise.

## 6. What the outcomes mean

- **recency** : the NH-03 "continuous > bounded" result is mostly a
  timing-to-evaluation effect ; a bounded episode placed late is as
  good or better. DR-0 amendment should then say "budget + schedule
  relative to the next demand", not "continuous consolidation".
- **spreading** : interleaving writes with learning is the active
  ingredient ; supports the continuous-consolidation reading of DR-0.
- **both / inconclusive** : report as such ; no DR-0 wording follows
  from this ablation alone.

## 7. Compute and host

PI constraint 2026-10-27 : **electron-server only, CPU**
(`MLX_DISABLE_COMPILE=1`, `nice 19`, **≤ 2 workers** — kiki-flow and
NH-01 share the machine ; macM1 offline). Numbers are **not mixed**
with the NH-03 macM1-GPU numbers (R1 : different backend) ; all
contrasts are within this run.

Timing probe (done before lock, seed 2000, 10 arms, 1 worker,
load ≈ 2.5) : 13.0 s / cell → 300 cells / 2 workers ≈ 33 min
(≤ 1 h with contention), well under the 12 h ceiling, so the full
N = 30 design runs unreduced. Workers split seeds 100–114 / 115–129 ;
their JSONs are merged and analysed by `run_nh03.py --mode merge`.

Disclosure : the probe produced one retention per arm on seed 2000
(slow : EB 0.628, ES 0.733, LB 0.963, LS 0.890, U 0.797 ; TTT : EB
0.624, ES 0.645, LB 0.913, LS 0.760, U 0.678). The statistic and the
decision rule (`ablation_analysis.py`, unit-tested) were written
before the probe ; nothing was changed after it. Seed 2000 is not
part of the analysis.

## 8. Provenance

- Driver : `experiments/nh03_ttt_vs_episodes/run_nh03.py`
  (`--mode ablation --ttt-lr 0.01 --seeds …`, then `--mode merge`)
- Analysis : `experiments/nh03_ttt_vs_episodes/ablation_analysis.py`
- Tests : `tests/unit/experiments/test_nh03_driver.py`
- Results (after the run) :
  `docs/results/2026-10-27-nh03-schedule-ablation.{md,json}`
