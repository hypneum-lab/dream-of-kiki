# NH-03 schedule ablation — recency vs spreading of consolidation writes

**Pre-registration** : `docs/milestones/2026-10-27-nh03-schedule-ablation.md`,
annotated tag `prereg-nh03-sched-20261027` (commit `6364719`), pushed
before the run.
**Run** : commit `6364719`, electron-server (CPU, MLX 0.31.1,
`MLX_DISABLE_COMPILE=1`, `nice 19`, 2 workers : seeds 100–114 and
115–129), 2026-10-27 17:48–18:26 CET, ≈ 37 min wall per worker
(load 3.4 at launch → 11.5 at the end, machine shared with kiki-flow
and NH-01). 300 cells (10 arms × 30 seeds), 0 excluded seeds,
every cell spent exactly 24 writes, every write traced to a logged
DE. Raw cells + analysis : `docs/results/2026-10-27-nh03-schedule-ablation.json`.
These CPU numbers are not mixed with the NH-03 macM1-GPU numbers ;
all contrasts below are within this run.

## Arm means (retention points, N = 30)

| substrate | early_block | early_spread | late_block | late_spread | uniform |
|---|---|---|---|---|---|
| TTT fast weights (primary) | 69.51 | 72.68 | **94.98** | 83.77 | 77.10 |
| slow replay + downscale (secondary) | 70.67 | 80.48 | **95.88** | 93.28 | 86.79 |

## Pre-registered effects (mean, 95 % bootstrap CI over seeds)

| effect | TTT | slow |
|---|---|---|
| recency R = late − early | **+18.28** [+17.10, +19.42] | **+19.01** [+18.06, +19.99] |
| spreading S = spread − block | **−4.02** [−4.99, −3.10] | +3.60 [+2.74, +4.48] |
| D = late_block − early_spread | **+22.30** [+20.78, +23.91] | +15.41 [+14.07, +16.79] |
| interaction (LB − EB) − (LS − ES) | +14.38 [+13.12, +15.77] | +12.41 [+11.40, +13.39] |
| **verdict (§5)** | **recency** | **recency** |

## Reading

1. **The NH-03 gain is a recency effect.** For the TTT substrate,
   moving the same 24 writes from the start to the end of each task
   adds 18.3 points, while spreading them *costs* 4.0 points on
   average. The single best schedule on both substrates is one bounded
   block at the end of the task (95–96 points), i.e. a discrete episode
   placed just before the next demand — not continuous consolidation.
2. Spreading helps only when the writes stay early (TTT early_spread −
   early_block = +3.2 [+2.5, +3.8] ; slow +9.8) ; once writes are late,
   bunching them is better (the large positive interaction). The
   "uniform" schedule of NH-03 sits in between because a part of its
   writes is late.
3. NH-03 replicates on this backend and these seeds : slow uniform −
   slow early_block = +16.1 [+15.0, +17.2] (NH-03 : +15.7) ;
   TTT uniform − slow early_block = +6.4 [+4.7, +8.2] (NH-03 primary :
   +4.7). At a late bounded clock the two substrates are close
   (TTT − slow = −0.9 [−1.9, 0.0]).
4. Consequence for DR-0 : the episode is defined neither by its
   budget alone nor by "continuity" ; at a fixed write budget what
   matters here is *when the writes land relative to the evaluation*.
   A DR-0 amendment should read "write budget + placement relative to
   the next demand" ; the NH-03 result does not support a
   "continuous consolidation beats episodes" reading.

## Limits

One toy benchmark (G4-ter, Split-FMNIST, retention measured on task 0
right after task 4), so "late" also means "close to the only
evaluation" : the benchmark rewards recency by construction. A
benchmark with evaluations at unannounced times, or a delay between
the last write and the evaluation, is needed before generalising.
One TTT variant (η = 0.01 from NH-03), one HP set (G4-ter C5), CPU
backend only, N = 30.

## Suggested follow-ups

- Delayed-evaluation variant (extra wake training on a 6th task, or
  a gap, after the last write) to test whether the late-block
  advantage survives when the evaluation is not imminent.
- Repeat the 2 × 2 on G4-quinto (CIFAR CNN).
