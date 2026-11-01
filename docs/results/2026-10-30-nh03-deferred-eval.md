# NH-03 deferred evaluation — does the recency advantage survive a write-free delay ?

**Pre-registration** : `docs/milestones/2026-10-30-nh03-deferred-eval.md`,
annotated tag `prereg-nh03-deferred-20261030` (tag object `fc89288`,
commit `aa80d1f`), pushed before the run.
**PI decision** : GO 2026-10-30 13:57, macM3 CPU, one host, one backend.
**Run** : commit `aa80d1f` (clean tree), macM3 (M3 Pro, MLX 0.31.1,
CPU backend, `MLX_DISABLE_COMPILE=1`, `nice 19`, 2 workers : seeds
200–214 and 215–229), launched by a wait script once NH-81 and the
M5 grid had both finished (two consecutive free checks 10 min apart),
2026-10-30 19:39:25–19:41:25 UTC, 112 s and 117 s wall per worker.
300 cells (10 arms × 30 seeds), 0 excluded seeds, every cell spent
exactly 24 writes, every write traced to a logged DE, no write during
the delay (asserted), delay steps {d0 : 0, d1 : 188, d3 : 564}.
Delay-task test accuracy : mean 0.997, min 0.996 (the delay is
learned). Raw cells + analysis :
`docs/results/2026-10-30-nh03-deferred-eval.json`. These numbers are
not mixed with #69 (macM1 GPU) or #70 (electron-server CPU).

## Arm means (retention points, N = 30)

| substrate | level | early_block | early_spread | late_block | late_spread | uniform |
|---|---|---|---|---|---|---|
| TTT (primary) | d0 | 67.60 | 71.03 | 93.98 | 81.75 | 75.14 |
| | d1 | 74.48 | 77.62 | 91.15 | 86.42 | 80.72 |
| | **d3** | 68.40 | 72.24 | **87.78** | 82.15 | 75.62 |
| slow (secondary) | d0 | 69.43 | 78.79 | 95.12 | 92.64 | 85.43 |
| | d1 | 75.93 | 82.65 | 94.65 | 92.24 | 86.79 |
| | d3 | 68.68 | 77.08 | 92.22 | 88.69 | 81.86 |

## Pre-registered effects (mean, 95 % bootstrap CI over seeds)

| effect | TTT | slow |
|---|---|---|
| R_d0 (control gate) | +18.55 [+17.45, +19.61] | +19.77 [+18.72, +20.87] |
| R_d1 | +12.74 [+11.94, +13.53] | +14.15 [+13.39, +14.96] |
| **R_d3 (primary)** | **+14.65 [+13.77, +15.51]** | +17.57 [+16.65, +18.52] |
| ΔR_d3 = R_d3 − R_d0 | −3.91 [−4.51, −3.25] | −2.20 [−2.96, −1.42] |
| S_d0 | −4.41 [−5.58, −3.35] | +3.44 [+2.46, +4.40] |
| S_d3 | −0.89 [−1.48, −0.27] | +2.44 [+1.74, +3.09] |
| D_d3 = LB − ES | +15.54 [+14.48, +16.64] | +15.13 [+14.28, +16.00] |
| control gate (§6.1) | passed | passed |
| **verdict (§6)** | **recency_survives** | recency_survives |
| secondary : attenuates | yes | yes |
| secondary : spreading_positive | no | yes |

Calibration check (not part of the rule, added under the lab's
null-calibration checklist) : paired sign-flip null of the per-seed R
(10 000 draws, RNG 20261030). For every substrate and level the
observed R exceeds the null P99 (TTT d3 : 14.65 vs P99 7.04) and
p = 1/10 001 (the smallest resolvable value) ; R > 0 on 30/30 seeds at
every level. The control gate is empirical (it could fail), and the
delay-task accuracy shows the delay was learned.

## Reading

1. **The late-write advantage is not only an imminent-evaluation
   effect on this benchmark.** After one full task of write-free
   training, `late` still beats `early` by 14.7 points on the TTT
   substrate (79 % of its `d0` size) and by 17.6 on the slow substrate
   (89 %). The pre-registered verdict is `recency_survives` on both.
2. Recency does attenuate (ΔR_d3 < 0 on both substrates), so part of
   the #70 effect was closeness to the evaluation, but the larger
   part persists.
3. The curve is not monotonic : R is smallest at `d1` and recovers at
   `d3`. The early arms gain about 4–7 points of task-0 accuracy
   during the first epoch of the delay, then lose them. This shape was
   not predicted and is reported descriptively ; no mechanism is
   claimed.
4. On TTT, the spreading cost shrinks from −4.4 to −0.9 but stays
   negative (prediction P4 held ; issue #78's "does S become positive
   for TTT" : no). On slow weights, S stays positive.
5. **Still no TTT-specific advantage.** At `d3` TTT is at or below
   slow on every schedule (TTT − slow : early_block −0.3 [−1.3, +0.9],
   late_block −4.4 [−5.4, −3.5], uniform −6.3 [−7.9, −4.6]). The NH-03
   primary contrast `ttt_uniform − slow_early_block` (+6.9 [+5.3,
   +8.8] at `d3`) still compares two schedules as well as two
   substrates, and must not be read as a substrate effect.
6. Predictions : P1 (gate passes) held ; P2 (attenuation) held ; P3
   was not directional ; P4 (S_d3 ≤ 0 for TTT) held.

## Limits

One toy benchmark (G4-ter MLP, Split-FMNIST), one delay type (task 4
under a fixed pixel permutation : same labels and class pair, so the
interference on task 0 is of a specific kind, and the distance match
with the `d0` early writes is in steps, not in content), one TTT
variant (η = 0.01), one HP set (G4-ter C5), one backend (macM3 CPU),
N = 30. Only task 0 is evaluated. "Survives one task of delay" says
nothing about longer delays, other interference, or other substrates.

## Suggested follow-ups

- A delay with new classes (e.g. a held-out dataset) and a longer
  delay (2–3 tasks), to see whether R keeps its `d3` level.
- The 2 × 2 × delay on G4-quinto (CIFAR CNN) before any DR-0 wording
  ("write budget + placement relative to the next demand").
