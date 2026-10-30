# Pre-registration (L1) — NH-03 deferred evaluation : does the recency advantage survive a write-free delay ?

**Project** : dreamOfkiki (Hypneum Lab)
**Parents** : NH-03 `docs/milestones/2026-10-25-nh03-ttt-vs-episodes.md`
  (tag `prereg-nh03-20261025`, PR #69) and the schedule ablation
  `docs/milestones/2026-10-27-nh03-schedule-ablation.md`
  (tag `prereg-nh03-sched-20261027`, PR #70) ; issue
  hypneum-lab/dream-of-kiki#78
**PI** : Clément Saillant — decision of 2026-10-30 13:39 : write the
  pre-registration of the deferred variant and cost it with a short
  pilot ; no confirmatory run in this step
**Level** : L1 (annotated git tag `prereg-nh03-deferred-20261030` on
  the commit adding this file, pushed before any confirmatory run ;
  not an OSF deposit)
**Date drafted** : 2026-10-30
**c_version** : `C-v0.25.0+PARTIAL`

## 1. Question

The schedule ablation (#70) found that moving the same 24 writes to
the end of each task adds R = +18.3 points [+17.1, +19.4] on the TTT
fast weights and +19.0 [+18.1, +20.0] on slow replay + downscale,
which explains the NH-03 +4.70 (#69) without any TTT-specific
advantage. On G4-ter the only evaluation of task 0 comes right after
task 4, so "late in the task" also means "just before the
evaluation" : the benchmark rewards recency by construction.

**Statement (EN, locked)** : On the G4-ter Split-FMNIST benchmark, at
an equal budget of 24 writes, the recency advantage of late
consolidation writes on the TTT fast-weight substrate (R, defined as
in the schedule ablation) remains larger than 3 retention points when
task 0 is evaluated after a write-free delay of one full task of wake
training.

## 2. How recency is neutralised

The benchmark, substrates, consolidation content, write budget and
the 10 arms are those of the schedule ablation, unchanged
(`ttt_*` / `slow_*` × `early_block`, `early_spread`, `late_block`,
`late_spread`, `uniform` ; G4-ter MLP 784-32-16-2, 3 epochs, batch
64, lr 0.01, beta FIFO 256 / 32 per task, G4-ter C5, TTT η = 0.01 ;
24 writes per cell, ledger-capped and asserted). Code :
`run_cell(..., delay_task=...)` ; without a delay task the cell is
bit-identical to the schedule ablation (unit test).

After task 4 and its interval's writes, the network trains on a
**6th "delay" task** with **no consolidation write and no beta push**
(plain wake SGD on the slow weights ; `W_f` is not in the wake
optimiser and stays frozen) :

- delay task = task 4 (classes {8, 9}, labels {0, 1}) under a fixed
  pixel permutation (`np.random.default_rng(20261030)`, identical for
  every seed and arm) : new input statistics, no task-0 image, no new
  class, same size (12 000 images → 564 wake steps over 3 epochs,
  batch order from the cell seed as for every task) ;
- task-0 test accuracy is read at three **delay levels** :
  `d0` = 0 steps (the schedule-ablation measurement, positive
  control), `d1` = 1 epoch (⌊564/3⌋ = 188 steps), `d3` = 3 epochs
  (564 steps, one full task — **primary**). The levels are nested
  checkpoints of the same trajectory (the delay writes nothing, so a
  read cannot affect what follows).

Why `d3` neutralises recency : at `d3` the writes of interval 4 under
`late_block` are as far from the evaluation (one full task of wake
training) as the `early_block` writes of interval 4 were from the
evaluation at `d0`. If R were only "closeness to the next evaluation",
it should shrink towards the residual offset between early and late
writes once both are a full task away ; if late writes carry a durable
benefit (the last item learned is consolidated without being
overwritten), R should survive.

Retention at level L = `acc_task0(L) / acc_initial` (same
`acc_initial`, same exclusion rule : seed dropped iff
`acc_initial < 0.5`, arm-free). The delay task's own test accuracy is
recorded (sanity : the delay must actually be learned).

## 3. Seeds and budget

- Confirmatory seeds **200–229 (N = 30)**, new, disjoint from 0–29
  (NH-03), 100–129 (ablation), 1000–1004 (NH-03 pilot), 2000
  (ablation probe) and the pilot seeds below (unit test).
- Pilot seeds 3000–3001 (§8), not analysed.
- Budget : 10 arms × 30 seeds = **300 cells**, 24 writes each, plus
  564 write-free delay steps and 2 extra task-0 evaluations per cell.

## 4. Statistic

Per substrate (`ttt` primary, `slow` secondary), delay level L and
seed, with retention points EB, ES, LB, LS at L :

- recency **R_L = (LB + LS − EB − ES) / 2**
- spreading **S_L = (ES + LS − EB − LB) / 2**
- **D_L = LB − ES**
- attenuation **ΔR_L = R_L − R_d0** (paired by seed), and ΔS_L likewise.

Estimate = mean over seeds ; 95 % percentile bootstrap CI over seeds
(10 000 resamples, RNG seed **20261030**). Also reported : arm means
per level, uniform − early_block per level, and (descriptive)
TTT − slow per schedule and per level, including the NH-03 primary
contrast `ttt_uniform − slow_early_block` at each level.
Implementation : `experiments/nh03_ttt_vs_episodes/deferred_analysis.py`
(unit-tested, written before the pilot).

## 5. Margin

The decision margin is **3 retention points**, unchanged from NH-03
(§6, catalogue anchor "continuous beats bounded by > 3 points") and
the schedule ablation (§5). It is kept for comparability : the
question is whether an effect that passed this margin at `d0` still
passes it at `d3`. No equivalence (TOST) test is used ; the only
other tests are simple sign tests on the CI.

## 6. Decision rule (primary = TTT substrate, level `d3`)

1. **Control gate** : R_d0 > 3 and CI_R_d0 lower bound > 0 (the
   schedule-ablation recency replicates on the new seeds). If the gate
   fails, the verdict is **"invalid_control"** and no conclusion on the
   delay is drawn.
2. **"recency_survives"** : R_d3 > 3 and CI_R_d3 lower bound > 0.
3. **"recency_below_margin"** : CI_R_d3 upper bound < 3.
4. **"inconclusive"** otherwise.

Secondary labels (reported for both substrates, no margin, no
multiplicity correction claimed) :

- **"attenuates"** : CI of ΔR_d3 entirely below 0 ;
- **"spreading_positive"** : CI of S_d3 entirely above 0 (issue #78 :
  "does S become positive for TTT ?").

The same rule is reported for the slow substrate as secondary. `d1`
is descriptive (shape of the curve).

## 7. Predictions (before the run) and what counts as a null

Predictions of the drafting author, recorded for transparency ; they
do not enter the decision :

- P1 : the control gate passes (R_d0 ≈ +18, as in #70).
- P2 : recency attenuates (ΔR_d3 < 0 on both substrates) — forgetting
  of task 0 under further interference is expected to be largest
  just after the last write.
- P3 : no directional prediction on "survives" vs "below margin" ;
  that is the question.
- P4 : S_d3 stays ≤ 0 for TTT (weak prediction).

**Null** : the label "recency_below_margin" on the TTT substrate,
i.e. after one task of write-free training the late-write advantage
is smaller than 3 points with 95 % confidence. It would mean that the
#70 recency effect is, on this benchmark, mostly a property of an
imminent evaluation, and that no schedule recommendation ("place the
episode late") follows from it. "inconclusive" is not a null and is
reported as such. "invalid_control" means the design failed, not that
the hypothesis is false.

## 8. Compute, host and pilot

**Pilot (done before the tag, role declared here)** : macM3 (M3 Pro
12 cores, 18 GB), `nice 19`, 2 workers, `MLX_DISABLE_COMPILE=1`,
seeds 3000 (GPU backend) and 3001 (CPU backend, `mx.cpu`), 10 arms
each, `--mode deferred_pilot`, commits `d809157` (first pass) and
`9a53f43` (timing passes ; the only change between them silences a
console print that crashed after the JSON was written). Its only purpose is
to time cells and check that the variant runs end to end (24 writes,
no write during the delay, delay task learned). **No scientific
conclusion is drawn from it** ; its retentions are not analysed and
the analysis code was not changed after it. Timing : see §8.1.

### 8.1 Pilot timing

Measured 2026-10-30 12:46–12:49 UTC, machine otherwise idle (the M5
workers were not running after a reboot), load ≈ 2.5 :

| configuration | per cell (mean / max) | 10 cells |
|---|---|---|
| 1 GPU worker + 1 CPU worker in parallel — GPU | 1.60 s / 2.22 s | 16.0 s |
| same — CPU | 0.67 s / 0.69 s | 6.7 s |
| 2 CPU workers in parallel | 0.76 s / 0.78 s | 7.6 s |

Checks : every cell spent exactly 24 writes, every write traced to a
logged DE, no write during the delay (asserted), delay steps
{d0 : 0, d1 : 188, d3 : 564}, delay-task test accuracy ≥ 0.997 in
every cell (the delay is learned), no excluded seed, merge and
deferred analysis run end to end.

Estimated confirmatory cost : 300 cells × 0.76 s / 2 workers ≈ 2 min
on macM3 CPU (≈ 8 min on one GPU worker) ; with the four M5 workers
running at `nice 19` on the same 12 cores, a ×3 slowdown still gives
< 10 min. Ceiling for the run : 2 h wall, beyond which it is stopped
and reported.

**Confirmatory host : PI decision, not fixed here.** Whatever the
host, all 300 cells run on one host and one MLX backend ; numbers are
not mixed with the #69 (macM1 GPU) or #70 (electron-server CPU) runs
(R1), and the `d0` replication is read within this run. Never
GrosMac, never KXKM ; `nice 19`, ≤ 2 workers on shared machines ;
workers split seeds 200–214 / 215–229, merged with `--mode merge`.

## 9. Deviations from issue #78

- Delay form fixed as a 6th task (permuted task 4) rather than
  "neutral steps" : without new input there is no interference to
  delay against, and re-training on task 4 would extend task 4
  rather than delay the evaluation.
- Three levels (0, 1 epoch, 3 epochs) as nested checkpoints of one
  trajectory, so the curve costs one delay per cell.
- Decision rule and margin : §5–§6.

## 10. Provenance

- Driver : `experiments/nh03_ttt_vs_episodes/run_nh03.py`
  (`--mode deferred --ttt-lr 0.01 --seeds …`, then `--mode merge`)
- Analysis : `experiments/nh03_ttt_vs_episodes/deferred_analysis.py`
- Tests : `tests/unit/experiments/test_nh03_driver.py`
- Results (after the run) :
  `docs/results/<date>-nh03-deferred-eval.{md,json}`
