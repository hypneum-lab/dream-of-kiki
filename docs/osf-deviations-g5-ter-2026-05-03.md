# OSF deviation log — G5-ter pilot — 2026-05-03

**Parent pre-registration** : `docs/osf-prereg-g5-ter-spiking-cnn.md`
**Date logged** : 2026-05-03
**PI** : Clement Saillant

This file is **append-only**. Each entry documents one deviation
from the locked pre-registration that occurred during the G5-ter
pilot run. Per `docs/CLAUDE.md`, dated immutables are corrected by
appending a superseding entry, never by rewriting the past.

## D-1 — CIFAR-10 train-shard subsampling (2026-05-03)

### Context

The pre-registration §4 (Dataset) implicitly assumes the full
Split-CIFAR-10 5-task shards (10 000 train + 2 000 test per task)
as returned by
`experiments.g4_quinto_test.cifar10_dataset.load_split_cifar10_5tasks_auto`.

The Plan G5-ter compute note estimated 30-60 min on M1 Max for the
40-cell Option B sweep, with an abort threshold at 90 min. On the
actual M1 Max (32 GB, pure-numpy substrate), one full cell measured
~6-7 min wall time (5 tasks × 2 epochs × 157 batches/task ×
~210 ms/STE step), which extrapolates to **~4-5 hours for 40 cells**
— well past the abort threshold.

### Deviation

For the milestone-publishing run, the train shard is subsampled
deterministically to **1500 examples per task** via a per-cell,
per-task numpy seed. The test shard is **left intact** (2 000
examples per task) so the retention denominator is comparable
across cells and across the cross-substrate aggregator (which loads
the G4-quinto Step 2 MLX milestone whose retention vector was
computed on the full test shard).

CLI flag : `--n-train-per-task 1500`. Default value (0) preserves
the full shard for any future Option A confirmatory follow-up that
has the compute envelope.

### Justification

1. The deviation envelope §9.1 inherited from G4-quinto ("HF
   parquet fallback") establishes the precedent that data-acquisition
   amendments are documented in this dated file rather than blocking
   the run.
2. The 4-arm × N=10 seeds × HP combo C5 design is preserved
   verbatim ; only the train-shard cardinality is reduced.
3. Power : N=10 per arm (own-substrate) gives a min detectable
   Hedges' g ≈ 1.27 at 80 % power, two-sided α = 0.0125. The G4-ter
   MLX richer-head reference effect was `g ≈ +2.77`, well above
   the detection floor ; reducing the train shard by a factor of
   ~6.7× should not move that effect below the floor by more than
   ~30 % in expectation.
4. Per-class balance is preserved by the deterministic random
   subsampling (each class is sampled in proportion to its
   prevalence in the canonical 5 000-per-class CIFAR-10 split).

### Recorded values

- Per-cell wall (measured smoke, 1 seed, P_equ arm, full 5 tasks at
  1500/task) : ~75 s
- Sweep wall extrapolation : 40 cells × ~75 s ≈ 50 min
- Aborted threshold not crossed.
- The subsample seed schedule is `seed + 100 * task_idx` per cell ;
  this is reproducible bit-for-bit per (`c_version`, `profile`,
  `seed`, `commit_sha`) tuple registered in `RunRegistry`.

### Effect on the H8 decision rule

None. The pre-registered thresholds (0.5 / 1.0 / 2.0) are LOCKED
and unchanged. The H8-A/B/C classifier reads the milestone JSONs
identically regardless of the train-shard cardinality used to
populate them. The aggregate verdict is logged with the
`n_train_per_task` provenance field so future replications can
reproduce the exact protocol.

### Cross-references

- Pre-reg : `docs/osf-prereg-g5-ter-spiking-cnn.md` §4, §5, §9
- G4-quinto deviation envelope precedent : `docs/osf-prereg-g4-quinto-pilot.md` §9.1
- Driver flag : `experiments/g5_ter_spiking_cnn/run_g5_ter.py:main` `--n-train-per-task`
- Plan compute note : `docs/superpowers/plans/2026-05-03-g5-ter-spiking-cnn-washout-test.md`
  "Compute / power note"
