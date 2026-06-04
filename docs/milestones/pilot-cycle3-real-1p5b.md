# Pilot cycle-3 — real 1.5B (20 seeds)

**Artifact** : `docs/milestones/pilot-cycle3-real-1p5b.json`
**Substrate** : `qwen3p5-1p5b-fp16` (real bf16/fp16 weights, not a mock)
**Harness** : `C-v0.7.0+PARTIAL`
**Verdict** : **GO — pipeline-validated, underpowered (1.5B pilot, NOT the 35B claim)**

## What this is

A real-weight 1.5B continual-learning ablation pilot over three profiles
(`p_min`, `p_equ`, `p_max`) × three benchmarks (`mmlu`, `hellaswag`,
`mega_v2`), 20 seeds × 3 profiles = **60 cells**, 100 samples per
benchmark. Run completed end-to-end (`failed_cells = 0`, wall
≈ 6934 s). This supersedes the earlier 5-seed run of the same pilot.

## Results (read honestly)

- **H1 (improvement over baseline)** : H0 rejected on all 3 profiles —
  `p_min` p ≈ 2.4e-20, `p_equ` p ≈ 1.1e-19, `p_max` p ≈ 1.5e-19
  (n = 20 each). The GO rule (≥ 2/3 profiles reject at α = 0.0125)
  is met 3/3.
- **The gain is carried by `mega_v2`.** `mmlu` and `hellaswag`
  pre/post are roughly flat; the consolidation signal is the
  `mega_v2` (cortical-update) delta. Label this a **1.5B pilot
  effect**, not the 35B claim.
- **E1 TOST (equivalence)** : SESOI = Hedges' g 0.29, α = 0.0167
  per test (Bonferroni). `equivalent = False` on all 3 profiles —
  expected, since the observed effect is large (this run is testing
  improvement, not equivalence).

## Power deficit (honest)

The powered TOST harness (E1) reports the cell count needed to
**support** an equivalence/null at the pre-registered SESOI:

| Quantity | Value |
|---|---|
| Target power | 0.80 |
| Required pairs for 80 % power | **≈ 141** |
| Seeds available | 20 |
| `adequately_powered` | **False** |

So the GO is **GO pipeline-validated, sub-powered** — 20 seeds is
far below the ~141 pairs required for nominal power. This is **not a
nominal-power proof** and must not be reported as one. It validates
that the real-weight pipeline runs end-to-end and produces a
registerable directional result on a 1.5B substrate.

## Scope discipline (R1 / DualVer)

- This is an **FC/pilot** result, not an **EC** claim at nominal power.
  No EC promotion. EC stays PARTIAL.
- Every number above resolves to a cell in the sibling JSON. No
  fabricated figures.
- The 35B real run remains **out of reach of the development machine
  (M1 Max, 32 GB)**; the `qwen3p6-35b-bf16-local` pin is turnkey
  (resolved via `KIKI_35B_BF16_PATH`) and ready to run on the Mac
  Studio. Until then, the 35B claim is **not** established by this
  pilot.

## Fixture notes

- **hellaswag** : a real 200-item `rowanz/val` fixture replaced a
  degenerate fallback (8 recycled items saturating at ≈ 1.0). With the
  real fixture hellaswag now reads ≈ 0.38–0.42 (sensible), and the
  consolidation gain is correctly isolated to `mega_v2`.
- **mmlu** : audited, **no silent fallback** — the loader raises
  `ValueError` when the fixture is missing. The 0.91 seen elsewhere is
  on a real ("sanity" subset) fixture, not a degenerate recycle.
