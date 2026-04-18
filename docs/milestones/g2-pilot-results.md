# G2 Pilot Results — Synthetic Benchmark

**Date** : 2026-04-18 (S9.5)
**Benchmark** : retained 50-item synthetic placeholder
**Hash** : c1b93b25dd84a95d455a5758ef1c73a2a92b606394ec429d4029635ef58ab3a0
**Seeds tested** : 42, 123, 7

## ⚠️ Caveat

These results validate the **measurement infrastructure**
(model → predictor → evaluate_retained → gate decision), NOT real
linguistic consolidation accuracy. The benchmark is synthetic
placeholder (S3.4); real mega-v2 retained set integration arrives
S10+ once dataset access is finalized.

## Results table

| Seed | baseline_acc | p_min_acc | delta |
|------|--------------|-----------|-------|
| 42   | 0.500        | 0.800     | +0.300 |
| 123  | 0.500        | 0.800     | +0.300 |
| 7    | 0.500        | 0.800     | +0.300 |

**delta mean** : +0.300
**delta range** : [+0.300, +0.300]

## Gate check

**Criterion** : delta ≥ -0.02 (G2 §7.2 — accuracy ≥ baseline − 2%)

**Result** : **PASS**

## Implications

If PASS (synthetic):
- Measurement pipeline validated end-to-end
- Path to GO-FULL clear once real benchmark arrives S10+
- G2 stays at GO-CONDITIONAL pending real data

If FAIL (synthetic):
- Pipeline bug — investigate evaluate_retained or swap_now logic
- Block GO-CONDITIONAL upgrade until pipeline corrected

## Raw data

JSON at `docs/milestones/g2-pilot-results.json` (see `scripts/pilot_g2.py`).

## Next steps

- S10+: integrate mega-v2 retained set (real linguistic data)
- S10+: re-run pilot with real predictors (MLX model inference,
  not synthetic predictors)
- S12+: G4 P_equ similar pilot if P_equ wired (S11.2)
