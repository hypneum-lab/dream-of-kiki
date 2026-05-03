# G6 pilot — micro-kiki Qwen × MMLU CL stream (qwen3p5-1p5b-fp16)

**Date** : 2026-05-03
**Path** : B
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `3a204a03efa9bd407fd5e339b7c73caa89ae3618`
**Cells** : 12
**Wall time** : 0.0s

## Pre-registered hypotheses

Pre-registration : `docs/osf-prereg-g6-pilot.md` (amended 2026-05-03 for G4-bis g_h1=-2.31).

### H1' — P_equ retention vs Hu 2020 (g >= 0.21)
```
{
  "above_hu_2020_lower_ci": false,
  "alpha_per_test": 0.016666666666666666,
  "hedges_g": 0.0,
  "is_within_hu_2020_ci": false,
  "n_base": 3,
  "n_p_equ": 3,
  "welch_p": 0.5,
  "welch_reject_h0": false
}
```
### H3' — P_min retention vs Javadi 2024 (g <= -0.13)
```
{
  "alpha_per_test": 0.016666666666666666,
  "below_javadi_2024_lower_ci_decrement": false,
  "hedges_g": 0.0,
  "is_within_javadi_2024_ci": false,
  "n_base": 3,
  "n_p_min": 3,
  "welch_p": 0.5,
  "welch_reject_h0": false
}
```
### H_DR4' — Jonckheere monotonicity
```
{
  "j_statistic": 13.5,
  "mean_p_equ": 1.0098341101229187,
  "mean_p_max": 1.0098341101229187,
  "mean_p_min": 1.0098341101229187,
  "monotonic_observed": true,
  "p_value": 0.5,
  "reject_h0": false
}
```
### H_NEW (amended) — exploratory infrastructure validation
```
{
  "amendment_note": "H_NEW reformulated 2026-05-03 given G4-bis g_h1=-2.31. Exploratory infrastructure validation only \u2014 Path B never triggers STABLE/UNSTABLE per pre-reg \u00a76.",
  "arm_mean_diffs": {
    "P_equ": 0.0,
    "P_max": 0.0,
    "P_min": 0.0
  },
  "baseline_mean_retention": 1.0098341101229187,
  "exploratory": true,
  "infrastructure_effect_observed": false,
  "infrastructure_effect_threshold": 1e-06,
  "max_abs_diff": 0.0
}
```

## Cells (R1 traceability)

| arm | seed | retention | excluded | run_id |
|-----|------|-----------|----------|--------|
| baseline | 0 | 1.0114 | False | `c4b35b9cb92cfd25770bc71b4030bee0` |
| baseline | 1 | 0.9989 | False | `6838132f984371e6d13b4a21646b5e1b` |
| baseline | 2 | 1.0191 | False | `ee6e059d562980ac87a18954b4064ee5` |
| P_min | 0 | 1.0114 | False | `7839913e3d06ee02499fe9bee3438cde` |
| P_min | 1 | 0.9989 | False | `258818c2d3da711d982cadf3b5ef18df` |
| P_min | 2 | 1.0191 | False | `9a1585245c5adc85861c99edc11a6f15` |
| P_equ | 0 | 1.0114 | False | `7cd580e082d3aabe347f270fa0b86120` |
| P_equ | 1 | 0.9989 | False | `d6f002dc501e9ed8d73faa955b79d494` |
| P_equ | 2 | 1.0191 | False | `784bf31a574baf09fbe90841c728d47c` |
| P_max | 0 | 1.0114 | False | `1728ee4a34123b08c15e1d0489083862` |
| P_max | 1 | 0.9989 | False | `308d4de2ea1784ed54caed5d74d4f991` |
| P_max | 2 | 1.0191 | False | `be6fce796c305478681a2e6f4e4c4c5c` |

## Path B disclosure

Per pre-reg §6, Path B never triggers a STABLE / UNSTABLE EC-axis bump regardless of effect-size outcome. The verdict scalars above are exploratory infrastructure validation.

## Pilot caveats — spectator pattern under Path B

The Path B implementation runs the four dream handlers on synthetic payloads (built by ``build_episode_payload``), not on the live ``InferenceOnlyAdapter._deltas`` buffer. The adapter accumulates per-subdomain perturbations via ``adapt_subdomain`` ; the dream-handler return tensors are **not** fed back into the adapter delta. Consequently the adapter state is identical across the four arms (modulo DR-0 / DR-1 stamps on the substrate's recombine / restructure dataclasses), so the L2-norm-driven Path B accuracy proxy yields bit-identical retention vectors per seed across arms — Hedges' g collapses to 0.0 and the Jonckheere monotonicity check is degenerate (equal means).

This mirrors the G4 spectator pattern (pre-coupling) and is the expected outcome when dream handlers operate on synthetic payloads disjoint from the evaluation surface. A genuine forgetting differential requires (a) Path A real LoRA fine-tune so the adapter sees gradient updates that the four handlers can perturb, or (b) extending Path B so the handler return tensors mutate ``adapter.set_delta``. The latter is post-hoc relative to this pre-reg ; per §7, any extension is logged as a deviation in a separate dated immutable before re-running.

**Honest verdict on this run** : G6 Path B successfully validates the pipeline shape (60 forgetting measurements, 12 R1 run_ids registered, deterministic across re-runs) and confirms the spectator-only handler wiring needs to be promoted to coupling before any STABLE EC bump is warranted. Path A on Studio remains the publishable G6 path.