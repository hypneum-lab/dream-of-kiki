# G6 pilot — micro-kiki Qwen × MMLU CL stream (qwen3p5-1p5b-fp16)

**Date** : 2026-05-03
**Path** : B
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `0a294bce0d0d39fc7c1c07a13ea3de742c75fa7a`
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
| baseline | 0 | 1.0114 | False | `a3726fe2be55913313dc357e8c16a09c` |
| baseline | 1 | 0.9989 | False | `25b051ef109d8237939827a513740b41` |
| baseline | 2 | 1.0191 | False | `03b2fdd6cd0f916c27222b4ab8d75d61` |
| P_min | 0 | 1.0114 | False | `3a8e21f5c98ab97dca57c2471cbf294a` |
| P_min | 1 | 0.9989 | False | `3119777b341166b882bb3c973e6a7502` |
| P_min | 2 | 1.0191 | False | `ab7b94debb4bd7f3b26342710f1d7c35` |
| P_equ | 0 | 1.0114 | False | `168470c0e88636fed4ee40f3d643a60f` |
| P_equ | 1 | 0.9989 | False | `21ecf7bf36b7405aea7f974dea1923ea` |
| P_equ | 2 | 1.0191 | False | `3ac4199c7be86b6188b7d2b2ac4c3e30` |
| P_max | 0 | 1.0114 | False | `2ec8635be3cc56efd3d10f586ce3ee29` |
| P_max | 1 | 0.9989 | False | `26b04352892e999557ab815af8548649` |
| P_max | 2 | 1.0191 | False | `36448c025b73bbf58869c7da1e9d3d97` |

## Path B disclosure

Per pre-reg §6, Path B never triggers a STABLE / UNSTABLE EC-axis bump regardless of effect-size outcome. The verdict scalars above are exploratory infrastructure validation.

## Pilot caveats — spectator pattern under Path B

The Path B implementation runs the four dream handlers on synthetic payloads (built by ``build_episode_payload``), not on the live ``InferenceOnlyAdapter._deltas`` buffer. The adapter accumulates per-subdomain perturbations via ``adapt_subdomain`` ; the dream-handler return tensors are **not** fed back into the adapter delta. Consequently the adapter state is identical across the four arms (modulo DR-0 / DR-1 stamps on the substrate's recombine / restructure dataclasses), so the L2-norm-driven Path B accuracy proxy yields bit-identical retention vectors per seed across arms — Hedges' g collapses to 0.0 and the Jonckheere monotonicity check is degenerate (equal means).

This mirrors the G4 spectator pattern (pre-coupling) and is the expected outcome when dream handlers operate on synthetic payloads disjoint from the evaluation surface. A genuine forgetting differential requires (a) Path A real LoRA fine-tune so the adapter sees gradient updates that the four handlers can perturb, or (b) extending Path B so the handler return tensors mutate ``adapter.set_delta``. The latter is post-hoc relative to this pre-reg ; per §7, any extension is logged as a deviation in a separate dated immutable before re-running.

**Honest verdict on this run** : G6 Path B successfully validates the pipeline shape (60 forgetting measurements, 12 R1 run_ids registered, deterministic across re-runs) and confirms the spectator-only handler wiring needs to be promoted to coupling before any STABLE EC bump is warranted. Path A on Studio remains the publishable G6 path.