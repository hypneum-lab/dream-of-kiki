# G4-quater Step 3 — H4-C RECOMBINE strategy + placebo

**Date** : 2026-05-03
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `83a6ae8d01b6339acc63e53b39faabf70632a0d1`
**Cells** : 1140 (3 strategies x 4 arms x 95 seeds)
**Wall time** : 2043.7s

## Pre-registered hypothesis

Pre-registration : `docs/osf-prereg-g4-quater-pilot.md`

### H4-C — RECOMBINE empirical-emptiness

Welch two-sided test of `retention(P_max with mog)` vs `retention(P_max with none)`. **Failing** to reject H0 at alpha = 0.05 / 3 = 0.0167 confirms H4-C : RECOMBINE is empirically empty at this scale (mog ≈ none).

- mean retention P_max (mog) : 0.7007 (N=95)
- mean retention P_max (none) : 0.7006 (N=95)
- Hedges' g (mog vs none) : 0.0021
- Welch t : 0.0143
- Welch p (two-sided, alpha = 0.0167) : 0.9886 -> fail_to_reject_h0 = True

**H4-C verdict** : RECOMBINE empty confirmed = True (positive empirical claim mog ≈ none if True).

*Honest reading* : Welch fail-to-reject = no evidence at this N for a difference between mog and none ; this is **the predicted outcome under H4-C**, framed as absence of evidence supporting RECOMBINE adding any signal beyond the REPLAY+DOWNSCALE coupling already provided by P_min.

### Secondary observation — AE strategy vs none placebo

- mean retention P_max (ae) : 0.7006 (N=95)
- mean retention P_max (none) : 0.7006 (N=95)
- Welch t : -0.0001
- Welch p (two-sided) : 1.0000

## Provenance

- Pre-registration : [docs/osf-prereg-g4-quater-pilot.md](../osf-prereg-g4-quater-pilot.md)
- Driver : `experiments/g4_quater_test/run_step3_recombine_strategies.py`
- Substrate : `experiments.g4_ter_hp_sweep.dream_wrap_hier.G4HierarchicalClassifier`
- Strategies : `experiments.g4_quater_test.recombine_strategies.sample_synthetic_latents`
- Run registry : `harness/storage/run_registry.RunRegistry` (db `.run_registry.sqlite`)
