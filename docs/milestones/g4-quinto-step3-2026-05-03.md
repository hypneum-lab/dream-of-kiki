# G4-quinto Step 3 — H5-C CNN+RECOMBINE strategy + placebo

**Date** : 2026-05-03
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `a6b942d647638b3090c155327481f76ec2a6ba80`
**Cells** : 360 (3 strategies x 4 arms x 30 seeds)
**Wall time** : 3057.1s

## Pre-registered hypothesis

Pre-registration : `docs/osf-prereg-g4-quinto-pilot.md`

### H5-C — universality of RECOMBINE-empty (CNN substrate)

Welch two-sided test of `retention(P_max with mog)` vs `retention(P_max with none)` on the small CNN. **Failing** to reject H0 at alpha = 0.05 / 3 = 0.0167 confirms H5-C : the G4-quater H4-C RECOMBINE-empty finding generalises across substrates (FMNIST 3-layer MLP -> CIFAR-CNN).

- mean retention P_max (mog) : 0.9842 (N=30)
- mean retention P_max (none) : 0.9845 (N=30)
- Hedges' g (mog vs none) : -0.0026
- Welch t : -0.0104
- Welch p (two-sided, alpha = 0.0167) : 0.9918 -> fail_to_reject_h0 = True

**H5-C verdict** : RECOMBINE empty confirmed = True (positive empirical claim mog ≈ none if True).

*Honest reading* : Welch fail-to-reject = absence of evidence at this N for a difference between mog and none — under H5-C specifically, this **is** the predicted positive empirical claim that RECOMBINE adds nothing measurable beyond REPLAY+DOWNSCALE on the CNN substrate at CIFAR-10 scale.

### Secondary observation — AE strategy vs none placebo

- mean retention P_max (ae) : 0.9840 (N=30)
- mean retention P_max (none) : 0.9845 (N=30)
- Welch t : -0.0181
- Welch p (two-sided) : 0.9857

## Provenance

- Pre-registration : [docs/osf-prereg-g4-quinto-pilot.md](../osf-prereg-g4-quinto-pilot.md)
- Driver : `experiments/g4_quinto_test/run_step3_cnn_recombine.py`
- Substrate : `experiments.g4_quinto_test.small_cnn.G4SmallCNN`
- Strategies : `experiments.g4_quater_test.recombine_strategies.sample_synthetic_latents`
- Run registry : `harness/storage/run_registry.RunRegistry` (db `.run_registry.sqlite`)
