# G4-septimo Step 1 — H6-B Tiny-ImageNet+RECOMBINE strategy + placebo

**Date** : 2026-05-04
**c_version** : `C-v0.12.0+PARTIAL`
**commit_sha** : `d52168a7e6e6ea45bf740140acf6ee65d08fb218`
**Cells** : 240 (2 strategies x 4 arms x 30 seeds)
**Wall time** : 14042.8s
**Smoke** : False

**Multi-class exclusion threshold** : acc_initial < 2 × random_chance = 0.10 (random_chance = 0.05 for n_classes = 20).

## Pre-registered hypothesis

Pre-registration : `docs/osf-prereg-g4-septimo-pilot.md`

### H6-B — universality of RECOMBINE-empty (Tiny-ImageNet, n_classes=20)

Welch two-sided test of `retention(P_max with mog)` vs `retention(P_max with none)` on the medium CNN with a 20-class per-task head. **Failing** to reject H0 at alpha = 0.05 confirms H6-B : the G4-sexto H6-A RECOMBINE-empty finding generalises to Tiny-ImageNet 200-class / 64×64 RGB scale (200 fine classes split into 10 tasks of 20 classes each).

- mean retention P_max (mog) : 0.3864 (N=29)
- mean retention P_max (none) : 0.3891 (N=29)
- Hedges' g (mog vs none) : -0.0246
- Welch t : -0.0950
- Welch p (two-sided, alpha = 0.0500) : 0.9247 -> fail_to_reject_h0 = True

**H6-B verdict** : RECOMBINE empty confirmed = True (positive empirical claim mog ≈ none if True).

*Honest reading* : Welch fail-to-reject = absence of evidence at this N for a difference between mog and none — under H6-B specifically, this **is** the predicted positive empirical claim that RECOMBINE adds nothing measurable beyond REPLAY+DOWNSCALE on the medium CNN substrate at Tiny-ImageNet 200-class / 64×64 RGB scale.

## Provenance

- Pre-registration : [docs/osf-prereg-g4-septimo-pilot.md](../osf-prereg-g4-septimo-pilot.md)
- Driver : `experiments/g4_septimo_test/run_step1_tiny_imagenet.py`
- Substrate : `experiments.g4_septimo_test.medium_cnn.G4MediumCNN` (n_classes=20)
- Loader : `experiments.g4_septimo_test.tiny_imagenet_dataset.load_split_tiny_imagenet_10tasks_auto`
- Strategies : `experiments.g4_quater_test.recombine_strategies.sample_synthetic_latents`
- Run registry : `harness/storage/run_registry.RunRegistry` (db `.run_registry.sqlite`)
