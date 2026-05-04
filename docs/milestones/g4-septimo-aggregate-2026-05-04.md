# G4-septimo aggregate verdict

**Date** : 2026-05-04
**Pre-registration** : [docs/osf-prereg-g4-septimo-pilot.md](../osf-prereg-g4-septimo-pilot.md)
**Sister pilot** : G4-sexto aggregate [docs/milestones/g4-sexto-aggregate-2026-05-03.md](./g4-sexto-aggregate-2026-05-03.md) (H6-A canonical source).

## Summary

- H6-A (CIFAR-100, 100-class scale) confirmed (from G4-sexto aggregate) : **True**
- H6-B (Tiny-ImageNet, 200-class / 64×64 RGB scale) confirmed : **True**
- H6-C (universality conjunction) state : **confirmed**
- H6-C confirmed : **True**
- H6-C partial : **False**
- H6-C falsified : **False**
- H5-C → H6-C universality extension (4 benchmarks × 4 substrates) : **True**

## H6-A — CIFAR-100 (n_classes=10 per task, G4SmallCNN) — from G4-sexto

- mean P_max (mog) : 0.3622
- mean P_max (none) : 0.3580
- Hedges' g (mog vs none) : 0.0570
- Welch t : 0.1966
- Welch p two-sided (alpha = 0.0167) : 0.8450
- fail_to_reject_h0 : True -> H6-A confirmed = True

## H6-B — Tiny-ImageNet (n_classes=20 per task, G4MediumCNN)

- mean P_max (mog) : 0.3864
- mean P_max (none) : 0.3891
- Hedges' g (mog vs none) : -0.0246
- Welch t : -0.0950
- Welch p two-sided (alpha = 0.0500) : 0.9247
- fail_to_reject_h0 : True -> H6-B confirmed = True

*Honest reading* : Welch fail-to-reject = absence of evidence at this N for a difference between mog and none — under H6-B specifically, this **is** the predicted positive empirical claim that RECOMBINE adds nothing measurable beyond REPLAY+DOWNSCALE on the medium CNN substrate at Tiny-ImageNet 200-class / 64×64 RGB scale.

## H6-C — universality conjunction (4 benchmarks × 4 substrates)

State : **confirmed**

Both H6-A (G4-sexto) and H6-B (G4-septimo) confirmed. The G4-quinto H5-C RECOMBINE-empty universality (FMNIST + CIFAR-10) is fully extended to {Split-FMNIST, Split-CIFAR-10, Split-CIFAR-100, Split-Tiny-ImageNet} × {3-layer MLP, 5-layer MLP, small CNN, medium CNN}. Framework-C claim 'richer ops yield richer consolidation' empirically refuted across the full pre-registered four-benchmark scope.

## Verdict — DR-4 evidence

Per pre-reg §6 : EC stays PARTIAL across all outcomes ; FC stays at C-v0.12.0. Under H6-C confirmed, the partial refutation of DR-4 established by G4-ter and universalised by G4-quinto / G4-sexto is extended to Tiny-ImageNet at 200-class / 64×64 RGB scale ; DR-4 evidence v0.6 amends the v0.5 G4-sexto addendum with the four-benchmark universality flag. Under H6-B falsified, the universality is shown to break at the Tiny-ImageNet scale and DR-4 evidence v0.6 records the boundary.
