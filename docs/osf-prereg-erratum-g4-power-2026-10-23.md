# Erratum: power sentences in locked G4 / G5-bis pre-registrations

**Date:** 2026-10-23
**Status:** erratum note. The locked pre-registrations listed below
are **not edited**; this file records the corrected arithmetic. It
should be filed on OSF next to them by the PI.
**Reproduction:** `scripts/prereg_power_2026_10.py`, section 2
(exact noncentral t, equal-variance two-sample design, equal n per
arm, 80 % power). Values are Cohen's d; Hedges' g = d × J(2n − 2).
The locked runs used Welch tests; the equal-variance noncentral t is
the standard planning approximation for them (as the locked text
itself states for quinto / sexto) and is exact only under equal
variances.

## What is wrong

| Locked file (line) | Locked sentence | Correct value |
|---|---|---|
| `docs/osf-prereg-g4-quinto-pilot.md:77` | N = 30 per arm at α = 0.0167 detects \|g\| ≥ 0.74 (Welch two-sided) | two-sided α = 0.0167: **d = 0.857** (g ≈ 0.846); one-sided: d = 0.782 (g ≈ 0.772) |
| `docs/osf-prereg-g4-sexto-pilot.md:110` | same sentence | same correction |
| `docs/osf-prereg-g4-septimo-pilot.md:80` | N = 30 at α = 0.05 detects \|g\| ≥ 0.74 (two-sided) | correct as a d value (d = 0.736); g ≈ 0.726 |
| `docs/osf-prereg-g4-octavo-pilot.md:80` | same as septimo | as septimo |
| `docs/osf-prereg-g5-bis-richer-esnn.md:106-107,141` | N = 10, two-sided α = 0.0125 detects g ≈ 1.27 | two-sided: **d = 1.638** (g ≈ 1.568); one-sided (the test G5-bis actually runs): d = 1.484 (g ≈ 1.421) |
| `docs/osf-prereg-g5-bis-richer-esnn.md:150` | N = 30 follow-up floor g ≈ 0.74 | at the G5-bis α = 0.0125: one-sided d = 0.814 (g ≈ 0.803), two-sided d = 0.886 (g ≈ 0.875) |

The 0.74 figure is the α = 0.05 two-sided value. It was carried into
the quinto / sexto pre-registrations after their α was lowered to
0.0167 by Bonferroni, without being recomputed. The 1.27 figure in
G5-bis does not match any of its stated settings.

## Consequences

- No verdict changes. Every affected hypothesis was decided by its
  pre-registered test and threshold, not by the power sentence.
- The quoted detection floors were optimistic. At N = 30 and
  α = 0.0167 an effect had to be about d = 0.86 (two-sided) to be
  detected with 80 % probability, not 0.74. Readings that cite
  "N = 30 detects |g| ≥ 0.74" as evidence that smaller effects are
  absent should cite the corrected floor. In particular, a
  non-rejection in those runs is absence of evidence below d ≈ 0.86,
  not evidence of absence.
- New drafts (G4-nono, DR-3 probe, E1 controls, 2026-10-23) use the
  corrected arithmetic.
