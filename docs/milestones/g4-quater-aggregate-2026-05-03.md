# G4-quater aggregate verdict

**Date** : 2026-05-03
**Pre-registration** : [docs/osf-prereg-g4-quater-pilot.md](../osf-prereg-g4-quater-pilot.md)

## Summary

- H4-A (substrate-depth) confirmed : **False**
- H4-B (HP-calibration) confirmed : **False**
- H4-C (RECOMBINE empirically empty) confirmed : **True**
- All three confirmed : False

## H4-A — substrate-depth

- mean P_min : 0.5959
- mean P_equ : 0.5958
- mean P_max : 0.5958
- monotonic_observed : False
- Jonckheere J : 13511.5000
- one-sided p (alpha = 0.0167) : 0.5137
- reject_h0 : False
- **H4-A confirmed** : False

## H4-B — HP-calibration (RESTRUCTURE factor sweep)

any_factor_recovers_ordering : False

### factor = 0.85
- mean P_min : 0.7065
- mean P_equ : 0.6554
- mean P_max : 0.6554
- monotonic_observed : False
- Jonckheere J : 1034.0000
- p (alpha = 0.0056) : 0.9904
- reject_h0 : False

### factor = 0.95
- mean P_min : 0.7065
- mean P_equ : 0.6582
- mean P_max : 0.6582
- monotonic_observed : False
- Jonckheere J : 1076.0000
- p (alpha = 0.0056) : 0.9788
- reject_h0 : False

### factor = 0.99
- mean P_min : 0.7065
- mean P_equ : 0.6589
- mean P_max : 0.6589
- monotonic_observed : False
- Jonckheere J : 1094.0000
- p (alpha = 0.0056) : 0.9710
- reject_h0 : False

**H4-B confirmed** : False

## H4-C — RECOMBINE empirical-emptiness

- mean P_max (mog) : 0.7007
- mean P_max (none) : 0.7006
- Hedges' g (mog vs none) : 0.0021
- Welch t : 0.0143
- Welch p two-sided (alpha = 0.0167) : 0.9886
- fail_to_reject_h0 : True -> H4-C confirmed = True

*Honest reading* : Welch fail-to-reject = no evidence at this N for a difference between mog and none ; the predicted outcome under H4-C, framed as positive empirical claim that RECOMBINE adds nothing beyond the REPLAY+DOWNSCALE coupling already provided by P_min.

### Secondary observation — AE strategy
- mean P_max (ae) : 0.7006
- mean P_max (none) : 0.7006
- Welch p two-sided : 1.0000

## Verdict — DR-4 evidence

Per pre-reg §6 : EC stays PARTIAL across all outcomes ; FC stays at C-v0.12.0. If H4-C is confirmed, the partial refutation of DR-4 established by G4-ter is **strengthened**, not weakened.
