# G4 — P_equ Functional Report

**Gate** : G4 (P_equ fonctionnel)
**Target week** : S12
**Status** : **CONDITIONAL — wiring complete, real ablation S13+**

## Gate criteria (from master spec §7.2)

- [ ] **Improvement criterion** : P_equ > P_min on ≥2 metrics with
  statistical significance
- [ ] **Stability criterion** : invariants all green for 7 consecutive
  days
- [ ] **DR-4 inclusion verified** : ops/channels P_min ⊆ P_equ

## Current evidence (S12)

### Implementation status

- ✅ DreamEpisode 5-tuple dataclass (S5.1)
- ✅ DreamRuntime scheduler with DR-0 log guarantee (S5.2)
- ✅ DR-0 + DR-1 + DR-3 + DR-4 axiom property tests passing (S5.3, S12.1)
- ✅ Replay operation skeleton + MLX backend (S5.4, S9.1)
- ✅ Downscale operation skeleton + MLX backend (S7.1, S9.2)
- ✅ Restructure operation skeleton (S10.1)
- ✅ Recombine operation skeleton with RNG injection (S11.1)
- ✅ S2 finite guard (S7.2)
- ✅ S3 topology guard (S10.2)
- ✅ Swap protocol skeleton with S1 + S2 guards (S7.3)
- ✅ Retained eval bridge (S9.3)
- ✅ P_min profile fully wired with swap_now (S7.4, S9.4)
- ✅ **P_equ profile fully wired** — 4 ops + 3 channels (S11.2)
- ✅ G2 P_min pilot PASS on synthetic benchmark (S9.5)

### Test coverage

- 86 tests total (target ≥90% coverage, currently ~95%)
- `kiki_oniric/dream/operations/`: replay, downscale, restructure,
  recombine all 100% line coverage
- `kiki_oniric/dream/guards/`: finite, topology fully tested with
  conformance tests
- `kiki_oniric/profiles/p_equ.py`: 100% on wiring + state tracking
- DR-3 Conformance Criterion conditions (1) typing + (2) axiom tests
  for DR-0/1/3/4 + (3) BLOCKING invariants enforceable — all green

### What is NOT yet measured

- No comparison vs P_min on retained benchmark (real ablation
  requires S13+ MLX-native restructure + recombine handlers)
- No 7-day continuous stability test (skeleton restructure +
  recombine ops are instantaneous; real ops will need real time)
- No statistical significance testing on M1.b/M2.b/M4.a (metrics
  not yet measured beyond pilot)

## Decision (S12)

**Branch GO-CONDITIONAL (default at S12 with wiring complete)** :
- Architecture validated, contracts in place, DR-4 verified
- Restructure + recombine are skeletons (no real graph mutation,
  no real VAE sampling)
- Defer real P_equ vs P_min comparison to S13+ once MLX-native
  restructure + recombine deliver real outputs
- Commit to G4 gate **conditional** on S13+ ablation evidence

**Branch GO-FULL** (if S13-S15 ablation completes faster than
expected) :
- Run baseline + P_min + P_equ on mega-v2 retained benchmark
- Measure M1.b, M2.b, M4.a (recomb quality on real latents)
- Statistical significance via Welch's t-test on 3 seeds
- Lock G4 fully if P_equ > P_min on ≥2 metrics with p < 0.05

**Branch NO-GO (re-scope)** :
- If S13-S15 measurements show P_equ ≤ P_min on most metrics
- Two sub-options:
  - **NO-GO-A** : skip P_max entirely (cycle-1 stops at P_equ
    documentation), ablation paper covers only P_min vs P_equ
  - **NO-GO-B** : Pivot A activated per master spec §7.3 (single-
    paper TMLR/ICLR workshop, framework deferred cycle 2)

## Action

S12 day end : adopt GO-CONDITIONAL (default). Document S13-S15
ablation path: integrate real MLX restructure + recombine, run
baseline + P_min + P_equ measurement, evaluate Welch's t-test.

## Comparison with G2 P_min report

G2 (P_min) is GO-CONDITIONAL because real benchmark mega-v2 not yet
integrated. G4 (P_equ) is GO-CONDITIONAL for the same reason **plus**
restructure + recombine handlers are still skeletons. S13+ resolves
both gaps simultaneously.
