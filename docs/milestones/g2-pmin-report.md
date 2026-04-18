# G2 — P_min Viability Report

**Gate** : G2 (P_min viable)
**Target week** : S8
**Status** : **PARTIAL — skeleton implementation, real evaluation S9+**

## Gate criteria (from master spec §7.2)

- [ ] **Accuracy criterion** : P_min retained accuracy ≥ baseline − 2%
- [ ] **Stability criterion** : runtime stable for 48 hours continuous

## Current evidence (S8)

### Implementation status

- ✅ DreamEpisode 5-tuple dataclass (S5.1)
- ✅ DreamRuntime scheduler with DR-0 log guarantee (S5.2)
- ✅ DR-0 + DR-1 property tests passing (S5.3)
- ✅ Replay operation skeleton (S5.4) — counts records, no weight mutation
- ✅ Downscale operation skeleton (S7.1) — records factor, no weight mutation
- ✅ S2 finite guard (S7.2)
- ✅ Swap protocol skeleton with S1 + S2 guards (S7.3)
- ✅ P_min profile wiring (replay + downscale on runtime) (S7.4)

### Test coverage

- 53 tests total (target ≥90% coverage, currently 96.28%)
- `kiki_oniric/dream/`: 100% coverage on episode, runtime, operations
- DR-0 + DR-1 + DR-3 axiom property tests passing
- S2 invariant conformance test passing

### What is NOT yet tested

- No MLX integration → no real weight matrix updates
- No retained benchmark consumption → S1 guard exercised on synthetic data only
- No multi-day runtime stability test (would require deployment)
- No comparison vs baseline accuracy on mega-v2 dataset

## Decision (S8)

**Branch GO-CONDITIONAL (default at S8 with skeleton)** :
- Code structure validated, contracts in place
- Defer real accuracy/stability measurement to S9+ once MLX wiring done
- Commit to G2 gate **conditional** on S9+ evidence

**Branch GO-FULL** (if MLX integration completes faster than expected) :
- Run baseline + P_min on mega-v2 retained benchmark
- Measure accuracy delta, confirm ≥ baseline − 2%
- Run 48h continuous stability test
- Lock G2 fully

**Branch NO-GO (Pivot A)** :
- If S9+ measurements show accuracy < baseline − 2%, OR runtime instability
- Activate Pivot A : single-paper TMLR/ICLR workshop on engineering results only
- Framework paper deferred to cycle 2

## Pilot results (S9.5)

Pilot measurement run on synthetic placeholder benchmark (3 seeds,
P_min vs mock baseline). See `g2-pilot-results.md` for full table.

**Pilot gate result** : **PASS** (delta ≥ −0.02 criterion)

**Decision update** : Branch GO-CONDITIONAL **maintained**
(infrastructure validated, real benchmark pending S10+). Path to
GO-FULL is clear : when real retained set is integrated S10+,
re-run pilot with real predictors. If gate passes on real data
3 consecutive runs, flip to GO-FULL.

## Action

S8 day end : decide GO-CONDITIONAL (default) and document path to GO-FULL in S9-S10.
