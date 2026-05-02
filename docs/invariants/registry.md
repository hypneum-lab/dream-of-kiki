# Invariants & Axioms Registry

**Authoritative source** for invariants I/S/K and axioms DR-0..DR-4.
Any code referencing an invariant MUST cite its code and version here.

## Family I (Information)

- **I1** Episodic conservation until consolidation — **BLOCKING**
  Every episode in β buffer gets consumed by a dream-episode within τ_max
  before purge. Enforced via T-Ops hourly cron query.
- **I2** Hierarchy traceability — **BLOCKING**
  Every TopologyDiff recorded with (DE_id, C_version, before_hash,
  after_hash). Enforced post-swap validator.
- **I3** Latent distributional drift bounded — **WARN**
  KL(p_recombined || p_awake) ≤ ε_drift per C-version. Measured via A.5
  experiment runner.

## Family S (Safety)

- **S1** Retained task non-regression — **BLOCKING**
  acc(post-swap) ≥ acc(pre-swap) − δ_regression. Enforced in swap
  protocol, aborts on fail.
- **S2** No NaN/Inf in W_scratch — **BLOCKING**
  All weights finite and bounded. Enforced pre-swap guard.
- **S3** Hierarchy guard — **BLOCKING**
  validate_topology(G_post) verifies species connectivity, no unwanted
  cycles, layer counts in bounds.
- **S4** Attention prior bounded — **WARN**
  Each prior component in [0, 1], sum ≤ budget_attention. Auto-clamp
  with log.

## Family K (Compute)

- **K1** Dream-episode budget respected — **BLOCKING** (per DE)
  FLOPs_actual ≤ budget.FLOPs and wall_time and energy. Enforced via
  A.4 runtime context manager.
- **K2** SO × fast-spindle phase-coupling within empirical CI —
  **WARN** (measurement-class invariant)
  For any substrate implementing the opt-in
  `PhaseCouplingObservable` Protocol, the measured coupling
  strength (Tort 2010-style mean-vector-length) must lie inside
  the 95 % CI [0.27, 0.39] reported by the eLife 2025 Bayesian
  meta-analysis (`elife2025bayesian` ; 23 studies, 297 effect
  sizes, BF > 58 vs. null, Egger phase-branch p = 0.59). Enforced
  by `tests/conformance/invariants/test_k2_coupling.py` against
  the synthetic substrate; real substrates plug in via the
  Protocol. Severity is WARN (not BLOCKING) because (a) only one
  meta-analysis pins the window, (b) substrate physiology can
  legitimately broaden the CI, (c) the synthetic substrate is the
  only canonical reference until S18+. Evidence stub:
  `docs/proofs/k2-coupling-evidence.md`.
- **K3** Swap latency bounded — **WARN**
  wall_clock(swap_atomic) ≤ K3_max (default 1s).
- **K4** Eval matrix coverage — **BLOCKING** (for tagging)
  MAJOR bump requires full stratified matrix executed before C-version
  tag. Enforced by T-Ops CI.

## Axioms DR-0..DR-4

- **DR-0** Accountability — every dream output is traceable to a DE
  with finite budget.
- **DR-1** Episodic conservation — formalizes I1 as axiom.
- **DR-2** Compositionality (proved 2026-04-21, v0.2 under precondition
  `¬(∃ i<j : π_i=RESTRUCTURE ∧ π_j=REPLAY)`) — see
  `docs/proofs/dr2-compositionality.md` (n-ary permutation form +
  generator-irreducibility sub-theorem §7). Fallback DR-2' retained
  with canonical order (replay < downscale < restructure < recombine)
  as stricter contract for P_min/P_equ/P_max profiles.
- **DR-3** Substrate-agnosticism — holds for any substrate satisfying
  Conformance Criterion: signature typing + axiom tests pass +
  BLOCKING invariants enforceable (S1, S2, S3, I1).
- **DR-4** Profile chain inclusion — ops and channels of P_min ⊆ P_equ
  ⊆ P_max.

## References

Full formal definitions in
`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`:
- §5 Invariants (I, S, K families)
- §6 Axioms (DR-0..DR-4 with proof sketches)
- §6.2 DR-3 Conformance Criterion (executable)
