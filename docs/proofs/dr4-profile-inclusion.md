# DR-4 Profile Inclusion — Draft

**Axiom statement** (from framework spec §6.2) :

```
DR-4 (Profile chain inclusion)
ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)
∧ channels(P_min) ⊆ channels(P_equ) ⊆ channels(P_max)
```

with **Lemma DR-4.L** : if P_min is valid (invariants satisfied) on
substrate S, then P_equ does not strictly worsen metrics monotone in
capacity.

---

## Profile definitions (recap from §3.1)

| Profile | Channels in | Channels out | Operations |
|---------|-------------|--------------|------------|
| P_min   | β           | 1            | {replay, downscale} |
| P_equ   | β + δ       | 1 + 3 + 4    | {replay, downscale, restructure, recombine_light} |
| P_max   | α + β + δ   | 1 + 2 + 3 + 4 | {replay, downscale, restructure, recombine_full} |

---

## Proof structure

### 1. Operations chain inclusion

We show `ops(P_min) ⊆ ops(P_equ) ⊆ ops(P_max)` element-wise.

**P_min ⊆ P_equ** :
- replay ∈ ops(P_min) and replay ∈ ops(P_equ) ✓
- downscale ∈ ops(P_min) and downscale ∈ ops(P_equ) ✓
- {replay, downscale} ⊆ {replay, downscale, restructure, recombine}
  by set inclusion. ∎

**P_equ ⊆ P_max** :
- replay, downscale, restructure ∈ both ✓
- recombine_light ⊆ recombine_full (full recombine subsumes the
  light variant — light is a parameter restriction of full)
- Therefore ops(P_equ) ⊆ ops(P_max). ∎

### 2. Channel chain inclusion

We show `channels(P_min) ⊆ channels(P_equ) ⊆ channels(P_max)`
element-wise on both directions (in-channels and out-channels).

**Channels in (awake → dream)** :
- {β} ⊆ {β, δ} ⊆ {α, β, δ} ✓ trivially by set inclusion.

**Channels out (dream → awake)** :
- {1} ⊆ {1, 3, 4} ⊆ {1, 2, 3, 4} ✓ trivially by set inclusion.

### 3. Combined

Both conditions hold ; DR-4 follows. ∎

---

## Lemma DR-4.L (monotonicity in capacity)

**Statement** : if P_min satisfies all invariants on substrate S,
then `best(P_equ, S) ≥ best(P_min, S)` for any metric M monotone in
capacity (where "monotone in capacity" means: more channels and ops
available → at least as much expressive power).

**Proof sketch** :

Let `R_min` be a P_min run on S that achieves metric value `M_min`.

Construct `R_equ` on S that mimics `R_min` :
- Use only the operations replay, downscale (subset of ops(P_equ))
- Use only channel β in, channel 1 out (subset of channels(P_equ))
- Ignore the additional channels δ, 3, 4 and operations restructure,
  recombine

By construction, `R_equ` produces the same trajectory as `R_min`
because the additional channels/ops are unused, and therefore
`M_equ(R_equ) = M_min(R_min) = M_min`.

Therefore `best(P_equ, S) ≥ M_equ(R_equ) = M_min = best(P_min, S)`. ∎

**Applies to metrics** : M1.b (avg accuracy), M2.b (RSA alignment) —
metrics that benefit from richer representations or more
consolidation operations.

**Does not apply to metrics** : M3.a (FLOPs ratio), M3.c (energy per
episode) — these are *cost* metrics where more capacity costs more,
not less.

---

## Status of each inclusion (S12.1)

| Inclusion | Status | Notes |
|-----------|--------|-------|
| ops(P_min) ⊆ ops(P_equ) | **Verified** | Test in test_dr4_profile_inclusion.py |
| channels(P_min) ⊆ channels(P_equ) | **Verified** | Same test |
| ops(P_equ) ⊆ ops(P_max) | **Pending** | P_max not yet wired (S13+) |
| channels(P_equ) ⊆ channels(P_max) | **Pending** | P_max not yet wired (S13+) |

The S12.1 axiom test therefore covers the P_min ⊆ P_equ chain
empirically. The P_equ ⊆ P_max chain will be added in S13+ when
P_max is wired. The proof above is structural and applies once
P_max definition is locked.

---

## Circulation

This proof is auxiliary to DR-2 (compositionality) — DR-4 is
structurally simpler and does not require external formal review.
Sub-agent `critic` review at draft time is sufficient. Final
incorporation into Paper 1 §6 framework section.
