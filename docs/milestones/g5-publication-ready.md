# G5 — PUBLICATION-READY Gate Report

**Gate** : G5 (PUBLICATION-READY for Paper 1 submission)
**Target week** : S18 (cycle 1)
**Status** : **PARTIAL — most criteria met, paper draft incomplete +
DR-2 review pending**

## Gate criteria (from framework spec §9 publication_ready_gate)

| Criterion | Status | Notes |
|-----------|--------|-------|
| coverage : 100% stratified cells in last MAJOR | ⚠️ PARTIAL | Synthetic ablation S15.3 PASS 3/4 ; real mega-v2 + real predictors deferred S20+ |
| seeds_per_cell : ≥3 | ✅ MET | 3 seeds (42, 123, 7) used in pilot G2 + G4 ablation |
| retained_regression_max_pct : ≤1% | ⚠️ N/A | Skeleton swap_now ; real regression check needs MLX-real model S20+ |
| consecutive_zero_blocking_days : 7 | ⚠️ N/A | No live deployment ; CI green continuously since repo public |
| dualver_status : `+STABLE` | ✅ MET | C-v0.5.0+STABLE current |
| pre_submission_reviews_min : ≥1 | ❌ PENDING | T-Col reviewer outreach in progress (action externe utilisateur) |
| axioms_proven : DR-0..DR-4 | 🟡 4/5 | DR-0 ✅ DR-1 ✅ DR-3 ✅ DR-4 ✅ ; DR-2 ⏳ pending external reviewer (G3 gate) |
| ablation_complete : P_min, P_equ, P_max | 🟡 2/3 | P_min ✅ wired + pilot ; P_equ ✅ wired + ablation S15.3 ; P_max skeleton only |
| paper_draft : complete | ⚠️ PARTIAL | Outline ✅ Abstract ✅ Intro ✅ Results section ⏳ S18.2 ; Discussion + Future Work ⏳ S20 |

## Decision (S18)

**Branch DEFER (default at S18 with PARTIAL)** :
- Architecture validated, infrastructure complete, statistical
  pipeline validated (synthetic G4 PASS)
- Paper draft skeleton ✅ but Results section + Discussion +
  Future Work ⏳
- DR-2 external review pending (action externe utilisateur)
- P_max skeleton only (cycle 2 deferred per SCOPE-DOWN)
- Real mega-v2 ablation pending S20+

**Branch GO-FULL** (if S20-S22 closes the gaps) :
- DR-2 reviewer confirms strict proof OR DR-2' fallback adopted
- Paper 1 final draft (Results + Discussion + Future Work) complete
- 3 consecutive PASS on real ablation
- Pre-submission review from T-Col.4 network ≥1 positive
- Tag framework C-v0.7.0+STABLE
- Submit Paper 1 arXiv → Nature HB

**Branch Pivot B** (if cycle-1 cannot close gates by S22) :
- See `docs/proofs/pivot-b-decision.md` (S18.3) for 3 sub-branches
- Likely outcomes : EXTEND-CYCLE-1 timeline OR DOWNGRADE-JOURNAL
  (Nature HB → PLoS Comp Bio / Cognitive Science) OR SCOPE-DOWN
  (single-paper TMLR/ICLR workshop, framework deferred cycle 2)

## Action S18

Adopt **DEFER** (default). Document path to GO-FULL in S20-S22.
Trigger T-Col reviewer follow-up (Q_CR.1 b) and OSF upload (action
externe utilisateur) as parallel critical-path items.

## Comparison with G2 + G4 reports

| Gate | Status | Blocker |
|------|--------|---------|
| G1 | ✅ LOCKED Branch A | Studyforrest fallback verified |
| G2 | 🟡 GO-CONDITIONAL | Real benchmark + real MLX inference S20+ |
| G3 | ⏳ PENDING | DR-2 external reviewer (action externe) |
| G4 | 🟡 GO-CONDITIONAL | Real ablation evidence S20+ |
| G5 | ⚠️ DEFER | Multiple criteria PARTIAL ; cycle-1 close S20-S22 |

## Next steps (S20-S22)

1. Real mega-v2 + real MLX inference wiring (closes G2 + G4)
2. DR-2 external reviewer feedback (closes G3)
3. Paper 1 Results + Discussion + Future Work draft
4. T-Col pre-submission network review ≥1 positive return
5. arXiv preprint then Nature HB submission
