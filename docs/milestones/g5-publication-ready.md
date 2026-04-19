# G5 — PUBLICATION-READY Gate Report

**Gate** : G5 (PUBLICATION-READY for Paper 1 submission)
**Target week** : S18 (cycle 1)
**Status** : **PARTIAL — most criteria met, paper draft incomplete +
DR-2 review pending**

## Gate criteria (from framework spec §9 publication_ready_gate)

| Criterion | Status | Notes |
|-----------|--------|-------|
| coverage : 100% stratified cells in last MAJOR | ⚠️ PARTIAL | Synthetic ablation S15.3 PASS 3/4 (run_id: `syn_s15_3_g4_synthetic_pipeline_v1`, dump: `docs/milestones/ablation-results.json`) ; real mega-v2 + real predictors deferred S20+ |
| seeds_per_cell : ≥3 | ✅ MET | 3 seeds (42, 123, 7) used in pilot G2 (run_id: `syn_g2_pmin_pipeline_v1`, dump: `docs/milestones/g2-pilot-results.md`) + G4 ablation (run_id: `syn_s15_3_g4_synthetic_pipeline_v1`) |
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
- DR-2 reviewer confirms the generated-semigroup proof published
  in `docs/proofs/dr2-compositionality.md` (DR-2 as stated is
  already proved ; the open item is the universal-property/freeness
  sub-claim, orthogonal to the three conjuncts of DR-2)
- Paper 1 final draft (Results + Discussion + Future Work) complete
- G7 ESNN conformance promoted in §5.6 and §8.3 (done 2026-04-19
  in the W-series revisions)
- Pre-submission review from T-Col.4 network ≥1 positive
- Tag framework C-v0.7.0+STABLE
- Submit Paper 1 arXiv → **PLOS Computational Biology** (primary
  target as of 2026-04-19 retarget from Nature Human Behaviour,
  per the cold-review analysis in
  `Business OS/paper1-cold-review-2026-04-19.md`)

**Branch Pivot B** (if cycle-1 cannot close gates by S22) :
- See `docs/proofs/pivot-b-decision.md` (S18.3) for 3 sub-branches
- Likely outcomes : EXTEND-CYCLE-1 timeline OR FALLBACK-VENUE
  (PLOS CB → Cognitive Science / Neural Computation / NeurIPS 2026
  workshop) OR SCOPE-DOWN (single-paper TMLR/ICLR workshop,
  framework deferred cycle 2)

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
5. arXiv preprint then PLOS Computational Biology submission
   (primary target retargeted 2026-04-19 from Nature Human
   Behaviour following scope-fit analysis ; PLOS CB accepts
   formal frameworks + computational-cognitive-modeling papers,
   open access, 4–6 week editorial turnaround)

## Submission decision (S22.1)

**Date** : TBD (S22 calendar)
**Status** : **PENDING action externe utilisateur**

### Branch selection

The G5 gate at S18 was status DEFER (multiple criteria PARTIAL).
At S22 closeout the user must choose between :

- **GO-FULL** (default if all criteria met by S22) :
  Submit Paper 1 to **PLOS Computational Biology** via the portal
  at https://journals.plos.org/ploscompbiol/ (retargeted
  2026-04-19 from Nature Human Behaviour after scope-fit
  analysis). Tag framework C-v0.7.0+STABLE post-DR-2 reviewer
  confirmation of the generated-semigroup proof. See
  `ops/plos-cb-submit-tracker.md` (to be added — template
  adapted from the Nature HB tracker) for the manual action
  checklist.

- **Pivot B** (if criteria not met) :
  Activate one of the three Pivot B branches per
  `docs/proofs/pivot-b-decision.md` :
  - **B-EXTEND** : extend cycle-1 timeline by 4-8 weeks (S22
    → S30), maintain PLOS CB target
  - **B-FALLBACK-VENUE** : retarget to Cognitive Science
    journal / Neural Computation / NeurIPS 2026 workshop
    (Theoretical Foundations of Continual Learning)
  - **B-SCOPE-DOWN** (Pivot A) : single-paper TMLR/ICLR
    workshop, framework deferred cycle 2

### Decision criteria recap

The branch selection consults the gate criteria status table
above. As a quick reference :

| Criterion | S18 status | S22 target |
|-----------|-----------|-----------|
| Coverage 100% MAJOR | PARTIAL | MET (real ablation S20+) |
| Seeds ≥3 | MET | MET |
| Retained regression ≤1% | N/A | MET (real model S20+) |
| Zero blocking 7d | N/A | MET (CI green) |
| DualVer +STABLE | MET | MET (or +STABLE-PRIME on Pivot) |
| Pre-submission review ≥1 | PENDING | MET (T-Col.4 outreach) |
| DR-0..DR-4 axioms | 4/5 | 5/5 (G3 reviewer close) |
| Ablation P_min, P_equ, P_max | 2/3 | 2/3 (P_max cycle 2) |
| Paper draft complete | PARTIAL | MET (S19+S20 done) |

### Outcome (fill at S22)

- **Branch chosen** : TBD
- **Date** : TBD
- **Framework version tagged** : TBD
- **Paper 1 journal target** : TBD
- **arXiv preprint reference** : TBD (see arxiv-submit-log.md)
- **Justification** (3-5 sentences) : TBD
