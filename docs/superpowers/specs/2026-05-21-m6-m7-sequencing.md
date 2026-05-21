# M6 + M7 sequencing decision

> **Context** : Wave 3b M5 shipped 2026-05-21 (PR #35, squash `ffe0aeb`)
> with a known DR-3 gap : `mlx_latent_diffusion.execute_profile` does
> not branch on `request.profile`. Issue [#36](https://github.com/hypneum-lab/dream-of-kiki/issues/36).
> Two milestones are open : M6 (paper §7.9 + OSF amendment + DualVer
> decision + ship-critic) and M7 (the substrate fix for #36).
>
> **Decision** : run **M7 first, then M6**.

## 1. Why M7 first

The OSF amendment Q6JYN-W3B carries a permanent DataCite DOI once
filed. The principle from `docs/CLAUDE.md` "Dated immutables" is :
*don't ship a primary record you'll have to amend again next week*.

Three concrete arguments :

1. **Paper §7.9 evidence quality.** Drafting §7.9 against the M5
   v2 milestone means writing "the substrate is non-discriminative
   under an EC intensity proxy" as the headline empirical claim. A
   reader of the published Wave 3b paper would treat that as the
   *real* result of the framework's diffusion substrate. We don't
   want that to be the public-record result if 3-5 days of substrate
   work yields a discriminative one.
2. **DualVer flip economics.** M6's STABLE/PARTIAL decision is
   gated by §12.3 of the framework-C spec, which requires
   conformance pass + axiom no-violation + ship-critic GO. Issue
   #36 is a DR-3 conformance gap — STABLE is impossible while it
   stays open. M6-first would force a `+PARTIAL` flip, then M7
   would force *another* DualVer bump (FC PATCH or MINOR) to flip
   to STABLE. Two flips means two CHANGELOG amendments, two
   `STATUS.md` "Prior :" prepends, and a 24-hour window where the
   public version disagrees with the substrate state.
3. **Ship-critic precedent.** `feedback_critic_before_ship.md`
   records 4 prior cases where the critic-before-ship gate caught
   a paper-rejection-grade issue. A critic-reviewed M6 with #36
   open would almost certainly flag the gap as a *spec contract
   not honoured* — and ask us to fix it before shipping. Doing M7
   first front-loads that fix.

## 2. What M6-first would cost

If we did M6 first as `+PARTIAL` :

- The OSF amendment filing is the only step with permanence cost.
  If the amendment text says "future work : substrate profile
  differentiation", that's defensible — but it weakens the Wave 3b
  claim and reads as a hedged result. A reader citing the DOI gets
  the hedged version forever.
- ~4 hours saved (no need to re-run the bench; the M5 v2 milestone
  is the cited evidence).
- M7 then ships as a follow-up amendment (Q6JYN-W3B-v2) or as a
  separate OSF record entirely.

The 4 hours saved aren't worth the permanent-record cost.

## 3. What M7-first costs

- 3-5 days substrate work (issue #36 fix : wire the 4 dream-side
  primitives through `execute_profile` per profile activation,
  surface real per-op metrics, FC MINOR bump).
- A re-run of the 450-cell bench on macM1 (~30 min wall).
- A new milestone `docs/milestones/wave3b-bench-2026-05-2X.{json,md,cells.jsonl}`
  (date pinned at re-run).
- The old `wave3b-bench-2026-05-21.{json,md,cells.jsonl}` stays
  in `docs/milestones/` as the dated immutable record (it documents
  the M5+quick-win state and was the formal evidence for issue #36).

## 4. Recovery posture

If M7 stalls past day 5 :

- Fall back to **M6 as `+PARTIAL`** using the existing 2026-05-21
  milestone. Paper §7.9 then explicitly frames the result as
  "harness shipped, profile-discrimination deferred to follow-up
  Wave 3c sub-cycle".
- File the OSF amendment with the hedged language.
- M7 becomes its own milestone (M7 → Wave 3c first deliverable).

This recovery is a safety valve, not the plan.

## 5. Sequence

```
M7 spec      → 2026-05-21-m7-substrate-dr3-design.md   (this PR)
M7 plan      → docs/superpowers/plans/2026-05-2X-m7-substrate-dr3.md
M7 implement → branch feat/m7-substrate-dr3, ~3-5 days, FC MINOR
M7 re-bench  → macM1, 30 min wall, new dated milestone
M6 spec      → 2026-05-21-m6-paper-osf-design.md       (this PR)
M6 plan      → docs/superpowers/plans/2026-05-2X-m6-paper-osf.md
M6 implement → paper §7.9 EN + FR + DR-3 evidence v1.0
              + ship-critic + OSF amendment + DualVer flip
```

Both M7 and M6 specs ship in this same PR so the user can review
the plan as a coherent whole before any code lands.

## 6. Self-review

- **Placeholder scan** : no TBD ; the dated path on the M7 re-bench
  is intentionally `2026-05-2X` (resolved at re-bench time).
- **Internal consistency** : the M7-first sequence is consistent
  across §1, §3, §5 ; the recovery posture §4 explicitly handles
  the stall case.
- **Scope check** : this decision doc is a *sequencing* spec — one
  decision, one rationale, one fallback. Two companion specs cover
  the actual deliverables. Fits the brainstorming-skill split.
