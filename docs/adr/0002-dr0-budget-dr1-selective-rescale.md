# ADR-0002: amend DR-0 (budget), DR-1 (selective) and pillar B (`downscale` to `rescale`), and call the axioms framework hypotheses

> Status: **Proposed** (2026-11-08). PI decisions D1-D8 closed on
> 2026-11-08 13:50 CET (listed in section 6); not frozen: the L1 tag
> and SWHID are gated by the roadmap reading lock (principle 6). Drafted by an agent from process records and the
> 2026-10-24 hypotheses review; no new run was made.
>
> Issue: hypneum-lab/dream-of-kiki#48 (milestone M1). Amendment:
> `docs/specs/amendments/2026-11-08-dr0-budget-dr1-selective-rescale.md`.
> Previous ADR: `0001-kxkm-derogation-e1-small-controls-p8c.md`.

## 1. Context

The framework-C spec (§6.2) states DR-0 as "every dream output traces
to a DE with a finite budget", where a DE is a bounded off-line slot;
DR-1 as "every buffered episode is offered to a DE within `τ_max`, then
purged"; and pillar B as `downscale`, a factor in (0, 1].

The 2026-10-24 review (`hypneum-lab/.github`,
`docs/research/2026-10-24-revue-hypotheses/README.md:78,146-150,273,304`)
reports convergent literature against each:

- consolidation moved to continuous fast-weight writes (Titans/Hope,
  TTT, MemoryBench), so a bounded slot is one schedule among others;
- selective tagging and raw retention beat replay-everything-then-purge
  (Denis 2026, Zhang 2605.12978);
- the WISE framework (Kinoshita 2025) makes the sign of homeostasis
  depend on the plasticity rule and the wake/sleep rate contrast, which
  contradicts a factor fixed in (0, 1].

Local evidence (three dated NH-03 results, 2026-10-25/27/30) shows that
at an equal write budget the placement of the writes relative to the
next demand moves retention by up to 15.7 points, with no
TTT-specific advantage.

## 2. Decision (proposed)

1. DR-0 becomes a recorded finite **budget** (bits erased, traffic,
   duration; joules reported on a separate axis) that selects the
   operator; a DE is bounded by its budget, not by a clock, and its
   schedule is a free parameter.
2. DR-1 becomes **selective** (weight above a declared threshold) with
   no mandatory purge.
3. Pillar B `downscale` becomes `rescale(f)` with a factor of
   indefinite sign; `downscale` stays as the special case f <= 1.
4. DR-0..DR-4 are called **framework hypotheses** in prose; identifiers
   are unchanged.
5. The change is an **additive amendment** (new L1 tag, SWHID, list of
   re-labelled runs). Frozen spec text, preregistrations and dated
   results are not edited. Code renames, with backward-compatible
   aliases, are a separate follow-up PR.

## 3. Consequences

- No registered run is invalidated or excluded from its own analysis;
  all existing runs are re-labelled "pre-amendment" and may not be cited
  for sign-indefinite rescale, bits/traffic budgets or selective DR-1
  (amendment, section 6).
- `ttt_fast_weights` stays a conforming C+ substrate; the amendment
  gives that conformance a budget reading, and makes no claim that TTT
  consolidates better.
- Follow-up work: code PR (aliases, `BudgetCap` bits/traffic fields,
  tests for the new forms, I1 restatement), docs PR (spec, Paper 1 for
  the ROADMAP Q2 reformulation), tests NH-05 / R9 H1 / H10 as separate
  preregistrations.
- Tension with `ROADMAP.md:130` (DR-0/DR-1 deferred after M3) and
  principle 6 (reading lock): the freeze cannot happen at M1 without
  either clearing the lock or a dated derogation (decision D1).

## 4. Alternatives considered

- Keep DR-0/DR-1 and add the new forms as separate hypotheses
  (decision D6 option c): no wording conflict, but two competing
  statements of the same contract.
- Rename `downscale` hard (no alias): breaks frozen scripts, golden
  hashes and run logs.
- Add `rescale` as a fifth operation: changes the monoid, DR-2 and DR-4
  operation sets.

## 5. Status log

| Date | Event |
|---|---|
| 2026-11-08 | Proposed with the L0 draft amendment (draft PR) |
| 2026-11-08 | PI decisions D1-D8 recorded; status stays Proposed |

## 6. PI decisions (2026-11-08)

- D1: merge as L0 now; L1 tag only after the reading lock; #48 moved
  to M3 by the PI.
- D2: the tag freezes hypothesis text and predictions only.
- D3: budget vector (B_bits, B_traffic, T); joules outside the statement.
- D4: f in (0, inf) with a per-profile cap.
- D5: `rescale` primary; `downscale` / `Operation.DOWNSCALE` aliases.
- D6: `weight(e)` and `theta` per profile, preregistered; I1 restated.
- D7: prose "framework hypothesis"; identifiers unchanged.
- D8: formal-axis MINOR (C-v0.26.0) via the follow-up spec PR.
