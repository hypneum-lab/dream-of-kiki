# M6 — Paper 2 §7.9 + DR-3 evidence v1.0 + OSF amendment design

> **Parent decision** : `2026-05-21-m6-m7-sequencing.md` — M6 runs
> *after* M7. The cited bench is the post-M7 v3 dump
> (`docs/milestones/wave3b-bench-2026-05-2X.{json,md,cells.jsonl}`),
> not the M5 v2 dump from 2026-05-21.
> **DualVer** : ship-critic-gated flip from `C-v0.15.0+PARTIAL`
> (post-M7) to `+STABLE` iff framework-C §12.3 holds. Otherwise stay
> `+PARTIAL` and document the deferred cell.

## 1. Goal

Close Wave 3b by producing the public-record artefacts :

- **`docs/papers/paper2/results.md` §7.9** : a new section drafting
  the Wave 3b latent-diffusion-substrate empirical story end-to-end,
  citing the post-M7 bench.
- **FR mirror** `docs/papers/paper2-fr/results.md` §7.9 (EN→FR rule
  per `CONTRIBUTING.md`).
- **`docs/proofs/dr3-diffusion-substrate-evidence.md` v1.0** : the
  behavioural-angle proof now backed by the M7 conformance test +
  the bench-v3 numbers, replacing the v0.1 PARTIAL stub.
- **OSF amendment Q6JYN-W3B** published with a DataCite DOI.
- **CHANGELOG** entry for the final DualVer state (STABLE if §12.3
  holds, else PARTIAL with the deferred-cell justification).
- **Ship-critic review** verdict GO before any of the above lands
  on `main`.

## 2. Decisions

### D1 — Bench cited

The Wave 3b paper §7.9 cites **the post-M7 bench v3**, not the
2026-05-21 M5+quick-win bench. The 2026-05-21 milestone stays in
`docs/milestones/` as the **dated immutable record** of the
intermediate state (the EC quick-win documented the substrate gap
formally and motivated M7) and is referenced once in §7.9 as the
audit trail for issue #36 — but the **headline numbers** are the
v3 numbers.

### D2 — DualVer flip gating

`framework-C §12.3` STABLE conditions :

1. All conformance tests pass on the latest commit.
2. No axiom violation in the test matrix.
3. Ship-critic GO verdict on the M6 PR.
4. At least one positive empirical claim of the substrate's
   profile-discrimination capability is registered in the bench.

If all four hold post-M7 bench v3, M6 flips to `+STABLE` (FC PATCH
`C-v0.15.0 → C-v0.15.1+STABLE`). Otherwise it stays
`C-v0.15.0+PARTIAL`, the CHANGELOG records the failing condition
explicitly, and §7.9 frames the result accordingly.

The acceptance test is **mechanical** : the M6 plan ships a CI
check `scripts/check_§12_3_stable_conditions.py` (or similar) that
returns a boolean. The bump is the boolean.

### D3 — Paper §7.9 structure

The §7.9 EN draft has 5 sub-sections, each capped at ~250-400
words :

- **§7.9.1 Substrate description.** Recap the diffusion substrate
  (Encoder + MLPDenoiser + NoiseSchedule + Trainer + Sampler), cite
  the M2 skeleton + M3 wiring + M5 harness + M7 profile binding
  milestones with their commits.
- **§7.9.2 Conformance posture.** State 8/8 typed primitives with
  Canal 4 documented no-op (cite v0.1 evidence) ; cite the M7
  conformance test asserting the profile-activation surface.
- **§7.9.3 Empirical setup.** N=30 seeds × 5 CIFAR-100 task-splits
  × 3 profiles, 450 cells on macM1 (cite the bench v3 milestone +
  the R1 reproducibility hashes).
- **§7.9.4 Headline result.** Whichever pattern bench v3 produced :
  positive H1/H2/H4 (point to the rejected null), or empty
  (extend the DR-4 v0.6 amendment's empirical-emptiness scope to
  the diffusion substrate, mirror the G4-sexto / -septimo paragraph
  style). The §7.9.4 *template* is drafted before bench v3 runs ;
  the *fill-in* happens after.
- **§7.9.5 Audit trail.** Briefly note the M5 → quick-win → issue
  #36 → M7 sequence, cite the 2026-05-21 milestone as the
  documented intermediate state, and link the OSF amendment.

EN draft ships, FR mirror follows in the same PR.

### D4 — DR-3 evidence v1.0

Replace `docs/proofs/dr3-diffusion-substrate-evidence.md` v0.1
PARTIAL with v1.0 :

- 8/8 primitives typed (the 7 + Canal 4 doc-no-op posture from
  v0.1 is preserved ; M7 adds the formal profile-activation
  surface as the missing piece).
- Cite the new `tests/conformance/axioms/test_dr3_diffusion_profile.py`
  as the conformance proof.
- Cite the bench v3 milestone as the behavioural-angle evidence.
- Status block flips to `v1.0 STABLE` iff D2's 4 conditions hold,
  else stays `v1.0 PARTIAL` with the same deferred-cell note.

### D5 — OSF amendment

The amendment text in `docs/osf-amendment-wave3b.md` (already drafted
2026-04-19, never filed) is updated to cite the final substrate
posture + bench-v3 evidence. Then filed against Q6JYN with a
DataCite DOI. Key changes from the 2026-04-19 draft :

- The R1 entries section updates with the **regenerated** golden
  hashes (M7 FC MINOR bump changed them ; the original draft cited
  pending-review hashes that no longer exist).
- The "Cross-machine R1" caveat references the MLX #3568 issue
  (M1 Max divergence) — already documented in the prior milestone.
- The "DualVer transition" line states the final state (STABLE or
  PARTIAL per D2).

Filing is the user's action (OSF web UI). M6 ships the text ready
to copy-paste ; the DOI is recorded in the CHANGELOG after filing.

### D6 — Ship-critic invocation

Per `feedback_critic_before_ship.md` (4 prior validations), the
critic-before-ship gate is mandatory. The critic-reviewer subagent
is dispatched with the full M6 PR diff in fresh context, scoped to :

- Does §7.9 match the bench v3 numbers exactly ?
- Does the DR-3 evidence v1.0 cite real, registered conformance
  tests ?
- Does the OSF amendment text match the actual repo state ?
- Does the DualVer flip honour §12.3 ?
- Are EN and FR mirrors consistent ?

If the critic finds a *blocking* issue, fix it and re-dispatch.
Treat the verdict as binding.

### D7 — Acceptance

M6 ships when :

1. EN §7.9 + FR §7.9 mirror both committed and reviewed.
2. DR-3 evidence v1.0 committed.
3. DualVer bump committed (STABLE or PARTIAL per D2) ;
   `CHANGELOG.md` updated ; `STATUS.md` "Prior :" prepended.
4. OSF amendment text finalised in `docs/osf-amendment-wave3b.md`,
   filing instructions handed to the user.
5. Ship-critic verdict = GO (or all blocking findings fixed and
   re-reviewed GO).
6. Full test suite green, ruff + mypy clean, coverage gate satisfied.

## 3. File touch map (M6 implementation plan target)

| File | Action | LoC est. |
|---|---|---|
| `docs/papers/paper2/results.md` | Add §7.9 (5 sub-sections, ~1500 words) | +250 |
| `docs/papers/paper2-fr/results.md` | FR mirror of §7.9 | +250 |
| `docs/proofs/dr3-diffusion-substrate-evidence.md` | Bump v0.1 → v1.0 | +60 −10 |
| `docs/osf-amendment-wave3b.md` | Finalise for filing | +30 |
| `CHANGELOG.md` | New section `[C-v0.15.1+STABLE]` or `[C-v0.15.0+PARTIAL]` (final) | +20 |
| `STATUS.md` | Prepend M6 closure entry | +1 (one long line) |
| `pyproject.toml` | SemVer bump (PATCH if STABLE flip, no-op if PARTIAL stays) | ±1 |
| `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §12.3 | Cite the new STABLE-condition check script (or fix an inconsistency surfaced by the critic) | +5..20 |
| `scripts/check_dualver_stable.py` | New : mechanical check returning bool for D2 | +60 |

EN→FR rule applies to any §12 spec edit.

## 4. Out of scope for M6

- Re-running the bench (that is M7's deliverable).
- New substrates (M6 ships *one* substrate's paper section).
- Cross-paper writing (`papers/byline-policy.md` etc.) — separate
  hygiene PR if needed.
- Wave 3c planning — separate spec.

## 5. Risks

| Ref | Risk | Mitigation |
|---|---|---|
| R1 | Bench v3 produces empty H1/H2/H4 → STABLE blocked | §7.9.4 template handles both branches ; OSF amendment is filed with PARTIAL framing if needed. M6 still ships. |
| R2 | Ship-critic finds a *blocking* issue (e.g. §7.9 number mismatch with the actual JSON) | Fix and re-dispatch ; precedent from the 4 prior cases shows this adds ~2 hours per cycle. |
| R3 | The OSF amendment text references commits / tags that haven't yet been pushed | The plan ships the amendment AFTER the M6 PR merges, so the cited SHAs are stable. |
| R4 | FR mirror drift (EN updated, FR stale) | The CONTRIBUTING.md rule blocks the PR ; the plan ships EN and FR in the same commit. |

## 6. Self-review

- **Placeholder scan** : no TBD ; "STABLE *or* PARTIAL" branches are
  intentional, with mechanical resolution (D2).
- **Internal consistency** : D1 (cite v3 bench) ↔ D3 (§7.9.4 cites
  v3 numbers) ↔ D5 (amendment cites regenerated R1) all reference
  the post-M7 state.
- **Scope check** : single milestone, single paper section, single
  amendment. Fits one writing-plans plan ; the ship-critic step is
  one task within that plan, not a separate spec.
- **Ambiguity** : §7.9.4 template-then-fill is acknowledged. The
  plan task drafting §7.9 starts with the template (executable
  before bench v3 finishes) and a second task fills in the numbers
  (sequenced after bench v3 lands).
