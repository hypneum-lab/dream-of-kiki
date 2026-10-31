# OSF Pre-registration Amendment — E1-small, 2026-10-30

**Target registration:** E1-small, `docs/osf-prereg-e1-small-2026-10-25.md`
(frozen, SHA-256
`6d47d8b71829d884d3a18dda84cd21aa080f1090101d62a4bafe6bc493208463`,
lock commit `963bb24`), parent OSF `10.17605/OSF.IO/Q6JYN`.
**Amendment type:** DISCLOSURE (exposure of the live run log, status of
the 35B launch rule, post-hoc analyses announced). No hypothesis,
estimand, threshold, sample size, corpus or code is changed.
**Status:** written 2026-10-30, PI authorisation of 2026-10-30 (about
15:10 CET). E1-small is **not yet filed on OSF**; this amendment is
filed with it and cites the deviation note below.
**Frozen draft:** not edited by this amendment.

---

## §1 Summary

E1-small is running on KXKM-AI. Its live run log prints per-cell
scores, and the PI read that log once on 2026-10-30 at 14:22 CET. The
same day the PI abandoned the 35B study, which makes the 35B GO /
POSTPONE rule (`docs/milestones/e1-35b-go-rule-2026-10-28.md`) moot.
This amendment declares the exposure, restates the run-before-filing
deviation, and announces three post-hoc analyses that will be reported
as exploratory, next to the registered results.

## §2 Exposure of the live run log

- **Run timeline.** The tuning and calibration phases ran on
  2026-10-29; their output, the tuning amendment
  (`docs/milestones/e1-small/e1-tuning-amendment-2026-10-29.json`),
  was frozen by PR #77, merged 2026-10-29T23:05:10Z. The confirmatory
  phase started on 2026-10-30 at 00:05:29 CET, 19 s after that merge.
  At 14:22 CET, 290 of 1,410 confirmatory cells were complete;
  completion is expected on 2026-11-01 around 21:30 CET.
- **What was exposed.** The live run log prints scores for each
  completed cell. They are visible to anyone reading the log since the
  confirmatory phase started. The exposure was found on 2026-10-30 at
  14:22 CET by a read-only SSH look at the log; nothing was changed on
  the host and the run was not touched.
- **No protocol decision after reading.** No protocol, code, corpus,
  tuning or analysis choice for E1-small has been made after that
  reading, and none will be. The registered analysis
  (`scripts/analyze_e1_cuda.py`) runs only on the complete cache
  (§8: no interim tests; partial caches are not analysed for the
  confirmatory family).
- **Why it no longer bears on a decision.** The only decision tied to
  E1-small numbers was the 35B launch rule. The PI abandoned the 35B
  study on 2026-10-30 (about 15:10 CET), on the basis of a re-reading
  of that rule (H14-A, one of its two triggers, contrasts `p_min`, SGD
  replay with RECOMBINE a no-op, with the prompt-only control: no
  dream operation is involved), and states that no E1-small
  confirmatory cell had been read by the PI at that point. The rule
  is withdrawn, not applied
  (`docs/milestones/e1-35b-go-rule-2026-10-28.md`, dated section).
  E1-small is reported as a stand-alone result.

## §2bis Addendum of 2026-10-31: earlier exposure to interim means

*Added 2026-10-31, PI authorisation of 2026-10-31. §2 above is kept
as written; this section corrects it.*

- **What §2 omits.** On 2026-10-30 at 13:11 CET the PI asked the
  assistant (Claude, an AI coding agent operating the run) for "first
  results". At 13:12:25 CET the assistant replied with descriptive
  per-profile summaries computed on the partial confirmatory cache
  (`e1-confirmatory-torch-Qwen_Qwen3.5-4B.cells.jsonl`, 266 / 1,410
  cells):
  - `p_min`: 141 / 141 cells (seeds 0-140), mean Δ = +0.057;
  - `p_equ`: 125 / 141 cells (seeds 0-124), mean Δ = +0.064;
  - mean wall time 177 s per cell, 0 collapse flags, 0 invariant
    violations, for both profiles.
  No other profile had any confirmatory cell at that time, so no
  registered contrast (H14-A, H14-B, H14-C or secondary) could be
  formed, and no test, interval or state was computed. The reply
  stated that no conclusion could be drawn before the run completes.
- **Correction.** The first PI exposure to confirmatory numbers is
  therefore 13:12 CET, not 14:22 CET, and it was to interim means of
  two profiles, not only to per-cell log lines. The statements that
  "no E1-small confirmatory cell had been read by the PI" when the
  35B study was abandoned (about 15:10 CET), in §2 above and in
  `docs/milestones/e1-35b-go-rule-2026-10-28.md`, are inaccurate and
  are corrected by this section.
- **Bearing.** The written basis of the 35B decision (the re-reading
  of H14-A, which contains no dream operation) does not use these
  numbers, and both means are uninformative about any contrast with a
  control. The exposure is nonetheless declared because it preceded
  the decision. The commitments of §2 stand: no protocol, code,
  corpus, tuning or analysis choice for E1-small has been or will be
  made after it, and the registered analysis runs on the complete
  cache only (§8). Computing interim per-profile means was itself a
  departure from the no-interim-analysis rule of §8, made by the
  assistant without a PI request for them; it is not repeated.

## §3 Run started before the OSF filing (already declared)

The run started on KXKM-AI before the pre-registration was filed on
OSF. This deviation was declared before the run, in
`docs/milestones/e1-small/deviation-run-before-osf-2026-10-29.md`
(PR #76, merged 2026-10-29T05:22:21Z; PI decision of 2026-10-29 06:13
CET). It is not declared in PR #77, which only freezes the tuning
amendment. The protocol is locked by commit `963bb24` (merge of PR
#72, 2026-10-28T16:57:07Z). E1-small, G4 and DR-3 are still not filed
on OSF at the time of writing.

## §4 Post-hoc analyses announced (exploratory)

These are announced before the confirmatory cache is complete and
before any analysis of it. They are **post-hoc** with respect to the
frozen protocol, spend no α, cannot change any registered state or
reading (§2, §7), and are reported in a separate exploratory section.

1. **Options-only control.** After the run, the base model
   (`Qwen/Qwen3.5-4B`, revision `851bf6e8…`) is scored on the
   confirmatory items with the question removed (answer options only),
   with the registered scoring rule (length-normalised log-probability
   of each option, `scripts/pilot_cycle3_cuda.py`, `_score_choice`).
   This measures how much of the base accuracy, and of each profile's
   change, can be reached from the options alone.
2. **Reading by source.** Every registered contrast is described per
   source (`daily_oracle`, `mmlu_cf`, `realtimeqa`) in addition to the
   per-source description already in the decision code. No per-source
   claim is made.
3. **Daily Oracle below chance.** In the frozen corpus manifest
   (`docs/milestones/e1-small/corpus-manifest-frozen-2026-10-28.json`,
   field `headroom.accuracy`), base accuracy on Daily Oracle (TUNING
   partition) is 0.185, below the 0.25 chance level of the four-choice
   format (§5.1). Two points are reported with the results, as
   observations made before the confirmatory analysis:
   - the correct option is the longest one in about 34 % of Daily
     Oracle items, which interacts with a length-normalised score;
   - DOWNSCALE (`SHRINK = 0.99`, `scripts/e1_profiles.py`; applied by
     `downscale` in `scripts/pilot_cycle3_cuda.py`) changes the weights
     roughly 35 times more than the SGD replay update does in this
     harness, so H14-B and H14-C readings are read with the DOWNSCALE
     caveat of §7 in mind.

None of these points changes the protocol: hypotheses, estimands,
δ, α, N = 141, profiles, tuning amendment and decision code stay as
frozen.

## §5 Paste into OSF amendment field

---

**Title:** E1-small: exposure of the live run log; 35B launch rule
withdrawn; post-hoc analyses announced

**Amendment summary:**

The E1-small confirmatory run (Qwen3.5-4B, KXKM-AI) started on
2026-10-30 at 00:05:29 CET. Its live log prints per-cell scores; the
PI read it once on 2026-10-30 at 14:22 CET (290 / 1,410 cells), without
intervening. No protocol decision has been or will be made after that
reading; the registered analysis runs on the complete cache only. The
35B study, whose launch depended on E1-small, was abandoned by the PI
on 2026-10-30, so the launch rule is withdrawn and E1-small is reported
as a stand-alone result. The run started before this registration was
filed, as declared on 2026-10-29 (deviation note, lock commit
`963bb24`). Three exploratory post-hoc analyses are announced: an
options-only control on the base model, a per-source reading of every
contrast, and a note on Daily Oracle base accuracy (0.185) below chance
(0.25). Nothing in the frozen protocol is changed.

---

## §6 Paste into OSF amendment field (addendum of 2026-10-31)

---

**Title:** E1-small: earlier exposure to interim means (addendum)

**Amendment summary:**

Correction to the amendment of 2026-10-30. Before the PI read the live
log at 14:22 CET, the AI agent operating the run had already reported
to the PI, at 13:12 CET on 2026-10-30, the interim mean change of the
two profiles that had confirmatory cells (`p_min`, 141 cells, +0.057;
`p_equ`, 125 cells, +0.064). No control profile had run, so no
registered contrast, test or state was computed. This exposure
preceded the PI decision to abandon the 35B study (about 15:10 CET);
the earlier statement that no confirmatory cell had been read at that
point is withdrawn. The decision's written basis does not use these
numbers. No protocol, code, corpus, tuning or analysis choice has been
or will be made after the exposure; the registered analysis runs on
the complete cache only.

---
