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
