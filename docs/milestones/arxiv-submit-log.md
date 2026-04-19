# arXiv Preprint Submission Log — Paper 1

**Cycle** : 1 (S21.1)
**Status** : **PENDING manual user action**

## Action checklist (user)

- [ ] Render `docs/papers/paper1/full-draft.md` to LaTeX via
      Quarto or pandoc :
      ```bash
      cd docs/papers/paper1
      quarto render full-draft.md --to latex
      # OR
      pandoc full-draft.md -o full-draft.tex \
        --bibliography=references.bib --citeproc
      ```
- [ ] Embed LaTeX figures (currently absent — see §7.x figure
      placeholders in `results-section.md` and full-draft notes)
- [ ] Generate PDF for visual review : `quarto render
      full-draft.md --to pdf` or `pdflatex full-draft.tex`
- [ ] Verify Nature HB length compliance (main ≤ 5000 words,
      supp unbounded). Trim per `full-draft.md` revision notes.
- [ ] Login to https://arxiv.org/submit
- [ ] Choose primary category : `cs.LG` (Machine Learning) +
      cross-list : `q-bio.NC` (Neurons and Cognition) +
      `cs.AI` (Artificial Intelligence)
- [ ] Upload sources : `full-draft.tex`, `references.bib`,
      figure files (if any), supplementary archive
- [ ] Verify on arXiv preview before final submit
- [ ] Receive arXiv ID (format `2604.XXXXX`)
- [ ] Update this log with arXiv ID + DOI + URL
- [ ] Tag git commit `arxiv-v0.1` for freeze :
      ```bash
      git tag arxiv-v0.1 -m "Paper 1 arXiv v0.1 submission"
      git push origin arxiv-v0.1
      ```

## arXiv submission identifiers (fill after submit)

- **arXiv ID** : TBD
- **DOI (arXiv-issued)** : TBD
- **URL** : TBD
- **Submitted on** : TBD
- **Version** : v1
- **Tag** : `arxiv-v0.1`

## Pre-submission verification

- [ ] All synthetic-data caveats explicit in §7 + §8
- [ ] OSF DOI inserted in §6.1 (currently pending OSF lock —
      action externe utilisateur)
- [ ] References.bib resolves all `\cite{}` calls without
      undefined-citation warnings
- [ ] Authorship byline `dreamOfkiki project contributors` with
      Saillant C. as corresponding ; affiliation L'Electron Rare
- [ ] Repo URL present in §5.5 :
      `github.com/electron-rare/dream-of-kiki`
- [ ] HuggingFace model URLs present (P_min + P_equ ; P_max
      cycle-2 placeholder)
- [ ] Zenodo DOI inserted (post-mint, see Methods §6.5)
- [ ] No AI attribution in author list, acknowledgments, or
      bibliography ; CONTRIBUTORS.md fully populated for the
      project byline disclosure

## Post-submission tracker

| Date | Event | Notes |
|------|-------|-------|
| TBD | arXiv submitted | (fill after action) |
| TBD | arXiv announced | (typically next business day) |
| TBD | Notification to T-Col.4 reviewers | (fill after annon.) |
| TBD | Reviewer feedback round 1 | (collect into reviewer-feedback.md S25.1) |

## Cross-references

- Source narrative : `docs/papers/paper1/full-draft.md`
- Section sources : `docs/papers/paper1/{abstract, introduction,
  background, methodology, results-section, discussion,
  future-work}.md`
- References : `docs/papers/paper1/references.bib`
- G5 PUBLICATION-READY gate : `docs/milestones/g5-publication-ready.md`
- Pivot B contingency : `docs/proofs/pivot-b-decision.md`

## Notes

- arXiv submission is intentionally pre-Nature HB submission :
  arXiv preprint provides a citable URL and DOI for outreach
  emails (T-Col reviewer recruitment, fMRI lab partnership) ;
  Nature HB submission follows in S22.1 (or Pivot B branch).
- arXiv allows version updates (v1 → v2 etc.) post-feedback ;
  significant revisions warrant a new arXiv version, not a
  silent edit.
- Should arXiv reject the cs.LG category placement, fall back to
  cs.AI primary with q-bio.NC cross-list.

## v2 plan (cycle-2 substrate evidence) — DEFERRED

**Status** : planned, **deferred until v1 acceptance / feedback
round 1**. Do not submit v2 before v1 is announced and feedback
collected.

**Scope of v2** : incorporate the cycle-2 cross-substrate
preliminary replication now referenced in Paper 1 §8.5
(`docs/papers/paper1/discussion.md`, mirror FR in
`docs/papers/paper1-fr/discussion.md`). The v2 update will add :

- Updated §8.5 "Cross-substrate preliminary replication" subsection
  already merged into the cycle-2 discussion draft.
- Pointer citations to the cycle-2 artifacts :
  - `docs/milestones/conformance-matrix.md` (C2.10 static matrix)
  - `docs/proofs/dr3-substrate-evidence.md` (C2.10 formal evidence)
  - `docs/milestones/cross-substrate-results.md` (C2.11 comparative
    H1-H4 MLX vs E-SNN)
- DualVer bump note : formal axis unchanged ; empirical axis bump
  from `C-v0.5.0+STABLE` to `C-v0.6.0+PARTIAL` reflecting two-
  substrate replication pipeline.
- Explicit **(synthetic substitute — not empirical claim)** flag
  carried on the §8.5 table caption so reviewers cannot mistake the
  cycle-2 addendum for a real-data replication.

**Trigger for v2 submission** : EITHER (a) first round of external
arXiv reviewer feedback collected into `reviewer-feedback.md`, OR
(b) cycle-3 substrate-specific predictor wiring lands and a real-
data replication becomes available — whichever first prompts a
substantive revision.

**Not** a v2 trigger : the cycle-2 synthetic-substitute dump alone
is not reason enough to bump arXiv ; it rides on the next
substantive revision.
