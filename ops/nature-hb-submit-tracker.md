# Nature Human Behaviour — Paper 1 Submission Tracker

**Status** : **PENDING manual user action** (S22+)
**Source narrative** : `docs/papers/paper1/full-draft.md`
**arXiv preprint** : see `docs/milestones/arxiv-submit-log.md`

## Portal

https://www.nature.com/nathumbehav/about/journal-policies →
Submit a manuscript

## Submission package checklist

- [ ] Cover letter (PDF, 1 page) — see template below
- [ ] Manuscript (PDF + LaTeX source from Quarto render)
- [ ] Supplementary information (proofs, full eval matrix,
      raw data Zenodo DOI)
- [ ] Figures (high-res PDF or EPS) — currently absent, must
      generate from results data + architecture diagram tool
- [ ] References.bib + bibliography style
- [ ] Author information (`dreamOfkiki project contributors`,
      Saillant C. corresponding, L'Electron Rare affiliation)
- [ ] Recommended reviewers (5 names from T-Col reviewer
      network — see `ops/formal-reviewer-recruitment.md`)
- [ ] Excluded reviewers (none expected — declare conflicts if
      relevant)
- [ ] Open access fee declaration (Nature HB hybrid model)
- [ ] Data availability statement (Zenodo DOI + GitHub URL +
      OSF DOI)
- [ ] Code availability statement (MIT, GitHub URL, frozen at
      arxiv-v0.1 tag)
- [ ] Ethics declaration (no human subjects ; Studyforrest
      derived data with PDDL license)
- [ ] Competing interests declaration

## Cover letter template (draft)

```
Dear Editor,

We submit "dreamOfkiki: A Substrate-Agnostic Formal Framework for
Dream-Based Knowledge Consolidation in Artificial Cognitive
Systems" for consideration at Nature Human Behaviour. The paper
presents the first executable formal framework for dream-based
consolidation in AI, with the following contributions :

1. Framework C with axioms DR-0..DR-4 including a Conformance
   Criterion enabling substrate-agnostic validation
2. kiki-oniric implementation on Apple Silicon MLX with three
   ablation profiles (P_min, P_equ, P_max)
3. Pre-registered hypotheses H1-H4 (OSF DOI : <pending>)
   evaluated under Welch / TOST / Jonckheere / one-sample t-test
   with Bonferroni correction
4. Open-source code (MIT, github.com/electron-rare/dream-of-kiki),
   reproducibility contract R1 (deterministic run_id), Zenodo
   artifacts (DOI : <pending>)

The paper bridges cognitive neuroscience theory (four pillars :
Walker/Stickgold, Tononi SHY, Friston FEP, Hobson/Solms) and
practical machine learning ablation, with a formal compositional
account of dream operations as a free non-commutative semigroup.
We believe Nature Human Behaviour readers will find both the
theoretical contribution (executable axioms + Conformance
Criterion) and the empirical methodology (pre-registration +
deterministic reproducibility) of broad interest.

The preprint is available on arXiv at <preprint URL>. We have not
submitted this work to any other journal and confirm that all
authors agree to the submission.

We suggest the following reviewers based on overlap with our
theoretical foundations and methodology :

1. <reviewer 1 name + affiliation> — expertise on <pillar/topic>
2. <reviewer 2 name + affiliation> — expertise on <pillar/topic>
3. <reviewer 3 name + affiliation> — expertise on <pillar/topic>
4. <reviewer 4 name + affiliation> — expertise on <pillar/topic>
5. <reviewer 5 name + affiliation> — expertise on <pillar/topic>

Yours sincerely,
Clement Saillant
on behalf of dreamOfkiki project contributors
clement@saillant.cc — L'Electron Rare, France
```

## Submission identifiers (fill after submit)

- **Manuscript ID** : TBD
- **Submission portal URL** : TBD (returned by portal)
- **Submitted on** : TBD
- **Editor handling** : TBD

## Review tracker

| Round | Date | Status | Reviewer comments | Action |
|-------|------|--------|-------------------|--------|
| Pre-submission (T-Col.4) | TBD | TODO | — | Collect via reviewer-feedback.md |
| Editor screen | TBD | TBD | — | TBD |
| Reviewer round 1 | TBD | TBD | — | TBD |
| Reviewer round 2 | TBD | TBD | — | TBD |
| Decision | TBD | TBD | — | TBD |

## Cross-references

- Cover letter draft : this file (template above)
- Source narrative : `docs/papers/paper1/full-draft.md`
- arXiv tracker : `docs/milestones/arxiv-submit-log.md`
- G5 gate report : `docs/milestones/g5-publication-ready.md`
- Pivot B contingency : `docs/proofs/pivot-b-decision.md`
- Reviewer recruitment : `ops/formal-reviewer-recruitment.md`
- T-Col outreach plan : `ops/tcol-outreach-plan.md`
