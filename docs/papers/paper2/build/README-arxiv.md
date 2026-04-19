# arXiv submission package — Paper 2

**Generated** : 2026-04-19 via pandoc 3.9.0.2 (cycle-2 C2.16
assembly)
**Source** : `docs/papers/paper2/full-draft.md`
**Output** : `build/full-draft.tex` (single-file LaTeX article
class)

## Status : synthetic-substitute methodology / replication paper

Paper 2 is a **methodology / replication paper**. Every
quantitative result in §7 is produced by a shared Python mock
predictor across two substrate registrations (MLX + E-SNN
numpy LIF skeleton). No Loihi-2 hardware or fMRI cohort data
participates in any result. See §2.3, §6.4, and §8.2 for the
full synthetic-substitute framing.

## Files in this build

| File | Purpose |
|------|---------|
| `full-draft.tex` | Main LaTeX source for arXiv upload (generated) |
| `README-arxiv.md` | This file (submission instructions) |
| `.gitignore` | Ignore LaTeX build artifacts |

## arXiv submission steps

1. **Verify LaTeX compiles** locally (optional) :
   ```bash
   cd docs/papers/paper2/build
   pdflatex full-draft.tex
   pdflatex full-draft.tex   # second pass for refs
   ```
   If you don't have LaTeX, **skip this step** — arXiv
   compiles the PDF on their server.

2. **Login to arXiv** : https://arxiv.org/submit
3. **Choose categories** :
   - Primary : `cs.LG` (Machine Learning)
   - Cross-list : `cs.AI`, `q-bio.NC` (Neurons and Cognition),
     `cs.NE` (Neural and Evolutionary Computing)
4. **Upload** :
   - `full-draft.tex` (main source)
   - `../references.bib` (bibliography — copy into build/ if
     arXiv requires single directory)
   - Figures (none yet — add before final submission)
5. **arXiv preview** : verify rendered PDF before final submit
6. **Submit** → receive arXiv ID `2604.XXXXX`

## Re-render command

If `full-draft.md` is updated, re-render :
```bash
cd docs/papers/paper2
pandoc full-draft.md -o build/full-draft.tex \
    --bibliography=references.bib --citeproc --standalone
```

## Cross-references from Paper 2 to repo artifacts

Every table and run_id in §7 round-trips to a repo artifact :

- Cross-substrate results :
  `docs/milestones/cross-substrate-results.md` (commit
  `fa0f26e`)
- Conformance matrix :
  `docs/milestones/conformance-matrix.md` (commit `fd54df7`)
- DR-3 evidence :
  `docs/proofs/dr3-substrate-evidence.md`
- G7 E-SNN conformance :
  `docs/milestones/g7-esnn-conformance.md`
- G8 P_max functional :
  `docs/milestones/g8-p-max-functional.md`
- G9 cycle-2 publication :
  `docs/milestones/g9-cycle2-publication.md`
- Framework C spec :
  `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`

## Known limitations

- Figures are absent. Add `\includegraphics{figures/X.pdf}`
  in section files + ship in `build/` for arXiv.
- BibTeX rendering uses `--citeproc` (built-in) rather than
  proper `bibtex` + `.bbl` ; for production submission consider
  `--natbib` or `--biblatex` with manual bibtex pass.
- Paper 1 cross-references use plain prose ("Paper 1 §X.Y")
  because the arXiv ID for Paper 1 is not yet locked. Once
  Paper 1 arXiv ID is assigned, replace with `\cite{}`.

## Pre-submission checklist

- [ ] All synthetic-substitute caveats explicit in §7 + §8 +
      abstract + §6.4
- [ ] OSF DOI inserted in §6.1 (currently pending OSF lock —
      inherited from Paper 1)
- [ ] References.bib resolves all `\cite{}` calls (14
      entries cycle-1 + 5 engineering cycle-2)
- [ ] Authorship byline `dreamOfkiki project contributors`
- [ ] Repo URL present
- [ ] Paper 1 cross-references updated once arXiv ID locked
- [ ] Zenodo DOI inserted for cycle-2 artifact bundle
- [ ] No AI attribution
- [ ] Figures embedded
- [ ] LaTeX compiles cleanly (if local LaTeX available)
- [ ] Pivot B contingency section reviewed if Paper 1
      acceptance delayed > 6 months (see §9.5)

## Sibling FR deposit (HAL)

A French full draft lives at
`docs/papers/paper2-fr/full-draft.md`. Target HAL deposit
after arXiv release, consistent with Paper 1 FR workflow.
