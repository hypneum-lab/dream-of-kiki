# Rebuttal Pocket — RecursiveMAS Concurrent-Work Question

**Status:** Internal pre-baked rebuttal material. Not part of submitted manuscript.
**Created:** 2026-05-10
**Trigger:** Use if a reviewer raises priority, novelty, or attribution concerns
relative to RecursiveMAS (Yang et al., arXiv:2604.25917, 2026-04-28).
**Owner:** C. Saillant (Hypneum Lab).

---

## 1. Public timeline of artefacts

| Date         | Event                                                      | Public artefact (immutable)                                                              |
|--------------|------------------------------------------------------------|------------------------------------------------------------------------------------------|
| 2026-04-20   | `GammaThetaMultiplexer` PR #2 merged into `nerve-wml/master` | Commit `77efb4d` on `hypneum-lab/nerve-wml`                                              |
| 2026-04-20   | Zenodo concept-DOI minted for `nerve-wml`                   | DOI `10.5281/zenodo.<concept>` (resolves to latest version)                              |
| 2026-04-23   | `nerve-wml v1.8.0` tagged & released to PyPI                | Git tag `v1.8.0`, PyPI release page                                                      |
| 2026-04-23   | `dreamofkiki v0.9.1` tag created                            | Git tag `v0.9.1` on `hypneum-lab/dream-of-kiki`                                          |
| **2026-04-28** | **RecursiveMAS arXiv preprint v1 submitted**                 | arXiv:2604.25917                                                                         |
| 2026-05-10   | This rebuttal pocket drafted                                | (internal)                                                                               |

**Key delta:** the `GammaThetaMultiplexer` is publicly merged, tagged, and
DOI-archived **5 to 8 days before** the RecursiveMAS preprint. There is no
plausible cycle by which either work could have copied the other.

---

## 2. Anticipated reviewer framings & responses

### R1 — "The contribution overlaps substantially with RecursiveMAS."

**Response.** The two works share the high-level intuition that *shared latent
state can replace token-level message passing between heterogeneous
components*, but diverge fundamentally on three axes that determine the
contribution's identity:

1. **Substrate.** RecursiveMAS introduces `RecursiveLink`, a learned
   two-layer linear projection with cosine-alignment loss
   $\mathcal{L}_{\text{in}} = 1 - \cos(\mathcal{R}_{\text{in}}(H), \mathrm{Emb}(y))$
   operating on transformer hidden states. Our `GammaThetaMultiplexer`
   implements information transfer via biophysically realistic
   phase-coupled oscillations, with no learned projection on the carrier.

2. **Gating mechanism.** RecursiveMAS gates information flow implicitly
   through gradient-descent-trained weights. We gate flow explicitly via
   gamma–theta phase coherence, which yields temporally-structured
   bandwidth allocation (multiplexed channels per theta cycle) that linear
   projections cannot express.

3. **Falsifiability scope.** RecursiveMAS produces engineering metrics
   (accuracy, tokens, latency). Our framework additionally yields
   neuroscientific predictions (Amedi-style dose–response curves,
   plasticity signatures, cross-frequency coupling spectra) that are
   verifiable against electrophysiological data. These predictions
   are pre-registered on OSF (`docs/osf-preregistration-draft.md`).

The convergence of two methodologically distinct approaches toward a
shared-latent-state architecture — from biophysical and pure-engineering
starting points respectively — is itself evidence for a structural property
of efficient multi-component computation.

### R2 — "RecursiveMAS should be cited as prior art; you ignored it."

**Response.** RecursiveMAS is cited in §Related Work (concurrent-work
paragraph) and tagged as `\citep{recursivemas2026}` in `references.bib`.
We explicitly note the temporal proximity ("released [date] after
nerve-wml v1.8.0") and discuss the three differentiation axes above. The
citation is also propagated to the `nerve-wml` README under "Related Work
— Concurrent / convergent" so that downstream users encounter the
positioning at the artefact level.

### R3 — "Why didn't you compare empirically against RecursiveMAS?"

**Response.** A direct empirical comparison is not methodologically
appropriate at this stage for three reasons:

1. **Different units of evaluation.** RecursiveMAS measures end-to-end
   accuracy on math/science/medical/code QA benchmarks against LLM
   baselines. Our framework measures conformance against
   electrophysiological signatures and Amedi dose-response curves on
   the bouba_sens benchmark. There is no shared evaluation surface.

2. **Different substrates.** RecursiveMAS requires a stack of pretrained
   transformers (Qwen3-1.7B, Llama-3.2-1B, Qwen2.5-Math-1.5B). Our
   substrate is a biophysical neural network simulator
   (`kiki_oniric`). A like-for-like swap of either component into the
   other's framework would constitute a separate research project.

3. **Release status of comparator.** As of submission, RecursiveMAS has
   released *demo inference code only* (see their Supported Features
   checklist, item ☑️ "Add Complete Inference Pipeline" not yet
   shipped, ☑️ "Add All Training Data" not yet shipped). A reproducible
   comparison would require the unreleased training pipeline.

We acknowledge cross-paradigm comparison as future work in §Future Work.

### R4 — "Your contribution is just RecursiveLink with neuroscience flavor text."

**Response.** This framing inverts the causal direction. The
`GammaThetaMultiplexer` was derived from first principles of
cross-frequency coupling literature (cf. references in §Background:
Lisman & Jensen 2013; Canolty & Knight 2010; Buzsáki 2006), implemented
and validated against biophysical kernels in `nerve-wml`, and merged to
public master *before* the RecursiveMAS preprint existed (see §1
timeline). Our derivation in §Methodology proceeds from neural
oscillations to a multiplexer formulation; we do not borrow the
RecursiveLink projection structure or the inner–outer training loop.
The mathematical objects are not interchangeable: RecursiveLink is a
trainable affine map; the multiplexer is a phase-coded carrier with no
learned weights on the channel itself.

### R5 — "Will you reproduce RecursiveMAS results to strengthen the comparison?"

**Response.** We commit to reproducing the Sequential-Light demo (Planner
Qwen3-1.7B + Critic Llama-3.2-1B + Solver Qwen2.5-Math-1.5B) on our
infrastructure and reporting the result in a revised version, *if* the
authors release the complete training pipeline before the camera-ready
deadline. Our hardware (MacStudio M3 Ultra 512 GB, RTX 4090 24 GB) is
sized for the released checkpoints.

---

## 3. Drop-in snippet for rebuttal letter

Use this paragraph verbatim in the response document if R1 or R2 is raised.

> We thank the reviewer for raising the relationship to RecursiveMAS
> (Yang et al., 2026), which we discuss in §Related Work as concurrent
> work. Both efforts converge on the high-level intuition that shared
> latent state outperforms token-level exchange between heterogeneous
> components, yet they differ fundamentally in (i) substrate —
> biophysical phase-coupled oscillations vs. learned linear projections
> over transformer hidden states; (ii) gating mechanism — explicit
> gamma–theta phase coherence vs. implicit gradient-descent weights;
> and (iii) falsifiability scope — neuroscientific predictions
> verifiable against electrophysiology vs. engineering benchmark
> accuracy. The `GammaThetaMultiplexer` was merged to public master on
> 2026-04-20 and tagged as `nerve-wml v1.8.0` on 2026-04-23, five days
> before the RecursiveMAS preprint, with corresponding Zenodo DOIs;
> independent derivation is therefore established. We view the
> convergence of the two approaches as mutual evidence for a structural
> property of multi-component computation rather than as competing
> claims.

---

## 4. Defensive pre-flight checklist (do before submission)

- [ ] §Related Work contains the RecursiveMAS paragraph (Variante A from
      drafting notes 2026-05-10).
- [ ] `references.bib` contains the `recursivemas2026` BibTeX entry.
- [ ] `nerve-wml/README.md` "Related Work — Concurrent / convergent"
      block is merged and tagged in a release ≥ v1.8.1.
- [ ] Zenodo DOI for `nerve-wml v1.8.0` is minted (not just the concept
      DOI) and resolves correctly.
- [ ] OSF pre-registration is timestamped before paper submission.
- [ ] This rebuttal pocket is reviewed and updated 2 weeks before the
      review-response deadline.

---

## 5. Escalation: if a reviewer asserts plagiarism

Unlikely given the public timeline, but in the worst case:

1. **Do not respond emotionally.** Cite §1 timeline with hyperlinks to
   commit `77efb4d`, PyPI release page, and Zenodo DOI.
2. **Quote the RecursiveMAS abstract back.** It does not mention
   oscillations, multiplexing, biophysics, or any concept from
   `nerve-wml`. The mathematical objects are disjoint.
3. **Offer the 12-author RecursiveMAS team a co-citation invitation.**
   If they wish to acknowledge `nerve-wml` in their v2, propose mutual
   related-work updates. This converts a perceived conflict into a
   collaborative signal that benefits both venues.
4. **If escalated to PC chairs:** provide the full git history of
   `nerve-wml/src/kiki_oniric/multiplexer.py` (or wherever the module
   lives at submission time) showing commit dates predating
   2026-04-28. Provide PyPI download statistics if available.

---

*End of rebuttal pocket. Do not include this file in the submitted
camera-ready package.*
