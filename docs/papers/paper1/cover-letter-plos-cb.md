# Cover letter — PLOS Computational Biology submission

**Manuscript** : *dreamOfkiki: A Substrate-Agnostic Formal
Framework for Dream-Based Knowledge Consolidation in Artificial
Cognitive Systems*

**Authors**: Saillant, Clément (corresponding
author: Clement Saillant, L'Electron Rare, France;
clement@saillant.cc)

**Date** : 2026-04-20
**Source** : `docs/papers/paper1/full-draft.md` v0.2 (rendered
from commit `22784f8`)

---

Dear PLOS Computational Biology Editorial Board,

We are pleased to submit our manuscript *"dreamOfkiki:
A Substrate-Agnostic Formal Framework for Dream-Based Knowledge
Consolidation in Artificial Cognitive Systems"* for your
consideration. The work introduces a fully formal,
substrate-agnostic framework for dream-based memory consolidation
in artificial cognitive systems, and reports cross-substrate
validation of that framework on two structurally divergent
implementations.

Catastrophic forgetting in artificial cognitive systems remains a
central obstacle for sequential, lifelong learning. The
neuroscience of sleep-dependent consolidation has been a fertile
inspiration for the field — Walker–Stickgold replay, Tononi's
synaptic-homeostasis hypothesis, Hobson and Solms' creative-REM
dreaming, Friston's free-energy account — but those four pillars
have never been integrated into a single composable theory. Our
contribution is exactly that integration. We axiomatise the
domain (DR-0 accountability, DR-1 episodic conservation, DR-2
compositionality, DR-3 substrate-agnosticism, DR-4 profile-chain
inclusion), prove DR-2 as a generated-semigroup theorem over four
canonical operations, and operationalise DR-3 via an executable
**Conformance Criterion** that any candidate substrate must
satisfy.

Three points distinguish this manuscript from prior continual-
learning work and align it with PLOS Computational Biology's
remit:

1. **Cross-substrate conformance, not single-implementation
   benchmarking.** §5.6 reports a conformance walkthrough on two
   substrates: an MLX gradient-based reference implementation
   (`kiki-oniric`) and a numpy-LIF thalamocortical spiking
   network (E-SNN). Both pass 9 axiom-property and
   invariant-enforcement tests each (27 assertions total ; gate
   G7 LOCKED). The framework is therefore demonstrably non-vacuous
   across formally distant substrates rather than tied to a
   single neural-network family.

2. **Pipeline-and-framework validation rather than synthetic
   p-value claims.** §7 explicitly does *not* report
   continual-learning hypothesis decisions on synthetic data.
   Instead, we validate (i) the pre-registered statistical
   pipeline under Bonferroni correction, (ii) BLOCKING-invariant
   fault injection on S1–S3, (iii) R1 reproducibility
   determinism, and (iv) cross-substrate portability numbers
   from a sibling project (Nerve-WML: 0% linear gap, 12.1%
   non-linear gap, Gate M merge retains 1.000). H1–H4 evaluation
   on real continual-learning benchmarks is the scope of the
   companion Paper 2.

3. **Open science by construction.** All code (MIT), specifications
   and proofs (CC-BY-4.0) are released openly at
   `github.com/hypneum-lab/dream-of-kiki`. Hypotheses are
   pre-registered on the Open Science Framework (DOI
   10.17605/OSF.IO/Q6JYN), the retained benchmarks ship with
   SHA-256 integrity files (contract R3), and every reported
   result resolves to a registered `run_id` deterministically
   keyed on `(c_version, profile, seed, commit_sha)` (contract
   R1). The reproducible-artifact pipeline is operational, not
   aspirational.

We believe PLOS Computational Biology is the natural venue for
this work. Its remit explicitly welcomes interdisciplinary
methodology papers at the intersection of neuroscience-inspired
modelling and computational-system design; the journal's
emphasis on reproducibility and open data aligns directly with
the framework-level discipline this manuscript embodies. Our
audience overlaps the readership that engaged with prior
sleep-consolidation modelling work in your pages, and the
substrate-agnosticism contribution speaks to the active community
working on neuromorphic implementations of biologically grounded
cognitive architectures.

The manuscript is original, has not been published elsewhere, and
is not under review at any other journal. The companion ablation
paper (Paper 2 — empirical H1–H4 evaluation on the MLX reference
substrate with mega-v2 continual-learning data and Studyforrest
fMRI alignment) is in preparation and will reference this
framework manuscript on submission.

**Suggested reviewers** (per PLOS CB submission norms — the
journal guidelines require a minimum of four suggested reviewers ;
placeholders below to fill from `ops/formal-reviewer-recruitment.md`
shortlist after COI screening against the 5-year PLOS lookback
rule and the PLOS competing-interests policy on institutional
overlap) :

- *Reviewer 1* : [PLACEHOLDER — sleep-consolidation modeller,
  e.g., Sen Cheng (Ruhr-Universität Bochum), Gordon Feld (ZI
  Mannheim), or a Walker-tradition cognitive-neuroscience lab]
- *Reviewer 2* : [PLACEHOLDER — continual-learning theorist with
  formal-method background, e.g., Samuel Gershman (Harvard CLS),
  Randy O'Reilly (UC Davis), or a predictive-coding modeller]
- *Reviewer 3* : [PLACEHOLDER — neuromorphic / spiking-network
  engineer who can adjudicate the E-SNN substrate conformance
  claim, e.g., Frédéric Alexandre (Inria Mnemosyne, Bordeaux)
  or a Loihi-2 / NORSE practitioner]
- *Reviewer 4* : [PLACEHOLDER — formal-methods / category-theory
  applied to cognitive science, to adjudicate the DR-2 monoid
  compositionality proof and the Conformance Criterion algebra,
  e.g., David Spivak (Topos Institute) or a reviewer from the
  Phillips & Wilson 2010 PLOS CB *Categorial Compositionality*
  lineage]

**Competing interests declaration** : all suggested reviewers
above are non-collaborators with Clément Saillant (no co-authored
publication in the past 5 years, per PLOS CB lookback rule) and
non-institutional (corresponding author is independent,
L'Electron Rare, France ; no formal academic affiliation).

Thank you for considering our submission. We look forward to the
review process and welcome any clarifying questions about the
formal framework, the cross-substrate conformance evidence, or
the reproducibility infrastructure.

Sincerely,

**Saillant, Clément**
*Corresponding author* : Clement Saillant
L'Electron Rare, France
clement@saillant.cc
ORCID : [0000-0002-8414-185X](https://orcid.org/0000-0002-8414-185X)
Repository : `github.com/hypneum-lab/dream-of-kiki`
Preregistration : OSF DOI 10.17605/OSF.IO/Q6JYN (primary) + 10.17605/OSF.IO/TPM5S (Amendment #1, Bonferroni family restructure, filed 2026-04-21)
