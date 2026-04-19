# Architecture — dreamOfkiki program and genial-lab repositories

*Last updated : 2026-04-19 (Paper 1 v0.2 release)*

This document maps how the four public repositories of the
**genial-lab** research organization compose into the **dreamOfkiki**
research program. It is intended as a reader's guide for :

- reviewers trying to verify what in Paper 1 is claimed vs. what is
  backed by code in which repository
- external collaborators who want to contribute to one specific
  substrate or component
- future maintainers who need to understand the dependency graph

## TL;DR

- `dream-of-kiki` is the **flagship** : it hosts the formal framework
  (axioms DR-0..DR-4, Conformance Criterion, primitives, profiles),
  the MLX `kiki-oniric` substrate (Track A), the E-SNN thalamocortical
  substrate (cycle-2 instantiation, already G7 LOCKED), and the two
  papers of cycle 1.
- `kiki-flow-research` is the **upstream numerical engine** : it
  provides the Wasserstein-gradient-flow solver, the species ontology
  (ρ_phono, ρ_lex, ρ_syntax, ρ_sem), and divergence estimators that
  the `dream-of-kiki` substrates consume.
- `nerve-wml` is the **reference cross-substrate measurement** : it
  hosts two concrete substrates (MlpWML dense, LifWML
  surrogate-gradient SNN) sharing a single `Nerve` Protocol, and the
  Gate W / Gate M measurements cited in Paper 1 §7.4 as independent
  evidence of the substrate-agnosticism principle.
- `micro-kiki` is a **deployable MoE-LoRA instance** : 35
  domain-expert LoRA adapters on a Qwen3.5 MoE base, with inference
  optimised for consumer GPUs. It is the "product-facing" artefact
  that consumes (will consume) the consolidated profiles produced by
  `dream-of-kiki`.

## Dependency graph

```
                  ┌──────────────────────────────┐
                  │  kiki-flow-research          │   upstream
                  │  Wasserstein solver          │   numerical
                  │  Species ontology            │   engine
                  │  Divergence estimators       │
                  └───────────────┬──────────────┘
                                  │
                                  │ numerical primitives
                                  ▼
┌────────────────────────────────────────────────────────────────────┐
│  dream-of-kiki (flagship)                                          │
│  ───────────────────────────                                       │
│  • Formal framework : axioms DR-0..DR-4, Conformance Criterion     │
│  • Substrates :                                                    │
│    — MLX kiki-oniric (Track A)                                     │
│    — E-SNN thalamocortical (G7 LOCKED)                             │
│  • Profiles : P_min ⊆ P_equ ⊆ P_max                                │
│  • Harness : run_registry (R1), invariant guards (S1/S2/S3)        │
│  • Papers : Paper 1 (PLOS CB target), Paper 2 (NeurIPS cycle 2)    │
└──────────┬─────────────────────────────────┬───────────────────────┘
           │                                 │
           │ Protocol-pattern inherited      │ cites §7.4
           │ cross-substrate portability     │
           ▼                                 ▼
┌────────────────────────┐      ┌────────────────────────────────────┐
│  nerve-wml             │      │  Paper 1 §7.4                      │
│  ──────────────────    │      │  (cross-substrate portability)     │
│  • Nerve Protocol      │      │                                    │
│  • MlpWML substrate    │◄──── │  cites MlpWML/LifWML polymorphism  │
│  • LifWML substrate    │      │  (0 % / 12.1 % gap), Gate M merge  │
│  • Gate W / Gate M     │      │  retains 1.000 of mock baseline.   │
│    measurements        │      │                                    │
└────────────────────────┘      └────────────────────────────────────┘

                                  │
                                  │ (future) consolidated profiles
                                  ▼
                   ┌──────────────────────────────┐
                   │  micro-kiki                  │   deployable
                   │  35 domain-expert MoE-LoRA   │   product
                   │  Qwen3.5 base, Q4_K_M infer  │   artefact
                   └──────────────────────────────┘
```

## Shared contracts

The genial-lab repositories align on a small number of cross-cutting
contracts. They are re-implemented per repo (by design, to keep each
repo self-contained) rather than imported from a shared Python
package.

### R1 — deterministic `run_id`

Every run is keyed by `(c_version, profile, seed, commit_sha,
benchmark_version)` and hashed to a 32-character SHA-256 prefix. The
same tuple always produces the same `run_id` ; a mismatch between
`run_id` expectation and outcome is a reproducibility regression.

Implemented in :

- `dream-of-kiki/harness/storage/run_registry.py`
- `kiki-flow-research/scripts/cl_llm_bench/harness/run_registry.py`
- `nerve-wml/harness/run_registry.py` (reduced form)

### R3 — artifact addressability by SHA-256

All benchmarks and registered artifacts carry a paired `.sha256`
integrity file. Mismatch raises `RetainedIntegrityError` and blocks
downstream evaluation.

Implemented in `dream-of-kiki/harness/benchmarks/` and in
`kiki-flow-research/scripts/cl_llm_bench/harness/`.

### DualVer versioning

Repositories that participate in the framework tag releases with the
`C-v<FC>+<EC>` DualVer scheme, where `FC` is the formal-consistency
axis (SemVer-like) and `EC` is the empirical-consistency axis
(STABLE / DIRTY / INVALIDATED). See
`dream-of-kiki/docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
§12 for the authoritative specification.

### Authorship and licensing

- Code : **MIT** (all four repos)
- Documentation and papers : **CC-BY-4.0**
- No AI co-authorship trailers
- Byline per repository : `<repo>-project contributors`, with
  corresponding-author credit to Clément Saillant
- Reviewer credits per paper in §Acknowledgments (not byline)

See `CONTRIBUTORS.md` in each repository for the detailed policy.

## What Paper 1 claims vs. where it lives

| Paper 1 claim | Backed by code in |
|---------------|-------------------|
| Axioms DR-0..DR-4 proved (DR-2 as generated-semigroup theorem) | `dream-of-kiki/docs/proofs/dr2-compositionality.md`, `dr4-profile-inclusion.md` |
| Conformance Criterion satisfied on MLX substrate | `dream-of-kiki/kiki_oniric/substrates/mlx_kiki_oniric.py` + `tests/conformance/axioms/` |
| Conformance Criterion satisfied on E-SNN substrate (G7 LOCKED) | `dream-of-kiki/kiki_oniric/substrates/esnn_thalamocortical.py` + `tests/conformance/axioms/test_dr3_esnn_substrate.py` |
| Cross-substrate portability (0 % / 12.1 % gap, Gate M merge = 1.000) | `nerve-wml/scripts/run_w2_hard.py`, `scripts/merge_pilot.py` ; full numbers in `nerve-wml/papers/paper1/main.tex` |
| Pre-registered hypotheses H1-H4 + Bonferroni correction | OSF [osf.io/q6jyn](https://osf.io/q6jyn), DOI 10.17605/OSF.IO/Q6JYN |
| Invariant guards S1/S2/S3 enforce atomic swap | `dream-of-kiki/kiki_oniric/dream/guards/` + `tests/invariants/` |
| R1 run-registry determinism | `dream-of-kiki/harness/storage/run_registry.py` |
| Wasserstein-flow numerical foundations | `kiki-flow-research/` (dependency of the MLX substrate) |
| Deployable MoE-LoRA artefact consuming the framework | `micro-kiki/` (future integration — cycle 2) |

## Integration status — honest assessment

- **`dream-of-kiki` ↔ `kiki-flow-research`** : shared ontology and
  numerical primitives, referenced implicitly by the MLX substrate.
  No unified pip package at cycle 1 ; each repo is
  installable standalone.
- **`dream-of-kiki` ↔ `nerve-wml`** : byline-aligned ("dreamOfkiki
  project contributors" vs "nerve-wml project contributors"),
  protocol-pattern shared (typed Protocol interfaces for
  substrate-swap), no shared Python import. Paper 1 §7.4 cites the
  nerve-wml measurements as independent corroboration.
- **`dream-of-kiki` → `micro-kiki`** : the consumer relationship is
  *declared* (micro-kiki targets the cognitive layer of
  dreamOfkiki) but *not yet wired* — the shared consolidation
  protocol is specified in the framework spec, implementation lands
  in cycle 2.
- **`kiki-flow-research` ↔ `nerve-wml`, `micro-kiki`** : siblings
  under the genial-lab org, minimal direct integration.

## Roadmap

### Cycle 1 (current, through S28)

- Paper 1 v0.2 → arXiv → PLOS Comp Bio submission
- Paper 2 draft (NeurIPS target)
- G7 LOCKED → cross-substrate walkthrough in Paper 1 §5.6

### Cycle 2 (post-S28)

- Unify the R1 run_registry across the four repos into a shared
  `genial-lab-harness` package
- Wire `micro-kiki` to consume consolidated profiles from the
  framework (close the currently-declared-not-wired link)
- Third conformant substrate (transformer-based or
  neuromorphic-hardware-backed) to strengthen DR-3 beyond the
  two-substrate cycle-1 evidence
- DR-2 freeness / universal-property proof (the sub-claim explicitly
  left open in Paper 1)

## References

- Framework spec :
  [`docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`](specs/2026-04-17-dreamofkiki-framework-C-design.md)
- Paper 1 draft :
  [`docs/papers/paper1/full-draft.md`](papers/paper1/full-draft.md)
- Paper 1 §5.6 cross-substrate conformance walkthrough
- Paper 1 §7.4 cross-substrate portability (cites nerve-wml)
- Paper 1 §8.4 prior art comparison (11 rows)
- OSF pre-registration : [osf.io/q6jyn](https://osf.io/q6jyn)
