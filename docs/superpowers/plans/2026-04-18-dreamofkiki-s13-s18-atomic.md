# dreamOfkiki S13-S18 Atomic Plan

> **Pour agents autonomes :** SKILL REQUIS — utiliser `superpowers:subagent-driven-development`. Les steps utilisent la syntaxe checkbox (`- [ ]`).

**Goal** : passer de **infrastructure validée + skeletons opérationnels** à **résultats empiriques publiables**. Phase ablation complète : real benchmark mega-v2, MLX-native restructure+recombine, baseline+P_min+P_equ measurements, statistical significance, full MAJOR eval matrix, paper 1 draft results section. Convergence vers gate G5 PUBLICATION-READY.

**Architecture** : 12 tasks atomiques (S13.1-S15.3, S16.1-S17.2, S18.1-S18.3) — 8 dev tasks + 4 docs/eval tasks. Mix MLX-native ops + concurrent dream worker (skeleton) + statistical eval + paper writing skeleton + G5 gate evaluation. Total attendu : 12 commits.

**Tech Stack** : Python 3.12+ uv, **MLX** (Apple Silicon), **scipy.stats** (Welch's t-test, TOST equivalence), **mega-v2** dataset (498K examples, 25 domains — actual file path TBD per access agreement S10+), pytest + hypothesis.

**Préréquis** :
- 64 commits dreamOfkiki, dernier `a5521e3 docs(milestone): G4 P_equ functional report`
- 86 tests passing, coverage 95.38%
- Framework C-v0.5.0+STABLE
- P_min + P_equ profiles fully wired
- DR-3 Conformance Criterion conditions (1)+(2)+(3) all green
- G2 + G4 gates GO-CONDITIONAL with clear path to GO-FULL
- DR-2 proof draft circulé (G3-draft), pending external reviewer feedback (action externe)

**Deferred to S19+** :
- Paper 1 final draft + arXiv submit
- Pre-submission network reviews
- OSF DOI integration in paper bibliography
- Paper 2 outline (P_max comparative ablation)

---

## Convention commits (validator-enforced)

- Subject ≤50 chars, format `<type>(<scope>): <description>`
- Scope ≥3 chars
- Body lines ≤72 chars, 2-3 paragraphs required
- NO AI attribution
- NO `--no-verify`

---

## File structure après S13-S18

```
dreamOfkiki/
├── kiki_oniric/
│   ├── dream/operations/
│   │   ├── restructure.py            ✅ skeleton → S13.1 MLX-native
│   │   ├── recombine.py              ✅ skeleton → S13.2 MLX-native VAE
│   │   └── concurrent.py             ← S14.1 (skeleton concurrent worker)
│   ├── eval/
│   │   ├── __init__.py               ← S15.1
│   │   ├── statistics.py             ← S15.1 (Welch's t-test, TOST)
│   │   └── ablation.py               ← S15.2 (baseline+P_min+P_equ harness)
│   └── profiles/
│       └── p_max.py                  ← S16.1 skeleton
├── scripts/
│   ├── pilot_g2.py                   ✅ existing
│   ├── ablation_g4.py                ← S15.3 (real ablation runner)
│   └── megav2_loader.py              ← S13.3 (mega-v2 dataset bridge)
├── harness/benchmarks/
│   └── mega_v2/                      ← S13.3
│       └── adapter.py                (mega-v2 → retained format)
├── tests/
│   ├── unit/
│   │   ├── test_restructure_mlx.py   ← S13.1
│   │   ├── test_recombine_mlx.py     ← S13.2
│   │   ├── test_megav2_loader.py     ← S13.3
│   │   ├── test_concurrent_worker.py ← S14.1
│   │   ├── test_statistics.py        ← S15.1
│   │   ├── test_ablation_runner.py   ← S15.2
│   │   ├── test_p_max.py             ← S16.1
│   │   └── test_p_max_partial.py     ← S16.2
│   └── conformance/
│       └── (no new conformance tests this phase)
├── docs/
│   ├── milestones/
│   │   ├── g4-pequ-report.md         ✅ existing → S15.3 update GO-FULL
│   │   ├── g5-publication-ready.md   ← S18.1 (G5 gate report)
│   │   └── ablation-results.md       ← S15.3 (raw data dump)
│   └── papers/
│       └── paper1/
│           ├── outline.md            ← S17.1
│           └── results-section.md    ← S18.2 (results draft)
└── results/                          ← gitignored, big files
    └── ablation-{date}/
        ├── runs.parquet
        ├── metrics.parquet
        └── statistics.json
```

---

# Task S13.1 — Restructure MLX-native (D-Friston FEP real)

**Goal** : remplacer `restructure_handler` skeleton (counter only) par version MLX qui modifie réellement la topologie d'un model — add layer, remove layer, reroute. Skeleton tracking préservé pour tests sans MLX.

**Files:**
- Modify : `kiki_oniric/dream/operations/restructure.py` (extend with `restructure_handler_mlx`)
- Create : `tests/unit/test_restructure_mlx.py`

## Pattern (abbreviated for plan density)

- TDD: 3 tests (add layer increases module count, remove decreases, reroute changes connectivity)
- Implementation: `restructure_handler_mlx(state, model)` operates on `model.named_modules()` — for "add", appends new `nn.Linear`; for "remove", deletes named module; for "reroute", swaps two modules' positions in forward chain
- Validate post-restructure topology via `topology.validate_topology()` (S10.2 guard)
- Commit: `feat(mlx): add restructure handler MLX backend` (45 chars)

Expected: 89 tests (86+3), coverage ≥90%.

---

# Task S13.2 — Recombine MLX-native VAE

**Goal** : remplacer `recombine_handler` skeleton (linear interp) par version VAE-light MLX : encoder-decoder simple, sample latent, decode. Skeleton préservé.

**Files:**
- Modify : `kiki_oniric/dream/operations/recombine.py` (extend with `recombine_handler_mlx`)
- Create : `tests/unit/test_recombine_mlx.py`

## Pattern

- TDD: 3 tests (sampled latent has correct dim, sampling diversity over N runs > threshold, deterministic with seed)
- Implementation: `recombine_handler_mlx(state, encoder, decoder, rng)` — encode latents, sample from N(μ, σ), decode
- Commit: `feat(mlx): add recombine handler MLX VAE` (40 chars)

Expected: 92 tests (89+3), coverage ≥90%.

---

# Task S13.3 — mega-v2 dataset loader bridge

**Goal** : adapter mega-v2 format (existing 498K examples, 25 domains) vers schema RetainedBenchmark. Skeleton scaffold qui charge un sous-ensemble (500 items stratified par domain) et produit `RetainedBenchmark` compatible avec `evaluate_retained`.

**Files:**
- Create : `scripts/megav2_loader.py` (one-shot extraction)
- Create : `harness/benchmarks/mega_v2/__init__.py` (empty)
- Create : `harness/benchmarks/mega_v2/adapter.py`
- Create : `tests/unit/test_megav2_loader.py`

## Pattern

- TDD: 4 tests (loads with mock dataset, stratification 20 items/domain × 25 = 500, hash check, RetainedBenchmark roundtrip)
- Implementation: adapter reads JSONL/parquet, samples N items per domain (default 20), computes SHA-256, writes to `harness/benchmarks/mega_v2/items.jsonl` + `.sha256`
- **Caveat real data**: if mega-v2 path not accessible, use **synthetic placeholder generator** that produces 500-item stratified set in same format. Document in adapter docstring.
- Commit: `feat(bench): add mega-v2 loader (real or synth)` (47 chars)

Expected: 96 tests (92+4), coverage ≥90%.

---

# Task S14.1 — Concurrent dream worker skeleton

**Goal** : skeleton du concurrent dream worker (asyncio + threading) qui run dream-episodes en parallèle de l'awake process. Skeleton ne fait pas vraiment d'asyncio (single-thread sequential pour cycle 1) mais expose l'API qui sera étendue cycle 2.

**Files:**
- Create : `kiki_oniric/dream/operations/concurrent.py`
- Create : `tests/unit/test_concurrent_worker.py`

## Pattern

- TDD: 3 tests (worker initializes, queue accepts episodes, drain returns results)
- Implementation: `ConcurrentDreamWorker(runtime, queue_size=128)` with `submit(episode) -> Future` + `drain() -> list[EpisodeLogEntry]` ; skeleton uses sync execution but Future API allows future async swap
- Commit: `feat(dream): concurrent worker skeleton` (40 chars)

Expected: 99 tests (96+3), coverage ≥90%.

---

# Task S15.1 — Statistical eval module

**Goal** : implémenter Welch's t-test (H1) + TOST equivalence (H2) + Jonckheere-Terpstra (H3) + one-sample t-test (H4) sur `MetricResult` data. Foundation pour ablation S15.2.

**Files:**
- Create : `kiki_oniric/eval/__init__.py` (empty)
- Create : `kiki_oniric/eval/statistics.py`
- Create : `tests/unit/test_statistics.py`

## Pattern

- TDD: 5 tests (Welch's t with synthetic data, TOST passes equivalence, TOST rejects far values, Jonckheere monotonic, one-sample t against budget threshold)
- Implementation: 4 functions wrapping `scipy.stats.ttest_ind(equal_var=False)`, custom TOST implementation, `scipy.stats.kendalltau` Jonckheere variant, `scipy.stats.ttest_1samp`
- Commit: `feat(eval): add stats tests for H1-H4` (39 chars)

Expected: 104 tests (99+5), coverage ≥90%.

---

# Task S15.2 — Ablation runner harness

**Goal** : harness qui exécute `(profile, seed, dataset)` matrix et collecte `MetricResult` rows. Foundation pour S15.3 real ablation.

**Files:**
- Create : `kiki_oniric/eval/ablation.py`
- Create : `tests/unit/test_ablation_runner.py`

## Pattern

- TDD: 3 tests (single-cell run produces MetricResult, multi-cell sweep covers grid, results dataframe schema)
- Implementation: `AblationRunner(profiles, seeds, benchmark)` with `.run() -> pd.DataFrame`. Each cell creates fresh profile, runs N dream-episodes, evaluates retained, records metrics
- Commit: `feat(eval): add ablation runner harness` (39 chars)

Expected: 107 tests (104+3), coverage ≥90%.

---

# Task S15.3 — G4 ablation real run + report update

**Goal** : run baseline + P_min + P_equ on mega-v2 (or synthetic-placeholder per S13.3 fallback). Update G4 report.

**Files:**
- Create : `scripts/ablation_g4.py`
- Create : `docs/milestones/ablation-results.md`
- Modify : `docs/milestones/g4-pequ-report.md`

## Pattern

- Script invokes `AblationRunner` on mega-v2 (or synthetic), 3 seeds × 3 profiles
- **Register batch run_id** : call
  `RunRegistry.register(c_version, profile="G4_ablation",
  seed=min(seeds), commit_sha=...)` from
  `harness.storage.run_registry` at the start of `main()` ; embed
  the returned `run_id` in every output row and in the JSON dump
  alongside `harness_version` + `is_synthetic` flag (S15.2 also
  registers a batch id inside `AblationRunner.run()` so direct
  callers get the same R1 contract)
- Propagate `seed` into `evaluate_retained(..., seed=s)` so the
  trace ledger matches the registered batch
- Computes Welch's t-test on M1.b (P_equ vs P_min), TOST on M3.c, Jonckheere on M2.b across profiles
- If P_equ > P_min on ≥2 metrics with p<0.05 → flip G4 to GO-FULL
- Commit: `docs(milestone): G4 ablation results + GO` (42 chars)

Expected: tests count unchanged (script-level run, no test additions), coverage ≥90%.

---

# Task S16.1 — P_max profile skeleton

**Goal** : skeleton P_max profile pour cycle 1 documentation. Mirror P_equ structure mais ajoute `recombine_full_handler` placeholder + `attention_prior` channel registration.

**Files:**
- Create : `kiki_oniric/profiles/p_max.py`
- Create : `tests/unit/test_p_max.py`

## Pattern

- TDD: 3 tests (instantiates, status="skeleton", unimplemented_ops=["recombine_full"])
- Implementation: `PMaxProfile` similar to `PEquProfile` but with `unimplemented_ops` flag
- Commit: `feat(profile): add P_max skeleton` (33 chars)

Expected: 110 tests (107+3), coverage ≥90%.

---

# Task S16.2 — DR-4 P_max axiom test (partial)

**Goal** : étendre `test_dr4_profile_inclusion.py` avec test `ops(P_equ) ⊆ ops(P_max)` même si P_max est skeleton (les ops sont déclarées dans `unimplemented_ops` ou dérivées de la structure du framework).

**Files:**
- Modify : `tests/conformance/axioms/test_dr4_profile_inclusion.py` (add P_max tests)

## Pattern

- 2 tests (P_equ ⊆ P_max ops via declared signatures, channels chain extends)
- Skeleton P_max declares its **target** ops/channels in metadata fields
- Commit: `test(dr4): extend P_equ ⊆ P_max chain check` (43 chars)

Expected: 112 tests (110+2), coverage ≥90%.

---

# Task S17.1 — Paper 1 outline draft

**Goal** : `papers/paper1/outline.md` avec structure complete : Abstract, Introduction, Background (4 piliers A/B/D/C), Framework (Conformance Criterion, axioms DR-0..DR-4 with proof refs), Implementation (kiki-oniric), Methodology (eval matrix, OSF pre-reg H1-H4), Results (ablation), Discussion, Future Work (E SNN cycle 2), References.

**Files:**
- Create : `docs/papers/paper1/__init__.md` (index)
- Create : `docs/papers/paper1/outline.md`

## Pattern

- Outline conforme à Nature Human Behaviour template (8-10 pages main + 30-50 supp)
- Cross-references vers all docs/proofs/* and docs/milestones/*
- Commit: `docs(paper1): add outline draft` (32 chars)

---

# Task S17.2 — Paper 1 abstract + intro draft

**Goal** : draft Abstract (250 words) + Introduction (~1.5 pages markdown).

**Files:**
- Create : `docs/papers/paper1/abstract.md`
- Create : `docs/papers/paper1/introduction.md`

## Pattern

- Abstract: contribution claim + result + significance, OSF pre-reg DOI placeholder
- Intro: position relative to prior art (van de Ven, Kirkpatrick EWC, Tononi SHY, Friston FEP), motivation, contribution roadmap
- Commit: `docs(paper1): add abstract + intro` (33 chars)

---

# Task S18.1 — G5 PUBLICATION-READY gate report

**Goal** : G5 gate evaluation. All criteria from framework spec §9 (publication_ready_gate). Skeleton report comparing actual state vs criteria.

**Files:**
- Create : `docs/milestones/g5-publication-ready.md`

## Pattern

- Status of each PUBLICATION-READY criterion (coverage, seeds, retained_regression, zero_blocking_days, dualver_status, pre_submission_reviews, axioms_proven, ablation_complete, paper_draft)
- Default at S18: PARTIAL — most criteria met, paper draft incomplete + DR-2 review pending
- Decision tree: GO-FULL S20 (after Paper 1 final), DEFER (extend cycle 1 timeline), Pivot B (retrograde to PLoS CB)
- Commit: `docs(milestone): G5 publication-ready gate` (42 chars)

---

# Task S18.2 — Paper 1 results section draft

**Goal** : draft Results section using ablation data from S15.3.

**Files:**
- Create : `docs/papers/paper1/results-section.md`

## Pattern

- 3 subsections: H1 forgetting (P_equ vs baseline t-test), H3 RSA monotonic, H4 energy ratio
- Tables with synthetic data placeholder (real data filled when ablation runs)
- Reference to OSF pre-reg + raw data Zenodo DOI placeholder
- Commit: `docs(paper1): add results section draft` (38 chars)

---

# Task S18.3 — Pivot B decision tree document

**Goal** : explicit Pivot B decision tree if G5 fails. Branches : EXTEND-CYCLE-1 (move S20+ submission later), DOWNGRADE-JOURNAL (Nature HB → PLoS CB), SCOPE-DOWN (Pivot A single-paper TMLR).

**Files:**
- Create : `docs/proofs/pivot-b-decision.md`

## Pattern

- 3 branches with conditions, actions, journal targets, framework version implications
- Cross-references to G5 report + master spec §7.3
- Commit: `docs(proof): add Pivot B decision tree` (38 chars)

---

# Self-review

**1. Spec coverage** :
- S13 MLX-native ops + mega-v2 → S13.1 + S13.2 + S13.3 ✅
- S14 concurrent worker → S14.1 ✅
- S15 statistical eval + ablation runner + G4 update → S15.1 + S15.2 + S15.3 ✅
- S16 P_max skeleton + DR-4 extension → S16.1 + S16.2 ✅
- S17 paper 1 outline + abstract+intro → S17.1 + S17.2 ✅
- S18 G5 gate + results draft + Pivot B → S18.1 + S18.2 + S18.3 ✅

**2. Placeholder scan** : Pattern abrégé délibéré pour S13.1+ tasks. Chaque task a son contrat (TDD pattern, files, expected test delta, commit subject pré-validé). Le subagent peut implémenter en suivant le pattern des phases précédentes.

**3. Type consistency** :
- `restructure_handler_mlx`, `recombine_handler_mlx` (S13.1, S13.2) → MLX backends, skeleton variants kept
- `mega_v2.adapter` produit `RetainedBenchmark` (S13.3) → consommé par `evaluate_retained`
- `ConcurrentDreamWorker` (S14.1) → API future-compatible, skeleton sync
- `eval.statistics` 4 functions H1-H4 (S15.1) → consommé par `ablation_runner` (S15.2) → consommé par `ablation_g4.py` script (S15.3)
- `PMaxProfile` (S16.1) → mirror P_equ avec unimplemented flag
- DR-4 axiom test étendu (S16.2) → réutilise pattern test_dr4_profile_inclusion.py

**4. Commit count** : 12 commits.

**5. Validator risks** : tous subjects pré-vérifiés ≤50 chars. Risque MLX : tests skip si MLX indisponible (déjà pattern `pytest.importorskip` éprouvé).

**6. Caveat majeur S13.3** : real mega-v2 access depends on user actions (pre-existing dataset path). Plan documents synthetic-placeholder fallback explicit.

---

**End of S13-S18 atomic plan.**

**Version** : v0.1.0
**Generated** : 2026-04-18 via refinement of S13-S18 from main plan
**Source** : `docs/superpowers/plans/2026-04-17-dreamofkiki-implementation.md`
