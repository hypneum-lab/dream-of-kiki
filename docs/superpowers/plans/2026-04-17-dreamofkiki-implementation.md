# dreamOfkiki Implementation Plan

> **Pour agents autonomes :** SKILL REQUIS — utiliser `superpowers:subagent-driven-development` (recommandé) ou `superpowers:executing-plans` pour implémenter tâche par tâche. Les steps utilisent la syntaxe checkbox (`- [ ]`) pour le tracking.

**Goal** : exécuter le programme de recherche dreamOfkiki cycle 1 — 2 papers scientifiques (Paper 1 framework formel → Nature HB, Paper 2 ablation empirique → NeurIPS/ICML/TMLR) sur 28 semaines (S1-S28), basé sur les specs validés `docs/specs/2026-04-17-dreamofkiki-*.md`.

**Architecture** : 4 tracks coordonnés (C framework, A kiki-oniric implementation, T-Ops infrastructure, T-Col external collaboration) orchestrés par meta-layer human-in-the-loop. Phases S1-S4 décomposées en tâches exécutables atomiques (TDD, commits fréquents) ; phases S5-S28 organisées en milestones + decision trees (car dépendent de résultats empiriques).

**Tech Stack** : Python 3.14 + uv (package manager), MLX (Apple Silicon inference/training), Hypothesis (property tests), pytest (unit + integration), DuckDB (metrics storage), Plotly (dashboard), Quarto (paper drafts), OSF + Zenodo (open science), GitHub Actions (CI).

**Compute topology** : Studio M3 Ultra 512GB (awake + P_min dream), KXKM-AI RTX 4090 24GB (P_equ/P_max dream), GrosMac M5 16GB (P_max recombination VAE).

---

## Notes de lecture

Ce plan est **dual-register** :
- **Phase 1 (S1-S4)** : tâches atomiques exécutables (2-5 min chacune) — directement actionnables par sub-agent ou inline
- **Phase 2 (S5-S28)** : milestones stratégiques avec decision trees — nécessite arbitrage humain aux gates G1-G6

Les checkboxes sur les tâches Phase 1 sont à cocher au fil de l'exécution. Les milestones Phase 2 sont à **planifier finement** à l'approche de leur début (ex: plan détaillé S5-S8 rédigé pendant S3-S4 avec les résultats S1-S2 intégrés).

---

## Structure du repo final

```
dreamOfkiki/
├── README.md                           ✅ (créé S1.0)
├── CONTRIBUTORS.md                     ✅ (créé S1.0)
├── .gitignore                          ✅ (créé S1.0)
├── pyproject.toml                      (S1.1)
├── uv.lock                             (S1.1, généré)
├── .github/workflows/                  (S1.5)
│   ├── ci.yml
│   └── repro-smoke.yml
├── docs/
│   ├── specs/                          ✅ (master + framework C)
│   ├── plans/                          (ce fichier)
│   ├── invariants/                     (S1.3)
│   │   └── registry.md
│   ├── glossary.md                     (S1.3)
│   ├── interfaces/                     (S2-S6)
│   │   ├── primitives.md
│   │   ├── eval-matrix.yaml
│   │   ├── experiment-contract.md
│   │   ├── proposal-template.md
│   │   └── fmri-schema.yaml
│   ├── feasibility/                    (S2)
│   │   └── studyforrest-rsa-note.md
│   └── proofs/                         (S5-S8)
│       └── dr2-compositionality.md
├── harness/                            (S1.2, S3-S6)
│   ├── __init__.py
│   ├── metrics/
│   │   ├── m1_forgetting.py
│   │   ├── m2_representational.py
│   │   ├── m3_efficiency.py
│   │   └── m4_emergence.py
│   ├── dispatcher/
│   │   ├── stratified_matrix.py
│   │   ├── run_builder.py
│   │   └── scheduler.py
│   ├── storage/
│   │   ├── run_registry.py
│   │   └── parquet_writer.py
│   ├── benchmarks/
│   │   ├── continual_learning/
│   │   ├── retained/
│   │   └── rsa_fmri/
│   ├── scorers/
│   │   └── teacher.sha256              (Qwen3.5-9B Q4_K_M hash)
│   └── cli/
│       └── dream_harness.py
├── kiki-oniric/                        (S1.6-S2, fork de kiki-flow-core)
│   ├── core/
│   │   ├── primitives.py               (Story 0 expose-primitives)
│   │   ├── episodic_buffer.py          (β)
│   │   ├── hierarchical_latents.py     (δ)
│   │   ├── raw_traces.py               (α, P_max only)
│   │   └── weights_snapshot.py         (γ)
│   ├── dream/
│   │   ├── runtime.py                  (S5-S7)
│   │   ├── swap_protocol.py            (S7)
│   │   ├── operations/
│   │   │   ├── replay.py
│   │   │   ├── downscale.py
│   │   │   ├── restructure.py
│   │   │   └── recombine.py
│   │   ├── episode.py                  (DE 5-tuple)
│   │   └── guards/
│   │       ├── s1_retained.py
│   │       ├── s2_finite.py
│   │       ├── s3_topology.py
│   │       └── s4_attention.py
│   └── profiles/
│       ├── p_min.py                    (S5-S6)
│       ├── p_equ.py                    (S7-S9)
│       └── p_max.py                    (S10-S12)
├── tests/
│   ├── conformance/                    (S5)
│   │   ├── axioms/
│   │   │   ├── test_dr0_accountability.py
│   │   │   ├── test_dr1_episodic_conservation.py
│   │   │   ├── test_dr2_compositionality.py
│   │   │   ├── test_dr3_substrate.py
│   │   │   └── test_dr4_profile_inclusion.py
│   │   └── invariants/
│   │       ├── test_i1_i2_i3.py
│   │       ├── test_s1_s2_s3_s4.py
│   │       └── test_k1_k3_k4.py
│   ├── unit/
│   ├── integration/
│   └── e2e/
├── papers/
│   ├── paper1-framework/                (S12-S20)
│   │   ├── draft/
│   │   ├── figures/
│   │   ├── formal-proofs/
│   │   └── submitted/                   (S20)
│   └── paper2-ablation/                 (S16-S24)
│       ├── draft/
│       ├── figures/
│       └── submitted/                   (post-Paper1-accept, hors cycle 1)
└── ops/                                 (T-Ops operational)
    ├── sync-pack-template.md
    ├── dream-sync-monday.md
    └── dashboard/
        ├── build.py
        └── templates/
```

---

# Phase 1 — Setup & Fondation (S1-S4, tâches atomiques)

## S1 — Bootstrap (semaine 1)

### Task S1.0 : Repo créé (déjà fait)

**Status** : ✅ **COMPLETED** avant ce plan.

**Files créés** :
- `README.md`, `CONTRIBUTORS.md`, `.gitignore`
- `docs/specs/2026-04-17-dreamofkiki-master-design.md`
- `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`

Commits : `01e40d4` (initial), `0f7f8ad` (DR-3 fix).

---

### Task S1.1 : Python project skeleton (uv)

**Files** :
- Create : `pyproject.toml`
- Create : `uv.lock` (généré)
- Create : `harness/__init__.py`

- [ ] **Step 1 : Créer pyproject.toml minimal**

```toml
[project]
name = "dreamofkiki"
version = "0.1.0"
description = "Substrate-agnostic formal framework for dream-based knowledge consolidation"
readme = "README.md"
requires-python = ">=3.12"
license = { text = "MIT" }
authors = [
    { name = "Clement Saillant", email = "clement@saillant.cc" }
]
dependencies = [
    "mlx>=0.18.0",
    "numpy>=2.0",
    "hypothesis>=6.100",
    "pytest>=8.0",
    "pytest-cov>=5.0",
    "duckdb>=1.0",
    "pyarrow>=17.0",
    "pandas>=2.2",
    "plotly>=5.20",
    "pyyaml>=6.0",
    "click>=8.1",
]

[project.optional-dependencies]
dev = [
    "ruff>=0.5",
    "mypy>=1.10",
    "pre-commit>=3.7",
]
fmri = [
    "nilearn>=0.10",
    "nibabel>=5.2",
    "scikit-learn>=1.5",
]
teacher = [
    "llama-cpp-python>=0.2.80",
]

[project.scripts]
dream-harness = "harness.cli.dream_harness:cli"

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.ruff]
line-length = 100
target-version = "py312"

[tool.mypy]
strict = true
python_version = "3.12"

[tool.pytest.ini_options]
testpaths = ["tests"]
addopts = "-v --cov=harness --cov=kiki_oniric --cov-report=term-missing --cov-fail-under=90"
```

- [ ] **Step 2 : Créer harness/__init__.py**

```python
"""dreamOfkiki shared evaluation harness."""

__version__ = "0.1.0"
```

- [ ] **Step 3 : Lancer uv sync**

Run : `cd /Users/electron/Documents/Projets/dreamOfkiki && uv sync`
Expected : création de `.venv/` et `uv.lock`, 0 errors.

- [ ] **Step 4 : Vérifier install**

Run : `uv run python -c "import harness; print(harness.__version__)"`
Expected : `0.1.0`

- [ ] **Step 5 : Commit**

```bash
git add pyproject.toml uv.lock harness/__init__.py
git commit -m "chore(setup): init Python project with uv"
```

---

### Task S1.2 : Harness core skeleton avec test failing

**Files** :
- Create : `harness/storage/run_registry.py`
- Create : `tests/unit/test_run_registry.py`

- [ ] **Step 1 : Écrire test failing**

Create `tests/unit/test_run_registry.py` :

```python
"""Unit tests for run registry (SQLite-backed)."""
from datetime import datetime
from pathlib import Path

import pytest

from harness.storage.run_registry import RunRegistry


@pytest.fixture
def tmp_registry(tmp_path: Path) -> RunRegistry:
    db_path = tmp_path / "runs.db"
    return RunRegistry(db_path=db_path)


def test_register_run_creates_entry(tmp_registry: RunRegistry) -> None:
    run_id = tmp_registry.register(
        c_version="C-v0.1.0+STABLE",
        profile="P_min",
        seed=42,
        commit_sha="abc123",
    )
    assert run_id is not None
    assert tmp_registry.get(run_id)["profile"] == "P_min"


def test_register_run_is_idempotent_for_same_inputs(tmp_registry: RunRegistry) -> None:
    args = dict(c_version="C-v0.1.0+STABLE", profile="P_equ", seed=1, commit_sha="def")
    run_id_1 = tmp_registry.register(**args)
    run_id_2 = tmp_registry.register(**args)
    assert run_id_1 == run_id_2  # Deterministic run_id for repro contract R1
```

- [ ] **Step 2 : Lancer test pour vérifier qu'il échoue**

Run : `uv run pytest tests/unit/test_run_registry.py -v`
Expected : FAIL avec `ModuleNotFoundError: harness.storage.run_registry`.

- [ ] **Step 3 : Implémenter RunRegistry minimal**

Create `harness/storage/__init__.py` (empty).
Create `harness/storage/run_registry.py` :

```python
"""Run registry — SQLite-backed, reproducibility contract R1."""
import hashlib
import sqlite3
from pathlib import Path
from typing import Any


class RunRegistry:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self._ensure_schema()

    def _ensure_schema(self) -> None:
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS runs (
                    run_id TEXT PRIMARY KEY,
                    c_version TEXT NOT NULL,
                    profile TEXT NOT NULL,
                    seed INTEGER NOT NULL,
                    commit_sha TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT (datetime('now'))
                )
            """)

    def _compute_run_id(self, c_version: str, profile: str, seed: int, commit_sha: str) -> str:
        key = f"{c_version}|{profile}|{seed}|{commit_sha}".encode()
        return hashlib.sha256(key).hexdigest()[:16]

    def register(self, c_version: str, profile: str, seed: int, commit_sha: str) -> str:
        run_id = self._compute_run_id(c_version, profile, seed, commit_sha)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT OR IGNORE INTO runs (run_id, c_version, profile, seed, commit_sha) "
                "VALUES (?, ?, ?, ?, ?)",
                (run_id, c_version, profile, seed, commit_sha),
            )
        return run_id

    def get(self, run_id: str) -> dict[str, Any]:
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            row = conn.execute("SELECT * FROM runs WHERE run_id = ?", (run_id,)).fetchone()
            if row is None:
                raise KeyError(f"run_id not found: {run_id}")
            return dict(row)
```

- [ ] **Step 4 : Lancer test, vérifier pass**

Run : `uv run pytest tests/unit/test_run_registry.py -v`
Expected : 2 passed.

- [ ] **Step 5 : Commit**

```bash
git add harness/storage/ tests/unit/test_run_registry.py
git commit -m "feat(harness): add RunRegistry with deterministic run_id"
```

---

### Task S1.3 : Invariants registry + glossary initiaux

**Files** :
- Create : `docs/invariants/registry.md`
- Create : `docs/glossary.md`

- [ ] **Step 1 : Écrire registry.md**

Create `docs/invariants/registry.md` avec copie des invariants depuis `framework-C-design.md` §5 + §6.2 axiomes DR + enforcement matrix. Structure :

```markdown
# Invariants & Axioms Registry

**Authoritative source** for invariants I/S/K and axioms DR-0..DR-4.
Any code referencing an invariant MUST cite its code and version here.

## Family I (Information)
- **I1** Episodic conservation until consolidation — BLOCKING
- **I2** Hierarchy traceability — BLOCKING
- **I3** Latent distributional drift bounded — WARN

## Family S (Safety)
- **S1** Retained task non-regression — BLOCKING
- **S2** No NaN/Inf in W_scratch — BLOCKING
- **S3** Hierarchy guard — BLOCKING
- **S4** Attention prior bounded — WARN

## Family K (Compute)
- **K1** Dream-episode budget respected — BLOCKING (per DE)
- **K3** Swap latency bounded — WARN
- **K4** Eval matrix coverage — BLOCKING (for tagging)

## Axioms DR-0..DR-4
- **DR-0** Accountability
- **DR-1** Episodic conservation
- **DR-2** Compositionality (or fallback DR-2' canonical order)
- **DR-3** Substrate-agnosticism (with Conformance Criterion)
- **DR-4** Profile chain inclusion

Full definitions in `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §5 et §6.
```

- [ ] **Step 2 : Écrire glossary.md**

Create `docs/glossary.md` :

```markdown
# Canonical Glossary (authoritative)

Any term appearing in specs, code, papers MUST match this glossary.

## Core entities
- **dreamOfkiki** — program name, logical identifier (camelCase for branding, prose, figures)
- **dream-of-kiki** — filesystem-safe technical name (kebab-case for repo paths)
- **kiki-oniric** — dedicated fork of kiki-flow-core for Track A experiments
- **kiki-flow-core** — source repo (not touched directly by dreamOfkiki — refer to jalonné rebase)

## Processes
- **awake process** — kiki real-time inference/training running on Studio (MLX)
- **dream process** — asynchronous offline consolidation running per profile topology
- **DE (dream-episode)** — atomic dream unit : 5-tuple `(trigger, input_slice, operation_set, output_delta, budget)`

## Data flows
- **α (alpha)** — raw traces firehose (P_max only)
- **β (beta)** — curated episodic buffer (all profiles)
- **γ (gamma)** — weights-only snapshot (fallback)
- **δ (delta)** — hierarchical latent snapshots (P_equ, P_max)
- **Canal 1** — weight_delta (dream → awake)
- **Canal 2** — latent_samples (dream → awake)
- **Canal 3** — hierarchy_chg (dream → awake)
- **Canal 4** — attention_prior (dream → awake)

## State copies (swap worktree)
- **W_awake** — active weights, read+write by awake
- **W_dream** — frozen snapshot, read-only by dream
- **W_scratch** — working copy modified by dream, becomes W_awake at swap

## Profiles
- **P_min** — β → 1 (minimal publishable)
- **P_equ** — β+δ → 1+3+4 (balanced, canonical)
- **P_max** — α+β+δ → 1+2+3+4 (maximalist)

## Versioning
- **DualVer** — `C-v<FC-MAJOR>.<FC-MINOR>.<FC-PATCH>+<EC-STATE>`
- **FC** — formal consistency (SemVer-like)
- **EC** — empirical consistency ∈ {STABLE, DIRTY, INVALIDATED}

## Gates
- **G1** S2 — T-Col fallback locked
- **G2** S8 — P_min viable
- **G3** S8 — DR-2 preuve peer-reviewed (+ G3-draft S6 circulé)
- **G4** S12 — P_equ fonctionnel
- **G5** S18 — PUBLICATION-READY
- **G6** S28 — amorce cycle 2 décision

## Maturity modes
- **RED** — ≥1 BLOCKING violé
- **GREEN** — BLOCKING respectés
- **PUBLICATION-READY** — GREEN + critères supplémentaires §9 framework
```

- [ ] **Step 3 : Vérifier cohérence avec specs**

Run : `grep -c "P_min\|P_equ\|P_max" docs/glossary.md docs/specs/*.md`
Expected : counts non-zéro, terminologie présente.

- [ ] **Step 4 : Commit**

```bash
git add docs/invariants/ docs/glossary.md
git commit -m "docs: add invariants registry and canonical glossary"
```

---

### Task S1.4 : T-Col outreach kickoff (livrable non-code)

**Files** :
- Create : `ops/tcol-outreach-plan.md`
- Create : `ops/proposal-template-draft.md`

- [ ] **Step 1 : Créer plan outreach**

Create `ops/tcol-outreach-plan.md` :

```markdown
# T-Col Outreach Plan (S1-S8 front-loaded)

## Cibles fMRI labs (M2.b RSA)

| Priorité | Lab | Contact | Data type | Status |
|----------|-----|---------|-----------|--------|
| 1 | Huth Lab (UT Austin) | alex@cs.utexas.edu | Narratives natural language | TODO S3 |
| 2 | Norman Lab (Princeton) | knorman@princeton.edu | Episodic memory | TODO S3 |
| 3 | Gallant Lab (UC Berkeley) | gallant@berkeley.edu | Natural stimuli | TODO S4 |

Fallback : **Studyforrest** (public, already available) — feasibility note S2 mandatory.

## Cibles relecteur formel (DR-2 preuve)

| Priorité | Candidat | Profil | Contact approach |
|----------|----------|--------|-------------------|
| 1 | Chercheur TCS familier monoïdes/catégories | Formel rigoureux | Via réseau académique perso |
| 2 | Cognitive modeling researcher | Interdisciplinaire | Via T-Col.2 outreach natural |
| 3 | Sub-agent critic + validator (fallback) | Automated | En backup si humain indisponible |

Cible : recruter S3-S5, relecture DR-2 draft S6-S8.

## Cible Intel NRC (cycle 2 only, non-blocking cycle 1)

- Outreach S9+ pour Loihi-2 access
- Proposal package basé sur Paper 1 arXiv (S18)
- Delai institutionnel typique 3-6 mois, aligné cycle 2 2027+
```

- [ ] **Step 2 : Draft template de proposal**

Create `ops/proposal-template-draft.md` :

```markdown
# Proposal Template (draft S1, finalize S3)

## Title
dreamOfkiki: A Substrate-Agnostic Formal Framework for Dream-Based Knowledge Consolidation — RSA fMRI Collaboration Request

## Research question (1 paragraph)
[À finaliser S3 avec lab-specific framing]

## Proposed collaboration
- **We provide** : kiki-oniric embeddings + stimuli mapping, formal framework C, open science infrastructure (OSF pre-reg, Zenodo DOIs)
- **We request** : fMRI data (task-based linguistic stimuli OR narrative comprehension dataset) compatible with RSA
- **Deliverable** : co-authorship Paper 1 (Nature HB / PLoS CB) as courtesy affiliation, full credit in CONTRIBUTORS.md

## Timeline
- S4 : proposal sent
- S5-S8 : negotiation + MoU
- S9-S14 : data analysis, RSA computation
- S15-S18 : pre-submission review involvement
- S20 : paper submitted

## Prior art
dreamOfkiki builds on Walker/Stickgold consolidation, Tononi SHY, Friston FEP, Hobson/Solms creative dreaming. Unique contribution : formal framework with executable axioms.

## Contact
Clement Saillant — L'Electron Rare — clement@saillant.cc
```

- [ ] **Step 3 : Commit**

```bash
git add ops/tcol-outreach-plan.md ops/proposal-template-draft.md
git commit -m "docs(tcol): add outreach plan and proposal template"
```

---

### Task S1.5 : CI GitHub Actions basique

**Files** :
- Create : `.github/workflows/ci.yml`

- [ ] **Step 1 : Créer workflow CI**

Create `.github/workflows/ci.yml` :

```yaml
name: CI

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4

      - name: Install uv
        uses: astral-sh/setup-uv@v3

      - name: Set up Python
        run: uv python install 3.12

      - name: Install dependencies
        run: uv sync --all-extras

      - name: Run lint
        run: uv run ruff check .

      - name: Run type check
        run: uv run mypy harness tests

      - name: Run tests with coverage
        run: uv run pytest --cov-fail-under=90

      - name: Invariant smoke tests (placeholder)
        run: echo "S5+: conformance/invariant tests will run here"
```

- [ ] **Step 2 : Commit**

```bash
git add .github/workflows/ci.yml
git commit -m "ci: add basic GitHub Actions workflow"
```

---

### Task S1.6 : Décision fork kiki-oniric (non-code, arbitrage)

**Files** :
- Create : `docs/fork-decision.md`

- [ ] **Step 1 : Documenter décision fork**

Create `docs/fork-decision.md` :

```markdown
# kiki-oniric fork decision

## Context
Master spec §3.1 et Q2.3 (c) : fork dédiée `kiki-oniric` de `kiki-flow-core`, isolation totale des 3 stories actives dans kiki-flow-research.

## Rebase policy (r3, jalonné)
- S1 : fork initial depuis kiki-flow-core `main` HEAD
- S8 : rebase mid-program (capture améliorations upstream majeures)
- S18 : rebase pre-paper final

## Source repo
- Parent : `~/Documents/Projets/kiki-flow-research/` (branche main)
- Fork target : `~/Documents/Projets/dreamOfkiki/kiki-oniric/` (nested dans dreamOfkiki repo, pas submodule — simplicité)

## Action S1.7
Cloner kiki-flow-core dans `kiki-oniric/` comme working copy, puis enregistrer le SHA source pour trace.
```

- [ ] **Step 2 : Commit**

```bash
git add docs/fork-decision.md
git commit -m "docs(fork): document kiki-oniric fork and rebase policy"
```

---

### Task S1.7 : Créer le dossier kiki-oniric (placeholder pour fork)

**Files** :
- Create : `kiki-oniric/README.md`
- Create : `kiki-oniric/.gitkeep`

- [ ] **Step 1 : Créer structure placeholder**

Create `kiki-oniric/README.md` :

```markdown
# kiki-oniric

Dedicated fork of `kiki-flow-core` for the dreamOfkiki research program (Track A).

**Status** : Skeleton S1. Fork content will be cloned S2 (Story 0 expose-primitives).

**Source** : `~/Documents/Projets/kiki-flow-research/` (kiki-flow-core parent), main HEAD at time of fork.

**Rebase policy** (r3 jalonné) : S1 fork, S8 mid-rebase, S18 pre-paper rebase.

**DO NOT** modify files in this directory directly — changes go through Track A planned tasks (Story 0 for expose-primitives, then P_min/P_equ/P_max implementations).
```

Create `kiki-oniric/.gitkeep` (empty file to keep directory tracked).

- [ ] **Step 2 : Commit**

```bash
git add kiki-oniric/
git commit -m "chore(kiki): add placeholder structure for kiki-oniric fork"
```

---

**S1 end-of-week review** :
- [ ] Repo bootstrapped : pyproject.toml, CI, specs + invariants registry + glossary
- [ ] RunRegistry minimal testé (TDD baseline)
- [ ] T-Col outreach plan drafted
- [ ] Fork decision documented
- [ ] Sync pack S1 rédigé : résumé semaine, invariants touchés (aucun encore), bloquants

---

## S2 — Story 0 + Feasibility + Gates préparation

### Task S2.1 : Écrire Studyforrest RSA feasibility note (critic #2, G1 S2)

**Files** :
- Create : `docs/feasibility/studyforrest-rsa-note.md`

**Livrable obligatoire** (décision critic issue #2). Sans go/no-go sur Studyforrest, M2.b est non-protégé.

- [ ] **Step 1 : Recherche Studyforrest dataset**

Run : `open https://www.studyforrest.org/`
Noter :
- ROIs disponibles (STG, IFG, AG nécessaires)
- Type de stimuli (narrative film — Forrest Gump)
- Format data (nifti, BIDS-compatible)
- Licence (ODC Open Data Commons, permissive)

- [ ] **Step 2 : Rédiger note de feasibility**

Create `docs/feasibility/studyforrest-rsa-note.md` :

```markdown
# Studyforrest RSA Feasibility for kiki-oniric M2.b

**Date** : 2026-04-XX (S2)
**Author** : Clement Saillant
**Status** : Decision required before S4 (framework §11 `fmri-schema.yaml` lock)

## Question

Can Studyforrest fMRI dataset support RSA alignment against kiki-oniric linguistic embeddings (M2.b metric)?

## Check list

1. **Linguistic ROIs coverage** : Studyforrest provides ROIs in STG, IFG, AG?
   - [ ] Verify via https://www.studyforrest.org/dataoverview.html
   - Expected : STG yes (auditory), IFG partial, AG partial

2. **Stimulus-embedding mappability** : Forrest Gump narrative stimuli projectable to kiki ortho species (ρ_phono, ρ_lex, ρ_syntax, ρ_sem)?
   - [ ] Identify temporal stimulus segmentation available
   - [ ] Check if transcripts are time-aligned at word or sentence level
   - Expected : word-level timing available via STOP (Studyforrest Transcript)

3. **RDM dimensionality** : enough stimuli for robust RDM estimation (≥50 conditions)?
   - [ ] Count distinct stimulus conditions in Studyforrest
   - Expected : ~1000+ word occurrences, sufficient

4. **Compute feasibility** : local Studio M3 Ultra can process in reasonable time?
   - [ ] Estimate compute via nilearn on 100-subj fMRI + RSA
   - Expected : feasible on Studio, 1-2 days per profile

## Decision tree

### Branch A — GO-STUDYFORREST (default if all 4 checks pass)
- Adopt Studyforrest as M2.b data source
- Lock `fmri-schema.yaml` S4
- Proceed with H3 hypothesis pre-registration unchanged

### Branch B — PIVOT-SYNTHETIC (if ROIs/stimuli inadequate)
- Create synthetic perceptual benchmark as proxy RSA
- Use controlled linguistic stimuli (existing datasets: EEG linguistic corpora)
- Adapt H3 hypothesis : "representational alignment with synthetic benchmark"
- Paper 1 framing shift : less "cognitive alignment" more "representational hierarchy"

### Branch C — DOWNGRADE-M2.b (if compute/data infeasible entirely)
- Move M2.b to Paper 1 Future Work section
- Remove M2.b from PUBLICATION-READY gate criteria (§9 framework spec amendment)
- Paper 1 focus shifts to formal framework + ablation on other 7 metrics
- Requires updating OSF pre-registration hypotheses : drop H3

## Action taken
- [ ] Decision recorded : [A / B / C]
- [ ] Commit decision + rationale
- [ ] Update `docs/interfaces/fmri-schema.yaml` (S4) accordingly
- [ ] Update `ops/tcol-outreach-plan.md` priority list
```

- [ ] **Step 3 : Exécuter les 4 checks**

Pour chaque check, documenter résultat dans la note (remplir les `[ ]` au fur et à mesure). Cette étape nécessite exploration web + possiblement download d'un sample Studyforrest.

- [ ] **Step 4 : Prendre la décision**

Remplir la section "Action taken" avec A, B, ou C.

- [ ] **Step 5 : Commit**

```bash
git add docs/feasibility/studyforrest-rsa-note.md
git commit -m "docs(feasibility): lock Studyforrest RSA decision (G1)"
```

**Milestone G1 locked**.

---

### Task S2.2 : Story 0 — Expose primitives typed Protocols

**Files** :
- Create : `kiki-oniric/core/primitives.py`
- Create : `kiki-oniric/core/__init__.py`
- Create : `tests/conformance/axioms/test_dr3_substrate.py`
- Create : `tests/conformance/__init__.py`, `tests/conformance/axioms/__init__.py`

**Livrable critic issue #3 partiel** : fournir les typed Protocols pour Conformance Criterion condition (1).

- [ ] **Step 1 : Écrire test DR-3 signature typing**

Create `tests/conformance/__init__.py` (empty), `tests/conformance/axioms/__init__.py` (empty).

Create `tests/conformance/axioms/test_dr3_substrate.py` :

```python
"""DR-3 Substrate-agnosticism — Conformance Criterion condition (1) signature typing."""
from typing import runtime_checkable

import pytest

from kiki_oniric.core.primitives import (
    AlphaStreamProtocol,
    BetaBufferProtocol,
    DeltaLatentsProtocol,
    GammaSnapshotProtocol,
    WeightDeltaChannel,
    LatentSampleChannel,
    HierarchyChangeChannel,
    AttentionPriorChannel,
)


def test_alpha_protocol_is_runtime_checkable() -> None:
    assert runtime_checkable(AlphaStreamProtocol) is AlphaStreamProtocol


def test_beta_protocol_is_runtime_checkable() -> None:
    assert runtime_checkable(BetaBufferProtocol) is BetaBufferProtocol


def test_all_protocols_declared() -> None:
    protocols = [
        AlphaStreamProtocol,
        BetaBufferProtocol,
        DeltaLatentsProtocol,
        GammaSnapshotProtocol,
        WeightDeltaChannel,
        LatentSampleChannel,
        HierarchyChangeChannel,
        AttentionPriorChannel,
    ]
    assert len(protocols) == 8  # 4 awake→dream + 4 dream→awake
    assert all(hasattr(p, "__protocol_attrs__") or hasattr(p, "_is_protocol") for p in protocols)
```

- [ ] **Step 2 : Lancer test, vérifier fail**

Run : `uv run pytest tests/conformance/axioms/test_dr3_substrate.py -v`
Expected : FAIL avec `ModuleNotFoundError: kiki_oniric.core.primitives`.

- [ ] **Step 3 : Implémenter primitives.py**

Create `kiki-oniric/__init__.py` (empty).
Create `kiki-oniric/core/__init__.py` (empty).

NOTE : Python package name uses underscores. Create a symlink or rename: the directory `kiki-oniric` has a dash. To import as `kiki_oniric`, create a namespace shim. Simpler solution : rename directory to `kiki_oniric`.

Run : `mv kiki-oniric kiki_oniric && git mv` workflow doesn't apply since we just created. Re-do:

```bash
cd /Users/electron/Documents/Projets/dreamOfkiki
git rm -r kiki-oniric  # placeholder from S1.7
mkdir -p kiki_oniric/core
```

Create `kiki_oniric/__init__.py` (empty).
Create `kiki_oniric/core/__init__.py` (empty).

Create `kiki_oniric/core/primitives.py` :

```python
"""Typed Protocol signatures for the 8 primitives of framework C.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §2.1
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path
from typing import Iterator, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray


# =================== Awake → Dream ===================

@runtime_checkable
class AlphaStreamProtocol(Protocol):
    """α — Raw traces firehose (P_max only)."""

    def append_trace(
        self,
        tokens: NDArray[np.int32],
        activations: NDArray[np.float32],
        attention_patterns: NDArray[np.float32],
        prediction_errors: NDArray[np.float32],
    ) -> None: ...

    def iter_traces(self) -> Iterator[dict[str, NDArray]]: ...


@runtime_checkable
class BetaBufferProtocol(Protocol):
    """β — Curated episodic buffer (all profiles)."""

    def append_record(
        self,
        context: str,
        outcome: str,
        saillance_score: float,
        timestamp: datetime,
    ) -> int: ...

    def fetch_unconsumed(self, limit: int) -> list[dict]: ...

    def mark_consumed(self, record_ids: list[int], de_id: str) -> None: ...


@runtime_checkable
class GammaSnapshotProtocol(Protocol):
    """γ — Weights-only snapshot (fallback/diagnostic)."""

    def get_checkpoint_path(self) -> Path: ...

    def get_checkpoint_sha256(self) -> str: ...


@runtime_checkable
class DeltaLatentsProtocol(Protocol):
    """δ — Hierarchical latent snapshots (P_equ, P_max)."""

    def snapshot(
        self,
        species_activations: dict[str, NDArray[np.float32]],
    ) -> int: ...

    def get_recent(self, n: int) -> list[dict[str, NDArray[np.float32]]]: ...


# =================== Dream → Awake ===================

@runtime_checkable
class WeightDeltaChannel(Protocol):
    """Canal 1 — Weight delta (via swap)."""

    def apply(
        self,
        lora_delta: dict[str, NDArray[np.float32]],
        fisher_bump: dict[str, NDArray[np.float32]] | None = None,
    ) -> None: ...


@runtime_checkable
class LatentSampleChannel(Protocol):
    """Canal 2 — Latent samples queue (consumed by awake data augmenter)."""

    def enqueue(self, species: str, latent_vector: NDArray[np.float32], provenance: str) -> None: ...

    def dequeue(self) -> dict | None: ...


@runtime_checkable
class HierarchyChangeChannel(Protocol):
    """Canal 3 — Topology diff (applied atomically at swap time)."""

    def apply_diff(self, diff: list[tuple[str, dict]]) -> None: ...


@runtime_checkable
class AttentionPriorChannel(Protocol):
    """Canal 4 — Attention prior tensor (copy at swap or live read-only)."""

    def set_prior(self, prior: NDArray[np.float32]) -> None: ...

    def get_prior(self) -> NDArray[np.float32] | None: ...
```

- [ ] **Step 4 : Lancer test, vérifier pass**

Run : `uv run pytest tests/conformance/axioms/test_dr3_substrate.py -v`
Expected : 3 passed.

- [ ] **Step 5 : Commit**

```bash
git add kiki_oniric/ tests/conformance/
git commit -m "feat(kiki): Story 0 expose typed Protocol primitives"
```

**Milestone DR-3 Conformance condition (1) fulfilled**.

---

### Task S2.3 : T-Col send outreach emails (non-code)

**Files** : aucun (action humaine)

- [ ] **Step 1 : Envoyer emails proposal**

3 emails (Huth, Norman, Gallant) basés sur `ops/proposal-template-draft.md` finalisé. Tracker dans `ops/tcol-outreach-plan.md` status column.

- [ ] **Step 2 : Commit update de l'outreach log**

```bash
git add ops/tcol-outreach-plan.md
git commit -m "docs(tcol): log outreach emails sent"
```

---

**S2 end-of-week review + Sync pack** :
- [ ] G1 locked (Studyforrest decision)
- [ ] Story 0 expose-primitives typed Protocols (DR-3 condition 1)
- [ ] T-Col emails sent to 3 labs
- [ ] No BLOCKING violations
- [ ] Sync pack S2 rédigé

---

## S3-S4 — Formalization C + T-Col negotiation

Phase de rédaction framework v0.3 → v0.5 + T-Col responses + formal reviewer recruitment (Q_CR.1).

### Task S3.1 : Formaliser primitives + invariants dans docs/interfaces/primitives.md

**Files** :
- Create : `docs/interfaces/primitives.md`
- Create : `docs/interfaces/eval-matrix.yaml`

- [ ] **Step 1 : Écrire primitives.md**

Copier + adapter depuis `framework-C-design.md` §2.1 en ajoutant :
- Mapping Protocol → classe concrète (sera remplie en S5)
- Contract tests planned locations
- Version DualVer du contrat

(Structure détaillée : ~400 lignes de contrat formel.)

- [ ] **Step 2 : Écrire eval-matrix.yaml (format machine-readable)**

Create `docs/interfaces/eval-matrix.yaml` :

```yaml
# Eval matrix stratification rules (C-v0.3+STABLE)
version: "C-v0.3.0+STABLE"

bump_rules:
  PATCH:
    obligatory: [touched_axis_metric × P_equ × 1_seed]
    optional: [others_skipped]

  MINOR:
    obligatory: [all_8_metrics × P_equ × 3_seeds]
    optional: [P_min_P_max_deferred_to_next_MAJOR]

  MAJOR:
    obligatory: [all_8_metrics × all_3_profiles × 3_seeds]  # 72 runs
    optional: [sensitivity_sweeps]

  EC_change:
    obligatory: [published_metrics_only]
    optional: [others_skipped]

publication_ready_gate:
  requires:
    - coverage: 100% stratified cells in last MAJOR bump
    - seeds: ≥3 per (profile, metric)
    - retained_regression: ≤1% on last 10 swaps
    - zero_blocking_days: ≥7 consecutive
    - dualver_status: "+STABLE"
    - pre_submission_reviews: ≥1 positive
    - axioms_proven: [DR-0, DR-1, DR-2, DR-3, DR-4]  # or DR-2' accepted
    - ablation_complete: [P_min, P_equ, P_max]
    - paper_draft: complete

metrics:
  M1.a: { name: forgetting_rate, family: continual_learning, det_strategy: seeded_rng }
  M1.b: { name: avg_accuracy, family: continual_learning, det_strategy: seeded_rng }
  M2.b: { name: rsa_fmri_alignment, family: representational, det_strategy: cpu_nilearn_seeded }
  M3.a: { name: flops_ratio, family: efficiency, det_strategy: mlx_static_profile }
  M3.b: { name: offline_gain, family: efficiency, det_strategy: flops_equiv_wallclock, pivot: true }
  M3.c: { name: energy_per_episode, family: efficiency, det_strategy: calibrated_proxy }
  M4.a: { name: recomb_quality, family: emergence, det_strategy: frozen_scorer_qwen3_5_9b_q4km }
  M4.b: { name: structure_discovery, family: emergence, det_strategy: permutation_test_seeded }
```

- [ ] **Step 3 : Commit**

```bash
git add docs/interfaces/
git commit -m "docs(interfaces): lock primitives.md and eval-matrix.yaml"
```

---

### Task S3.2 : OSF pre-registration H1-H4

**Files** :
- Create : `docs/osf-preregistration-draft.md`

- [ ] **Step 1 : Rédiger pré-enregistrement H1-H4**

Copier hypothèses depuis master spec §5.4 + affiner avec statistical tests prévus. Structure OSF standard : study design, hypotheses, pre-specified analyses, power calculations, data exclusion rules.

- [ ] **Step 2 : Upload vers OSF.io**

Action manuelle via https://osf.io : créer projet "dreamOfkiki", upload pré-enregistrement, lock date.

- [ ] **Step 3 : Récupérer OSF DOI et commit**

```bash
git add docs/osf-preregistration-draft.md
git commit -m "docs(osf): lock H1-H4 pre-registration [DOI: 10.17605/OSF.IO/XXXXX]"
```

---

### Task S3.3 : Recruter relecteur formel DR-2 (Q_CR.1 b)

**Files** :
- Create : `ops/formal-reviewer-recruitment.md`

- [ ] **Step 1 : Identifier 3 candidats**

Chercheurs TCS / monoids / categories theory / cognitive modeling formels dans réseau perso. Rédiger email template de demande (offrir co-auteur courtoisie sur Paper 1 pour 1-2 heures de relecture DR-2 draft S6).

- [ ] **Step 2 : Envoyer demandes S3-S5**

- [ ] **Step 3 : Track status dans ops/**

Similar pattern à tcol-outreach-plan.md.

- [ ] **Step 4 : Commit**

```bash
git add ops/formal-reviewer-recruitment.md
git commit -m "docs(tcol): recruit formal reviewer for DR-2 proof (Q_CR.1)"
```

---

### Task S3.4 : Retained benchmark 500 items gelé

**Files** :
- Create : `harness/benchmarks/retained/items.jsonl`
- Create : `harness/benchmarks/retained/retained.py`
- Create : `tests/unit/test_retained_benchmark.py`

Pattern TDD standard. (Détails omis pour concision — même forme que S1.2.)

Sélection des 500 items : subset stratifié de mega-v2 (498K examples) couvrant les 25 domaines. Gel par SHA-256 hash du fichier JSONL.

---

### Task S4.1 : fMRI schema lock + interface contract

**Files** :
- Create : `docs/interfaces/fmri-schema.yaml`

Dépend de la décision S2 (Branch A/B/C). Structure dépend de la décision :
- Branch A : schema Studyforrest-compatible
- Branch B : schema synthetic benchmark
- Branch C : schema absent (M2.b downgraded), placeholder note

---

### Task S4.2 : C v0.5 draft circulé

**Files** :
- Modify : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` → bump DualVer en header
- Create : `docs/drafts/C-v0.5-changelog.md`

Contenu : revisions framework basées sur findings S1-S4. Commit avec bump DualVer explicite :

```bash
git commit -m "refactor(spec): bump framework to C-v0.5.0+STABLE"
```

---

**S4 end-of-week + G1 retrospective** :
- [ ] Interface `primitives.md`, `eval-matrix.yaml`, `fmri-schema.yaml` locked
- [ ] OSF pre-registration live
- [ ] Formal reviewer recruited (ou fallback critic-only documenté)
- [ ] Retained benchmark frozen
- [ ] C-v0.5.0+STABLE tagged

---

# Phase 2 — Milestones stratégiques (S5-S28)

À partir d'ici, le plan devient **moins granulaire** car les tâches dépendent des résultats de Phase 1. Chaque milestone est décrit en termes de **livrables attendus + decision tree**. Les tâches atomiques seront raffinées au début de chaque semaine correspondante par un sub-agent planner dédié.

## S5 — Dream runtime skeleton + property tests DR-0/DR-1

**Objectif** : implémenter le scaffolding du dream runtime et commencer les property tests des axiomes (Conformance Criterion condition 2).

**Livrables** :
- `kiki_oniric/dream/runtime.py` — boucle principale dream process
- `kiki_oniric/dream/episode.py` — classe `DreamEpisode` 5-tuple
- `tests/conformance/axioms/test_dr0_accountability.py` — property tests
- `tests/conformance/axioms/test_dr1_episodic_conservation.py` — property tests

**TDD pattern** à suivre pour chaque test file : write failing test (Hypothesis-based) → implement → pass → commit.

**Commit convention** : `feat(dream): …` ou `test(conformance): …`.

---

## S6 — DR-2 compositionality draft + P_min implementation start + G3-draft

**Objectif** : livrer la **première version circulée** de la preuve DR-2 (G3-draft, critic issue #1). Commencer P_min.

**Livrables** :
- `docs/proofs/dr2-compositionality.md` — draft complet avec proof by cases
- Circulation au relecteur formel externe (recruté S3-S5)
- `kiki_oniric/profiles/p_min.py` — skeleton
- `kiki_oniric/dream/operations/replay.py` — première opération implémentée
- `kiki_oniric/dream/operations/downscale.py` — seconde opération

**Decision point S6** :
- **Si DR-2 proof sketch converge** → poursuivre vers G3 final S8
- **Si difficultés majeures** → amorcer fallback DR-2' dès S7, documenter decision dans `docs/proofs/dr2-decision-log.md`

---

## S7 — P_min fonctionnel + swap protocol v1

**Objectif** : premier runtime complet P_min avec swap worktree fonctionnel.

**Livrables** :
- `kiki_oniric/dream/swap_protocol.py` — 8-step swap avec guards S1-S4
- `kiki_oniric/dream/guards/s1_retained.py`, `s2_finite.py`, `s3_topology.py`, `s4_attention.py`
- Tests intégration end-to-end P_min
- Première mesure M1.a, M3.a, M3.b sur P_min vs baseline

---

## S8 — G2 + G3 gates

**G2 — P_min viable** :
- Critère : accuracy P_min ≥ baseline − 2% sur retained benchmark, runtime stable 48h
- **Si GO** → proceed S9-S12 P_equ
- **Si NO-GO** → Pivot A (single-paper, framework simplifié, cible TMLR)

**G3 — DR-2 preuve peer-reviewed** :
- Critère : preuve DR-2 validée par relecteur externe (ou fallback DR-2' adopté avec rationale)
- **Si GO** → proceed vers Paper 1 rigueur maximale
- **Si NO-GO strict DR-2** → adopter DR-2', Paper 1 semi-formel (pivot B lite)
- **Si NO-GO même DR-2'** → Pivot B full : Paper 1 rétrogradé Cognitive Systems Research

**Decision tree documenté** : `docs/proofs/g3-decision-log.md` rédigé à S8, décision tracée.

---

## S9-S12 — P_equ implementation + G4 gate

**Objectif** : ablation P_equ fonctionnelle avec hierarchy restructure (D-Friston) et attention prior.

**Livrables** :
- `kiki_oniric/profiles/p_equ.py`
- `kiki_oniric/dream/operations/restructure.py`
- `kiki_oniric/dream/operations/recombine.py` (version light pour P_equ)
- Cross-machine sync Studio↔KXKM-AI fonctionnel (NFS + sync batch)
- Full E3+E4 metric suite running sur P_equ

**G4 — P_equ fonctionnel S12** :
- Critère : P_equ > P_min sur ≥2 métriques avec significance statistique, invariants all green 7 jours consécutifs
- **Si GO** → proceed S13-S18 P_max + ablation complète
- **Si NO-GO** → Skip P_max, publier uniquement P_min vs P_equ (Paper 2 simplifié)

---

## S13-S17 — P_max + full ablation

**Objectif** : ablation complète 3 profils × 8 métriques × 3 seeds + pipeline 3-machines.

**Livrables** :
- `kiki_oniric/profiles/p_max.py`
- `kiki_oniric/dream/operations/recombine.py` version full (VAE/diffusion)
- GrosMac M5 integré comme recombination worker
- Full E3+E4 results sur les 3 profils
- Dashboard `dream.saillant.cc` live avec ablation results
- Statistical significance tests par paire de profils

---

## S18 — G5 PUBLICATION-READY gate + Paper 1 arXiv

**G5 check automatisé** via T-Ops CI :
- [ ] Coverage eval matrix 100% (last MAJOR bump)
- [ ] 3 seeds minimum
- [ ] Retained non-regression ≤1% sur last 10 swaps
- [ ] 7 jours consécutifs zero BLOCKING
- [ ] DualVer status `+STABLE`
- [ ] ≥1 pre-submission review positive
- [ ] DR-0..DR-4 formalized + DR-2 (ou DR-2') proven/adopted
- [ ] Ablation complete 3 profils
- [ ] Paper 1 draft complet, no TODO

**Si tous cochés** → Paper 1 arXiv S18, triggering pre-submission network T-Col.4 final review.

---

## S19-S20 — Paper 1 revisions + submit Nature HB

**Objectif** : intégrer feedback T-Col.4, soumettre Paper 1.

---

## S21-S24 — Paper 2 draft + buffer

**Objectif** : Paper 2 draft complet (gelé en attente acceptation Paper 1 per S1-séquentielle).

**Livrables** :
- `papers/paper2-ablation/draft/paper.md` complet
- Figures finalisées
- Code artifact release : Zenodo DOI
- HuggingFace models release : `clemsail/kiki-oniric-P_{min,equ,max}`

---

## S25-S28 — Buffer + G6 cycle-2 decision

**Objectif** : maintenance pipeline, response aux early reviews Paper 1 si rapide, decision G6.

**G6 — Amorcer cycle 2 ?** :
- Si bandwidth disponible + Paper 1 submitted cleanly → démarrer T-Col.5 Intel NRC outreach pour E-SNN cycle 2
- Si bandwidth consumé → cycle 2 repoussé, focus L'Electron Rare business

---

# Annexe A — Convention de commits

Format conventional commits avec scope :
- `feat(<scope>): …` — nouvelle fonctionnalité
- `fix(<scope>): …` — correction bug
- `refactor(<scope>): …` — refactoring sans changement de comportement
- `test(<scope>): …` — ajout/modification tests
- `docs(<scope>): …` — documentation
- `ci: …` — CI/CD
- `chore(<scope>): …` — tooling, setup

Scopes communs : `setup`, `spec`, `harness`, `kiki`, `dream`, `tcol`, `ops`, `fork`, `osf`, `feasibility`, `interfaces`.

Règles validator :
- Subject ≤50 chars
- Body lines ≤72 chars
- Pas d'attribution AI

---

# Annexe B — Dream-sync Monday template

Chaque lundi à 9h, chaque track owner rédige un sync pack dans `ops/sync-packs/<YYYY-MM-DD>-<track>.md` :

```markdown
# Sync pack <track> <YYYY-MM-DD>

## Diffs cette semaine
- ...

## Invariants respectés / violés / à discuter
- I1: ok | S2: pending | K1: ❌ (détails)

## Bloquants
- ...

## Décision demandée
- ...

## Plan semaine prochaine
- ...
```

T-Ops cron dimanche soir envoie rappel ; lundi 9h agrégateur compile les 4 tracks + vue programme health.

---

# Annexe C — Mapping critic findings → tâches du plan

| Critic finding | Tâche du plan | Semaine |
|----------------|---------------|---------|
| Issue #1 (DR-2 proof load + G3-draft) | S6 — DR-2 draft circulation + S8 — G3 final | S6, S8 |
| Issue #2 (Studyforrest feasibility) | Task S2.1 — feasibility note avec decision tree A/B/C | S2 |
| Issue #3 (DR-3 conformance) | ✅ Déjà fixé dans framework spec §6.2 | Fait |
| Q_CR.1 (reviewer externe) | Task S3.3 — recrutement S3-S5 | S3-S5 |
| Q_CR.2 (Studyforrest check) | Inclus Task S2.1 | S2 |
| Q_CR.3 (pivot B) | Decision tree G3 `docs/proofs/g3-decision-log.md` | S8 |

---

# Self-review checklist

- [x] Spec coverage : chaque section des 2 specs a une tâche correspondante dans le plan
- [x] Pas de placeholders TBD/TODO non-intentionnels (seulement `- [ ]` checkboxes qui sont syntaxe standard)
- [x] Type consistency : nommages Python alignés (`kiki_oniric`, pas `kiki-oniric` pour package)
- [x] Critic findings intégrés (Annexe C)
- [x] Commit convention respectée (Annexe A avec rules validator)
- [x] Dual-register documenté (Phase 1 atomique, Phase 2 stratégique)

---

**End of implementation plan.**

**Version** : v0.1.0-draft
**Generated** : 2026-04-17 (brainstorming session) via superpowers:writing-plans skill
**Source specs** :
- `docs/specs/2026-04-17-dreamofkiki-master-design.md`
- `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` (with DR-3 fix 0f7f8ad)
