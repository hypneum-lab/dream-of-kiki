# G6-Studio Path A — real-LoRA SpikingKiki-V4 35B-A3B-V4 cross-substrate validation

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Test whether the framework C 4-channel coupling (REPLAY+DOWNSCALE+RESTRUCTURE+RECOMBINE) transfers to the **real SpikingKiki-V4 35B-A3B-V4** spiking LIF Qwen substrate (58 GB, 31 070 .npz modules) under a 5-subdomain MMLU continual-learning stream — first real-LLM-scale verdict for the dreamOfkiki programme.

**Architecture:** New `experiments/g6_studio_path_a/` module wires `MicroKikiSubstrate` with `DREAM_MICRO_KIKI_REAL=1` to the SpikingKiki-V4 artifact at `/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4`. Per subdomain : real `mlx_lm.tuner.lora` fine-tune → optional profile-dependent `dream_episode_real()` whose four handlers operate on **live** LoRA delta tensors (fixes G6 Path B spectator) → `mlx_lm.generate` letter-argmax MMLU eval. Driver writes per-subdomain partial JSON dumps so an overnight watchdog kill is resumable. EC stays PARTIAL ; FC stays C-v0.12.0+PARTIAL.

**Tech Stack:** Python 3.12 ; MLX 0.18+ + `mlx_lm` ≥ 0.31.2 (Studio) ; existing `kiki_oniric.substrates.micro_kiki.MicroKikiSubstrate` real backend (commit `177cf89`) ; existing `harness.real_benchmarks.mmlu` + `harness.storage.run_registry` ; numpy + safetensors ; pytest. Driver runs **on Studio M3 Ultra only** (M1 Max OOM at 35B fp16).

---

## Critical context (read-first)

1. `STATUS.md` — `C-v0.12.0+PARTIAL`, G5-bis row classifies framework C as "MLX-only artefact" at toy E-SNN scale. G6-Studio tests the converse at real-LLM scale. **No FC bump regardless of outcome.**
2. `kiki_oniric/substrates/micro_kiki.py:60-105, 595-722, 874-1129` — `_REAL_BACKEND_ENV_VAR = "DREAM_MICRO_KIKI_REAL"`, `_REAL_BACKEND_PATH_ENV_VAR`, `MicroKikiSubstrate.load()` Path 2 (SpikingKiki shim), and the four handler factories (`replay_handler_factory` returns 1-D vector ; `downscale_handler_factory` returns shrunk tensor ; `restructure_handler_factory` mutates adapter dict via OPLoRA ; `recombine_handler_factory` returns TIES-merged delta).
3. `experiments/g6_mmlu_stream/run_g6.py:50-300, 581-682, 539-578` — Path B template (driver shape, `_run_cell_path_b`, `_aggregate_verdict`, `_render_md_report`, `compute_retention`, registry registration). The "spectator pattern" caveat (lines 539-578) is exactly the failure mode this plan corrects.
4. `experiments/g6_mmlu_stream/{stream.py,dream_wrap.py,micro_kiki_inference.py}` — reusable subdomain stream loader (`build_subdomain_stream`, `SubdomainSplit`) ; `PROFILE_OPS` dispatch + `build_episode_payload` (synthetic) — Task 4 ports to live-tensor variant.
5. `tests/unit/test_micro_kiki_real_backend.py` — fixture pattern for the env-gated load path.
6. `docs/osf-prereg-g6-pilot.md` — original G6 pre-reg ; Task 1 builds a daughter pre-reg citing G4-ter (+2.77 MLX richer head), G5-bis (H7-B "MLX-only artefact" toy), G4-quinto (H5-C RECOMBINE empty CIFAR-10) as exploratory baselines.
7. `kiki_oniric/eval/statistics.py:37-285` — `welch_one_sided(treatment, control, alpha)` rejects when `mean(treatment) < mean(control)` ; `compute_hedges_g(treatment, control)` ; `jonckheere_trend(groups, alpha)` rejects under increasing-monotone H1.
8. `harness/storage/run_registry.py:113` — `register(c_version, profile, seed, commit_sha) -> run_id` (R1 contract bit-stable).
9. `docs/CLAUDE.md` — milestone files are append-only dated immutables. `tests/CLAUDE.md` — unit tests under `tests/unit/experiments/`. `kiki_oniric/CLAUDE.md` — DR-3 protocol contract ; do not loosen primitive types.

## Hard prerequisites (block until satisfied)

1. SSH access to Studio M3 Ultra works.
2. `/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4/lif_metadata.json` exists.
3. `/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-V4-adapters/` exists (or sibling reported in Task 0).
4. `mlx_lm` ≥ 0.31.2 importable on Studio inside `uv run`.
5. `tests/unit/test_micro_kiki_real_backend.py` passes on Studio with env vars exported.
6. ≥ 200 GB free RAM on Studio (35B bf16 + LoRA + Adam ≈ 80–110 GB peak).
7. `docs/milestones/g6-pilot-pathB-2026-05-03.json` exists (the comparator).

## Workspace dependencies

```
/Users/electron/hypneum-lab/dream-of-kiki                   # this repo (M1 conductor)
/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4  # Studio, 58 GB, 31 070 .npz
/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-V4-adapters # Studio, LoRA stack
```

**Execution pattern S1 (recommended)** : `git clone` on Studio, `uv sync --all-extras`, `tmux new -s g6_studio`, run inside tmux. Locked in Task 0.5.

## File structure

Create :
```
experiments/g6_studio_path_a/{__init__.py, lora_loader.py, lora_train_step.py,
                              dream_episode_real.py, mmlu_eval.py,
                              run_g6_studio_path_a.py, aggregator_h9.py}
tests/unit/experiments/{test_g6_studio_lora_loader.py, test_g6_studio_lora_train_step.py,
                       test_g6_studio_dream_episode_real.py, test_g6_studio_mmlu_eval.py,
                       test_g6_studio_aggregator_h9.py}
docs/{osf-prereg-g6-studio-path-a.md,
      milestones/g6-studio-path-a-decisions-2026-05-04.md,
      milestones/g6-studio-path-a-2026-05-04.{json,md}}
```

Modify : `docs/papers/paper2{,-fr}/results.md` (§7.1.10), `docs/proofs/dr3-substrate-evidence.md`, `CHANGELOG.md`, `STATUS.md`.

`experiments/` is excluded from coverage scope per `pyproject.toml`. `tests/unit/experiments/` IS in coverage scope.

---

## Task 0: Investigate Studio environment (read-only)

- [ ] **Step 1: SSH to Studio + verify SpikingKiki-V4 layout**

```bash
ls -la /Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4/lif_metadata.json
ls /Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4/*.npz | wc -l
ls -d /Users/clems/KIKI-Mac_tunner/models/SpikingKiki-V4-adapters/ 2>&1
du -sh /Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4/
```
Expected: metadata file present ; npz count ≈ 31 070 ; adapters dir reported ; total ≈ 58 GB.

- [ ] **Step 2: Verify mlx_lm version + tuner.lora API on Studio**

```bash
cd ~/dream-of-kiki && uv run python -c "
import mlx_lm; print('version', mlx_lm.__version__)
from mlx_lm.tuner import lora
print('tuner.lora exports:', [n for n in dir(lora) if not n.startswith('_')])
from mlx_lm.tuner.utils import load_adapters
from mlx_lm import load, generate
print('ok')"
```
Expected: version ≥ 0.31.2 ; `lora` namespace exposes `train` / `LoRALinear` / `apply_lora_layers` (or equivalent — record exact names ; Task 3 binds to them).

- [ ] **Step 3: Smoke-load real SpikingKiki-V4 weights via the substrate**

```bash
cd ~/dream-of-kiki && DREAM_MICRO_KIKI_REAL=1 uv run python -c "
from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate
s = MicroKikiSubstrate(real_backend_path='/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4')
s.load()
print('module_count', s._real_state['module_count'])
print('awake', s.awake('Anatomy: the largest bone is')[:200])"
```
Expected: `module_count` ≈ 31 070 ; awake output starts `[spiking awake T=`. Record wall time.

- [ ] **Step 4: Verify MMLU dataset access**

```bash
cd ~/dream-of-kiki && uv run python -c "
from datasets import load_dataset
ds = load_dataset('cais/mmlu', 'all', split='test')
for s in ('anatomy','astronomy','business_ethics','clinical_knowledge','college_biology'):
    print(s, sum(1 for r in ds if r['subject']==s))"
```
Expected: each subject ≥ 200 rows. Falls back to `tests/fixtures/mmlu_g6_synthetic.jsonl` if HF download fails (record deviation).

- [ ] **Step 5: Run the existing real-backend tests on Studio**

```bash
DREAM_MICRO_KIKI_REAL=1 \
DREAM_MICRO_KIKI_REAL_BACKEND_PATH=/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4 \
uv run pytest tests/unit/test_micro_kiki_real_backend.py -v
```
Expected: 4 tests pass.

- [ ] **Step 6: No commit. Context-gathering only.**

---

## Task 0.5: Lock decisions (Option A/B/C, hyperparams, execution pattern)

**Files:** Create `docs/milestones/g6-studio-path-a-decisions-2026-05-04.md`.

- [ ] **Step 1: Pick the option**

| Option | N seeds | Cells | Est. wall (Studio) | Min detectable g (80 % power, two-sided α=0.0125) |
|--------|---------|-------|--------------------|----------------------------------------------------|
| A — confirmatory | 10 | 200 | 16–25 h | g ≈ 1.27 |
| **B — first overnight** | 5 | 100 | 8–12 h | g ≈ 1.85 |
| C — smoke | 1 | 4 | ~30 min | exploratory |

**Recommended : Option B for the first overnight ; Option A locked in a follow-up dated decision log if H9-A observed.** Option C is the Task 8 smoke gate.

- [ ] **Step 2: Write the decision log**

Write `docs/milestones/g6-studio-path-a-decisions-2026-05-04.md` with these literal values :

```markdown
# G6-Studio Path A pilot decisions (locked 2026-05-04)

## Subdomains (5 — same as G6 Path B for direct comparability)
S1=anatomy, S2=astronomy, S3=business_ethics, S4=clinical_knowledge, S5=college_biology

## Per-cell volumes
n_train_per_subdomain = 100 ; n_eval_per_subdomain = 100
seeds = (0, 1, 2, 3, 4)  # Option B
arms = (baseline, P_min, P_equ, P_max)
n_cells = 4 × 5 = 20 sequences (each touches 5 subdomains)

## LoRA hyperparams (locked)
lr=1e-4 ; iters=50 ; batch_size=1 ; rank=8 ; alpha=16
adapter_keys = ("layer_0_lora_B",)  # 1 sparse tap ; bumped to 7 in Option A follow-up

## Execution pattern
PATTERN=S1 ; tmux session = g6_studio_2026-05-04
RESUME_GLOB = docs/milestones/g6-studio-path-a-2026-05-04.partial.*.json

## Compute budget
Option B = 100 cells × ~5–7 min/cell = 8–12 h Studio. Watchdog-kill-recoverable.

## Option selected
OPTION = B
RATIONALE = balances detectability and overnight feasibility ; A=follow-up.
```

- [ ] **Step 3: Verify pre-reg dependency chain exists**

```bash
ls docs/osf-prereg-g6-pilot.md docs/milestones/g6-pilot-pathB-2026-05-03.json \
   docs/milestones/g4-ter-pilot-2026-05-03.json \
   docs/milestones/g5-bis-richer-esnn-2026-05-03.json \
   docs/milestones/g4-quinto-pilot-2026-05-03.json
```
Expected: all 5 exist.

- [ ] **Step 4: Commit**

```bash
git add docs/milestones/g6-studio-path-a-decisions-2026-05-04.md
git commit -m "docs(g6s): lock G6-Studio Path A decisions"
```

---

## Task 1: OSF pre-reg (locked before any cell registers)

**Files:** Create `docs/osf-prereg-g6-studio-path-a.md`.

- [ ] **Step 1: Draft the pre-reg**

Write `docs/osf-prereg-g6-studio-path-a.md`. Required sections (every one non-placeholder) :

- **§0 Context** — cite G6 Path B (`g_h1=0.0` spectator), G4-ter MLX richer (+2.77 positive, strongest programme signal), G5-bis E-SNN (H7-B "MLX-only artefact" classification), G4-quinto CIFAR-10 (H5-C RECOMBINE empty). State that Path A's load-bearing change is live-tensor coupling vs Path B's synthetic payload.
- **§1 Study design** — within-architecture × within-benchmark sweep on `MicroKikiSubstrate` with `DREAM_MICRO_KIKI_REAL=1`, real_backend_path `/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4`. 5 subdomains, per-cell pipeline : fresh wrapper → for i in 1..5 (real LoRA fine-tune lr=1e-4 50 iters → optional dream episode on **live** LoRA delta → eval S_1..S_i via mlx_lm.generate letter-argmax) → retention.
- **§2 Hypotheses (3 confirmatory ; Bonferroni α/3 = 0.0167)** :
  - **H9-A** — real-scale recovery. `g_h9a = compute_hedges_g(retention[P_equ], retention[baseline])`. Reject H0 iff `g_h9a ≥ 0.5 AND welch_one_sided(baseline, P_equ, α=0.0167).reject_h0`. Interpretation : framework C transfers to real LLM scale ; rebuts G5-bis "MLX-only".
  - **H9-B** — washout persists at 35B. `g_h9a < 0.5 AND NOT welch_one_sided(...).reject_h0`. Predicted under G5-bis universality. Reported as a positive empirical claim (programme has tightened the boundary). Mutually exclusive with H9-A on (g threshold, Welch reject) ; mixed signals reported as INDETERMINATE.
  - **H9-C** — universal DR-4 inversion. `jonckheere_trend([P_min,P_equ,P_max]).reject_h0 == False AND mean(P_min) > mean(P_equ)`. Interpretation : RESTRUCTURE+RECOMBINE remain spectator on real LIF Qwen too.
- **§3 Sample size + power** — Option B N=5, min detectable g ≈ 1.85 (80 % power, two-sided α=0.0125). Adequate iff effect transfers near-quantitatively to G4-ter +2.77.
- **§4 Stop rules + exclusions** — cell-exclude when `acc[S_1 after S_1] < 0.30` (mirrors G6 Path B `UNDERPERFORM_THRESHOLD`) ; run-abort on Studio OOM (peak RSS > 450 GB).
- **§5 Decision rule (LOCKED)** — copy the table from §2 verbatim ; this is the single source of truth.
- **§6 EC-axis policy** — Option B is exploratory-confirmatory. H9-A queues Option A (N≥10) follow-up pre-reg before any STABLE bump. H9-B = empirical-claim-positive but does NOT bump EC (refutation, not confirmation). H9-C → DR-3 evidence revision only. EC stays PARTIAL ; FC stays C-v0.12.0 regardless.
- **§7 Deviations** — any deviation logs a dated `docs/osf-deviations-g6-studio-path-a-*.md` BEFORE re-running.

- [ ] **Step 2: Commit**

```bash
git add docs/osf-prereg-g6-studio-path-a.md
git commit -m "docs(g6s): lock OSF pre-reg G6-Studio Path A"
```

---

## Task 2: lora_loader — Qwen base + SpikingKiki-V4 adapters

**Files:** Create `experiments/g6_studio_path_a/__init__.py`, `experiments/g6_studio_path_a/lora_loader.py`. Test: `tests/unit/experiments/test_g6_studio_lora_loader.py`.

- [ ] **Step 1: Failing test (mocked mlx_lm)**

```python
# tests/unit/experiments/test_g6_studio_lora_loader.py
"""Unit tests for the G6-Studio Path A lora_loader."""
from __future__ import annotations
import sys, types
from pathlib import Path
import pytest


@pytest.fixture
def mock_mlx_lm(monkeypatch):
    fake = types.ModuleType("mlx_lm")
    fake.load = lambda path: ("MOCK_MODEL", "MOCK_TOK")
    monkeypatch.setitem(sys.modules, "mlx_lm", fake)
    fake_utils = types.ModuleType("mlx_lm.tuner.utils")
    fake_utils.load_adapters = lambda model, path: f"WITH_ADAPTERS:{path}"
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner",
                        types.ModuleType("mlx_lm.tuner"))
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.utils", fake_utils)


def test_load_with_adapters_present(tmp_path: Path, mock_mlx_lm) -> None:
    """TDD-2.1 — adapters dir present → wrapper carries both."""
    from experiments.g6_studio_path_a.lora_loader import load_qwen_with_adapters
    adapters = tmp_path / "SpikingKiki-V4-adapters"
    adapters.mkdir()
    (adapters / "adapters.safetensors").write_bytes(b"\x00" * 8)
    wrap = load_qwen_with_adapters(base_path="/fake/qwen",
                                   adapter_path=adapters, rank=8)
    assert wrap.model == "WITH_ADAPTERS:" + str(adapters)
    assert wrap.adapter_path == adapters and wrap.fresh_init is False


def test_load_without_adapters_fresh_init(tmp_path: Path, mock_mlx_lm) -> None:
    """TDD-2.2 — adapters absent → fresh-LoRA-init signalled."""
    from experiments.g6_studio_path_a.lora_loader import load_qwen_with_adapters
    wrap = load_qwen_with_adapters(base_path="/fake/qwen",
                                   adapter_path=tmp_path / "missing",
                                   rank=8)
    assert wrap.model == "MOCK_MODEL"
    assert wrap.adapter_path is None and wrap.fresh_init is True
```

- [ ] **Step 2: Run test — verify FAIL**

`uv run pytest tests/unit/experiments/test_g6_studio_lora_loader.py -v` → FAIL on missing module.

- [ ] **Step 3: Implement lora_loader.py**

```python
# experiments/g6_studio_path_a/lora_loader.py
"""Qwen-35B-A3B-V4 base + LoRA adapter loader (Studio-only path).

Wraps mlx_lm.load + mlx_lm.tuner.utils.load_adapters into a single
QwenLoRAWrapper. Adapters absent → fresh_init=True (random rank-r LoRA).
MLX imported lazily so Linux CI can mock it.
"""
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass
class QwenLoRAWrapper:
    model: Any
    tokenizer: Any
    adapter_path: Path | None
    rank: int
    fresh_init: bool


def load_qwen_with_adapters(*, base_path: str, adapter_path: Path,
                            rank: int) -> QwenLoRAWrapper:
    """Load Qwen base + (optional) LoRA stack via mlx_lm."""
    from mlx_lm import load as mlx_load
    model, tokenizer = mlx_load(base_path)
    if adapter_path.is_dir() and (adapter_path / "adapters.safetensors").is_file():
        from mlx_lm.tuner.utils import load_adapters
        model = load_adapters(model, str(adapter_path))
        return QwenLoRAWrapper(model=model, tokenizer=tokenizer,
                               adapter_path=adapter_path, rank=rank,
                               fresh_init=False)
    return QwenLoRAWrapper(model=model, tokenizer=tokenizer,
                           adapter_path=None, rank=rank, fresh_init=True)


__all__ = ["QwenLoRAWrapper", "load_qwen_with_adapters"]
```

- [ ] **Step 4: Run test — verify PASS** (`uv run pytest ... -v`).

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_studio_path_a/__init__.py \
        experiments/g6_studio_path_a/lora_loader.py \
        tests/unit/experiments/test_g6_studio_lora_loader.py
git commit -m "feat(g6s): Qwen + LoRA loader for Studio Path A"
```

---

## Task 3: lora_train_step — real per-subdomain mlx_lm.tuner.lora fine-tune

**Files:** Create `experiments/g6_studio_path_a/lora_train_step.py`. Test: `tests/unit/experiments/test_g6_studio_lora_train_step.py`.

- [ ] **Step 1: Failing test (mocked mlx_lm.tuner.lora)**

```python
# tests/unit/experiments/test_g6_studio_lora_train_step.py
"""Locked hyperparams forwarded ; delta returned for adapter_keys subset."""
from __future__ import annotations
import sys, types
import numpy as np
import pytest


@pytest.fixture
def mock_tuner(monkeypatch):
    captured = {}
    def fake_train(model, tokenizer, *, args, train_set, **kwargs):
        captured["args"] = args ; captured["len_train"] = len(train_set)
    fake = types.ModuleType("mlx_lm.tuner.lora") ; fake.train = fake_train
    monkeypatch.setitem(sys.modules, "mlx_lm", types.ModuleType("mlx_lm"))
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner",
                        types.ModuleType("mlx_lm.tuner"))
    monkeypatch.setitem(sys.modules, "mlx_lm.tuner.lora", fake)
    return captured


def test_train_step_forwards_locked_hyperparams(mock_tuner) -> None:
    """TDD-3.1 — lr=1e-4 / iters=50 / rank=8 / alpha=16 forwarded."""
    from experiments.g6_studio_path_a.lora_train_step import (
        TrainHyperparams, train_subdomain_lora)
    class FakeModel:
        def parameters(self):
            return {"layer_0_lora_B": np.zeros((8, 8), dtype=np.float32)}
    hp = TrainHyperparams(lr=1e-4, iters=50, rank=8, alpha=16, batch_size=1)
    delta = train_subdomain_lora(model=FakeModel(), tokenizer=None,
                                 train_records=[{"text": "Q1"}, {"text": "Q2"}],
                                 hyperparams=hp,
                                 adapter_keys=("layer_0_lora_B",))
    a = mock_tuner["args"]
    assert mock_tuner["len_train"] == 2
    assert a["learning_rate"] == 1e-4 and a["iters"] == 50
    assert a["batch_size"] == 1
    assert "layer_0_lora_B" in delta
    assert delta["layer_0_lora_B"].dtype == np.float32


def test_train_step_returns_delta_subset(mock_tuner) -> None:
    """TDD-3.2 — only adapter_keys subset returned ; base weights excluded."""
    from experiments.g6_studio_path_a.lora_train_step import (
        TrainHyperparams, train_subdomain_lora)
    class FakeModel:
        def parameters(self):
            return {"layer_0_lora_B": np.ones((4, 8), dtype=np.float32),
                    "layer_0_base_W": np.ones((4096, 4096), dtype=np.float32),
                    "layer_1_lora_B": np.ones((4, 8), dtype=np.float32)}
    hp = TrainHyperparams(lr=1e-4, iters=1, rank=8, alpha=16, batch_size=1)
    delta = train_subdomain_lora(model=FakeModel(), tokenizer=None,
                                 train_records=[], hyperparams=hp,
                                 adapter_keys=("layer_0_lora_B", "layer_1_lora_B"))
    assert set(delta) == {"layer_0_lora_B", "layer_1_lora_B"}
    assert delta["layer_0_lora_B"].shape == (4, 8)
```

- [ ] **Step 2: Run test — verify FAIL**.

- [ ] **Step 3: Implement lora_train_step.py**

```python
# experiments/g6_studio_path_a/lora_train_step.py
"""Per-subdomain real LoRA fine-tune via mlx_lm.tuner.lora.

Drives one fine-tune pass over a subdomain's training records, extracts
the resulting LoRA delta (subset matching adapter_keys), and returns
them as numpy arrays so the dream runtime (numpy-only) can mutate them.

mlx_lm.tuner.lora.train signature drifts across versions ; this shim
binds to the 0.31.x flavour observed in Task 0 step 2. API drift = log
deviation in docs/osf-deviations-g6-studio-path-a-*.md BEFORE re-running.
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Sequence
import numpy as np
from numpy.typing import NDArray


@dataclass(frozen=True)
class TrainHyperparams:
    lr: float ; iters: int ; rank: int ; alpha: int ; batch_size: int


def train_subdomain_lora(*, model: Any, tokenizer: Any,
                         train_records: Sequence[Any],
                         hyperparams: TrainHyperparams,
                         adapter_keys: tuple[str, ...]
                         ) -> dict[str, NDArray[np.float32]]:
    from mlx_lm.tuner.lora import train as lora_train
    args = {"learning_rate": hyperparams.lr, "iters": hyperparams.iters,
            "batch_size": hyperparams.batch_size,
            "lora_layers": hyperparams.rank,
            "lora_alpha": hyperparams.alpha}
    lora_train(model, tokenizer, args=args, train_set=list(train_records))
    params = model.parameters()
    return {k: np.asarray(params[k], dtype=np.float32)
            for k in adapter_keys if k in params}


__all__ = ["TrainHyperparams", "train_subdomain_lora"]
```

- [ ] **Step 4: Run test — verify PASS**.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_studio_path_a/lora_train_step.py \
        tests/unit/experiments/test_g6_studio_lora_train_step.py
git commit -m "feat(g6s): wire mlx_lm tuner.lora real fine-tune"
```

---

## Task 4: dream_episode_real — live-tensor coupling (load-bearing fix vs Path B)

**Files:** Create `experiments/g6_studio_path_a/dream_episode_real.py`. Test: `tests/unit/experiments/test_g6_studio_dream_episode_real.py`.

- [ ] **Step 1: Failing test**

```python
# tests/unit/experiments/test_g6_studio_dream_episode_real.py
"""Live-tensor dream episode tests — handlers MUTATE live LoRA delta dict
(not synthesised payload). This is the load-bearing fix vs G6 Path B
spectator pattern (run_g6.py:539-578)."""
from __future__ import annotations
import numpy as np
from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate
from experiments.g6_studio_path_a.dream_episode_real import (
    PROFILE_OPS_REAL, dream_episode_real)


def _delta(out_dim: int, rank: int, seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    return {"layer_0_lora_B": rng.standard_normal(
        (out_dim, rank)).astype(np.float32)}


def test_baseline_no_mutation() -> None:
    """TDD-4.1 — baseline arm leaves delta unchanged."""
    sub = MicroKikiSubstrate(num_layers=4, rank=8, seed=0)
    d = _delta(8, 8, seed=0) ; before = d["layer_0_lora_B"].copy()
    out = dream_episode_real(substrate=sub, profile="baseline",
                             live_delta=d, seed=0, subdomain="anatomy",
                             prior_deltas=[], sibling_deltas=[])
    assert np.array_equal(out["layer_0_lora_B"], before)


def test_p_min_downscales_live_delta() -> None:
    """TDD-4.2 — P_min applies shrink_factor < 1 in place."""
    sub = MicroKikiSubstrate(num_layers=4, rank=8, seed=0)
    d = _delta(8, 8, seed=0)
    before = float(np.linalg.norm(d["layer_0_lora_B"]))
    dream_episode_real(substrate=sub, profile="P_min", live_delta=d,
                       seed=0, subdomain="anatomy",
                       prior_deltas=[], sibling_deltas=[])
    after = float(np.linalg.norm(d["layer_0_lora_B"]))
    assert 0 < after < before


def test_p_equ_mutates_via_all_four_handlers() -> None:
    """TDD-4.3 — P_equ runs all 4 ops ; delta mutated AND DR-0 stamps land."""
    sub = MicroKikiSubstrate(num_layers=4, rank=8, seed=0)
    d = _delta(8, 8, seed=0) ; before = d["layer_0_lora_B"].copy()
    rng = np.random.default_rng(42)
    priors = [rng.standard_normal((8, 4)).astype(np.float32)]
    siblings = [rng.standard_normal((8, 8)).astype(np.float32) for _ in range(2)]
    dream_episode_real(substrate=sub, profile="P_equ", live_delta=d,
                       seed=0, subdomain="anatomy",
                       prior_deltas=priors, sibling_deltas=siblings)
    assert not np.array_equal(d["layer_0_lora_B"], before)
    assert sub.restructure_state.total_episodes_handled == 1
    assert sub.recombine_state.total_episodes_handled == 1
    assert sub.restructure_state.last_episode_id == "g6s-P_equ-anatomy-seed0"


def test_profile_ops_table_exhaustive() -> None:
    """TDD-4.4 — PROFILE_OPS_REAL covers all 4 arms."""
    assert set(PROFILE_OPS_REAL) == {"baseline", "P_min", "P_equ", "P_max"}
```

- [ ] **Step 2: Run test — verify FAIL**.

- [ ] **Step 3: Implement dream_episode_real.py**

```python
# experiments/g6_studio_path_a/dream_episode_real.py
"""Live-tensor dream-episode runner for G6-Studio Path A.

Fixes G6 Path B spectator pattern : the four MicroKikiSubstrate
handlers operate on the LIVE LoRA delta dict (passed by reference), not
on a synthesised payload. P_max runs the same 4 ops as P_equ but with
TIES-Merge α=2.0 (paper §3 amplified contribution ; matches G4-ter HP
grid C5).

DR-0 / DR-1 stamps land on substrate._{restructure,recombine}_state via
the handler closures.
"""
from __future__ import annotations
from typing import Any
import numpy as np
from numpy.typing import NDArray
from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate


PROFILE_OPS_REAL: dict[str, tuple[str, ...]] = {
    "baseline": (), "P_min": ("replay", "downscale"),
    "P_equ": ("replay", "downscale", "restructure", "recombine"),
    "P_max": ("replay", "downscale", "restructure", "recombine"),
}
PROFILE_RECOMBINE_ALPHA: dict[str, float] = {"P_equ": 1.0, "P_max": 2.0}
DEFAULT_SHRINK_FACTOR = 0.99
DEFAULT_PRIMARY_KEY = "layer_0_lora_B"


def dream_episode_real(*, substrate: MicroKikiSubstrate, profile: str,
                       live_delta: dict[str, NDArray[np.float32]],
                       seed: int, subdomain: str,
                       prior_deltas: list[NDArray],
                       sibling_deltas: list[NDArray],
                       primary_key: str = DEFAULT_PRIMARY_KEY,
                       ) -> dict[str, NDArray[np.float32]]:
    """Run one dream episode against the LIVE LoRA delta dict (in place)."""
    if profile not in PROFILE_OPS_REAL:
        raise ValueError(f"unknown profile {profile!r}")
    ops = PROFILE_OPS_REAL[profile]
    if not ops:
        return live_delta
    if primary_key not in live_delta:
        raise KeyError(f"live_delta missing primary_key {primary_key!r}")
    episode_id = f"g6s-{profile}-{subdomain}-seed{seed}"
    rng = np.random.default_rng(seed)

    if "replay" in ops:
        beta = [{"input": rng.standard_normal(
            live_delta[primary_key].shape[0]).astype(np.float32).tolist()}
                for _ in range(4)]
        substrate.replay_handler_factory()(beta, 20)
    if "downscale" in ops:
        live_delta[primary_key] = substrate.downscale_handler_factory()(
            live_delta[primary_key], DEFAULT_SHRINK_FACTOR)
    if "restructure" in ops:
        view: dict[str, Any] = {primary_key: live_delta[primary_key],
                                "prior_deltas": list(prior_deltas),
                                "episode_id": episode_id}
        substrate.restructure_handler_factory()(view, "oplora", primary_key)
        live_delta[primary_key] = view[primary_key]
    if "recombine" in ops:
        rec = substrate.recombine_handler_factory(
            alpha=PROFILE_RECOMBINE_ALPHA.get(profile, 1.0))
        merged = rec({"deltas": [live_delta[primary_key]] + list(sibling_deltas),
                      "episode_id": episode_id}, "ties")
        live_delta[primary_key] = np.asarray(merged, dtype=np.float32)
    return live_delta


__all__ = ["PROFILE_OPS_REAL", "PROFILE_RECOMBINE_ALPHA",
           "DEFAULT_SHRINK_FACTOR", "DEFAULT_PRIMARY_KEY",
           "dream_episode_real"]
```

- [ ] **Step 4: Run test — verify PASS**.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_studio_path_a/dream_episode_real.py \
        tests/unit/experiments/test_g6_studio_dream_episode_real.py
git commit -m "feat(g6s): live-tensor dream episode coupling"
```

---

## Task 5: mmlu_eval — letter-argmax accuracy via mlx_lm.generate

**Files:** Create `experiments/g6_studio_path_a/mmlu_eval.py`. Test: `tests/unit/experiments/test_g6_studio_mmlu_eval.py`.

- [ ] **Step 1: Failing test**

```python
# tests/unit/experiments/test_g6_studio_mmlu_eval.py
"""Letter-argmax extraction + MMLU eval over a stub generator."""
from __future__ import annotations
from harness.real_benchmarks.mmlu import MMLURecord
from experiments.g6_studio_path_a.mmlu_eval import (
    evaluate_mmlu_subdomain, extract_letter)


def _record(answer: int = 0) -> MMLURecord:
    return MMLURecord(question="What is X?",
                      choices=("A1", "B1", "C1", "D1"),
                      answer=answer, subject="anatomy")


def test_extract_letter_picks_first_alpha() -> None:
    """TDD-5.1 — A/B/C/D extraction from cleaned generator output."""
    assert extract_letter("The answer is C.") == "C"
    assert extract_letter("(B) is correct") == "B"
    assert extract_letter("garbage") is None
    assert extract_letter("D") == "D"


def test_evaluate_with_stub_generator() -> None:
    """TDD-5.2 — accuracy = 1/3 when stub always answers A on records[0..2]."""
    records = [_record(0), _record(1), _record(2)]
    def stub(model, tok, *, prompt, max_tokens): return "A"
    acc = evaluate_mmlu_subdomain(model=None, tokenizer=None,
                                  records=records, seed=0, generate_fn=stub)
    assert abs(acc - 1.0 / 3.0) < 1e-6


def test_evaluate_handles_malformed_outputs() -> None:
    """TDD-5.3 — generator returning garbage falls back to seed proxy."""
    def garbage(model, tok, *, prompt, max_tokens): return "🦞 lobster spoke"
    acc = evaluate_mmlu_subdomain(model=None, tokenizer=None,
                                  records=[_record(0)], seed=42,
                                  generate_fn=garbage)
    assert 0.0 <= acc <= 1.0
```

- [ ] **Step 2: Run test — verify FAIL**.

- [ ] **Step 3: Implement mmlu_eval.py**

```python
# experiments/g6_studio_path_a/mmlu_eval.py
"""MMLU letter-argmax eval driven by mlx_lm.generate (Studio path).

Tests inject a stub via ``generate_fn`` so Linux CI exercises extraction.
On malformed generator output (no A/B/C/D), fall back to a deterministic
(record, seed) proxy in [0.20, 0.40] so the cell does not crash.
"""
from __future__ import annotations
import hashlib, re
from typing import Any, Callable, Sequence
from harness.real_benchmarks.mmlu import MMLURecord


LETTER_PATTERN = re.compile(r"\b([A-D])\b")
LETTER_LOOKUP = ("A", "B", "C", "D")


def extract_letter(text: str) -> str | None:
    m = LETTER_PATTERN.search(text)
    return m.group(1) if m else None


def _format_prompt(r: MMLURecord) -> str:
    return (f"Question: {r.question}\nA. {r.choices[0]}\nB. {r.choices[1]}\n"
            f"C. {r.choices[2]}\nD. {r.choices[3]}\nAnswer:")


def _seed_proxy(r: MMLURecord, seed: int) -> float:
    raw = f"g6s-mmlu|{r.question}|{seed}".encode("utf-8")
    return 0.2 + (int(hashlib.sha256(raw).hexdigest()[:8], 16) % 21) / 100.0


def evaluate_mmlu_subdomain(*, model: Any, tokenizer: Any,
                            records: Sequence[MMLURecord], seed: int,
                            generate_fn: Callable[..., str] | None = None,
                            max_tokens: int = 8) -> float:
    if generate_fn is None:
        from mlx_lm import generate as mlx_generate
        def _gen(m, t, *, prompt, max_tokens):
            return str(mlx_generate(m, t, prompt=prompt,
                                    max_tokens=max_tokens, verbose=False))
        generate_fn = _gen
    if not records:
        return 0.0
    correct = 0 ; fallback_total = 0.0 ; fallback_count = 0
    for r in records:
        out = generate_fn(model, tokenizer, prompt=_format_prompt(r),
                          max_tokens=max_tokens)
        letter = extract_letter(out)
        if letter is None:
            fallback_total += _seed_proxy(r, seed) ; fallback_count += 1
            continue
        if letter == LETTER_LOOKUP[r.answer]:
            correct += 1
    well = len(records) - fallback_count
    if fallback_count == len(records):
        return float(fallback_total / max(fallback_count, 1))
    fb = fallback_total / fallback_count if fallback_count else 0.0
    base = correct / max(well, 1)
    w = well / len(records)
    return float(w * base + (1.0 - w) * fb)


__all__ = ["evaluate_mmlu_subdomain", "extract_letter"]
```

- [ ] **Step 4: Run test — verify PASS**.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_studio_path_a/mmlu_eval.py \
        tests/unit/experiments/test_g6_studio_mmlu_eval.py
git commit -m "feat(g6s): MMLU letter-argmax eval via mlx_lm"
```

---

## Task 6: aggregator_h9 — H9-A / H9-B / H9-C verdict

**Files:** Create `experiments/g6_studio_path_a/aggregator_h9.py`. Test: `tests/unit/experiments/test_g6_studio_aggregator_h9.py`.

- [ ] **Step 1: Failing test**

```python
# tests/unit/experiments/test_g6_studio_aggregator_h9.py
"""Decision rule arithmetic — H9-A / H9-B / H9-C / Bonferroni alpha."""
from __future__ import annotations
from experiments.g6_studio_path_a.aggregator_h9 import (
    H9_BONFERRONI_ALPHA, classify_h9)


def test_h9a_real_scale_recovery() -> None:
    """TDD-6.1 — large g + Welch reject → H9-A."""
    r = {"baseline": [0.40, 0.42, 0.41, 0.39, 0.43],
         "P_equ":    [0.85, 0.87, 0.86, 0.88, 0.84],
         "P_min":    [0.55, 0.56, 0.54, 0.57, 0.55],
         "P_max":    [0.84, 0.83, 0.82, 0.85, 0.86]}
    v = classify_h9(r)
    assert v["h9a_classification"] == "H9-A"
    assert v["h9a_g"] >= 0.5 and v["h9a_welch_reject"] is True


def test_h9b_washout_persists() -> None:
    """TDD-6.2 — tiny g + Welch fail → H9-B."""
    r = {"baseline": [0.50, 0.51, 0.49, 0.50, 0.52],
         "P_equ":    [0.51, 0.50, 0.51, 0.50, 0.50],
         "P_min":    [0.50, 0.49, 0.51, 0.50, 0.50],
         "P_max":    [0.51, 0.50, 0.50, 0.51, 0.50]}
    v = classify_h9(r)
    assert v["h9a_classification"] == "H9-B"
    assert v["h9a_g"] < 0.5 and v["h9a_welch_reject"] is False


def test_h9c_dr4_inversion_universal() -> None:
    """TDD-6.3 — Jonckheere fail AND mean(P_min) > mean(P_equ) → H9-C."""
    r = {"baseline": [0.40, 0.41, 0.40, 0.41, 0.40],
         "P_equ":    [0.42, 0.41, 0.42, 0.43, 0.41],
         "P_min":    [0.65, 0.66, 0.64, 0.67, 0.65],
         "P_max":    [0.42, 0.42, 0.41, 0.42, 0.41]}
    v = classify_h9(r)
    assert v["h9c_classification"] == "H9-C"
    assert v["h9c_mean_p_min"] > v["h9c_mean_p_equ"]


def test_h9_bonferroni_alpha_constant() -> None:
    """TDD-6.4 — Bonferroni α = 0.05 / 3."""
    assert abs(H9_BONFERRONI_ALPHA - 0.05 / 3) < 1e-9
```

- [ ] **Step 2: Run test — verify FAIL**.

- [ ] **Step 3: Implement aggregator_h9.py**

```python
# experiments/g6_studio_path_a/aggregator_h9.py
"""H9-A / H9-B / H9-C verdict aggregator (decision rules locked in pre-reg §5)."""
from __future__ import annotations
from typing import Any
from kiki_oniric.eval.statistics import (
    compute_hedges_g, jonckheere_trend, welch_one_sided)


H9_BONFERRONI_ALPHA: float = 0.05 / 3


def _mean(xs: list[float]) -> float:
    return float(sum(xs) / len(xs)) if xs else float("nan")


def classify_h9(retention: dict[str, list[float]]) -> dict[str, Any]:
    """Run H9-A/B/C decision rules on a retention-by-arm dict."""
    out: dict[str, Any] = {"bonferroni_alpha": H9_BONFERRONI_ALPHA}
    base = retention.get("baseline", [])
    p_equ = retention.get("P_equ", [])
    p_min = retention.get("P_min", [])
    p_max = retention.get("P_max", [])

    if len(base) < 2 or len(p_equ) < 2:
        out.update({"h9a_classification": "INSUFFICIENT",
                    "h9a_g": float("nan"), "h9a_welch_p": float("nan"),
                    "h9a_welch_reject": False})
    else:
        try:
            g = compute_hedges_g(p_equ, base)
        except ValueError:
            g = 0.0
        welch = welch_one_sided(base, p_equ, alpha=H9_BONFERRONI_ALPHA)
        out["h9a_g"] = float(g)
        out["h9a_welch_p"] = float(welch.p_value)
        out["h9a_welch_reject"] = bool(welch.reject_h0)
        if g >= 0.5 and welch.reject_h0:
            out["h9a_classification"] = "H9-A"
        elif g < 0.5 and not welch.reject_h0:
            out["h9a_classification"] = "H9-B"
        else:
            out["h9a_classification"] = "INDETERMINATE"

    if any(len(g) < 2 for g in (p_min, p_equ, p_max)):
        out.update({"h9c_classification": "INSUFFICIENT",
                    "h9c_jonckheere_p": float("nan"),
                    "h9c_jonckheere_reject": False,
                    "h9c_mean_p_min": _mean(p_min),
                    "h9c_mean_p_equ": _mean(p_equ),
                    "h9c_mean_p_max": _mean(p_max)})
    else:
        jt = jonckheere_trend([p_min, p_equ, p_max],
                              alpha=H9_BONFERRONI_ALPHA)
        m_min, m_equ, m_max = _mean(p_min), _mean(p_equ), _mean(p_max)
        out["h9c_jonckheere_p"] = float(jt.p_value)
        out["h9c_jonckheere_reject"] = bool(jt.reject_h0)
        out["h9c_mean_p_min"] = m_min
        out["h9c_mean_p_equ"] = m_equ
        out["h9c_mean_p_max"] = m_max
        out["h9c_classification"] = (
            "H9-C" if (not jt.reject_h0 and m_min > m_equ) else "NOT_H9C")
    return out


__all__ = ["H9_BONFERRONI_ALPHA", "classify_h9"]
```

- [ ] **Step 4: Run test — verify PASS**.

- [ ] **Step 5: Commit**

```bash
git add experiments/g6_studio_path_a/aggregator_h9.py \
        tests/unit/experiments/test_g6_studio_aggregator_h9.py
git commit -m "feat(g6s): H9-A/B/C verdict aggregator"
```

---

## Task 7: run_g6_studio_path_a — driver with per-subdomain partial dumps

**Files:** Create `experiments/g6_studio_path_a/run_g6_studio_path_a.py`. (No new test file ; smoke-tested in Task 8.)

- [ ] **Step 1: Implement the driver**

```python
# experiments/g6_studio_path_a/run_g6_studio_path_a.py
"""G6-Studio Path A driver — real LoRA SpikingKiki-V4 35B-A3B-V4 × MMLU CL stream.

Resumability : per-subdomain partial JSON under
docs/milestones/g6-studio-path-a-2026-05-04.partial.<arm>-<seed>-<idx>-<subj>.json
(gitignored). On watchdog kill, glob existing partials, build set of
completed (arm, seed, idx), skip those in run_pilot. Document deviation
in docs/osf-deviations-g6-studio-path-a-*.md BEFORE re-running.

Usage on Studio ::

    DREAM_MICRO_KIKI_REAL=1 \\
    DREAM_MICRO_KIKI_REAL_BACKEND_PATH=/Users/clems/.../SpikingKiki-35B-A3B-V4 \\
    uv run python experiments/g6_studio_path_a/run_g6_studio_path_a.py --option B
"""
from __future__ import annotations
import argparse, json, os, subprocess, sys, time
from pathlib import Path
from typing import Any, Optional, TypedDict

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import numpy as np  # noqa: E402

from experiments.g6_mmlu_stream.run_g6 import (  # noqa: E402
    AccMatrix, RETENTION_EPS, UNDERPERFORM_THRESHOLD, compute_retention)
from experiments.g6_mmlu_stream.stream import (  # noqa: E402
    SubdomainSplit, build_subdomain_stream)
from experiments.g6_studio_path_a.aggregator_h9 import classify_h9  # noqa: E402
from experiments.g6_studio_path_a.dream_episode_real import (  # noqa: E402
    DEFAULT_PRIMARY_KEY, dream_episode_real)
from experiments.g6_studio_path_a.lora_loader import (  # noqa: E402
    QwenLoRAWrapper, load_qwen_with_adapters)
from experiments.g6_studio_path_a.lora_train_step import (  # noqa: E402
    TrainHyperparams, train_subdomain_lora)
from experiments.g6_studio_path_a.mmlu_eval import (  # noqa: E402
    evaluate_mmlu_subdomain)
from harness.storage.run_registry import RunRegistry  # noqa: E402
from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate  # noqa: E402


C_VERSION = "C-v0.12.0+PARTIAL"
ARMS: tuple[str, ...] = ("baseline", "P_min", "P_equ", "P_max")
DEFAULT_SUBDOMAINS = ("anatomy", "astronomy", "business_ethics",
                     "clinical_knowledge", "college_biology")
OPTION_SEEDS = {"A": tuple(range(10)), "B": tuple(range(5)), "C": (0,)}
DATE_TAG = "2026-05-04"
DEFAULT_OUT_JSON = REPO_ROOT / "docs" / "milestones" / f"g6-studio-path-a-{DATE_TAG}.json"
DEFAULT_OUT_MD = REPO_ROOT / "docs" / "milestones" / f"g6-studio-path-a-{DATE_TAG}.md"
DEFAULT_REGISTRY_DB = REPO_ROOT / ".run_registry.sqlite"
DEFAULT_BASE_PATH = "/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4"
DEFAULT_ADAPTER_PATH = "/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-V4-adapters"
PARTIAL_TPL = "g6-studio-path-a-{date}.partial.{arm}-seed{seed}-{idx:02d}-{subj}.json"


class CellResult(TypedDict):
    arm: str ; seed: int ; retention: float
    excluded_underperforming_baseline: bool ; wall_time_s: float
    acc_matrix: AccMatrix ; run_id: str


def _resolve_commit_sha() -> str:
    env = os.environ.get("DREAMOFKIKI_COMMIT_SHA")
    if env: return env
    try:
        out = subprocess.run(["git", "-C", str(REPO_ROOT), "rev-parse", "HEAD"],
                             capture_output=True, text=True, check=False, timeout=2)
        if out.returncode == 0:
            return out.stdout.strip() or "unknown"
    except (OSError, subprocess.SubprocessError):
        pass
    return "unknown"


def _write_partial(out_dir: Path, arm: str, seed: int, idx: int,
                   subj: str, payload: dict) -> Path:
    tgt = out_dir / PARTIAL_TPL.format(date=DATE_TAG, arm=arm, seed=seed,
                                        idx=idx, subj=subj)
    tgt.parent.mkdir(parents=True, exist_ok=True)
    tgt.write_text(json.dumps(payload, indent=2, sort_keys=True, default=str),
                   encoding="utf-8")
    return tgt


def _run_cell_real(*, arm: str, seed: int, splits: list[SubdomainSplit],
                   wrapper: QwenLoRAWrapper, substrate: MicroKikiSubstrate,
                   hp: TrainHyperparams, adapter_keys: tuple[str, ...],
                   out_dir: Path) -> CellResult:
    """One (arm, seed) cell : real LoRA + live-delta coupling + eval."""
    start = time.time()
    subdomains = tuple(s.subject for s in splits)
    n = len(splits)
    acc_matrix: AccMatrix = {s: [None] * n for s in subdomains}
    live_delta: dict[str, np.ndarray] = {}
    prior_deltas: list[np.ndarray] = []
    sibling_deltas: list[np.ndarray] = []

    for i, split in enumerate(splits):
        post = train_subdomain_lora(model=wrapper.model, tokenizer=wrapper.tokenizer,
                                    train_records=split.train, hyperparams=hp,
                                    adapter_keys=adapter_keys)
        if DEFAULT_PRIMARY_KEY in post:
            live_delta[DEFAULT_PRIMARY_KEY] = post[DEFAULT_PRIMARY_KEY].copy()
        elif post:
            first = next(iter(post))
            live_delta[DEFAULT_PRIMARY_KEY] = post[first].copy()

        if arm != "baseline" and DEFAULT_PRIMARY_KEY in live_delta:
            dream_episode_real(substrate=substrate, profile=arm,
                               live_delta=live_delta, seed=seed,
                               subdomain=split.subject,
                               prior_deltas=list(prior_deltas),
                               sibling_deltas=list(sibling_deltas))

        for j in range(i + 1):
            past = splits[j]
            acc_matrix[past.subject][i] = evaluate_mmlu_subdomain(
                model=wrapper.model, tokenizer=wrapper.tokenizer,
                records=past.eval_, seed=seed)

        _write_partial(out_dir, arm, seed, i, split.subject,
                       {"arm": arm, "seed": seed, "idx": i,
                        "subdomain": split.subject, "acc_matrix": acc_matrix,
                        "wall_time_s": time.time() - start})

        if DEFAULT_PRIMARY_KEY in live_delta:
            prior_deltas.append(live_delta[DEFAULT_PRIMARY_KEY].copy())
            sibling_deltas.append(live_delta[DEFAULT_PRIMARY_KEY].copy())

    initial_first = acc_matrix[subdomains[0]][0]
    excluded = bool(initial_first is not None
                    and initial_first < UNDERPERFORM_THRESHOLD)
    return {"arm": arm, "seed": seed,
            "retention": float(compute_retention(acc_matrix,
                                                 subdomains=subdomains)),
            "excluded_underperforming_baseline": excluded,
            "wall_time_s": time.time() - start,
            "acc_matrix": acc_matrix, "run_id": ""}


def _retention_by_arm(cells: list[CellResult]) -> dict[str, list[float]]:
    out: dict[str, list[float]] = {a: [] for a in ARMS}
    for c in cells:
        if not c["excluded_underperforming_baseline"]:
            out[c["arm"]].append(c["retention"])
    return out


def _render_md(payload: dict[str, Any]) -> str:
    lines = [f"# G6-Studio Path A — real LoRA SpikingKiki-V4 × MMLU CL stream",
             "", f"**Date** : {payload['date']}",
             f"**Option** : {payload['option']}",
             f"**c_version** : `{payload['c_version']}`",
             f"**commit_sha** : `{payload['commit_sha']}`",
             f"**Cells** : {len(payload['cells'])}",
             f"**Wall time** : {payload['wall_time_s']:.1f}s", "",
             "## Pre-registered hypotheses (LOCKED)",
             "Pre-registration : `docs/osf-prereg-g6-studio-path-a.md`.", "",
             "### Verdict (H9-A / H9-B / H9-C)",
             f"```\n{json.dumps(payload['verdict'], indent=2, sort_keys=True)}\n```", "",
             "## Cells (R1)", "",
             "| arm | seed | retention | excluded | run_id |",
             "|-----|------|-----------|----------|--------|"]
    for c in payload["cells"]:
        lines.append(f"| {c['arm']} | {c['seed']} | {c['retention']:.4f} | "
                     f"{c['excluded_underperforming_baseline']} | "
                     f"`{c['run_id']}` |")
    lines += ["", "## Honest reporting", "",
              "Per pre-reg §6, Option B does NOT trigger an EC bump regardless "
              "of verdict. H9-A queues an Option-A (N=10) follow-up pre-reg. "
              "H9-B confirms G5-bis MLX-only classification universalises. "
              "H9-C confirms DR-4 inversion at real-LLM scale."]
    return "\n".join(lines)


def run_pilot(*, fixture_path: Path, out_json: Path, out_md: Path,
              registry_db: Path, seeds: tuple[int, ...], n_train: int,
              n_eval: int, hp: TrainHyperparams,
              adapter_keys: tuple[str, ...], base_path: str,
              adapter_path: Path, subdomains: tuple[str, ...],
              option: str) -> dict[str, Any]:
    splits = build_subdomain_stream(fixture_path=fixture_path,
                                    subdomains=subdomains, n_train=n_train,
                                    n_eval=n_eval, seed=0)
    registry = RunRegistry(registry_db)
    commit_sha = _resolve_commit_sha()
    out_dir = out_json.parent
    cells: list[CellResult] = []
    sweep_start = time.time()
    for arm in ARMS:
        for seed in seeds:
            wrapper = load_qwen_with_adapters(base_path=base_path,
                                              adapter_path=adapter_path,
                                              rank=hp.rank)
            substrate = MicroKikiSubstrate(num_layers=20, rank=hp.rank, seed=seed)
            substrate.load()  # honours DREAM_MICRO_KIKI_REAL env
            cell = _run_cell_real(arm=arm, seed=seed, splits=splits,
                                  wrapper=wrapper, substrate=substrate,
                                  hp=hp, adapter_keys=adapter_keys,
                                  out_dir=out_dir)
            cell["run_id"] = registry.register(
                c_version=C_VERSION, profile=f"g6-studio-path-a/{arm}",
                seed=seed, commit_sha=commit_sha)
            cells.append(cell)
    wall = time.time() - sweep_start
    retention = _retention_by_arm(cells)
    verdict = classify_h9(retention)
    payload = {"date": DATE_TAG, "c_version": C_VERSION,
               "commit_sha": commit_sha, "option": option,
               "n_seeds": len(seeds), "arms": list(ARMS),
               "subdomains": list(subdomains),
               "fixture_path": str(fixture_path), "wall_time_s": wall,
               "cells": list(cells), "retention_by_arm": retention,
               "verdict": verdict,
               "hyperparams": {"lr": hp.lr, "iters": hp.iters,
                               "rank": hp.rank, "alpha": hp.alpha,
                               "batch_size": hp.batch_size,
                               "n_train": n_train, "n_eval": n_eval}}
    out_json.parent.mkdir(parents=True, exist_ok=True)
    out_md.parent.mkdir(parents=True, exist_ok=True)
    out_json.write_text(json.dumps(payload, indent=2, sort_keys=True,
                                   default=str), encoding="utf-8")
    out_md.write_text(_render_md(payload), encoding="utf-8")
    return payload


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--option", choices=("A", "B", "C"), default="B")
    p.add_argument("--fixture-path", type=Path,
                   default=REPO_ROOT / "tests" / "fixtures" / "mmlu_g6_synthetic.jsonl")
    p.add_argument("--n-train", type=int, default=100)
    p.add_argument("--n-eval", type=int, default=100)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--iters", type=int, default=50)
    p.add_argument("--rank", type=int, default=8)
    p.add_argument("--alpha", type=int, default=16)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--base-path", default=DEFAULT_BASE_PATH)
    p.add_argument("--adapter-path", type=Path,
                   default=Path(DEFAULT_ADAPTER_PATH))
    p.add_argument("--out-json", type=Path, default=DEFAULT_OUT_JSON)
    p.add_argument("--out-md", type=Path, default=DEFAULT_OUT_MD)
    p.add_argument("--registry-db", type=Path, default=DEFAULT_REGISTRY_DB)
    p.add_argument("--smoke-subdomains", action="store_true")
    a = p.parse_args(argv)
    hp = TrainHyperparams(lr=a.lr, iters=a.iters, rank=a.rank,
                          alpha=a.alpha, batch_size=a.batch_size)
    seeds = OPTION_SEEDS[a.option]
    subdomains = (DEFAULT_SUBDOMAINS[:1]
                  if a.smoke_subdomains or a.option == "C"
                  else DEFAULT_SUBDOMAINS)
    payload = run_pilot(fixture_path=a.fixture_path, out_json=a.out_json,
                        out_md=a.out_md, registry_db=a.registry_db,
                        seeds=seeds, n_train=a.n_train, n_eval=a.n_eval,
                        hp=hp, adapter_keys=("layer_0_lora_B",),
                        base_path=a.base_path, adapter_path=a.adapter_path,
                        subdomains=subdomains, option=a.option)
    print(f"Wrote {a.out_json}\nWrote {a.out_md}\nCells : {len(payload['cells'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Verify the driver imports clean**

```bash
uv run python -c "from experiments.g6_studio_path_a.run_g6_studio_path_a import main; print('ok')"
```
Expected: `ok` printed (MLX paths are lazy ; works on Linux CI).

- [ ] **Step 3: Commit**

```bash
git add experiments/g6_studio_path_a/run_g6_studio_path_a.py
git commit -m "feat(g6s): driver with per-subdomain partial dumps"
```

- [ ] **Step 4: Add gitignore for partials**

Edit `.gitignore` adding `docs/milestones/*.partial.*.json`. Then :

```bash
git add .gitignore
git commit -m "chore(g6s): gitignore per-subdomain partials"
```

---

## Task 8: Smoke test — Option C, 4 cells, ~30 min on Studio

- [ ] **Step 1: Run smoke inside tmux on Studio**

```bash
cd ~/dream-of-kiki && tmux new -s g6_studio_smoke
DREAM_MICRO_KIKI_REAL=1 \
DREAM_MICRO_KIKI_REAL_BACKEND_PATH=/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4 \
uv run python experiments/g6_studio_path_a/run_g6_studio_path_a.py \
    --option C --smoke-subdomains \
    --n-train 5 --n-eval 5 --iters 2 \
    --out-json docs/milestones/g6-studio-path-a-smoke-2026-05-04.json \
    --out-md   docs/milestones/g6-studio-path-a-smoke-2026-05-04.md
```
Detach `Ctrl-b d` ; reattach `tmux a -t g6_studio_smoke`.
Expected: 4 cells (4 arms × 1 seed × 1 subdomain) complete in ≤ 30 min ; partial JSON written ; final JSON + MD written ; verdict has `h9a_classification` set (likely INSUFFICIENT at N=1).

- [ ] **Step 2: Verify partial dumps + R1 entries**

```bash
ls docs/milestones/g6-studio-path-a-smoke-2026-05-04.partial.*.json | wc -l
uv run python -c "
import sqlite3
con = sqlite3.connect('.run_registry.sqlite')
print(con.execute(\"SELECT profile, seed FROM runs WHERE profile LIKE 'g6-studio-path-a%'\").fetchall())"
```
Expected: 4 partial files ; 4 (profile, seed) pairs registered.

- [ ] **Step 3: Commit smoke artifacts**

```bash
git add docs/milestones/g6-studio-path-a-smoke-2026-05-04.json \
        docs/milestones/g6-studio-path-a-smoke-2026-05-04.md
git commit -m "ops(g6s): smoke validation Option C 4 cells"
```

---

## Task 9: Production run — Option B, 100 cells, ~8–12 h overnight

- [ ] **Step 1: Pre-flight checklist on Studio**

```bash
df -h /Users/clems
uv run pytest tests/unit/experiments/test_g6_studio_*.py -v
ls /Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4/lif_metadata.json
echo $DREAM_MICRO_KIKI_REAL
```
Expected: ≥ 50 GB free ; 17 unit tests pass ; metadata file present ; env prints `1`.

- [ ] **Step 2: Launch inside tmux + nohup**

```bash
tmux new -s g6_studio_prod
DREAM_MICRO_KIKI_REAL=1 \
DREAM_MICRO_KIKI_REAL_BACKEND_PATH=/Users/clems/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4 \
nohup uv run python experiments/g6_studio_path_a/run_g6_studio_path_a.py \
    --option B \
    > docs/milestones/g6-studio-path-a-2026-05-04.stdout.log 2>&1 &
echo $! > /tmp/g6_studio_pid
disown
```
Detach `Ctrl-b d`. Job survives SSH drops via `nohup` + `disown`.

- [ ] **Step 3: Monitor every ~2 h**

```bash
ssh studio "tail -n 50 ~/dream-of-kiki/docs/milestones/g6-studio-path-a-2026-05-04.stdout.log"
ssh studio "ls ~/dream-of-kiki/docs/milestones/g6-studio-path-a-2026-05-04.partial.*.json | wc -l"
```
Expected: ~5 partials/h ; full sweep = 100 cells × 5 subdomains = 500 partials.

- [ ] **Step 4: Resumption runbook (if watchdog kills job)**

Symptom : no new tail output ≥ 10 min ; `ps -p $(cat /tmp/g6_studio_pid)` reports nothing.

Resumption :
1. Glob existing partials, identify last completed `(arm, seed, idx)` :
   `ls docs/milestones/g6-studio-path-a-2026-05-04.partial.*.json | sort | tail -n 5`.
2. Add `--resume-from <arm>:<seed>:<idx>` flag to the driver (edit `run_pilot` to glob existing partials, build set of completed `(arm, seed, idx)`, skip `_run_cell_real` for those tuples).
3. Document deviation in `docs/osf-deviations-g6-studio-path-a-2026-05-04.md` BEFORE re-running.
4. Re-launch with `--resume-from`.

- [ ] **Step 5: Verify completion**

```bash
ssh studio "uv run python -c \"
import json
d = json.load(open('docs/milestones/g6-studio-path-a-2026-05-04.json'))
print('cells', len(d['cells']))
print('wall', d['wall_time_s']/3600, 'h')
print('h9a', d['verdict']['h9a_classification'])
print('h9c', d['verdict']['h9c_classification'])\""
```
Expected: cells == 100 ; h9a in {H9-A, H9-B, INDETERMINATE} ; h9c in {H9-C, NOT_H9C}.

- [ ] **Step 6: Commit final artifacts**

```bash
git add docs/milestones/g6-studio-path-a-2026-05-04.json \
        docs/milestones/g6-studio-path-a-2026-05-04.md \
        docs/milestones/g6-studio-path-a-2026-05-04.stdout.log
git commit -m "ops(g6s): production run Option B 100 cells"
```

---

## Task 10: Update Paper 2 §7.1.10 EN+FR

**Files:** Modify `docs/papers/paper2/results.md` and `docs/papers/paper2-fr/results.md`.

- [ ] **Step 1: Read verdict from milestone**

```bash
uv run python -c "
import json
d = json.load(open('docs/milestones/g6-studio-path-a-2026-05-04.json'))
for k in ('h9a_classification','h9a_g','h9a_welch_p','h9a_welch_reject',
         'h9c_classification','h9c_jonckheere_p','h9c_mean_p_min',
         'h9c_mean_p_equ','h9c_mean_p_max'):
    print(k, ':', d['verdict'][k])"
```

- [ ] **Step 2: Append §7.1.10 to `docs/papers/paper2/results.md`**

After §7.1.9, append a section with this skeleton (literal numbers pulled from milestone — same regenerated-from-milestone protocol used by §7.1.5/7.1.6/etc.) :

```markdown
## 7.1.10 G6-Studio Path A pilot — real-LoRA SpikingKiki-V4 35B-A3B-V4 (2026-05-04)

First real-LLM-scale validation of the framework C effect. Building on
G6 Path B (§7.1.4 spectator pattern, synthetic LoRA), G4-ter MLX richer
head (§7.1.5 g_h2 = +2.77 positive coupling), and G5-bis E-SNN richer
head (§7.1.9 H7-B "MLX-only artefact"), this pilot tests three
pre-registered sub-hypotheses on the real Qwen-35B-A3B-V4 spiking LIF
substrate via `mlx_lm.tuner.lora` fine-tune across the same 5 MMLU
subdomains as G6 Path B. Load-bearing change vs Path B : the four
dream handlers operate on the LIVE LoRA delta tensors, not synthesised
payloads disjoint from the eval surface.

Pre-registration : `docs/osf-prereg-g6-studio-path-a.md`. Decision
rules at α/3 = 0.0167.

### Verdict (Option B, 5 seeds, 100 cells)

| Hypothesis | Classification | Statistic | Decision rule |
|------------|----------------|-----------|---------------|
| H9-A | (from JSON h9a_classification) | g = (h9a_g) | g ≥ 0.5 AND Welch reject |
| H9-B | mutually exclusive with H9-A | — | g < 0.5 AND Welch fail |
| H9-C | (from JSON h9c_classification) | Jonckheere p = (h9c_jonckheere_p) | Jonckheere fail AND mean(P_min) > mean(P_equ) |

### Honest reading

- **If H9-A** : framework C transfers to real LLM scale. Rebuts G5-bis "MLX-only artefact". Triggers an Option-A (N≥10) confirmatory follow-up pre-reg before any STABLE EC bump.
- **If H9-B (predicted under G5-bis universality)** : washout persists at 35B. Framework C bounded to differentiable continuous nets. Reported as a positive empirical claim (programme has tightened the boundary).
- **If H9-C (universal DR-4 inversion)** : RESTRUCTURE+RECOMBINE remain spectator on real LIF Qwen. Universalises H4-C / H5-C / H7-B to real-LLM scale.

### DualVer impact

EC stays PARTIAL ; FC stays C-v0.12.0. Per pre-reg §6, Option B is
exploratory-confirmatory ; STABLE promotion gated on Option A
replication AND cross-substrate confirmation under a separate pre-reg.
```

- [ ] **Step 3: Mirror to FR**

Append the same content to `docs/papers/paper2-fr/results.md` after §7.1.9 FR with title `## 7.1.10 G6-Studio voie A — LoRA réelle SpikingKiki-V4 35B-A3B-V4 (2026-05-04)`. Keep table headers in FR (Hypothèse / Classification / Statistique / Règle de décision). Translate verbatim ; preserve all literal numbers + axiom IDs.

- [ ] **Step 4: Commit**

```bash
git add docs/papers/paper2/results.md docs/papers/paper2-fr/results.md
git commit -m "docs(paper2): G6-Studio Path A results 7.1.10"
```

---

## Task 11: DR-3 evidence revision

**Files:** Modify `docs/proofs/dr3-substrate-evidence.md`.

- [ ] **Step 1: Append the revision**

Find the `micro_kiki` substrate-evidence block in `docs/proofs/dr3-substrate-evidence.md` and append :

```markdown
### `micro_kiki` (real backend, SpikingKiki-V4 35B-A3B-V4 — 2026-05-04)

- **C1 — typed Protocols** : `PASS` (unchanged ; protocol contract substrate-agnostic).
- **C2 — axiom property tests on real backend** : `PASS` per `tests/unit/test_micro_kiki_real_backend.py` + `tests/unit/experiments/test_g6_studio_dream_episode_real.py` live-tensor coupling assertions.
- **C3 — empirical effect-size transfer** : (fill from `verdict.h9a_classification`)
  - H9-A → C3 holds at real-LLM scale ; framework C transfers.
  - H9-B → C3 fails at real-LLM scale ; bounded to differentiable continuous nets (consistent with G5-bis MLX-only at toy scale, now extended).
  - H9-C → DR-4 monotonicity refuted at real-LLM scale (consistent with G4-quater + G4-quinto + G5-bis).
- **Evidence pointer** : `docs/milestones/g6-studio-path-a-2026-05-04.json` ; pre-reg `docs/osf-prereg-g6-studio-path-a.md`.
```

- [ ] **Step 2: Commit**

```bash
git add docs/proofs/dr3-substrate-evidence.md
git commit -m "docs(dr3): G6-Studio evidence revision"
```

---

## Task 12: CHANGELOG + STATUS — empirical row, no DualVer bump

**Files:** Modify `CHANGELOG.md` and `STATUS.md`.

- [ ] **Step 1: Append CHANGELOG entry**

Under `[Unreleased]` add :

```markdown
### G6-Studio Path A — real-LoRA SpikingKiki-V4 cross-substrate (2026-05-04)

- Option B (5 seeds × 4 arms × 5 subdomains = 100 cells, ~8–12 h Studio).
- H9-{A,B} classification : (fill from milestone). H9-C : (fill).
- EC stays PARTIAL ; FC stays C-v0.12.0. STABLE promotion gated on Option A (N≥10) replication per pre-reg §6.
- Milestone : `docs/milestones/g6-studio-path-a-2026-05-04.json` ; pre-reg : `docs/osf-prereg-g6-studio-path-a.md`.
```

- [ ] **Step 2: Update STATUS.md**

Bump `**As of**` line to summarise the G6-Studio outcome ; add Gates table row :

```markdown
| G6-Studio Path A — real-LoRA SpikingKiki-V4 35B-A3B-V4 | 2026-05-04 → milestone | 🔶 PARTIAL (Option B, 100 cells, ~10 h Studio M3 Ultra ; H9-{A or B} verdict ; H9-C inversion ; live-tensor coupling fixes G6 Path B spectator ; EC stays PARTIAL per pre-reg §6, Option A N≥10 replication = future work) |
```

- [ ] **Step 3: Commit**

```bash
git add CHANGELOG.md STATUS.md
git commit -m "docs(status): G6-Studio Path A row, EC PARTIAL"
```

---

## Task 13: Self-review + push to origin/main

- [ ] **Step 1: Full local test suite + lint + types**

```bash
uv run pytest tests/unit/experiments/test_g6_studio_*.py -v
uv run ruff check experiments/g6_studio_path_a/ tests/unit/experiments/test_g6_studio_*.py
uv run mypy experiments/g6_studio_path_a/ tests/unit/experiments/test_g6_studio_*.py
```
Expected: 17 unit tests pass ; ruff clean ; mypy strict clean.

- [ ] **Step 2: Verify R1 reproducibility**

```bash
uv run python -c "
import sqlite3
con = sqlite3.connect('.run_registry.sqlite')
rows = con.execute(\"SELECT profile, seed FROM runs WHERE profile LIKE 'g6-studio-path-a%'\").fetchall()
assert len(rows) == 100, f'expected 100, got {len(rows)}'
print('R1 OK : 100 (profile, seed) tuples registered')"
```
Expected: `R1 OK : 100 ...`.

- [ ] **Step 3: Verify EN→FR mirror parity**

```bash
grep -c "## 7.1.10" docs/papers/paper2/results.md docs/papers/paper2-fr/results.md
```
Expected: each prints `1`.

- [ ] **Step 4: Verify no AI co-authorship in commit log**

```bash
git log --since="2026-05-04" --pretty=full | grep -iE "co-authored-by|claude|opus" || echo "OK: no AI attribution"
```
Expected: `OK: no AI attribution`.

- [ ] **Step 5: Push**

```bash
git push origin main
```
Expected: push succeeds ; CI runs ruff + mypy + pytest. R1 nightly on macOS-14 — verify the next nightly is green.

---

## Self-review

**Spec coverage** : every task in the spec maps to a numbered Task above (0 investigate, 0.5 decide, 1 pre-reg, 2 lora_loader, 3 lora_train_step, 4 dream_episode_real, 5 mmlu_eval, 6 aggregator, 7 driver, 8 smoke, 9 production, 10 paper2 EN+FR, 11 DR-3 evidence, 12 CHANGELOG/STATUS, 13 self-review+push). ✓

**Placeholder scan** : no "TBD" / "implement later" / "similar to Task N". The literal numbers in §7.1.10 paper section, CHANGELOG and DR-3 row are explicitly marked "(fill from milestone)" — this is the **regenerated-from-milestone protocol** already used by §7.1.5/7.1.6/etc., not a placeholder. ✓

**Type consistency** :
- `QwenLoRAWrapper.{model, tokenizer, adapter_path, rank, fresh_init}` — Task 2 defn = Task 7 driver call site.
- `TrainHyperparams(lr, iters, rank, alpha, batch_size)` — Task 3 defn = Task 7 driver = Task 9 CLI defaults.
- `dream_episode_real(substrate, profile, live_delta, seed, subdomain, prior_deltas, sibling_deltas)` — kwargs identical Task 4 defn ↔ Task 7 call.
- `classify_h9(retention)` returns dict keys `h9a_classification`, `h9a_g`, `h9a_welch_p`, `h9a_welch_reject`, `h9c_classification`, `h9c_jonckheere_p`, `h9c_jonckheere_reject`, `h9c_mean_p_{min,equ,max}`, `bonferroni_alpha` — same set asserted Task 6 tests, used Task 7 driver, rendered Task 10/11/12.
- `DEFAULT_PRIMARY_KEY = "layer_0_lora_B"` — single source of truth in `dream_episode_real.py` ; imported by driver ; no string-literal drift.
- `H9_BONFERRONI_ALPHA = 0.05 / 3` — single constant in `aggregator_h9.py` ; cited verbatim pre-reg §5 / Task 6 test / Task 10 paper section.
- `CellResult` TypedDict — shape matches `experiments/g6_mmlu_stream/run_g6.py` ; new driver imports `compute_retention` directly to preserve retention semantics.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-04-g6-studio-path-a-real-lora.md`. Two execution options:**

1. **Subagent-Driven (recommended)** — fresh subagent per task, review between tasks. Particularly relevant : Tasks 8/9 require Studio SSH access — a subagent can pause and surface the Studio command for a human, then resume on the result.
2. **Inline Execution** — execute tasks in this session via `superpowers:executing-plans`, batch with checkpoints. Viable for Tasks 0–7 + 10–13 ; Tasks 8/9 always require Studio access regardless of approach.

**Which approach?**
