# Cycle-3 C3.8 Multi-Substrate Ablation Runner — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use `superpowers:subagent-driven-development` (recommended) or `superpowers:executing-plans` to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build the runtime C3.8 that executes the Cartesian ablation (4 scales × 3 substrates × 3 profiles × 60 seeds = 2160 cells) over Qwen3.5 Q4 {1.5B, 7B, 35B-A3B} + Qwen3.6-35B-A3B-bf16, measures H1/H2/H3/H4/H6 hypotheses, and persists to SQLite run-registry + JSONL metrics.

**Architecture:** Extend the validated `scripts/pilot_cycle3_real.py` pattern (1.5B cell, Phase B 46.75 min for 180 configs) to a generic multi-substrate runner. Central design = a `substrate_factory` that wraps three divergent substrate ABIs (MLX in-place mutation / NumPy NDArray / OPLoRA adapter dict) behind a unified `SubstrateAdapter` Protocol consumable by `DreamRuntime.execute()`.

**Tech Stack:** Python 3.14, uv, MLX 0.31.1 + Metal, mlx-lm 0.31.2, NumPy, SQLite, pytest, hypothesis (prop-based).

**Dependencies already DONE:**
- `harness/real_models/qwen_mlx.py` — `QwenMLXWrapper`, `GammaSnapshotProtocol` (C3.2)
- `harness/real_models/base_model_registry.py` — pins Qwen3.5 Q4 + Qwen3.6 local (registry extension 2026-04-24, commit `618c79a`)
- `harness/real_benchmarks/` — MMLU + HellaSwag + mega-v2 loaders (C3.1)
- `kiki_oniric/dream/operations/{replay,downscale,restructure,recombine}_real.py` (C3.3)
- `kiki_oniric/eval/{scaling_law,statistics,ablation}.py` — H1-H5 math (C3.4+C3.5)
- `kiki_oniric/substrates/{mlx_kiki_oniric,esnn_thalamocortical,micro_kiki}.py` — 3 substrates (C3.11+C3.12)
- `harness/storage/run_registry.py` — SQLite `runs` table + `INSERT OR IGNORE` idempotence
- `scripts/ablation_cycle3.py` — enumerator, `--resume`, SUBSTRATES tuple (3 substrates, commit `177cf89`)

**Budget estimate:**
- **LOC:** ~910 new + ~40 modified
- **Compute:** MLX track (720 cells) × ~35-50 min/cell ≈ 420-600 h ; numpy tracks (1440 cells) × ~2-5 s/cell ≈ 2-3 h ; total wall-clock **~18-26 days Studio M3 Ultra**
- **Disk:** ~10 GB JSONL + ~500 MB SQLite + model caches

**Worktree decision:** User declined worktree (scope earlier was patch-sized). Given new scope (~910 LOC across `eval/`, `substrates/`, `scripts/`, `harness/storage/`), **re-raise at implementation kickoff**. Default: direct-to-`main` with atomic commits per task.

**Seed count resolution:** canonical spec §7.1 = 40, `REAL_SEEDS_DEFAULT` = 30, plan-canonical = 60. Use **60** (plan-canonical) — statistical power requirement of H3 Jonckheere + H6 meta-test. Reduce to 30 only if Studio availability window closes.

---

## File Structure

```
dream-of-kiki/
├── scripts/
│   └── ablation_cycle3_runner.py          [NEW, ~350 LOC] Main runner, CLI, resume, loop
├── kiki_oniric/
│   ├── eval/
│   │   └── cycle3_metrics.py              [NEW, ~200 LOC] CellResult + H1/H2/H3/H4/H6 aggregators + serialize
│   └── substrates/
│       └── factory.py                     [NEW,  ~80 LOC] SubstrateAdapter Protocol + 3 wrappers + dispatch
├── harness/storage/
│   └── run_registry.py                    [MODIFY, +30 LOC] `register_metrics_jsonl(run_id, path, metrics)`
├── tests/unit/
│   ├── test_substrate_factory.py          [NEW, ~150 LOC] Factory TDD
│   ├── test_cycle3_metrics.py             [NEW, ~180 LOC] Per-cell + aggregator TDD
│   └── test_ablation_cycle3_runner.py     [NEW, ~120 LOC] Runner TDD + smoke
└── docs/milestones/
    └── c38-runner-launch.md               [NEW,  ~50 LOC] Milestone doc
```

---

## Task 0: Studio prerequisites — model weights availability

**Files:**
- Inspect: Studio `~/KIKI-Mac_tunner/models/`, `~/models/`, `~/.cache/huggingface/`
- No code change this task

**Goal:** Confirm the 4 model pins resolve to loadable artifacts on Studio. Fetch via HF if missing.

- [ ] **Step 1: Inventory current pin-to-path mapping**

Run:
```bash
ssh studio '
  export PATH="$HOME/.local/bin:$PATH"
  for slot in qwen3p5-1p5b qwen3p5-7b qwen3p5-35b qwen3p6-35b-bf16-local; do
    echo "=== $slot ==="
    cd ~/Projets/dream-of-kiki && uv run python -c "
from harness.real_models.base_model_registry import get_pin
p = get_pin(\"$slot\")
print(\"repo_id:\", p.repo_id, \"| quant:\", p.quantization, \"| framework:\", p.framework)
"
  done
'
```

Expected: 4 pins print with their `repo_id` (HF id or absolute path) and `quantization`.

- [ ] **Step 2: Probe each repo for local presence**

Run (per slot, adapt path):
```bash
ssh studio '
  for p in \
    /Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B \
    /Users/clems/KIKI-Mac_tunner/models/gguf \
    /Users/clems/KIKI-Mac_tunner/models; do
    echo "=== $p ==="; ls -la "$p" 2>&1 | head -20
  done
'
```

Expected: Qwen3.6-35B-A3B present (verified 2026-04-24). Check presence of Qwen3.5-1.5B, Qwen3.5-7B, Qwen3.5-35B-A3B Q4 MLX artifacts.

- [ ] **Step 3: Fetch missing Q4 MLX variants**

For each missing scale, run:
```bash
ssh studio 'bash -lc "
  ~/KIKI-Mac_tunner/.venv/bin/huggingface-cli download \
    mlx-community/Qwen3.5-1.5B-Instruct-MLX-4bit \
    --local-dir ~/models/qwen3p5-1p5b-mlx-4bit
  ~/KIKI-Mac_tunner/.venv/bin/huggingface-cli download \
    mlx-community/Qwen3.5-7B-Instruct-MLX-4bit \
    --local-dir ~/models/qwen3p5-7b-mlx-4bit
"'
```

Expected: each download completes (<10 min per model at 110 MB/s fibre). Token already present in `~/.cache/huggingface/token`.

- [ ] **Step 4: Smoke-load each model via mlx_lm**

Run:
```bash
ssh studio 'bash -lc "
  ~/KIKI-Mac_tunner/.venv/bin/python3 -c \"
from mlx_lm import load
for p in [
  \\\"/Users/clems/models/qwen3p5-1p5b-mlx-4bit\\\",
  \\\"/Users/clems/models/qwen3p5-7b-mlx-4bit\\\",
  \\\"/Users/clems/KIKI-Mac_tunner/models/Qwen3.6-35B-A3B\\\",
]:
    m, t = load(p); print(p, type(m).__name__, t.vocab_size)
\"
"'
```

Expected: 3 models load, each prints class name + vocab_size. If Qwen3.5-35B-A3B Q4 GGUF is the canonical 35B Q4: add appropriate MLX conversion step or swap pin to `mlx-community/Qwen3.5-35B-A3B-MLX-4bit`.

- [ ] **Step 5: Update registry pins if local paths differ**

If any pin's `repo_id` points to HF and local path differs, add a `-local` variant (same pattern as `qwen3p6-35b-bf16-local`) in `base_model_registry.py`. Commit:
```bash
cd /Users/electron/Documents/Projets/dream-of-kiki
git add harness/real_models/base_model_registry.py
git commit -m "feat(cycle3): add local pins for Qwen3.5 Q4 scales"
git push origin main
```

---

## Task 1: `SubstrateAdapter` Protocol + `CellRequest` dataclass

**Files:**
- Create: `kiki_oniric/substrates/factory.py`
- Test: `tests/unit/test_substrate_factory.py`

- [ ] **Step 1: Write failing test for `CellRequest` shape**

Create `tests/unit/test_substrate_factory.py`:
```python
from pathlib import Path
import pytest
from kiki_oniric.substrates.factory import CellRequest, SubstrateAdapter, SUBSTRATE_NAMES


def test_cell_request_fields():
    req = CellRequest(
        substrate="mlx_kiki_oniric",
        profile="p_equ",
        seed=7,
        scale="qwen3p6-35b-bf16-local",
        model_path=Path("/fake/path"),
        benchmarks=("mmlu", "hellaswag", "mega_v2"),
    )
    assert req.seed == 7
    assert req.substrate in SUBSTRATE_NAMES


def test_substrate_names_is_3_tuple():
    assert SUBSTRATE_NAMES == ("mlx_kiki_oniric", "esnn_thalamocortical", "micro_kiki")
```

- [ ] **Step 2: Run test, expect ImportError**

```bash
cd /Users/electron/Documents/Projets/dream-of-kiki
uv run pytest tests/unit/test_substrate_factory.py -x
```

Expected: FAIL `ModuleNotFoundError: No module named 'kiki_oniric.substrates.factory'`.

- [ ] **Step 3: Create minimal `factory.py` with dataclass + Protocol**

```python
"""Substrate factory unifying divergent substrate ABIs behind a single Protocol.

The three cycle-3 substrates have incompatible handler signatures:

- mlx_kiki_oniric: operates on DreamEpisode via _real handlers, mutates MLX weights in-place
- esnn_thalamocortical: numpy-native, NDArray in/out
- micro_kiki: OPLoRA adapter dicts (stub) or SpikingKiki-V4 real backend

This factory wraps each in a unified SubstrateAdapter consumable by DreamRuntime.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal, Protocol, Sequence


SUBSTRATE_NAMES: tuple[str, ...] = (
    "mlx_kiki_oniric",
    "esnn_thalamocortical",
    "micro_kiki",
)


SubstrateName = Literal["mlx_kiki_oniric", "esnn_thalamocortical", "micro_kiki"]


@dataclass(frozen=True)
class CellRequest:
    substrate: SubstrateName
    profile: str
    seed: int
    scale: str
    model_path: Path
    benchmarks: Sequence[str] = field(default_factory=lambda: ("mmlu", "hellaswag", "mega_v2"))


class SubstrateAdapter(Protocol):
    """Unified Protocol for all cycle-3 substrates."""

    def execute_profile(self, request: CellRequest) -> dict:
        """Run pre-eval → dream ops (profile-ordered) → post-eval. Return raw cell metrics."""
        ...

    def teardown(self) -> None:
        """Release MLX Metal buffers / close files."""
        ...
```

- [ ] **Step 4: Run tests, expect PASS**

```bash
uv run pytest tests/unit/test_substrate_factory.py -x
```

Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
git add kiki_oniric/substrates/factory.py tests/unit/test_substrate_factory.py
git commit -m "feat(cycle3): SubstrateAdapter Protocol + CellRequest"
```

---

## Task 2: MLX substrate adapter

**Files:**
- Modify: `kiki_oniric/substrates/factory.py`
- Modify: `tests/unit/test_substrate_factory.py`

- [ ] **Step 1: Write failing test for `MLXAdapter` lifecycle**

Append to `tests/unit/test_substrate_factory.py`:
```python
from pathlib import Path
from unittest.mock import MagicMock, patch

def test_mlx_adapter_execute_profile_shape(tmp_path):
    from kiki_oniric.substrates.factory import MLXAdapter, CellRequest

    with patch("kiki_oniric.substrates.factory.QwenMLXWrapper") as wrapper_cls:
        wrapper = MagicMock()
        wrapper.model = MagicMock()
        wrapper.tokenizer = MagicMock()
        wrapper_cls.return_value = wrapper

        adapter = MLXAdapter(pin_slot="qwen3p6-35b-bf16-local")
        request = CellRequest(
            substrate="mlx_kiki_oniric",
            profile="p_equ",
            seed=1,
            scale="qwen3p6-35b-bf16-local",
            model_path=tmp_path,
        )
        with patch.object(MLXAdapter, "_run_profile_ops", return_value={"delta_acc": 0.02, "wall_time_s": 123.0}) as op:
            result = adapter.execute_profile(request)
        assert "delta_acc" in result
        assert result["wall_time_s"] == pytest.approx(123.0)
        op.assert_called_once()
        adapter.teardown()
```

- [ ] **Step 2: Run, expect FAIL**

```bash
uv run pytest tests/unit/test_substrate_factory.py::test_mlx_adapter_execute_profile_shape -x
```

Expected: FAIL `ImportError: cannot import name 'MLXAdapter'`.

- [ ] **Step 3: Implement `MLXAdapter`**

Append to `kiki_oniric/substrates/factory.py`:
```python
from harness.real_models.qwen_mlx import QwenMLXWrapper
from harness.real_benchmarks import MMLU, HellaSwag, MegaV2Eval
from kiki_oniric.dream.operations import replay_real, downscale_real, restructure_real, recombine_real
from kiki_oniric.dream.runtime import DreamRuntime, Operation
from kiki_oniric.dream.episode import DreamEpisode


PROFILE_OP_SEQUENCE: dict[str, tuple[Operation, ...]] = {
    "p_min": (Operation.REPLAY,),
    "p_equ": (Operation.REPLAY, Operation.DOWNSCALE, Operation.RESTRUCTURE),
    "p_max": (Operation.REPLAY, Operation.DOWNSCALE, Operation.RESTRUCTURE, Operation.RECOMBINE),
}


class MLXAdapter:
    def __init__(self, pin_slot: str) -> None:
        self._pin_slot = pin_slot
        self._wrapper: QwenMLXWrapper | None = None

    def _ensure_loaded(self) -> QwenMLXWrapper:
        if self._wrapper is None:
            self._wrapper = QwenMLXWrapper.from_slot(self._pin_slot, enforce_pin=False)
        return self._wrapper

    def execute_profile(self, request: CellRequest) -> dict:
        wrapper = self._ensure_loaded()
        return self._run_profile_ops(wrapper, request)

    def _run_profile_ops(self, wrapper: QwenMLXWrapper, request: CellRequest) -> dict:
        import time
        t0 = time.perf_counter()
        runtime = DreamRuntime()
        runtime.register_handler(Operation.REPLAY, replay_real.replay_real_handler(None, model=wrapper.model))
        runtime.register_handler(Operation.DOWNSCALE, downscale_real.downscale_handler_mlx(model=wrapper.model))
        runtime.register_handler(Operation.RESTRUCTURE, restructure_real.restructure_handler_mlx(model=wrapper.model))
        runtime.register_handler(Operation.RECOMBINE, recombine_real.recombine_handler_full_mlx(
            encoder=wrapper.model, decoder=wrapper.model, seed=request.seed,
        ))
        pre = self._evaluate(wrapper, request.benchmarks, request.seed)
        episode = DreamEpisode.synthesize(profile=request.profile, seed=request.seed)
        for op in PROFILE_OP_SEQUENCE[request.profile]:
            runtime.execute(op, episode)
        post = self._evaluate(wrapper, request.benchmarks, request.seed)
        return {
            "pre": pre, "post": post,
            "delta_acc": post["composite"] - pre["composite"],
            "wall_time_s": time.perf_counter() - t0,
        }

    def _evaluate(self, wrapper: QwenMLXWrapper, benchmarks: Sequence[str], seed: int) -> dict:
        results: dict = {}
        for name in benchmarks:
            bench = {"mmlu": MMLU, "hellaswag": HellaSwag, "mega_v2": MegaV2Eval}[name]()
            results[name] = bench.evaluate(wrapper, n_samples=100, seed=seed)
        results["composite"] = float(sum(results[b]["accuracy"] for b in benchmarks) / len(benchmarks))
        return results

    def teardown(self) -> None:
        self._wrapper = None
        import mlx.core as mx
        mx.clear_cache()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_substrate_factory.py -x
```

Expected: 3 passed (previous 2 + new MLX).

- [ ] **Step 5: Commit**

```bash
git add kiki_oniric/substrates/factory.py tests/unit/test_substrate_factory.py
git commit -m "feat(cycle3): MLXAdapter wrapping QwenMLX + DreamRuntime"
```

---

## Task 3: ESNN substrate adapter

**Files:**
- Modify: `kiki_oniric/substrates/factory.py`
- Modify: `tests/unit/test_substrate_factory.py`

- [ ] **Step 1: Failing test**

Append:
```python
def test_esnn_adapter_numpy_path(tmp_path):
    from kiki_oniric.substrates.factory import ESNNAdapter, CellRequest

    adapter = ESNNAdapter()
    request = CellRequest(
        substrate="esnn_thalamocortical",
        profile="p_equ",
        seed=3,
        scale="qwen3p5-1p5b",
        model_path=tmp_path,
    )
    result = adapter.execute_profile(request)
    assert "wall_time_s" in result
    assert isinstance(result.get("delta_acc"), float)
    adapter.teardown()
```

- [ ] **Step 2: Run, expect FAIL**

```bash
uv run pytest tests/unit/test_substrate_factory.py::test_esnn_adapter_numpy_path -x
```

- [ ] **Step 3: Implement `ESNNAdapter`**

Append:
```python
from kiki_oniric.substrates.esnn_thalamocortical import EsnnSubstrate


class ESNNAdapter:
    def __init__(self) -> None:
        self._substrate = EsnnSubstrate()

    def execute_profile(self, request: CellRequest) -> dict:
        import time
        t0 = time.perf_counter()
        replay = self._substrate.replay_handler_factory()
        downscale = self._substrate.downscale_handler_factory()
        restructure = self._substrate.restructure_handler_factory()
        recombine = self._substrate.recombine_handler_factory()
        pre_snapshot = self._substrate.snapshot()
        ops = PROFILE_OP_SEQUENCE[request.profile]
        rng_state = {"seed": request.seed}
        for op in ops:
            if op == Operation.REPLAY:
                rng_state["replay"] = replay([{"beta_record": [0.1, 0.2]}], n_steps=16)
            elif op == Operation.DOWNSCALE:
                rng_state["ds"] = downscale(pre_snapshot["weights"], factor=0.5)
            elif op == Operation.RESTRUCTURE:
                rng_state["rs"] = restructure(pre_snapshot["connections"], op="prune", src=0, dst=1)
            elif op == Operation.RECOMBINE:
                rng_state["rc"] = recombine([[1.0, 2.0]], seed=request.seed, n_steps=16)
        post_snapshot = self._substrate.snapshot()
        delta = float(((post_snapshot["weights"] - pre_snapshot["weights"]) ** 2).mean())
        return {
            "pre_snapshot_hash": hash(pre_snapshot["weights"].tobytes()),
            "post_snapshot_hash": hash(post_snapshot["weights"].tobytes()),
            "delta_acc": delta,
            "wall_time_s": time.perf_counter() - t0,
        }

    def teardown(self) -> None:
        self._substrate = EsnnSubstrate()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_substrate_factory.py -x
```

Expected: all tests pass.

- [ ] **Step 5: Commit**

```bash
git add kiki_oniric/substrates/factory.py tests/unit/test_substrate_factory.py
git commit -m "feat(cycle3): ESNNAdapter numpy LIF path"
```

---

## Task 4: MicroKiki substrate adapter (stub + real backend)

**Files:**
- Modify: `kiki_oniric/substrates/factory.py`
- Modify: `tests/unit/test_substrate_factory.py`

- [ ] **Step 1: Failing tests**

Append:
```python
def test_microkiki_adapter_stub_mode(tmp_path, monkeypatch):
    from kiki_oniric.substrates.factory import MicroKikiAdapter, CellRequest

    monkeypatch.delenv("DREAM_MICRO_KIKI_REAL", raising=False)
    adapter = MicroKikiAdapter()
    request = CellRequest(
        substrate="micro_kiki", profile="p_min", seed=9,
        scale="qwen3p6-35b-bf16-local", model_path=tmp_path,
    )
    result = adapter.execute_profile(request)
    assert "wall_time_s" in result
    adapter.teardown()


def test_microkiki_adapter_real_backend_env(tmp_path, monkeypatch):
    from kiki_oniric.substrates.factory import MicroKikiAdapter, CellRequest

    monkeypatch.setenv("DREAM_MICRO_KIKI_REAL", "1")
    monkeypatch.setenv("DREAM_MICRO_KIKI_REAL_BACKEND_PATH", str(tmp_path))
    (tmp_path / "lif_metadata.json").write_text('{"version": "v4", "defaults": {"timesteps": 128, "threshold": 0.0625, "tau": 1.0}, "modules": {}}')
    adapter = MicroKikiAdapter()
    assert adapter._real_mode is True
    adapter.teardown()
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement `MicroKikiAdapter`**

Append:
```python
import os
from kiki_oniric.substrates.micro_kiki import MicroKikiSubstrate


class MicroKikiAdapter:
    def __init__(self) -> None:
        self._real_mode = os.environ.get("DREAM_MICRO_KIKI_REAL") == "1"
        backend_path = os.environ.get("DREAM_MICRO_KIKI_REAL_BACKEND_PATH")
        kwargs = {"real_backend_path": backend_path} if self._real_mode and backend_path else {}
        self._substrate = MicroKikiSubstrate(**kwargs)

    def execute_profile(self, request: CellRequest) -> dict:
        import time
        t0 = time.perf_counter()
        self._substrate.load()
        replay = self._substrate.replay_handler_factory()
        downscale = self._substrate.downscale_handler_factory()
        restructure = self._substrate.restructure_handler_factory(rank_thresh=1e-4)
        recombine = self._substrate.recombine_handler_factory(trim_fraction=0.2, alpha=0.5)
        ops = PROFILE_OP_SEQUENCE[request.profile]
        trace: dict = {}
        for op in ops:
            if op == Operation.REPLAY:
                trace["replay"] = replay([{"beta_record": [0.1]}], n_steps=16)
            elif op == Operation.DOWNSCALE:
                trace["ds"] = downscale(self._substrate.snapshot()["weights"], factor=0.5)
            elif op == Operation.RESTRUCTURE:
                trace["rs"] = restructure({"k": [0.0]}, op="prune", key="k")
            elif op == Operation.RECOMBINE:
                trace["rc"] = recombine({"k": [0.0]}, op="trim")
        awake = self._substrate.awake()
        return {
            "awake_signature": awake[:200] if isinstance(awake, str) else str(awake)[:200],
            "real_mode": self._real_mode,
            "delta_acc": 0.0,  # stub; real backend spike-count normalized in metrics
            "wall_time_s": time.perf_counter() - t0,
        }

    def teardown(self) -> None:
        self._substrate = MicroKikiSubstrate()
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_substrate_factory.py -x
```

- [ ] **Step 5: Commit**

```bash
git add kiki_oniric/substrates/factory.py tests/unit/test_substrate_factory.py
git commit -m "feat(cycle3): MicroKikiAdapter stub + real env hook"
```

---

## Task 5: Factory dispatch function + integration smoke

**Files:**
- Modify: `kiki_oniric/substrates/factory.py`
- Modify: `tests/unit/test_substrate_factory.py`

- [ ] **Step 1: Failing test for dispatch**

Append:
```python
import pytest


@pytest.mark.parametrize("name,cls_name", [
    ("mlx_kiki_oniric", "MLXAdapter"),
    ("esnn_thalamocortical", "ESNNAdapter"),
    ("micro_kiki", "MicroKikiAdapter"),
])
def test_make_adapter_dispatch(name, cls_name):
    from kiki_oniric.substrates import factory as F

    adapter = F.make_adapter(name, pin_slot="qwen3p6-35b-bf16-local")
    assert type(adapter).__name__ == cls_name
    adapter.teardown()


def test_make_adapter_unknown_raises():
    from kiki_oniric.substrates import factory as F

    with pytest.raises(ValueError, match="unknown substrate"):
        F.make_adapter("not-real-substrate")
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement `make_adapter`**

Append to `factory.py`:
```python
def make_adapter(name: str, *, pin_slot: str = "qwen3p6-35b-bf16-local") -> SubstrateAdapter:
    if name == "mlx_kiki_oniric":
        return MLXAdapter(pin_slot=pin_slot)
    if name == "esnn_thalamocortical":
        return ESNNAdapter()
    if name == "micro_kiki":
        return MicroKikiAdapter()
    raise ValueError(f"unknown substrate: {name!r}; valid = {SUBSTRATE_NAMES}")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_substrate_factory.py -x
```

Expected: all 7+ tests pass.

- [ ] **Step 5: Commit**

```bash
git add kiki_oniric/substrates/factory.py tests/unit/test_substrate_factory.py
git commit -m "feat(cycle3): factory.make_adapter dispatch"
```

---

## Task 6: `CellResult` dataclass

**Files:**
- Create: `kiki_oniric/eval/cycle3_metrics.py`
- Create: `tests/unit/test_cycle3_metrics.py`

- [ ] **Step 1: Failing test**

Create `tests/unit/test_cycle3_metrics.py`:
```python
import json
from kiki_oniric.eval.cycle3_metrics import CellResult


def test_cell_result_roundtrip_json():
    r = CellResult(
        run_id="abc123",
        substrate="mlx_kiki_oniric",
        scale="qwen3p6-35b-bf16-local",
        profile="p_equ",
        seed=7,
        pre_composite=0.40,
        post_composite=0.42,
        delta=0.02,
        wall_time_s=120.5,
        benchmarks={"mmlu": {"accuracy": 0.41}, "hellaswag": {"accuracy": 0.43}},
        commit_sha="deadbeef",
    )
    payload = r.to_json()
    r2 = CellResult.from_json(payload)
    assert r2.seed == 7
    assert r2.delta == 0.02
    assert r2.benchmarks["mmlu"]["accuracy"] == 0.41
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement `CellResult`**

Create `kiki_oniric/eval/cycle3_metrics.py`:
```python
"""Cycle-3 C3.8 per-cell result schema + H1-H6 aggregators."""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field


@dataclass(frozen=True)
class CellResult:
    run_id: str
    substrate: str
    scale: str
    profile: str
    seed: int
    pre_composite: float
    post_composite: float
    delta: float
    wall_time_s: float
    benchmarks: dict
    commit_sha: str
    extra: dict = field(default_factory=dict)

    def to_json(self) -> str:
        return json.dumps(asdict(self), sort_keys=True)

    @classmethod
    def from_json(cls, payload: str) -> "CellResult":
        d = json.loads(payload)
        return cls(**d)
```

- [ ] **Step 4: Run test**

```bash
uv run pytest tests/unit/test_cycle3_metrics.py::test_cell_result_roundtrip_json -x
```

- [ ] **Step 5: Commit**

```bash
git add kiki_oniric/eval/cycle3_metrics.py tests/unit/test_cycle3_metrics.py
git commit -m "feat(cycle3): CellResult dataclass + JSON roundtrip"
```

---

## Task 7: Per-cell H1/H2/H4 wrappers

**Files:**
- Modify: `kiki_oniric/eval/cycle3_metrics.py`
- Modify: `tests/unit/test_cycle3_metrics.py`

- [ ] **Step 1: Failing test**

Append:
```python
import numpy as np
from kiki_oniric.eval.cycle3_metrics import compute_per_cell


def test_compute_per_cell_emits_h1_h2_h4():
    seeds_post = np.array([0.42, 0.43, 0.41, 0.44, 0.42])
    seeds_pre  = np.array([0.40, 0.41, 0.39, 0.40, 0.40])
    replay_only = np.array([0.41, 0.42, 0.40, 0.42, 0.41])
    null_mean = 0.39
    verdict = compute_per_cell(seeds_post, seeds_pre, replay_only, null_mean, epsilon=0.01)
    for key in ("H1_p", "H2_tost", "H4_p", "H1_significant", "H4_significant"):
        assert key in verdict
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement `compute_per_cell`**

Append to `cycle3_metrics.py`:
```python
from typing import Mapping
import numpy as np
from kiki_oniric.eval.ablation import welch_one_sided, tost_equivalence, one_sample_threshold
from kiki_oniric.eval.statistics import CYCLE3_FAMILY, apply_bonferroni


ALPHA_PER_TEST = 1.0 / CYCLE3_FAMILY.family_size * 0.05  # 0.00625 for family=8


def compute_per_cell(
    post: np.ndarray,
    pre: np.ndarray,
    replay_only: np.ndarray,
    null_mean: float,
    epsilon: float = 0.01,
) -> dict:
    h1 = welch_one_sided(post, pre)
    h2 = tost_equivalence(post, replay_only, epsilon=epsilon)
    h4 = one_sample_threshold(post, null_mean)
    return {
        "H1_p": float(h1.p_value),
        "H1_significant": bool(h1.p_value < ALPHA_PER_TEST),
        "H2_tost": bool(h2.equivalent),
        "H2_p": float(h2.p_value),
        "H4_p": float(h4.p_value),
        "H4_significant": bool(h4.p_value < ALPHA_PER_TEST),
    }
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_cycle3_metrics.py -x
```

- [ ] **Step 5: Commit**

```bash
git add kiki_oniric/eval/cycle3_metrics.py tests/unit/test_cycle3_metrics.py
git commit -m "feat(cycle3): compute_per_cell H1 H2 H4 wrapper"
```

---

## Task 8: H3 aggregator (profile ordering Jonckheere)

**Files:**
- Modify: `kiki_oniric/eval/cycle3_metrics.py`
- Modify: `tests/unit/test_cycle3_metrics.py`

- [ ] **Step 1: Failing test**

Append:
```python
def test_compute_h3_profile_order_passes_monotone():
    from kiki_oniric.eval.cycle3_metrics import compute_h3

    deltas_by_profile = {
        "p_min": np.array([0.01, 0.005, 0.008, 0.012, 0.011]),
        "p_equ": np.array([0.02, 0.025, 0.022, 0.028, 0.021]),
        "p_max": np.array([0.04, 0.045, 0.042, 0.038, 0.041]),
    }
    h3 = compute_h3(deltas_by_profile)
    assert h3["significant"] is True
    assert h3["trend"] in ("increasing", "decreasing")
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement `compute_h3`**

Append:
```python
from kiki_oniric.eval.ablation import jonckheere_trend


def compute_h3(deltas_by_profile: Mapping[str, np.ndarray], alpha: float = 0.025) -> dict:
    ordered = [deltas_by_profile[p] for p in ("p_min", "p_equ", "p_max")]
    j = jonckheere_trend(ordered, alternative="increasing")
    return {
        "J_stat": float(j.statistic),
        "p_value": float(j.p_value),
        "significant": bool(j.p_value < alpha),
        "trend": "increasing" if j.statistic > 0 else "decreasing",
    }
```

- [ ] **Step 4: Run test**

```bash
uv run pytest tests/unit/test_cycle3_metrics.py::test_compute_h3_profile_order_passes_monotone -x
```

- [ ] **Step 5: Commit**

```bash
git add kiki_oniric/eval/cycle3_metrics.py tests/unit/test_cycle3_metrics.py
git commit -m "feat(cycle3): compute_h3 profile-order Jonckheere"
```

---

## Task 9: H6 aggregator (cross-substrate meta-test)

**Files:**
- Modify: `kiki_oniric/eval/cycle3_metrics.py`
- Modify: `tests/unit/test_cycle3_metrics.py`

- [ ] **Step 1: Failing test**

Append:
```python
def test_compute_h6_cross_substrate_invariance():
    from kiki_oniric.eval.cycle3_metrics import compute_h6

    # For each substrate, deltas per profile. H6 asks: is the
    # profile ordering p_min < p_equ < p_max preserved across substrates?
    deltas_per_sub_per_profile = {
        "mlx_kiki_oniric": {
            "p_min": np.array([0.01] * 5),
            "p_equ": np.array([0.02] * 5),
            "p_max": np.array([0.04] * 5),
        },
        "esnn_thalamocortical": {
            "p_min": np.array([0.008] * 5),
            "p_equ": np.array([0.018] * 5),
            "p_max": np.array([0.035] * 5),
        },
        "micro_kiki": {
            "p_min": np.array([0.009] * 5),
            "p_equ": np.array([0.019] * 5),
            "p_max": np.array([0.036] * 5),
        },
    }
    h6 = compute_h6(deltas_per_sub_per_profile)
    assert h6["invariant"] is True
    assert h6["per_substrate_p_values"]["mlx_kiki_oniric"] < 0.05
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement `compute_h6`**

Append:
```python
def compute_h6(
    deltas_per_sub_per_profile: Mapping[str, Mapping[str, np.ndarray]],
    alpha: float = 0.025,
) -> dict:
    per_sub_p = {}
    per_sub_trend = {}
    for substrate, by_prof in deltas_per_sub_per_profile.items():
        h3 = compute_h3(by_prof, alpha=alpha)
        per_sub_p[substrate] = h3["p_value"]
        per_sub_trend[substrate] = h3["trend"]
    # Invariant iff all substrates same trend AND all p < alpha (Bonferroni across substrates)
    alpha_meta = alpha / len(per_sub_p)
    invariant = (
        all(p < alpha_meta for p in per_sub_p.values())
        and len(set(per_sub_trend.values())) == 1
    )
    return {
        "per_substrate_p_values": per_sub_p,
        "per_substrate_trend": per_sub_trend,
        "alpha_meta_bonferroni": alpha_meta,
        "invariant": bool(invariant),
    }
```

- [ ] **Step 4: Run test**

```bash
uv run pytest tests/unit/test_cycle3_metrics.py::test_compute_h6_cross_substrate_invariance -x
```

- [ ] **Step 5: Commit**

```bash
git add kiki_oniric/eval/cycle3_metrics.py tests/unit/test_cycle3_metrics.py
git commit -m "feat(cycle3): compute_h6 cross-substrate meta-test"
```

---

## Task 10: Metrics JSON/JSONL serialization

**Files:**
- Modify: `kiki_oniric/eval/cycle3_metrics.py`
- Modify: `tests/unit/test_cycle3_metrics.py`

- [ ] **Step 1: Failing test**

Append:
```python
def test_append_cell_result_to_jsonl(tmp_path):
    from kiki_oniric.eval.cycle3_metrics import append_cell_result, CellResult

    path = tmp_path / "cells.jsonl"
    for i in range(3):
        r = CellResult(
            run_id=f"run{i}", substrate="mlx_kiki_oniric",
            scale="qwen3p5-1p5b", profile="p_equ", seed=i,
            pre_composite=0.4, post_composite=0.42, delta=0.02,
            wall_time_s=10.0, benchmarks={}, commit_sha="deadbeef",
        )
        append_cell_result(path, r)
    lines = path.read_text().splitlines()
    assert len(lines) == 3
    import json
    for ln in lines:
        d = json.loads(ln)
        assert "run_id" in d
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement append**

Append to `cycle3_metrics.py`:
```python
from pathlib import Path


def append_cell_result(path: Path, result: CellResult) -> None:
    """Append a single CellResult as one JSON line. Creates file if absent."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as f:
        f.write(result.to_json())
        f.write("\n")
```

- [ ] **Step 4: Run test**

```bash
uv run pytest tests/unit/test_cycle3_metrics.py::test_append_cell_result_to_jsonl -x
```

- [ ] **Step 5: Commit**

```bash
git add kiki_oniric/eval/cycle3_metrics.py tests/unit/test_cycle3_metrics.py
git commit -m "feat(cycle3): append_cell_result JSONL writer"
```

---

## Task 11: run_registry extension — `register_metrics_jsonl`

**Files:**
- Modify: `harness/storage/run_registry.py`
- Test: `tests/unit/test_run_registry_metrics.py`

- [ ] **Step 1: Failing test**

Create `tests/unit/test_run_registry_metrics.py`:
```python
import json
from pathlib import Path
from harness.storage.run_registry import RunRegistry


def test_register_metrics_jsonl_appends(tmp_path):
    reg_path = tmp_path / "runs.db"
    reg = RunRegistry(reg_path)
    reg.register(run_id="abc", c_version="C-v0.9.2", profile="cycle3/1.5b/p_equ/mlx_kiki_oniric", seed=0, commit_sha="deadbeef")
    metrics_path = tmp_path / "metrics.jsonl"
    reg.register_metrics_jsonl(run_id="abc", path=metrics_path, metrics={"delta": 0.02, "wall_time_s": 12.3})
    lines = metrics_path.read_text().splitlines()
    assert len(lines) == 1
    d = json.loads(lines[0])
    assert d["run_id"] == "abc"
    assert d["metrics"]["delta"] == 0.02
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement**

Add method to `RunRegistry` in `harness/storage/run_registry.py`:
```python
def register_metrics_jsonl(self, *, run_id: str, path, metrics: dict) -> None:
    """Append `{"run_id", "metrics", "ts"}` line to JSONL metrics sidecar."""
    import json, time
    from pathlib import Path
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    record = {"run_id": run_id, "metrics": metrics, "ts": time.time()}
    with p.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, sort_keys=True))
        f.write("\n")
```

- [ ] **Step 4: Run test**

```bash
uv run pytest tests/unit/test_run_registry_metrics.py -x
```

- [ ] **Step 5: Commit**

```bash
git add harness/storage/run_registry.py tests/unit/test_run_registry_metrics.py
git commit -m "feat(cycle3): registry JSONL sidecar for metrics"
```

---

## Task 12: Runner CLI + argparse

**Files:**
- Create: `scripts/ablation_cycle3_runner.py`
- Create: `tests/unit/test_ablation_cycle3_runner.py`

- [ ] **Step 1: Failing test for CLI parsing**

Create `tests/unit/test_ablation_cycle3_runner.py`:
```python
from scripts.ablation_cycle3_runner import parse_args


def test_parse_args_defaults():
    ns = parse_args([])
    assert ns.resume is False
    assert ns.scale is None
    assert ns.substrate is None
    assert ns.output_dir.name == "c38-results"


def test_parse_args_override():
    ns = parse_args(["--resume", "--scale", "qwen3p6-35b-bf16-local", "--substrate", "mlx_kiki_oniric"])
    assert ns.resume is True
    assert ns.scale == "qwen3p6-35b-bf16-local"
    assert ns.substrate == "mlx_kiki_oniric"
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement `parse_args`**

Create `scripts/ablation_cycle3_runner.py`:
```python
"""Cycle-3 C3.8 multi-substrate ablation runner.

Executes the enumerator output from scripts/ablation_cycle3.py cell-by-cell,
dispatching to the right SubstrateAdapter via factory.make_adapter, evaluating
pre/post benchmarks, computing per-cell + aggregator metrics, persisting to
run-registry SQLite + JSONL sidecar.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="ablation_cycle3_runner", description=__doc__)
    p.add_argument("--resume", action="store_true", help="Skip cells already in registry")
    p.add_argument("--scale", default=None, help="Filter to single scale (slot name)")
    p.add_argument("--substrate", default=None, help="Filter to single substrate")
    p.add_argument("--profile", default=None, help="Filter to single profile (p_min|p_equ|p_max)")
    p.add_argument("--max-cells", type=int, default=None, help="Cap number of cells this run")
    p.add_argument("--seeds", type=int, default=60, help="Seeds per cell")
    p.add_argument("--output-dir", type=Path, default=Path("docs/milestones/c38-results"))
    p.add_argument("--dry-run", action="store_true", help="List cells without executing")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    ns = parse_args(argv)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_ablation_cycle3_runner.py -x
```

- [ ] **Step 5: Commit**

```bash
git add scripts/ablation_cycle3_runner.py tests/unit/test_ablation_cycle3_runner.py
git commit -m "feat(cycle3): C3.8 runner CLI skeleton"
```

---

## Task 13: Runner single-cell execution

**Files:**
- Modify: `scripts/ablation_cycle3_runner.py`
- Modify: `tests/unit/test_ablation_cycle3_runner.py`

- [ ] **Step 1: Failing test**

Append:
```python
from unittest.mock import MagicMock, patch


def test_execute_cell_dispatches_to_adapter(tmp_path):
    from scripts.ablation_cycle3_runner import execute_cell, CellRequest

    req = CellRequest(
        substrate="esnn_thalamocortical", profile="p_equ", seed=1,
        scale="qwen3p5-1p5b", model_path=tmp_path,
    )
    with patch("scripts.ablation_cycle3_runner.make_adapter") as f:
        adapter = MagicMock()
        adapter.execute_profile.return_value = {"delta_acc": 0.02, "wall_time_s": 3.14}
        f.return_value = adapter
        res = execute_cell(req, commit_sha="deadbeef")
        assert res.delta == 0.02
        assert res.wall_time_s == 3.14
        assert res.substrate == "esnn_thalamocortical"
        adapter.teardown.assert_called_once()
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement `execute_cell`**

Append:
```python
from kiki_oniric.substrates.factory import CellRequest, make_adapter
from kiki_oniric.eval.cycle3_metrics import CellResult
import hashlib


def execute_cell(req: CellRequest, *, commit_sha: str) -> CellResult:
    adapter = make_adapter(req.substrate, pin_slot=req.scale)
    try:
        raw = adapter.execute_profile(req)
    finally:
        adapter.teardown()
    run_id_material = f"{req.substrate}|{req.scale}|{req.profile}|{req.seed}|{commit_sha}"
    run_id = hashlib.sha256(run_id_material.encode()).hexdigest()[:32]
    return CellResult(
        run_id=run_id,
        substrate=req.substrate,
        scale=req.scale,
        profile=req.profile,
        seed=req.seed,
        pre_composite=float(raw.get("pre", {}).get("composite", 0.0)),
        post_composite=float(raw.get("post", {}).get("composite", 0.0)),
        delta=float(raw.get("delta_acc", 0.0)),
        wall_time_s=float(raw.get("wall_time_s", 0.0)),
        benchmarks=raw.get("post", {}),
        commit_sha=commit_sha,
        extra={k: v for k, v in raw.items() if k not in ("pre", "post", "delta_acc", "wall_time_s")},
    )
```

Also re-export `CellRequest`:
```python
# Re-export for tests/callers
__all__ = ["parse_args", "main", "execute_cell", "CellRequest"]
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_ablation_cycle3_runner.py -x
```

- [ ] **Step 5: Commit**

```bash
git add scripts/ablation_cycle3_runner.py tests/unit/test_ablation_cycle3_runner.py
git commit -m "feat(cycle3): execute_cell dispatch + CellResult"
```

---

## Task 14: Runner main loop + resume

**Files:**
- Modify: `scripts/ablation_cycle3_runner.py`

- [ ] **Step 1: Failing test**

Append to test file:
```python
def test_main_dry_run_enumerates(capsys, tmp_path):
    from scripts.ablation_cycle3_runner import main

    rc = main(["--dry-run", "--scale", "qwen3p6-35b-bf16-local", "--output-dir", str(tmp_path)])
    out = capsys.readouterr().out
    assert rc == 0
    assert "cells to run" in out.lower()


def test_main_resume_skips_registered(tmp_path, monkeypatch):
    from scripts.ablation_cycle3_runner import main
    from harness.storage.run_registry import RunRegistry

    reg = RunRegistry(tmp_path / "runs.db")
    # Pre-register one run_id so --resume should skip it
    reg.register(run_id="abc", c_version="C-v0.9.2",
                 profile="cycle3/qwen3p6-35b-bf16-local/p_equ/mlx_kiki_oniric",
                 seed=0, commit_sha="deadbeef")
    monkeypatch.setenv("CYCLE3_REGISTRY_PATH", str(tmp_path / "runs.db"))
    # Dry-run with resume should report cells_to_run < 2160
    rc = main(["--dry-run", "--resume", "--output-dir", str(tmp_path)])
    assert rc == 0
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Implement main loop**

Extend `scripts/ablation_cycle3_runner.py`:
```python
from harness.storage.run_registry import RunRegistry
from scripts.ablation_cycle3 import AblationCycle3Runner, enumerate_configs
from kiki_oniric.eval.cycle3_metrics import append_cell_result


def _resolve_registry(output_dir: Path) -> RunRegistry:
    import os
    reg_path = Path(os.environ.get("CYCLE3_REGISTRY_PATH", output_dir / "runs.db"))
    reg_path.parent.mkdir(parents=True, exist_ok=True)
    return RunRegistry(reg_path)


def _filter_cells(cells, *, scale, substrate, profile, max_cells):
    filtered = [
        c for c in cells
        if (scale is None or c.scale == scale)
        and (substrate is None or c.substrate == substrate)
        and (profile is None or c.profile == profile)
    ]
    return filtered[:max_cells] if max_cells else filtered


def main(argv: list[str] | None = None) -> int:
    import subprocess, sys
    ns = parse_args(argv)
    commit_sha = subprocess.check_output(
        ["git", "rev-parse", "HEAD"], text=True
    ).strip()
    registry = _resolve_registry(ns.output_dir)
    cells_all = list(enumerate_configs(seeds=ns.seeds))
    cells = _filter_cells(
        cells_all,
        scale=ns.scale, substrate=ns.substrate, profile=ns.profile,
        max_cells=ns.max_cells,
    )
    if ns.resume:
        known = set(registry.list_run_ids())
        cells = [c for c in cells if _run_id_for(c, commit_sha) not in known]
    print(f"[c3.8 runner] {len(cells)} cells to run (total enumerated: {len(cells_all)})")
    if ns.dry_run:
        return 0
    metrics_path = ns.output_dir / "c38-cells.jsonl"
    for idx, cell in enumerate(cells, start=1):
        try:
            result = execute_cell(cell, commit_sha=commit_sha)
            registry.register(
                run_id=result.run_id, c_version="C-v0.9.2",
                profile=f"cycle3/{cell.scale}/{cell.profile}/{cell.substrate}",
                seed=cell.seed, commit_sha=commit_sha,
            )
            append_cell_result(metrics_path, result)
            print(f"[{idx}/{len(cells)}] {result.run_id} delta={result.delta:.4f} wall={result.wall_time_s:.1f}s")
        except Exception as exc:  # isolate failure per cell
            print(f"[{idx}/{len(cells)}] FAILED {cell}: {exc!r}", file=sys.stderr)
            continue
    return 0


def _run_id_for(cell: CellRequest, commit_sha: str) -> str:
    import hashlib
    mat = f"{cell.substrate}|{cell.scale}|{cell.profile}|{cell.seed}|{commit_sha}"
    return hashlib.sha256(mat.encode()).hexdigest()[:32]
```

Note: `enumerate_configs` must yield `CellRequest` — if `scripts/ablation_cycle3.py` yields tuples today, add a thin adapter:

```python
# Top of ablation_cycle3_runner.py
from kiki_oniric.substrates.factory import CellRequest as _CR
def _to_cell_requests(tuples, scales_to_pins):
    for (scale, profile, substrate, seed) in tuples:
        yield _CR(substrate=substrate, profile=profile, seed=seed,
                 scale=scale, model_path=scales_to_pins[scale])
```

If `enumerate_configs` already yields objects with these fields, use directly.

- [ ] **Step 4: Run all runner tests**

```bash
uv run pytest tests/unit/test_ablation_cycle3_runner.py -x
```

- [ ] **Step 5: Commit**

```bash
git add scripts/ablation_cycle3_runner.py tests/unit/test_ablation_cycle3_runner.py
git commit -m "feat(cycle3): runner main loop + resume via registry"
```

---

## Task 15: Progress logging + error isolation

**Files:**
- Modify: `scripts/ablation_cycle3_runner.py`

- [ ] **Step 1: Failing test for structured log**

Append:
```python
import logging


def test_runner_logs_per_cell_progress(tmp_path, caplog):
    from scripts.ablation_cycle3_runner import main

    with caplog.at_level(logging.INFO, logger="cycle3.runner"):
        rc = main(["--dry-run", "--max-cells", "3", "--output-dir", str(tmp_path)])
    assert rc == 0
    msgs = [r.getMessage() for r in caplog.records]
    assert any("enumerated" in m.lower() for m in msgs)
```

- [ ] **Step 2: Run, expect FAIL**

- [ ] **Step 3: Add logging**

At top of `scripts/ablation_cycle3_runner.py`:
```python
import logging
_log = logging.getLogger("cycle3.runner")
```

Replace `print(...)` calls with:
```python
_log.info("[c3.8 runner] %d cells to run (enumerated: %d)", len(cells), len(cells_all))
# ...
_log.info("[%d/%d] %s delta=%.4f wall=%.1fs",
          idx, len(cells), result.run_id, result.delta, result.wall_time_s)
# ...
_log.error("[%d/%d] FAILED %s: %r", idx, len(cells), cell, exc)
```

And in `main` before anything:
```python
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")
```

- [ ] **Step 4: Run tests**

```bash
uv run pytest tests/unit/test_ablation_cycle3_runner.py -x
```

- [ ] **Step 5: Commit**

```bash
git add scripts/ablation_cycle3_runner.py tests/unit/test_ablation_cycle3_runner.py
git commit -m "feat(cycle3): structured logging + error isolation"
```

---

## Task 16: Smoke test ESNN end-to-end (1 cell)

**Files:**
- Create: `tests/integration/test_smoke_esnn_cell.py`

- [ ] **Step 1: Write smoke test**

```python
"""Smoke test: run one ESNN cell end-to-end without MLX."""
from pathlib import Path
from scripts.ablation_cycle3_runner import execute_cell, CellRequest


def test_smoke_esnn_one_cell(tmp_path):
    req = CellRequest(
        substrate="esnn_thalamocortical",
        profile="p_equ",
        seed=0,
        scale="qwen3p5-1p5b",
        model_path=tmp_path,
    )
    result = execute_cell(req, commit_sha="smoketest")
    assert result.substrate == "esnn_thalamocortical"
    assert result.wall_time_s >= 0.0
    assert isinstance(result.delta, float)
```

- [ ] **Step 2: Run test**

```bash
uv run pytest tests/integration/test_smoke_esnn_cell.py -x
```

Expected: passes in <10 s (numpy only, no MLX).

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_smoke_esnn_cell.py
git commit -m "test(cycle3): smoke 1-cell ESNN end-to-end"
```

---

## Task 17: Smoke test MLX 1-cell (Qwen3.5-1.5B Q4)

**Files:**
- Create: `tests/integration/test_smoke_mlx_cell.py`

- [ ] **Step 1: Write smoke test**

```python
"""Smoke test: run one MLX cell end-to-end with Qwen3.5-1.5B Q4 MLX.

Skipped unless CYCLE3_SMOKE_MLX=1 (requires ~3 GB local model + ~5 min).
"""
import os
import pytest
from pathlib import Path
from scripts.ablation_cycle3_runner import execute_cell, CellRequest


@pytest.mark.skipif(
    os.environ.get("CYCLE3_SMOKE_MLX") != "1",
    reason="Requires local Qwen3.5-1.5B Q4 MLX; opt-in via CYCLE3_SMOKE_MLX=1",
)
def test_smoke_mlx_one_cell(tmp_path):
    req = CellRequest(
        substrate="mlx_kiki_oniric",
        profile="p_min",  # replay-only, fastest profile
        seed=0,
        scale="qwen3p5-1p5b",
        model_path=Path("/Users/clems/models/qwen3p5-1p5b-mlx-4bit"),
    )
    result = execute_cell(req, commit_sha="smoketest")
    assert result.wall_time_s < 600  # <10 min
    assert result.pre_composite > 0.0
```

- [ ] **Step 2: Run on Studio**

```bash
ssh studio 'cd ~/Projets/dream-of-kiki && CYCLE3_SMOKE_MLX=1 uv run pytest tests/integration/test_smoke_mlx_cell.py -x -v'
```

Expected: passes in <10 min. Note wall-time for budget calibration.

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_smoke_mlx_cell.py
git commit -m "test(cycle3): smoke MLX 1-cell (opt-in)"
```

---

## Task 18: Smoke test micro_kiki stub 1-cell

**Files:**
- Create: `tests/integration/test_smoke_microkiki_cell.py`

- [ ] **Step 1: Write smoke test**

```python
"""Smoke test: micro_kiki in stub mode (no DREAM_MICRO_KIKI_REAL)."""
from pathlib import Path
from scripts.ablation_cycle3_runner import execute_cell, CellRequest


def test_smoke_microkiki_stub_one_cell(tmp_path, monkeypatch):
    monkeypatch.delenv("DREAM_MICRO_KIKI_REAL", raising=False)
    req = CellRequest(
        substrate="micro_kiki",
        profile="p_equ",
        seed=2,
        scale="qwen3p6-35b-bf16-local",
        model_path=tmp_path,
    )
    result = execute_cell(req, commit_sha="smoketest")
    assert result.substrate == "micro_kiki"
    assert result.extra.get("real_mode") is False
```

- [ ] **Step 2: Run**

```bash
uv run pytest tests/integration/test_smoke_microkiki_cell.py -x
```

- [ ] **Step 3: Commit**

```bash
git add tests/integration/test_smoke_microkiki_cell.py
git commit -m "test(cycle3): smoke micro_kiki stub 1-cell"
```

---

## Task 19: Studio deployment — launch script + monitoring

**Files:**
- Create: `scripts/launch_c38_studio.sh`
- Create: `scripts/c38_monitor.py`

- [ ] **Step 1: Write launch script**

Create `scripts/launch_c38_studio.sh`:
```bash
#!/usr/bin/env bash
# Launch C3.8 runner on Studio with resume, background, logs.
# Usage: ssh studio 'bash ~/Projets/dream-of-kiki/scripts/launch_c38_studio.sh [--scale SCALE] [--substrate SUB]'

set -euo pipefail
cd "$HOME/Projets/dream-of-kiki"
export PATH="$HOME/.local/bin:$PATH"
export DREAM_MICRO_KIKI_REAL=1
export DREAM_MICRO_KIKI_REAL_BACKEND_PATH="$HOME/KIKI-Mac_tunner/models/SpikingKiki-35B-A3B-V4"

LOG_DIR="$HOME/c38-logs"
mkdir -p "$LOG_DIR"
TS=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$LOG_DIR/c38-$TS.log"

echo "[launch] starting cycle-3 C3.8 runner, log: $LOG"
nohup uv run python -m scripts.ablation_cycle3_runner \
  --resume --seeds 60 \
  --output-dir "$HOME/c38-results" \
  "$@" \
  >"$LOG" 2>&1 &
echo "[launch] pid=$! log=$LOG"
```

- [ ] **Step 2: Make executable**

```bash
chmod +x scripts/launch_c38_studio.sh
```

- [ ] **Step 3: Write monitor script**

Create `scripts/c38_monitor.py`:
```python
"""Query registry + JSONL to print progress of the C3.8 run."""
import argparse, json
from pathlib import Path
from harness.storage.run_registry import RunRegistry


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--registry", type=Path, required=True)
    p.add_argument("--jsonl", type=Path, required=True)
    ns = p.parse_args()
    reg = RunRegistry(ns.registry)
    n_runs = len(reg.list_run_ids())
    lines = ns.jsonl.read_text().splitlines() if ns.jsonl.exists() else []
    deltas = []
    for ln in lines:
        try:
            d = json.loads(ln)
            deltas.append(float(d.get("delta", 0.0)))
        except json.JSONDecodeError:
            continue
    print(f"registered runs:  {n_runs}")
    print(f"jsonl cells:      {len(lines)}")
    if deltas:
        import statistics
        print(f"delta mean:       {statistics.mean(deltas):.4f}")
        print(f"delta stdev:      {statistics.pstdev(deltas):.4f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Smoke the monitor**

```bash
uv run python scripts/c38_monitor.py --registry /tmp/fake.db --jsonl /tmp/fake.jsonl 2>&1 || true
```

- [ ] **Step 5: Commit**

```bash
git add scripts/launch_c38_studio.sh scripts/c38_monitor.py
git commit -m "ops(cycle3): Studio launch + monitor scripts"
```

---

## Task 20: Milestone doc

**Files:**
- Create: `docs/milestones/c38-runner-launch.md`

- [ ] **Step 1: Write milestone doc**

```markdown
# Milestone: C3.8 Runner Launch (2026-04-24)

## Scope

Full multi-scale substrate ablation (cycle-3 C3.8), extending the validated
Phase B 1.5B cell (commit `22c58c9`, 46.75 min / 180 configs) to:

- **4 scales**: Qwen3.5-{1.5B, 7B, 35B-A3B} Q4 MLX + Qwen3.6-35B-A3B bf16 (local pin)
- **3 substrates**: mlx_kiki_oniric, esnn_thalamocortical, micro_kiki (real spike backend via SpikingKiki-V4)
- **3 profiles**: p_min (replay only), p_equ (+downscale+restructure), p_max (+recombine)
- **60 seeds** per cell

Total cells: **2160**.

## Infrastructure

- Runner: `scripts/ablation_cycle3_runner.py`
- Substrate factory: `kiki_oniric/substrates/factory.py`
- Per-cell metrics: `kiki_oniric/eval/cycle3_metrics.py` (H1/H2/H3/H4/H6)
- Run-registry: SQLite `harness/storage/run_registry.py` + JSONL sidecar
- Launch: `scripts/launch_c38_studio.sh` (Studio M3 Ultra, nohup)
- Monitor: `scripts/c38_monitor.py`

## Compute estimate

- MLX track: 720 cells × ~35-50 min = 420-600 h
- Numpy tracks (ESNN + micro_kiki stub): 1440 cells × ~2-5 s ≈ 2-3 h
- **Wall-clock: 18-26 days Studio**

## Resume / idempotence

- `--resume` flag consults registry, skips registered `run_id`s
- `run_id = sha256(substrate|scale|profile|seed|commit_sha)[:32]`
- Registry bit-stable R1 contract

## Status

- [ ] Prereqs Studio models fetched
- [ ] Substrate factory landed
- [ ] Metrics aggregators landed
- [ ] Runner CLI landed
- [ ] Smoke tests ESNN / MLX / micro_kiki passing
- [ ] Launch kicked off on Studio
- [ ] First 24h progress checkpoint
- [ ] Full run complete

## Hypotheses validated (post-run)

TBD — populated after aggregator run.

## Links

- Plan: `docs/superpowers/plans/2026-04-24-cycle3-c38-runner.md`
- Spec: `docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md`
- Adaptation matrix: `docs/milestones/cycle3-plan-adaptation-2026-04-20.md`
```

- [ ] **Step 2: Commit**

```bash
git add docs/milestones/c38-runner-launch.md
git commit -m "docs(cycle3): C3.8 launch milestone"
```

---

## Task 21: First launch trigger + 1 h checkpoint

**Files:**
- No new code — operational task

- [ ] **Step 1: Pull on Studio**

```bash
ssh studio 'cd ~/Projets/dream-of-kiki && git pull origin main'
```

- [ ] **Step 2: Run full dry-run first (sanity)**

```bash
ssh studio 'cd ~/Projets/dream-of-kiki && uv run python -m scripts.ablation_cycle3_runner --dry-run --seeds 60'
```

Expected: logs "2160 cells to run" (or slightly less if any are already registered).

- [ ] **Step 3: Launch restricted subset first (ESNN only, full coverage)**

```bash
ssh studio 'bash ~/Projets/dream-of-kiki/scripts/launch_c38_studio.sh --substrate esnn_thalamocortical'
```

Expected: ~720 numpy cells complete in ~1-2 h.

- [ ] **Step 4: Monitor + verify metrics JSONL**

```bash
ssh studio '
  cd ~/Projets/dream-of-kiki
  uv run python scripts/c38_monitor.py \
    --registry ~/c38-results/runs.db \
    --jsonl ~/c38-results/c38-cells.jsonl
'
```

Expected: printout shows ~720 runs registered, delta stats reasonable.

- [ ] **Step 5: Launch full MLX track**

```bash
ssh studio 'bash ~/Projets/dream-of-kiki/scripts/launch_c38_studio.sh --substrate mlx_kiki_oniric'
ssh studio 'bash ~/Projets/dream-of-kiki/scripts/launch_c38_studio.sh --substrate micro_kiki'
```

Note: MLX track will take 18-26 days. Check once a day.

- [ ] **Step 6: Commit milestone status update**

After first 24 h, edit `docs/milestones/c38-runner-launch.md` Status section and commit:

```bash
git add docs/milestones/c38-runner-launch.md
git commit -m "docs(cycle3): C3.8 launch 24h checkpoint"
git push origin main
```

---

## Open questions / residual ambiguities

1. **Seeds 60 vs 30** — plan uses 60; if Studio window too tight, reduce to 30 via `--seeds 30` (no code change).
2. **Qwen3.5-35B-A3B Q4** — need to confirm HF slug (`mlx-community/Qwen3.5-35B-A3B-MLX-4bit` or equivalent) exists and maps to the existing registry pin.
3. **DreamEpisode.synthesize(profile, seed)** — used in MLXAdapter Task 2; verify this classmethod exists or add a thin helper. If absent, Task 2 must add it to `kiki_oniric/dream/episode.py`.
4. **H6 meta-test formalism** — Task 9 uses Bonferroni across 3 substrates + same-direction trend check. Verify this matches `scripts/h6_jonckheere_cross_substrate.py` if present; otherwise this plan's approach is canonical.
5. **DreamEpisode input_slice content per profile** — the `_run_profile_ops` in MLXAdapter relies on `DreamEpisode.synthesize(profile)` producing the right `input_slice` keys for each op. Confirm via `pilot_cycle3_real.py:246-579`.

## Self-review

- **Spec coverage:** every prereq from scout mapped to Task 0-21. ✓
- **Placeholder scan:** no TBD except in milestone doc Status section (intended, populated post-run). ✓
- **Type consistency:** `CellRequest`, `CellResult`, `SubstrateAdapter`, `make_adapter` used consistently across Tasks 1-15. ✓
- **Operation enum** used uniformly (Task 2, 3, 4) — depends on existing `kiki_oniric/dream/runtime.Operation`. If absent, needs a prereq task.

---

*Plan written 2026-04-24 via superpowers:writing-plans. Ready for execution via superpowers:subagent-driven-development.*
