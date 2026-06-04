# Wave 3b M5 — Full bench run Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `mlx_latent_diffusion` substrate runnable as a full 450-cell ablation bench (3 profiles × 30 seeds × 5 CIFAR-100 task-splits) on Studio M3 Ultra, with crash-safe resume, R1 output hashing, and a deterministic milestone dump.

**Architecture:** Three coupled changes — (1) a production CIFAR-100 split loader behind the `smoke=False` branch, sourcing from HuggingFace `datasets`; (2) the diffusion substrate consuming loader-supplied batches instead of synthetic latents; (3) the `ablation_cycle3_diffusion.py` runner gaining a per-cell prod execution loop, `--resume`, and a milestone aggregation step. The bench launch itself is a separate future session (spec §5 pre-flight runbook).

**Tech Stack:** Python 3.12+, `uv`, MLX (Apple Silicon), HuggingFace `datasets`, `pytest`, `RunRegistry` (sqlite).

**Source spec:** `docs/superpowers/specs/2026-05-21-wave3b-m5-bench-design.md` (revised 2026-05-21, §9 audit corrections are authoritative).

**Branch:** create `feat/wave3b-m5-bench` off `main`. Do NOT implement on `main`.

---

## File Structure

| File | Responsibility | Task |
|---|---|---|
| `harness/diffusion_eval/cifar100_split_loader.py` | Prod CIFAR-100 partition + batching behind `smoke=False` | 2 |
| `scripts/ablation_cycle3_diffusion.py` | CLI flags, `_CellRequest`, prod execute loop, resume, milestone call | 1, 4, 6, 8 |
| `kiki_oniric/substrates/mlx_latent_diffusion.py` | `execute_profile` consumes loader batches | 5 |
| `harness/diffusion_eval/milestone.py` | New: milestone aggregation (stats + H1/H2/H4 verdicts + JSON/md) | 8 |
| `tests/unit/diffusion_eval/` | New dir: prod-loader tests | 3 |
| `tests/unit/scripts/test_ablation_cycle3_resume.py` | New: resume-skip test | 7 |
| `conftest.py` (repo root) | Add `tests/unit/diffusion_eval` to `collect_ignore_glob` | 3 |
| `CHANGELOG.md`, `pyproject.toml` | Empirical bullet + SemVer 0.22.1 → 0.22.2 | 9 |

**Pre-existing facts (from the 2026-05-21 audit — do not re-discover):**
- Loader constants: `N_TASKS=5`, `N_CLASSES_PER_TASK=20`, `RAW_FEATURE_DIM=3072`, `PROD_N_TRAIN_PER_TASK=5_000`, `PROD_N_VAL_PER_TASK=1_000`. Helpers `task_classes(task_idx)`, `_derive_task_keys(task_idx, seed)`, `_smoke_batches(...)` already exist.
- `SplitCifar100Batch` is `@dataclass(frozen=True)` with fields `features`, `labels`, `task_idx`.
- `load_split_cifar100(task_idx, batch_size=32, seed=0, *, smoke=False, split="train")` returns an `Iterable[SplitCifar100Batch]`. The `smoke=False` branch raises `FileNotFoundError` at lines 244-252.
- Runner: `DEFAULT_SEEDS = tuple(range(60))` (line 80); hand-rolled `_parse_cli` (lines 275-311) with `--smoke/--resume/--dry-run/--max-runs` only; `_CellRequest` is a plain `@dataclass` with `seed/profile/task_idx`; the `[plan-only]` block is lines 364-378; `_r1_hash_metrics(metrics)` (lines 255-272) is the canonical-hash helper; `enumerate_configs` yields cells; `run_one_cell` materialises the loader.
- `RunRegistry`: `register(c_version, profile, seed, commit_sha) -> run_id` (idempotent), `register_output_hash(run_id, output_hash)`, `get_output_hash(run_id)` (raises `KeyError`), `_compute_run_id(...)`. No `query`/`register_run`.
- `mlx_latent_diffusion.py` `execute_profile(self, request)` builds its own synthetic latents (lines 169-181), returns a metrics dict with keys `replay_rate, downscale_norm, restructure_sum, recombine_rate, delta_acc, wall_time_s, synthetic, profile, seed, substrate, substrate_version`.
- `pyproject.toml` is already at `0.22.1`.

---

### Task 0: Branch setup

**Files:** none (git only)

- [ ] **Step 1: Create the feature branch**

Run:
```bash
cd ~/Documents/Projets/dream-of-kiki
git checkout main
git checkout -b feat/wave3b-m5-bench
git branch --show-current
```
Expected: `feat/wave3b-m5-bench`

---

### Task 1: CLI flags + seed-count reconciliation

**Files:**
- Modify: `scripts/ablation_cycle3_diffusion.py` (`DEFAULT_SEEDS` line 80; `_parse_cli` lines 275-311)
- Test: `tests/unit/scripts/test_ablation_cycle3_resume.py` (created in Task 7 — CLI parsing asserted there)

- [ ] **Step 1: Patch `DEFAULT_SEEDS` to 30**

Read `scripts/ablation_cycle3_diffusion.py` first. Replace line 80:
```python
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(60))
```
with:
```python
# N=30 per Wave 3b plan §4 M5 acceptance criterion (authoritative).
# Spec docs/superpowers/specs/2026-05-21-wave3b-m5-bench-design.md D1.
DEFAULT_SEEDS: tuple[int, ...] = tuple(range(30))
```

- [ ] **Step 2: Extend `_parse_cli` with three flags**

In `_parse_cli`, add parsing for `--num-seeds <int>`, `--task-idx <int>`, `--output <path>`. Follow the existing `--max-runs` integer-parsing pattern. The resulting `opts` dict must gain keys `num_seeds` (int or `None`), `task_idx` (int or `None`), `output` (`str` or `None`). Add this block alongside the existing flag handling (adapt to the exact loop structure you see in the file):

```python
        elif arg == "--num-seeds":
            idx += 1
            value = int(argv[idx])
            if not 0 < value <= 30:
                raise SystemExit("--num-seeds must be in 1..30")
            opts["num_seeds"] = value
        elif arg == "--task-idx":
            idx += 1
            value = int(argv[idx])
            if not 0 <= value < N_TASKS:
                raise SystemExit(f"--task-idx must be in 0..{N_TASKS - 1}")
            opts["task_idx"] = value
        elif arg == "--output":
            idx += 1
            opts["output"] = argv[idx]
```

Initialise the three new keys to `None` in the `opts` dict default (where `smoke/resume/dry_run/max_runs` are initialised).

- [ ] **Step 3: Run the existing runner smoke to confirm no regression**

Run: `uv run python scripts/ablation_cycle3_diffusion.py --smoke`
Expected: exits 0, prints the smoke envelope (1 cell), no traceback.

- [ ] **Step 4: Commit**

```bash
git add scripts/ablation_cycle3_diffusion.py
git commit -m "$(cat <<'EOF'
feat(harness): M5 seed count and CLI flags

DEFAULT_SEEDS to range(30) per plan section 4. Add --num-seeds,
--task-idx, --output to the hand-rolled _parse_cli.
EOF
)"
```

---

### Task 2: CIFAR-100 production loader

**Files:**
- Modify: `harness/diffusion_eval/cifar100_split_loader.py` (replace the `FileNotFoundError` block at lines 244-252; add `_prod_batches` helper)
- Test: `tests/unit/diffusion_eval/test_cifar100_prod_loader.py` (Task 3)

- [ ] **Step 1: Write the failing test (placeholder happy-path)**

This step's test is fully specified in Task 3. Skip to Step 2 — the loader is exercised by Task 3's tests. (Task 2 and Task 3 are paired: implement the loader, then the tests. If using TDD strictly, do Task 3 Step 1-2 first to get a red test, then return here.)

- [ ] **Step 2: Add the `_prod_batches` helper**

Read `cifar100_split_loader.py` fully first (note the exact bodies of `task_classes`, `_derive_task_keys`, `_smoke_batches`, and the imports). Add this helper just below `_smoke_batches`:

```python
def _prod_batches(
    *,
    task_idx: int,
    seed: int,
    batch_size: int,
    split: str,
) -> Iterable["SplitCifar100Batch"]:
    """Real Split-CIFAR-100 batches for one task window.

    Loads ``uoft-cs/cifar100`` from the HuggingFace cache, keeps only
    the 20 fine-label classes of ``task_idx`` (via ``task_classes``),
    remaps labels into the task-local 0..19 range, and yields
    fixed-size batches in a seed-deterministic order.
    """
    import mlx.core as mx
    import numpy as np

    try:
        from datasets import load_dataset
    except ImportError as exc:  # pragma: no cover - env guard
        raise FileNotFoundError(
            "The 'datasets' package is required for the M5 prod "
            "loader. Install with: uv sync --all-extras"
        ) from exc

    hf_split = "train" if split == "train" else "test"
    try:
        dataset = load_dataset("uoft-cs/cifar100", split=hf_split)
    except (FileNotFoundError, ConnectionError, OSError) as exc:
        raise FileNotFoundError(
            "Split-CIFAR-100 prod data not in the HuggingFace cache. "
            "Run: huggingface-cli download uoft-cs/cifar100 "
            "--repo-type dataset"
        ) from exc

    keep_classes = list(task_classes(task_idx))
    class_to_local = {c: i for i, c in enumerate(keep_classes)}

    fine = np.asarray(dataset["fine_label"], dtype=np.int64)
    mask = np.isin(fine, keep_classes)
    rows = np.nonzero(mask)[0]

    cap = PROD_N_TRAIN_PER_TASK if split == "train" else PROD_N_VAL_PER_TASK

    images = dataset.with_format("numpy")["img"]
    feats = np.stack([np.asarray(images[i], dtype=np.uint8) for i in rows])
    feats = feats.reshape(len(rows), RAW_FEATURE_DIM).astype(np.float32) / 255.0
    labels = np.array(
        [class_to_local[int(fine[i])] for i in rows], dtype=np.int32
    )

    feature_key, _ = _derive_task_keys(task_idx, seed)
    perm_key = mx.random.split(feature_key, num=2)[1]
    order = np.asarray(mx.random.permutation(len(rows), key=perm_key))
    order = order[:cap]

    feats = feats[order]
    labels = labels[order]

    for start in range(0, len(order), batch_size):
        stop = start + batch_size
        yield SplitCifar100Batch(
            features=mx.array(feats[start:stop]),
            labels=mx.array(labels[start:stop]),
            task_idx=task_idx,
        )
```

- [ ] **Step 3: Replace the `FileNotFoundError` prod block**

Replace lines 244-252 (the `# Prod path (M5) …` comment + `raise FileNotFoundError(...)`) with:

```python
    # Prod path (M5) — real CIFAR-100 from the HuggingFace cache.
    split_seed = seed if split == "train" else seed + 10_007
    return _prod_batches(
        task_idx=task_idx,
        seed=split_seed,
        batch_size=batch_size,
        split=split,
    )
```

- [ ] **Step 4: Run Task 3 tests**

Run: `uv run pytest tests/unit/diffusion_eval/ -v --no-cov`
Expected: PASS (after Task 3 is implemented). If Task 3 not yet done, defer to Task 3 Step 4.

- [ ] **Step 5: Commit**

```bash
git add harness/diffusion_eval/cifar100_split_loader.py
git commit -m "$(cat <<'EOF'
feat(harness): CIFAR-100 prod loader for M5 bench

Replace the M4 FileNotFoundError stub with _prod_batches: load
uoft-cs/cifar100 from the HF cache, keep the 20-class task window
via task_classes, remap labels to 0..19, yield seed-deterministic
batches. Offline misses raise with the download command.
EOF
)"
```

---

### Task 3: Production loader tests

**Files:**
- Create: `tests/unit/diffusion_eval/__init__.py` (empty)
- Create: `tests/unit/diffusion_eval/test_cifar100_prod_loader.py`
- Modify: `conftest.py` (repo root — add the new test dir to `collect_ignore_glob`)

- [ ] **Step 1: Create the test directory and file**

Create `tests/unit/diffusion_eval/__init__.py` empty. Create `tests/unit/diffusion_eval/test_cifar100_prod_loader.py`:

```python
"""M5 prod CIFAR-100 split loader — happy path, offline, determinism."""

from __future__ import annotations

import pytest

from harness.diffusion_eval.cifar100_split_loader import (
    N_CLASSES_PER_TASK,
    RAW_FEATURE_DIM,
    SplitCifar100Batch,
    load_split_cifar100,
)

pytestmark = pytest.mark.skipif(
    __import__("importlib").util.find_spec("datasets") is None,
    reason="datasets package not installed",
)


def _has_cifar100_cache() -> bool:
    try:
        from datasets import load_dataset

        load_dataset("uoft-cs/cifar100", split="test")
    except Exception:  # noqa: BLE001 - any cache miss disables the test
        return False
    return True


requires_cache = pytest.mark.skipif(
    not _has_cifar100_cache(),
    reason="uoft-cs/cifar100 not in the HuggingFace cache",
)


@requires_cache
def test_prod_loader_happy_path() -> None:
    batches = list(
        load_split_cifar100(task_idx=0, batch_size=32, seed=0,
                             smoke=False, split="val")
    )
    assert batches, "prod loader yielded no batches"
    first = batches[0]
    assert isinstance(first, SplitCifar100Batch)
    assert first.task_idx == 0
    assert first.features.shape[1] == RAW_FEATURE_DIM
    labels = [int(lbl) for b in batches for lbl in b.labels.tolist()]
    assert min(labels) >= 0
    assert max(labels) < N_CLASSES_PER_TASK


@requires_cache
def test_prod_loader_is_seed_deterministic() -> None:
    def _hashes(seed: int) -> list[tuple]:
        return [
            (tuple(b.features.reshape(-1)[:8].tolist()),
             tuple(b.labels.tolist()))
            for b in load_split_cifar100(
                task_idx=1, batch_size=32, seed=seed,
                smoke=False, split="val")
        ]

    assert _hashes(0) == _hashes(0)
    assert _hashes(0) != _hashes(1)


def test_prod_loader_offline_raises_with_hint(monkeypatch) -> None:
    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setenv("HF_HOME", "/tmp/dreamofkiki-m5-empty-cache")
    with pytest.raises(FileNotFoundError, match="huggingface-cli download"):
        list(
            load_split_cifar100(task_idx=0, batch_size=32, seed=0,
                                smoke=False, split="val")
        )
```

- [ ] **Step 2: Run the tests to verify they fail (loader not yet done)**

Run: `uv run pytest tests/unit/diffusion_eval/ -v --no-cov`
Expected: if Task 2 not done — FAIL/ERROR on import or `FileNotFoundError` text mismatch. If Task 2 done — PASS (cache tests may SKIP).

- [ ] **Step 3: Add the test dir to `conftest.py` `collect_ignore_glob`**

Read the repo-root `conftest.py`. Add `"tests/unit/diffusion_eval/*"` to the `collect_ignore_glob` list (the MLX-only-test skip list, per CLAUDE.md PR #28 convention).

- [ ] **Step 4: Run the full narrow scope**

Run: `uv run pytest tests/unit/diffusion_eval/ -v --no-cov`
Expected: PASS — `test_prod_loader_offline_raises_with_hint` always runs; the two `@requires_cache` tests PASS or SKIP depending on cache.

- [ ] **Step 5: Commit**

```bash
git add tests/unit/diffusion_eval/ conftest.py
git commit -m "$(cat <<'EOF'
test(harness): CIFAR-100 prod loader tests

Happy path, seed determinism, offline-raises-with-hint. Add the
new test dir to conftest collect_ignore_glob so Linux CI skips
the MLX-only path.
EOF
)"
```

---

### Task 4: Extend `_CellRequest` with loader batches

**Files:**
- Modify: `scripts/ablation_cycle3_diffusion.py` (`_CellRequest` lines 175-186)

- [ ] **Step 1: Add the `loader_batches` field**

Read the current `_CellRequest`. Replace it with:

```python
@dataclass
class _CellRequest:
    seed: int
    profile: str
    task_idx: int
    loader_batches: tuple = ()
```

`loader_batches` holds a materialised `tuple[SplitCifar100Batch, ...]`; it defaults to `()` so the smoke path and any existing caller stay valid.

- [ ] **Step 2: Run the smoke path to confirm no regression**

Run: `uv run python scripts/ablation_cycle3_diffusion.py --smoke`
Expected: exits 0, no traceback.

- [ ] **Step 3: Commit**

```bash
git add scripts/ablation_cycle3_diffusion.py
git commit -m "$(cat <<'EOF'
feat(harness): carry loader batches on _CellRequest

Add a defaulted loader_batches tuple so the prod execute loop can
hand real CIFAR-100 batches to the substrate.
EOF
)"
```

---

### Task 5: Substrate consumes loader batches

**Files:**
- Modify: `kiki_oniric/substrates/mlx_latent_diffusion.py` (`execute_profile` lines 139-221)
- Test: `tests/unit/test_mlx_latent_diffusion_adapter.py` (existing — extend)

- [ ] **Step 1: Write the failing test**

Append to `tests/unit/test_mlx_latent_diffusion_adapter.py`:

```python
def test_execute_profile_consumes_loader_batches() -> None:
    import mlx.core as mx

    from kiki_oniric.substrates.mlx_latent_diffusion import (
        MLXLatentDiffusionSubstrate,
    )

    class _Req:
        seed = 0
        profile = "p_min"
        task_idx = 0
        loader_batches = (
            type("B", (), {
                "features": mx.zeros((8, 3072)),
                "labels": mx.zeros((8,), dtype=mx.int32),
                "task_idx": 0,
            })(),
        )

    substrate = MLXLatentDiffusionSubstrate()
    metrics = substrate.execute_profile(_Req())
    assert metrics["synthetic"] is False
    assert "replay_rate" in metrics
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_mlx_latent_diffusion_adapter.py::test_execute_profile_consumes_loader_batches -v --no-cov`
Expected: FAIL — `metrics["synthetic"]` is `True`.

- [ ] **Step 3: Make `execute_profile` consume `loader_batches`**

Read `mlx_latent_diffusion.py` lines 139-221. The current code (lines 169-181) builds a synthetic `dataset` list of `mx.random.normal` tensors. Replace that synthetic-dataset construction with a branch:

```python
        loader_batches = getattr(request, "loader_batches", ())
        if loader_batches:
            # Project raw CIFAR-100 features into the latent space
            # with the substrate's encoder; one latent batch per
            # loader batch.
            dataset = [
                self._encode_features(batch.features)
                for batch in loader_batches
            ]
            synthetic = False
        else:
            root = mx.random.key(seed)
            train_root, sample_root, data_root = mx.random.split(root, num=3)
            d_latent = self.config.d_latent
            n_batches = 4
            batch_size = 8
            data_keys = mx.random.split(data_root, num=n_batches)
            dataset = [
                mx.random.normal(shape=(batch_size, d_latent), key=k)
                for k in data_keys
            ]
            synthetic = True
```

Set `metrics["synthetic"] = synthetic` instead of the hard-coded `True`. Add a small `_encode_features` method that pushes `(B, RAW_FEATURE_DIM)` through the substrate's existing `Encoder` (`from kiki_oniric.substrates._diffusion import Encoder`) down to `config.d_latent`:

```python
    def _encode_features(self, features: "mx.array") -> "mx.array":
        """Project raw (B, 3072) CIFAR features to (B, d_latent)."""
        return self._encoder(features)
```

If the substrate does not already hold an `Encoder` instance, construct one in `__init__`/`components()` sized `RAW_FEATURE_DIM -> config.d_latent` and store it as `self._encoder`. Reuse the `_diffusion.Encoder` constructor signature you see in `kiki_oniric/substrates/_diffusion/`.

> If the encoder wiring proves larger than ~40 LOC, report `DONE_WITH_CONCERNS` — the spec §4 budgeted +20..40 LOC here; a bigger change means the substrate's encoder API needs its own task.

- [ ] **Step 4: Run the test to verify it passes**

Run: `uv run pytest tests/unit/test_mlx_latent_diffusion_adapter.py -v --no-cov`
Expected: PASS (all tests in the file).

- [ ] **Step 5: Run the diffusion R1 reproducibility test**

Run: `uv run pytest tests/reproducibility/test_r1_diffusion.py -v --no-cov`
Expected: PASS — the synthetic branch is unchanged, so existing R1 hashes hold.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/substrates/mlx_latent_diffusion.py tests/unit/test_mlx_latent_diffusion_adapter.py
git commit -m "$(cat <<'EOF'
feat(substrate): diffusion consumes loader batches

execute_profile encodes request.loader_batches into latents when
present; falls back to the synthetic driver otherwise. The
synthetic R1 path is byte-unchanged.
EOF
)"
```

---

### Task 6: Prod execute loop + resume

**Files:**
- Modify: `scripts/ablation_cycle3_diffusion.py` (replace `[plan-only]` block lines 364-378; add `_run_prod_grid`)

- [ ] **Step 1: Add the per-cell JSONL sidecar writer + resume helper**

Add these helpers near `_r1_hash_metrics`:

```python
def _resume_skip(registry, run_id: str) -> bool:
    """True if this cell already has a registered output hash."""
    try:
        registry.get_output_hash(run_id)
    except KeyError:
        return False
    return True


def _append_cell_row(jsonl_path: Path, row: dict[str, object]) -> None:
    with jsonl_path.open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(row, sort_keys=True) + "\n")
```

- [ ] **Step 2: Write `_run_prod_grid`**

Add this function (it replaces the body of the `[plan-only]` branch):

```python
def _run_prod_grid(
    configs: "list[_CellRequest]",
    registry,
    commit_sha: str,
    *,
    resume: bool,
    output_path: Path,
) -> int:
    """Execute every prod cell: load, run, hash, register, dump."""
    import resource

    substrate = _build_substrate()
    cells_path = output_path.with_suffix(".cells.jsonl")
    done = 0
    for cfg in configs:
        run_id = registry._compute_run_id(
            HARNESS_VERSION, cfg.profile, cfg.seed, commit_sha
        )
        if resume and _resume_skip(registry, run_id):
            done += 1
            continue

        batches = tuple(
            load_split_cifar100(
                cfg.task_idx, batch_size=32, seed=cfg.seed,
                smoke=False, split="train",
            )
        )
        cell = _CellRequest(
            seed=cfg.seed, profile=cfg.profile,
            task_idx=cfg.task_idx, loader_batches=batches,
        )
        wall_start = time.monotonic()
        metrics = substrate.execute_profile(cell)
        wall_s = time.monotonic() - wall_start

        output_hash = _r1_hash_metrics(metrics)
        registry.register(
            HARNESS_VERSION, cfg.profile, cfg.seed, commit_sha
        )
        registry.register_output_hash(run_id, output_hash)

        peak_rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        peak_rss_mb = int(peak_rss_kb / 1024)  # macOS reports bytes->KB
        _append_cell_row(cells_path, {
            "run_id": run_id,
            "c_version": HARNESS_VERSION,
            "profile": cfg.profile,
            "seed": cfg.seed,
            "task_idx": cfg.task_idx,
            "output_hash": output_hash,
            "replay_rate": metrics["replay_rate"],
            "downscale_norm": metrics["downscale_norm"],
            "restructure_sum": metrics["restructure_sum"],
            "recombine_rate": metrics["recombine_rate"],
            "delta_acc": metrics["delta_acc"],
            "wall_s": wall_s,
            "peak_rss_mb": peak_rss_mb,
            "slow_cell": wall_s > 120.0,
        })
        done += 1
        print(
            f"[m5] {done}/{len(configs)} "
            f"{cfg.profile} seed={cfg.seed} task={cfg.task_idx} "
            f"wall={wall_s:.1f}s hash={output_hash[:12]}"
        )
    return done
```

Notes for the implementer:
- `_build_substrate()` — reuse the existing substrate-construction helper the file already uses in `run_one_cell`; if it is inlined there, extract it. Use a single substrate instance for the whole grid (memory: drop `batches` each iteration — they go out of scope).
- `time` and `json` are already imported in this file (`_r1_hash_metrics` uses `json`); add `import time` if absent.
- `load_split_cifar100` must be imported at module top from `harness.diffusion_eval.cifar100_split_loader`.

- [ ] **Step 3: Replace the `[plan-only]` block**

In `main()`, replace lines 364-378 (the `if not opts["smoke"]:` envelope-only block ending in `return 0`) with:

```python
    if not opts["smoke"]:
        for cfg in configs:
            registry.register(
                c_version=HARNESS_VERSION,
                profile=_registry_profile_tag(cfg),
                seed=cfg.seed,
                commit_sha=commit_sha,
            )
        output_path = Path(
            opts["output"]
            or REPO_ROOT / "docs" / "milestones"
            / "wave3b-bench-pending.json"
        )
        done = _run_prod_grid(
            configs, registry, commit_sha,
            resume=bool(opts["resume"]), output_path=output_path,
        )
        write_milestone(output_path, registry, commit_sha)
        print(f"[m5] complete: {done}/{len(configs)} cells")
        return 0
```

`write_milestone` is delivered in Task 8 — import it from `harness.diffusion_eval.milestone`.

- [ ] **Step 4: Honour `--num-seeds` and `--task-idx` in grid construction**

Where `configs` is built (the `enumerate_configs()` call in `main()`), apply the new flags: if `opts["num_seeds"]` is set, pass `seeds=tuple(range(opts["num_seeds"]))` to `enumerate_configs`; if `opts["task_idx"]` is set, filter `configs` to that task. Show the actual filter:

```python
        if opts["task_idx"] is not None:
            configs = [c for c in configs if c.task_idx == opts["task_idx"]]
```

- [ ] **Step 5: Dry-run the prod grid (no execution)**

Run: `uv run python scripts/ablation_cycle3_diffusion.py --dry-run`
Expected: exits 0, prints the 450-cell envelope, does not execute cells.

- [ ] **Step 6: Commit**

```bash
git add scripts/ablation_cycle3_diffusion.py
git commit -m "$(cat <<'EOF'
feat(harness): M5 prod execute loop with resume

Replace the plan-only early-exit with _run_prod_grid: per-cell
load, execute_profile, _r1_hash_metrics, register_output_hash,
JSONL sidecar. --resume skips cells with a stored hash.
EOF
)"
```

---

### Task 7: Resume-skip test

**Files:**
- Create: `tests/unit/scripts/test_ablation_cycle3_resume.py`

- [ ] **Step 1: Write the test**

```python
"""M5 runner — resume skips cells that already have an output hash."""

from __future__ import annotations

from pathlib import Path

from scripts.ablation_cycle3_diffusion import _resume_skip
from harness.storage.run_registry import RunRegistry


def test_resume_skip_true_when_hash_present(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "r.sqlite")
    run_id = registry.register("C-v0.14.0+PARTIAL", "p_min", 0, "abc123")
    registry.register_output_hash(run_id, "deadbeef" * 8)
    assert _resume_skip(registry, run_id) is True


def test_resume_skip_false_when_hash_absent(tmp_path: Path) -> None:
    registry = RunRegistry(tmp_path / "r.sqlite")
    run_id = registry.register("C-v0.14.0+PARTIAL", "p_min", 1, "abc123")
    assert _resume_skip(registry, run_id) is False


def test_cli_num_seeds_rejects_over_30() -> None:
    from scripts.ablation_cycle3_diffusion import _parse_cli

    try:
        _parse_cli(["--num-seeds", "60"])
    except SystemExit:
        return
    raise AssertionError("--num-seeds 60 should raise SystemExit")
```

- [ ] **Step 2: Run to verify it passes**

Run: `uv run pytest tests/unit/scripts/test_ablation_cycle3_resume.py -v --no-cov`
Expected: PASS (3 tests). If `_parse_cli` is not importable with a list arg, adapt the call to the real signature observed in the file.

- [ ] **Step 3: Commit**

```bash
git add tests/unit/scripts/test_ablation_cycle3_resume.py
git commit -m "$(cat <<'EOF'
test(harness): M5 resume-skip and CLI bound tests
EOF
)"
```

---

### Task 8: Milestone aggregation

**Files:**
- Create: `harness/diffusion_eval/milestone.py`
- Test: `tests/unit/diffusion_eval/test_milestone.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/diffusion_eval/test_milestone.py`:

```python
"""M5 milestone aggregation — deterministic bytes, per-profile stats."""

from __future__ import annotations

import json
from pathlib import Path

from harness.diffusion_eval.milestone import aggregate_cells


def _fake_cells() -> list[dict]:
    cells = []
    for profile in ("p_min", "p_equ", "p_max"):
        for seed in range(30):
            for task in range(5):
                cells.append({
                    "profile": profile, "seed": seed, "task_idx": task,
                    "replay_rate": 0.5 + 0.001 * seed,
                    "downscale_norm": 0.9,
                    "restructure_sum": 1.0,
                    "recombine_rate": 0.3,
                    "delta_acc": 0.1,
                    "wall_s": 40.0, "slow_cell": False,
                })
    return cells


def test_aggregate_is_deterministic() -> None:
    cells = _fake_cells()
    a = json.dumps(aggregate_cells(cells), sort_keys=True)
    b = json.dumps(aggregate_cells(cells), sort_keys=True)
    assert a == b


def test_aggregate_has_per_profile_stats() -> None:
    summary = aggregate_cells(_fake_cells())
    assert set(summary["profiles"]) == {"p_min", "p_equ", "p_max"}
    p_min = summary["profiles"]["p_min"]
    assert p_min["n_cells"] == 150
    assert "replay_rate" in p_min["stats"]
    assert "mean" in p_min["stats"]["replay_rate"]
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/diffusion_eval/test_milestone.py -v --no-cov`
Expected: FAIL — `harness.diffusion_eval.milestone` does not exist.

- [ ] **Step 3: Implement `harness/diffusion_eval/milestone.py`**

```python
"""Wave 3b M5 milestone aggregation.

Reads per-cell records (the JSONL sidecar or a registry scan) and
produces the deterministic ``wave3b-bench-YYYY-MM-DD.json`` summary:
per-profile descriptive stats, H1/H2/H4 verdicts, and the
cross-substrate consistency table.
"""

from __future__ import annotations

import json
import statistics
from pathlib import Path
from typing import Any

_METRIC_KEYS = (
    "replay_rate", "downscale_norm", "restructure_sum",
    "recombine_rate", "delta_acc",
)


def _describe(values: list[float]) -> dict[str, float]:
    n = len(values)
    mean = statistics.fmean(values)
    std = statistics.pstdev(values) if n > 1 else 0.0
    return {"n": n, "mean": mean, "std": std,
            "min": min(values), "max": max(values)}


def aggregate_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """Pure aggregation — deterministic given the same cell list."""
    profiles: dict[str, Any] = {}
    for profile in sorted({c["profile"] for c in cells}):
        rows = [c for c in cells if c["profile"] == profile]
        stats = {
            key: _describe(sorted(float(r[key]) for r in rows))
            for key in _METRIC_KEYS
        }
        profiles[profile] = {"n_cells": len(rows), "stats": stats}
    return {
        "c_version": "C-v0.14.0+PARTIAL",
        "substrate": "mlx_latent_diffusion",
        "total_cells": len(cells),
        "profiles": profiles,
    }


def _hypothesis_verdicts(cells: list[dict[str, Any]]) -> dict[str, Any]:
    """H1/H2/H4 stat-test verdicts over real per-cell metrics.

    Bonferroni alpha = 0.05 / 4 = 0.0125. Uses the same primitives
    as ablation_cycle2._run_h1_h4 but fed real arrays, not the
    synthetic stand-ins (p_max_smoke / hard-coded energy_ratios).
    """
    from kiki_oniric.eval.statistics import (
        one_sample_threshold,
        tost_equivalence,
        welch_one_sided,
    )

    def _col(profile: str, key: str) -> list[float]:
        return [float(c[key]) for c in cells if c["profile"] == profile]

    alpha = 0.0125
    verdicts: dict[str, Any] = {}
    for profile in sorted({c["profile"] for c in cells}):
        h1 = welch_one_sided(
            treatment=_col(profile, "replay_rate"),
            control=_col("p_min", "replay_rate"), alpha=alpha,
        )
        h2 = tost_equivalence(
            treatment=_col(profile, "downscale_norm"),
            control=_col("p_equ", "downscale_norm"),
            epsilon=0.05, alpha=alpha,
        )
        h4 = one_sample_threshold(
            sample=_col(profile, "recombine_rate"),
            threshold=0.0, alpha=alpha,
        )
        verdicts[profile] = {
            "H1_replay": {"test": h1.test_name, "p": h1.p_value,
                          "reject_h0": h1.reject_h0},
            "H2_downscale": {"test": h2.test_name, "p": h2.p_value,
                             "reject_h0": h2.reject_h0},
            "H4_recombine": {"test": h4.test_name, "p": h4.p_value,
                             "reject_h0": h4.reject_h0},
        }
    return verdicts


def write_milestone(output_path: Path, registry: Any,
                    commit_sha: str) -> dict[str, Any]:
    """Read the JSONL sidecar, aggregate, write the JSON + md.

    Deterministic: two runs over the same sidecar produce identical
    bytes (sorted keys, sorted cell order).
    """
    cells_path = Path(output_path).with_suffix(".cells.jsonl")
    cells: list[dict[str, Any]] = []
    if cells_path.exists():
        with cells_path.open(encoding="utf-8") as fh:
            cells = [json.loads(line) for line in fh if line.strip()]
    cells.sort(key=lambda c: (c["profile"], c["seed"], c["task_idx"]))

    summary = aggregate_cells(cells)
    summary["commit_sha"] = commit_sha
    summary["hypotheses"] = _hypothesis_verdicts(cells) if cells else {}

    Path(output_path).write_text(
        json.dumps(summary, sort_keys=True, indent=2) + "\n",
        encoding="utf-8",
    )
    md_path = Path(output_path).with_suffix(".md")
    md_path.write_text(_render_md(summary), encoding="utf-8")
    return summary


def _render_md(summary: dict[str, Any]) -> str:
    lines = [
        "# Wave 3b M5 bench milestone",
        "",
        f"- substrate: `{summary['substrate']}`",
        f"- c_version: `{summary['c_version']}`",
        f"- commit: `{summary.get('commit_sha', 'n/a')}`",
        f"- total cells: {summary['total_cells']}",
        "",
        "## Per-profile cell counts",
        "",
        "| profile | n_cells |",
        "|---|---|",
    ]
    for profile, body in sorted(summary["profiles"].items()):
        lines.append(f"| {profile} | {body['n_cells']} |")
    lines.append("")
    return "\n".join(lines)
```

> If `welch_one_sided` / `tost_equivalence` / `one_sample_threshold` signatures differ from the `ablation_cycle2._run_h1_h4` call sites, match the real signature you find in `kiki_oniric/eval/statistics.py` and report `DONE_WITH_CONCERNS` noting the deviation.

- [ ] **Step 4: Run the tests**

Run: `uv run pytest tests/unit/diffusion_eval/test_milestone.py -v --no-cov`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add harness/diffusion_eval/milestone.py tests/unit/diffusion_eval/test_milestone.py
git commit -m "$(cat <<'EOF'
feat(harness): M5 milestone aggregation

aggregate_cells gives deterministic per-profile descriptive
stats; write_milestone adds H1/H2/H4 verdicts over real per-cell
arrays and emits the JSON + md milestone.
EOF
)"
```

---

### Task 9: Version bump + CHANGELOG + full suite

**Files:**
- Modify: `pyproject.toml`, `CHANGELOG.md`

- [ ] **Step 1: Bump the SemVer alias**

In `pyproject.toml`, change `version = "0.22.1"` to `version = "0.22.2"`. No framework-C FC bump — M5 is empirical-axis only.

- [ ] **Step 2: Add the CHANGELOG bullet**

Under the `[Unreleased]` section of `CHANGELOG.md`, add an `### Added` (or extend the existing one) bullet:

```markdown
- Wave 3b M5: production CIFAR-100 split loader and the
  `ablation_cycle3_diffusion.py` prod execute loop (450-cell grid,
  `--resume`, R1 output hashing, milestone aggregation). EC bench
  only — no framework-C axiom touched.
```

- [ ] **Step 3: Run the full test suite**

Run: `uv run pytest`
Expected: PASS, coverage gate satisfied (macOS Apple Silicon ≥ 90 %). New MLX-only loader tests SKIP on Linux via `collect_ignore_glob`.

- [ ] **Step 4: Lint + type check**

Run: `uv run ruff check . && uv run mypy harness tests`
Expected: clean.

- [ ] **Step 5: Commit**

```bash
git add pyproject.toml CHANGELOG.md
git commit -m "$(cat <<'EOF'
chore: bump SemVer to 0.22.2 for M5 bench

Empirical-axis only; no framework-C formal bump.
EOF
)"
```

---

## Post-implementation: bench launch (separate future session)

The code is bench-ready after Task 9. The actual 6-8 h launch on
Studio M3 Ultra is **out of scope for this plan** — it follows the
spec §5 pre-flight checklist:

1. `git status` clean, commit SHA pinned.
2. `uv sync --all-extras` exits 0 on Studio.
3. `huggingface-cli download uoft-cs/cifar100 --repo-type dataset`.
4. Smoke: `uv run python scripts/ablation_cycle3_diffusion.py --num-seeds 1 --task-idx 0` exits 0 in < 5 min.
5. `tmux new -s wave3b-m5-bench`, then inside:
   `nohup uv run python scripts/ablation_cycle3_diffusion.py --num-seeds 30 --resume --output docs/milestones/wave3b-bench-2026-MM-DD.json 2>&1 | tee logs/wave3b-m5-bench.log`
6. Watch via `ssh studio` + `tail -F logs/wave3b-m5-bench.log`.

Acceptance is spec §7 D7 (450/450 hashes, R1 within-machine,
deterministic milestone bytes, consistency table populated).

---

## Self-Review

**Spec coverage:**
- D1 seed count → Task 1. D2 loader source → Task 2. D3 runner prod path → Tasks 4, 6. D4 launch → Post-implementation section. D5 wall budget → `slow_cell` flag in Task 6 (cell row). D6 dump schema → Task 6 (per-cell JSONL) + Task 8 (aggregate). D7 acceptance → Post-implementation + Task 8 determinism test. §4 touch map → Tasks 1-9 cover every row. §8 conftest glob → Task 3 Step 3.
- Gap noted: D7.3 (R1 within-machine re-run on Studio) and D7.5 (cross-substrate consistency table populated against real `mlx_kiki_oniric` cells) are launch-session checks, not code — covered by the Post-implementation section, not a code task. The cross-substrate *code path* (`_cross_substrate_consistency`) is referenced in the spec but the milestone in Task 8 only emits per-profile verdicts; wiring the live `mlx_kiki_oniric` comparison needs those reference cells in the registry first (spec §9 ambiguity note). **Flagged: if the reference cells are absent at launch, add a +1 h re-run, or extend `milestone.py` with a consistency table in a follow-up task.**

**Placeholder scan:** no "TBD"/"TODO"/"add error handling". Every code step shows the code. `wave3b-bench-2026-MM-DD` is a deliberate run-date placeholder resolved at launch.

**Type consistency:** `_CellRequest.loader_batches` (Task 4) ↔ consumed by `execute_profile` via `getattr(request, "loader_batches", ())` (Task 5) ↔ populated in `_run_prod_grid` (Task 6). `_resume_skip` (Task 6) ↔ tested (Task 7). `aggregate_cells` / `write_milestone` (Task 8) ↔ imported in Task 6 Step 3. `_r1_hash_metrics` is the existing helper, used unchanged.

**Known soft spots (implementer must verify against disk):** the exact `_parse_cli` loop shape, `_smoke_batches` import surface, `_build_substrate`/`run_one_cell` helper extraction, the `_diffusion.Encoder` constructor signature, and the `kiki_oniric.eval.statistics` primitive signatures. Each is called out inline in the relevant task with a `DONE_WITH_CONCERNS` instruction.
