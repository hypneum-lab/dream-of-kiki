# B1a — LoRA Model Abstraction Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a LoRA-adapter model abstraction — frozen base weights plus trainable named A/B adapters — that `replay` (B1b) can run a low-rank gradient step on.

**Architecture:** A new MLX module `kiki_oniric/substrates/micro_kiki/lora_model.py` with two `nn.Module` classes: `LoRALinear` (a single LoRA-adapted linear layer) and `LoRAModel` (a named stack of them). No training, no `WeightUpdate` emission, no runtime wiring — B1a is a standalone, independently testable model component.

**Tech Stack:** Python 3.12, `uv`, MLX (`mlx.core`, `mlx.nn`), pytest, mypy.

**Spec:** `docs/superpowers/specs/2026-05-19-b1a-lora-model-design.md`

**MLX notes for the implementer:** MLX `nn.Module` registers every `mx.array` attribute as a parameter; `module.freeze(recurse=False, keys=[...])` excludes named params from `trainable_parameters()`. Random keys are functional: `mx.random.key(seed)` → key array, `mx.random.split(key, n)` → `n` sub-keys, `mx.random.normal(shape=..., key=...)` / `mx.random.uniform(low=, high=, shape=, key=)`. Follow the repo's `TinyMLP` pattern (`tests/unit/test_replay_op_mlx.py`) for mlx-dynamic `# type: ignore` comments. MLX is a hard dependency (pinned `mlx>=0.31,<0.32`); no `importorskip` needed in `lora_model.py` itself, but the test file uses `pytest.importorskip` like the existing MLX tests.

---

## File Structure

- **Create** `kiki_oniric/substrates/micro_kiki/lora_model.py` — `LoRALinear` + `LoRAModel`.
- **Create** `tests/unit/test_lora_model.py` — unit tests.
- **Modify** `CHANGELOG.md` — `[C-v0.15.0+PARTIAL]` entry.
- **Modify** `pyproject.toml` — version `0.12.0 → 0.13.0`.
- **Modify** `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` — note the new substrate component.

---

## Task 1: `LoRALinear` layer

**Files:**
- Create: `kiki_oniric/substrates/micro_kiki/lora_model.py`
- Test: `tests/unit/test_lora_model.py`

- [ ] **Step 1: Write the failing test**

Create `tests/unit/test_lora_model.py`:

```python
"""Unit tests for the LoRA-adapter model abstraction (B1a)."""
from __future__ import annotations

import pytest

if True:  # mlx is a hard dep; importorskip mirrors the repo's MLX tests
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

from kiki_oniric.substrates.micro_kiki.lora_model import (
    LoRALinear,
    LoRAModel,
)


def test_lora_linear_shapes() -> None:
    layer = LoRALinear(4, 8, rank=2, alpha=4.0, key=mx.random.key(0))
    assert layer.base_weight.shape == (8, 4)
    assert layer.lora_a.shape == (2, 4)
    assert layer.lora_b.shape == (8, 2)
    assert layer.bias.shape == (8,)


def test_lora_linear_b_init_is_zero() -> None:
    layer = LoRALinear(4, 8, rank=2, alpha=4.0, key=mx.random.key(0))
    assert bool(mx.all(layer.lora_b == 0.0).item())


def test_lora_linear_initial_forward_equals_base() -> None:
    layer = LoRALinear(4, 8, rank=2, alpha=4.0, key=mx.random.key(1))
    x = mx.array([[0.1, 0.2, 0.3, 0.4]])
    got = layer(x)
    expected = x @ layer.base_weight.T + layer.bias
    assert bool(mx.allclose(got, expected).item())


def test_lora_linear_nonzero_b_changes_output() -> None:
    layer = LoRALinear(4, 8, rank=2, alpha=4.0, key=mx.random.key(2))
    x = mx.array([[0.1, 0.2, 0.3, 0.4]])
    base_out = layer(x)
    layer.lora_b = mx.ones((8, 2))
    assert not bool(mx.allclose(layer(x), base_out).item())


def test_lora_linear_scale_applied() -> None:
    layer = LoRALinear(3, 3, rank=1, alpha=6.0, key=mx.random.key(3))
    layer.lora_a = mx.ones((1, 3))
    layer.lora_b = mx.ones((3, 1))
    x = mx.array([[1.0, 0.0, 0.0]])
    # delta = scale * (B @ A); scale = alpha/rank = 6.0
    delta = 6.0 * (layer.lora_b @ layer.lora_a)
    expected = x @ (layer.base_weight + delta).T + layer.bias
    assert bool(mx.allclose(layer(x), expected).item())


def test_lora_linear_base_weight_is_frozen() -> None:
    layer = LoRALinear(4, 8, rank=2, alpha=4.0, key=mx.random.key(4))
    trainable = dict(nn.utils.tree_flatten(layer.trainable_parameters()))
    assert "base_weight" not in trainable
    assert "bias" not in trainable
    assert "lora_a" in trainable
    assert "lora_b" in trainable


def test_lora_linear_rejects_nonpositive_rank() -> None:
    with pytest.raises(ValueError, match="rank"):
        LoRALinear(4, 8, rank=0, alpha=4.0, key=mx.random.key(0))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_lora_model.py -v`
Expected: FAIL — `ModuleNotFoundError: kiki_oniric.substrates.micro_kiki.lora_model`

- [ ] **Step 3: Write `LoRALinear`**

Create `kiki_oniric/substrates/micro_kiki/lora_model.py`:

```python
"""LoRA-adapter model abstraction (B1a, issue #15).

A `LoRALinear` layer carries a frozen base weight `W0` and a
trainable low-rank adapter pair `(A, B)`; its effective weight is
`W0 + (alpha/rank) * (B @ A)`. `LoRAModel` stacks named
`LoRALinear` layers and exposes only the adapters as the
trainable surface — so a downstream gradient step (sub-project
B1b: `replay`) touches the adapters and nothing else.

B1a builds the model only. No training, no channel output.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

__all__ = ["LoRALinear", "LoRAModel"]


class LoRALinear(nn.Module):  # type: ignore[misc]  # mlx.nn dynamic
    """A LoRA-adapted linear layer.

    Base weight `W0` (and optional bias) are frozen; the rank-`r`
    adapters `lora_a` (r, in) and `lora_b` (out, r) are trainable.
    Standard LoRA init: `A` seeded-random, `B` zeros — so the
    initial effective weight equals `W0`.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        rank: int,
        alpha: float,
        *,
        bias: bool = True,
        key: mx.array | None = None,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError(f"rank must be positive, got {rank}")
        self.in_features = in_features
        self.out_features = out_features
        self.rank = rank
        self.alpha = alpha
        self.scale = alpha / rank
        self.use_bias = bias

        if key is None:
            key = mx.random.key(0)
        k_base, k_a = mx.random.split(key, 2)

        bound = 1.0 / (in_features ** 0.5)
        self.base_weight = mx.random.uniform(
            low=-bound,
            high=bound,
            shape=(out_features, in_features),
            key=k_base,
        )
        frozen = ["base_weight"]
        if bias:
            self.bias = mx.zeros((out_features,))
            frozen.append("bias")

        self.lora_a = mx.random.normal(
            shape=(rank, in_features), key=k_a
        ) * bound
        self.lora_b = mx.zeros((out_features, rank))

        self.freeze(recurse=False, keys=frozen)

    def __call__(self, x: mx.array) -> mx.array:
        delta = self.scale * (self.lora_b @ self.lora_a)
        y = x @ (self.base_weight + delta).T
        if self.use_bias:
            y = y + self.bias
        return y


class LoRAModel(nn.Module):  # type: ignore[misc]  # mlx.nn dynamic
    """A feed-forward stack of named `LoRALinear` layers.

    `layer_sizes` is the sequence of widths, e.g. `(4, 8, 2)` →
    two layers `layer0` (4→8) and `layer1` (8→2). ReLU is applied
    between layers, not after the last.
    """

    def __init__(
        self,
        layer_sizes: tuple[int, ...],
        rank: int,
        alpha: float,
        *,
        seed: int = 0,
    ) -> None:
        super().__init__()
        if len(layer_sizes) < 2:
            raise ValueError(
                f"layer_sizes needs >= 2 widths, got {layer_sizes}"
            )
        n = len(layer_sizes) - 1
        keys = mx.random.split(mx.random.key(seed), n)
        self.layers = [
            LoRALinear(
                layer_sizes[i],
                layer_sizes[i + 1],
                rank,
                alpha,
                key=keys[i],
            )
            for i in range(n)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        last = len(self.layers) - 1
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < last:
                x = nn.relu(x)
        return x

    def adapter_parameters(self) -> dict[str, mx.array]:
        """Return ONLY the trainable adapter arrays, keyed by
        `layer<i>.lora_a` / `layer<i>.lora_b`. Base weights are
        excluded — this is the surface a B1b gradient step trains.
        """
        out: dict[str, mx.array] = {}
        for i, layer in enumerate(self.layers):
            out[f"layer{i}.lora_a"] = layer.lora_a
            out[f"layer{i}.lora_b"] = layer.lora_b
        return out
```

Note: if MLX 0.31's `freeze` signature differs, adjust so that `trainable_parameters()` excludes `base_weight`/`bias` — that behaviour is what `test_lora_linear_base_weight_is_frozen` verifies.

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_lora_model.py -v`
Expected: PASS — 7 tests (the `LoRAModel` tests come in Task 2; only the `LoRALinear` ones run so far — all 7 listed `test_lora_linear_*` pass).

- [ ] **Step 5: Commit**

```bash
git add kiki_oniric/substrates/micro_kiki/lora_model.py tests/unit/test_lora_model.py
git commit -m "feat: add LoRALinear adapter layer"
```

Commit body:
```
feat: add LoRALinear adapter layer

B1a / issue #15. LoRALinear: a linear layer with a frozen base
weight and a trainable rank-r adapter pair (A random, B zero),
effective weight W0 + (alpha/rank) * B@A. No training yet.
```

---

## Task 2: `LoRAModel` stack

**Files:**
- Modify: `kiki_oniric/substrates/micro_kiki/lora_model.py` (already has `LoRAModel` from Task 1's code block)
- Test: `tests/unit/test_lora_model.py`

Note: Task 1's implementation code block already includes `LoRAModel`. Task 2 adds the tests that exercise it. If `LoRAModel` was not written in Task 1, write it now exactly as shown in Task 1 Step 3.

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_lora_model.py`:

```python
def test_lora_model_forward_shape() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    x = mx.array([[0.1, 0.2, 0.3, 0.4]])
    assert model(x).shape == (1, 2)


def test_lora_model_adapter_parameters_keys() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    params = model.adapter_parameters()
    assert set(params) == {
        "layer0.lora_a",
        "layer0.lora_b",
        "layer1.lora_a",
        "layer1.lora_b",
    }
    assert params["layer0.lora_a"].shape == (2, 4)
    assert params["layer1.lora_b"].shape == (2, 2)


def test_lora_model_adapter_parameters_excludes_base() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    for name in model.adapter_parameters():
        assert "base_weight" not in name
        assert "bias" not in name


def test_lora_model_is_deterministic_under_seed() -> None:
    m1 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=7)
    m2 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=7)
    assert bool(
        mx.allclose(m1.layers[0].base_weight, m2.layers[0].base_weight).item()
    )
    assert bool(
        mx.allclose(m1.layers[1].lora_a, m2.layers[1].lora_a).item()
    )


def test_lora_model_rejects_too_few_sizes() -> None:
    with pytest.raises(ValueError, match="layer_sizes"):
        LoRAModel((4,), rank=2, alpha=4.0, seed=0)
```

- [ ] **Step 2: Run tests to verify they pass (or fail then pass)**

Run: `uv run pytest tests/unit/test_lora_model.py -v`
Expected: if `LoRAModel` was implemented in Task 1, all 12 tests PASS. If not, these 5 FAIL first — then implement `LoRAModel` (Task 1 Step 3 code) and re-run to PASS.

- [ ] **Step 3: Run mypy + the full suite**

Run: `uv run mypy harness tests`
Expected: `Success: no issues found` (the new test file is under `tests/`, so it is type-checked).
Run: `uv run pytest -q`
Expected: full suite passes, 0 failures.

- [ ] **Step 4: Commit**

```bash
git add kiki_oniric/substrates/micro_kiki/lora_model.py tests/unit/test_lora_model.py
git commit -m "feat: add LoRAModel named adapter stack"
```

Commit body:
```
feat: add LoRAModel named adapter stack

B1a / issue #15. LoRAModel stacks named LoRALinear layers with a
ReLU between them and exposes adapter_parameters() — the named
A/B arrays only, the trainable surface B1b will gradient-step.
```

---

## Task 3: Documentation and DualVer sync

**Files:**
- Modify: `CHANGELOG.md`, `pyproject.toml`, `uv.lock`
- Modify: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md`

- [ ] **Step 1: Add the CHANGELOG entry**

Insert at the top of the `CHANGELOG.md` body (above the most recent `[C-v0.14.0+PARTIAL]` entry):

```markdown
## [C-v0.15.0+PARTIAL] — 2026-05-19 — LoRA model abstraction (B1a)

### Formal axis (FC) — MINOR (v0.14.0 → v0.15.0)

- **New module** `kiki_oniric/substrates/micro_kiki/lora_model.py`:
  `LoRALinear` (a linear layer with a frozen base weight and a
  trainable rank-r adapter pair `A`/`B`, effective weight
  `W0 + (alpha/rank) * B@A`) and `LoRAModel` (a named stack of
  `LoRALinear` exposing `adapter_parameters()`).
- Standard LoRA init: `A` seeded-random, `B` zeros — the initial
  effective weight equals the base weight.
- Sub-project B1a of issue #15: the model abstraction `replay`
  (B1b) will gradient-step to emit a low-rank `WeightUpdate`. No
  training, no channel output, no runtime wiring in B1a.

### Empirical axis (EC) — UNCHANGED (PARTIAL)

- New substrate component only; no op, axiom, or empirical claim.
  EC stays `+PARTIAL`.

### Packaging

- `pyproject.toml` version bumped `0.12.0 → 0.13.0`.
```

Match the formatting of the surrounding entries.

- [ ] **Step 2: Bump `pyproject.toml`**

Change `version = "0.12.0"` to `version = "0.13.0"`.

- [ ] **Step 3: Sync the lockfile**

Run: `uv lock`
Expected: `uv.lock` updates the `dreamofkiki` pin to `0.13.0`.

- [ ] **Step 4: Note the component in the framework-C spec**

In `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §4.2 (the replay operation), add one sentence noting the substrate component:

```markdown
The low-rank substrate `replay` operates on is the LoRA-adapter
model `kiki_oniric/substrates/micro_kiki/lora_model.py`
(`LoRALinear` / `LoRAModel`) — frozen base weights, trainable
A/B adapters (B1a, issue #15).
```

Apply the equivalent French sentence at the matching location in `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` (code identifiers stay in original form).

- [ ] **Step 5: Verify**

Run: `uv run pytest -q` — full suite passes (docs/version change, no code touched).
Run: `uv run mypy harness tests` — `Success`.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md pyproject.toml uv.lock docs/specs/2026-04-17-dreamofkiki-framework-C-design.md docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md
git commit -m "docs: sync spec for B1a LoRA model"
```

Commit body:
```
docs: sync spec for B1a LoRA model

B1a / issue #15. Add the C-v0.15.0+PARTIAL changelog entry, note
the LoRA-adapter model component in framework-C spec section 4.2
(EN + FR), and bump the package version to 0.13.0.
```

---

## Task 4: Final verification

**Files:** none — verification only.

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -q`
Expected: all pass, 0 failures, coverage gate met.

- [ ] **Step 2: Type check**

Run: `uv run mypy harness tests`
Expected: `Success: no issues found`.

- [ ] **Step 3: Lint**

Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 4: Confirm working tree is clean**

Run: `git status --short`
Expected: empty (if `tests/reproducibility/golden_hashes.json` shows modified, that is unrelated R1 metadata drift — restore it with `git checkout -- tests/reproducibility/golden_hashes.json`).

- [ ] **Step 5: Update issue #15**

Add a comment on issue #15: B1a complete (the LoRA-adapter model abstraction), B1b (replay does LoRA-only SGD on it and emits the `WeightUpdate`) is now unblocked.

---

## Self-Review

- **Spec coverage:** `LoRALinear` (Task 1) ✓; `LoRAModel` + `adapter_parameters()` (Tasks 1-2) ✓; standard LoRA init `A` random / `B` zero (Task 1, `test_lora_linear_b_init_is_zero` + `test_lora_linear_initial_forward_equals_base`) ✓; `alpha/rank` scaling (Task 1, `test_lora_linear_scale_applied`) ✓; base-weight freezing (Task 1, `test_lora_linear_base_weight_is_frozen`) ✓; explicit seeding (Task 1-2, `key=` / `seed=`, `test_lora_model_is_deterministic_under_seed`) ✓; CHANGELOG + spec + FR + version (Task 3) ✓; verification (Task 4) ✓. All six acceptance criteria from the spec have a covering task.
- **Placeholder scan:** no TBD/TODO; the MLX `freeze`-API caveat in Task 1 Step 3 is an explicit fallback instruction with a concrete acceptance check, not a vague placeholder.
- **Type consistency:** `LoRALinear(in_features, out_features, rank, alpha, *, bias, key)`, `LoRAModel(layer_sizes, rank, alpha, *, seed)`, attributes `base_weight` / `bias` / `lora_a` / `lora_b` / `scale` / `use_bias`, and `adapter_parameters()` keys `layer<i>.lora_a` / `layer<i>.lora_b` are used identically across Tasks 1-3.
