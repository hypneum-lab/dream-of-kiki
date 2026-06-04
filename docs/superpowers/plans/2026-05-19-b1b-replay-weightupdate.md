# B1b — `replay` emits a `WeightUpdate` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `replay` operation run a LoRA-only gradient step on a `LoRAModel` and return a real channel-1 `WeightUpdate`.

**Architecture:** A new `replay_lora_handler` factory in `kiki_oniric/dream/operations/replay_real.py` snapshots the model's named A/B adapters, runs one SGD step (the base weight is frozen by B1a, so MLX trains only the adapters), and returns `WeightUpdate(lora_delta=<per-adapter deltas>, fisher_bump=None)`. `DreamRuntime.execute()` (from B0) already captures the return value into `EpisodeLogEntry.channel_outputs`. The MLX→numpy delta conversion lives in a helper in `lora_model.py` so `replay_real.py` stays MLX-only per the `operations/CLAUDE.md` rule.

**Tech Stack:** Python 3.12, `uv`, MLX (`mlx.core`/`mlx.nn`/`mlx.optimizers`), numpy, pytest, mypy.

**Spec:** `docs/superpowers/specs/2026-05-19-b1b-replay-weightupdate-design.md`

**Convention note:** `operations/CLAUDE.md` says `_real.py` is MLX-only — no `numpy` import. `WeightUpdate.lora_delta` is `dict[str, NDArray[np.float32]]` (numpy). Resolution: the MLX→numpy delta conversion is a helper `adapter_delta()` in `kiki_oniric/substrates/micro_kiki/lora_model.py` (a `substrates/` module where numpy is allowed — `oplora.py`/`ties.py` are pure numpy). `replay_real.py` only does MLX work and *imports the name* `adapter_delta`; it writes no `import numpy`.

**LoRA fact (do not mistake for a bug):** with B1a's standard init `B = 0`, the first SGD step has `∂loss/∂A = scale·Bᵀ·… = 0`, so after one step `ΔA` is exactly zero and only `ΔB` moves. One `replay` episode = one step, so the emitted `lora_delta` has zero `lora_a` deltas and non-zero `lora_b` deltas. Tests assert "at least one adapter moved", not "all".

---

## File Structure

- **Modify** `kiki_oniric/substrates/micro_kiki/lora_model.py` — add the `adapter_delta()` helper (MLX snapshots → numpy delta dict).
- **Modify** `tests/unit/test_lora_model.py` — tests for `adapter_delta()`.
- **Modify** `kiki_oniric/dream/operations/replay_real.py` — add `replay_lora_handler` + `_flop_estimate_lora`.
- **Create** `tests/unit/test_replay_lora.py` — end-to-end handler tests via `DreamRuntime`.
- **Modify** `CHANGELOG.md`, `pyproject.toml`, `uv.lock` — `[C-v0.16.0+PARTIAL]`, version `0.14.0`.
- **Modify** `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` — §4.2 note.

---

## Task 1: `adapter_delta()` helper

**Files:**
- Modify: `kiki_oniric/substrates/micro_kiki/lora_model.py`
- Test: `tests/unit/test_lora_model.py`

- [ ] **Step 1: Write the failing tests**

Append to `tests/unit/test_lora_model.py`:

```python
def test_adapter_delta_computes_float32_difference() -> None:
    before = {"layer0.lora_a": mx.zeros((2, 4))}
    after = {"layer0.lora_a": mx.ones((2, 4))}
    delta = adapter_delta(before, after)
    assert delta["layer0.lora_a"].dtype == np.float32
    assert delta["layer0.lora_a"].shape == (2, 4)
    assert bool((delta["layer0.lora_a"] == 1.0).all())


def test_adapter_delta_rejects_key_mismatch() -> None:
    with pytest.raises(ValueError, match="keys"):
        adapter_delta({"a": mx.zeros((1,))}, {"b": mx.zeros((1,))})
```

Update the test file's imports: it currently imports `numpy as np` already (used by existing tests); add `adapter_delta` to the `from kiki_oniric.substrates.micro_kiki.lora_model import (...)` block alongside `LoRALinear, LoRAModel`.

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_lora_model.py -k adapter_delta -v`
Expected: FAIL — `ImportError: cannot import name 'adapter_delta'`.

- [ ] **Step 3: Implement `adapter_delta()`**

In `kiki_oniric/substrates/micro_kiki/lora_model.py`, add `numpy` imports near the top (after the existing `import mlx.core as mx` / `import mlx.nn as nn`):

```python
import numpy as np
from numpy.typing import NDArray
```

Add `"adapter_delta"` to the module's `__all__` list. Add this function at the end of the module:

```python
def adapter_delta(
    before: dict[str, mx.array],
    after: dict[str, mx.array],
) -> dict[str, NDArray[np.float32]]:
    """Per-adapter delta ``after - before`` as float32 numpy arrays.

    ``before`` / ``after`` are `LoRAModel.adapter_parameters()`
    snapshots taken around a gradient step; their key sets must
    match. The MLX→numpy conversion lives here (a `substrates/`
    module) so MLX-only op modules need not import numpy.
    """
    if before.keys() != after.keys():
        raise ValueError(
            "adapter_delta: before/after adapter keys differ"
        )
    return {
        k: np.asarray(after[k] - before[k], dtype=np.float32)
        for k in before
    }
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_lora_model.py -k adapter_delta -v`
Expected: PASS — 2 tests.

- [ ] **Step 5: Run the full lora_model test file + mypy**

Run: `uv run pytest tests/unit/test_lora_model.py -v` — expect all pass (14 prior + 2 new = 16).
Run: `uv run mypy harness tests` — expect `Success`.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/substrates/micro_kiki/lora_model.py tests/unit/test_lora_model.py
git commit -m "feat: add adapter_delta LoRA snapshot helper"
```

Commit body:
```
feat: add adapter_delta LoRA snapshot helper

B1b / issue #15. adapter_delta() diffs two adapter_parameters()
snapshots into a float32 numpy delta dict. Lives in lora_model
so MLX-only op modules need no numpy import.
```

---

## Task 2: `replay_lora_handler`

**Files:**
- Modify: `kiki_oniric/dream/operations/replay_real.py`
- Test: `tests/unit/test_replay_lora.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/unit/test_replay_lora.py`:

```python
"""Unit tests for the LoRA replay handler (B1b)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import mlx.core as mx
else:
    mx = pytest.importorskip("mlx.core")

from kiki_oniric.dream.channels import WeightUpdate
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.replay_real import (
    ReplayRealState,
    replay_lora_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime
from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


def _records() -> list[dict[str, list[float]]]:
    return [
        {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
        {"x": [0.5, 0.6, 0.7, 0.8], "y": [0.0, 1.0]},
    ]


def _episode(records: list[dict[str, list[float]]]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"beta_records": records},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-lora",
    )


def _run(model: LoRAModel, records: list[dict[str, list[float]]]):
    state = ReplayRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.REPLAY,
        replay_lora_handler(state, model=model, lr=0.05),
    )
    runtime.execute(_episode(records))
    return state, runtime.log[-1]


def test_replay_lora_emits_weight_update() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _, entry = _run(model, _records())
    assert isinstance(entry.channel_outputs[0], WeightUpdate)


def test_replay_lora_delta_keys_match_adapters() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _, entry = _run(model, _records())
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    assert set(out.lora_delta) == set(model.adapter_parameters())


def test_replay_lora_delta_is_finite_float32() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _, entry = _run(model, _records())
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    for arr in out.lora_delta.values():
        assert arr.dtype == np.float32
        assert bool(np.isfinite(arr).all())
    # B is zero-init, so the first step moves lora_b but not lora_a;
    # at least one adapter must have changed.
    assert any(bool(np.any(arr != 0.0)) for arr in out.lora_delta.values())


def test_replay_lora_deltas_match_adapter_transition() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    before = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    _, entry = _run(model, _records())
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    after = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    for k in before:
        np.testing.assert_allclose(
            before[k] + out.lora_delta[k], after[k],
            rtol=1e-5, atol=1e-6,
        )


def test_replay_lora_composed_effective_delta() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    layer0 = model.layers[0]
    a_before = np.asarray(layer0.lora_a, dtype=np.float32)
    b_before = np.asarray(layer0.lora_b, dtype=np.float32)
    scale = layer0.scale
    _, entry = _run(model, _records())
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    a_after = a_before + out.lora_delta["layer0.lora_a"]
    b_after = b_before + out.lora_delta["layer0.lora_b"]
    # Recomposing the emitted low-rank deltas reproduces the dense
    # effective-weight change the model actually underwent.
    composed = scale * (b_after @ a_after - b_before @ a_before)
    actual = scale * (
        np.asarray(model.layers[0].lora_b, dtype=np.float32)
        @ np.asarray(model.layers[0].lora_a, dtype=np.float32)
        - b_before @ a_before
    )
    np.testing.assert_allclose(composed, actual, rtol=1e-5, atol=1e-6)


def test_replay_lora_empty_records_returns_none() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    state, entry = _run(model, [])
    assert entry.channel_outputs[0] is None
    assert state.last_compute_flops == 0
    assert state.last_loss is None


def test_replay_lora_tags_k1_flops() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    state, _ = _run(model, _records())
    assert state.last_compute_flops > 0
    assert state.total_compute_flops == state.last_compute_flops
    assert state.total_records_consumed == 2


def test_replay_lora_is_deterministic() -> None:
    m1 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=3)
    m2 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=3)
    _, e1 = _run(m1, _records())
    _, e2 = _run(m2, _records())
    o1, o2 = e1.channel_outputs[0], e2.channel_outputs[0]
    assert isinstance(o1, WeightUpdate) and isinstance(o2, WeightUpdate)
    for k in o1.lora_delta:
        np.testing.assert_array_equal(o1.lora_delta[k], o2.lora_delta[k])


def test_replay_lora_rejects_malformed_record() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    state = ReplayRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.REPLAY, replay_lora_handler(state, model=model),
    )
    with pytest.raises(ValueError, match="missing"):
        runtime.execute(_episode([{"x": [0.1, 0.2, 0.3, 0.4]}]))
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_replay_lora.py -v`
Expected: FAIL — `ImportError: cannot import name 'replay_lora_handler'`.

- [ ] **Step 3: Implement the handler in `replay_real.py`**

In `kiki_oniric/dream/operations/replay_real.py`, add `_flop_estimate_lora` after the existing `_flop_estimate`:

```python
def _flop_estimate_lora(model, n_records: int) -> int:
    """Rough FLOP count for a LoRA-adapter replay step.

    Dominated by the per-layer low-rank product ``B @ A`` (forward +
    backward) over ``n_records``. Smaller than a full-weight step.
    """
    per_record = sum(
        2 * layer.rank * layer.in_features * layer.out_features
        for layer in model.layers
    )
    return max(per_record * n_records, 1)
```

Add the handler factory after `replay_real_handler`:

```python
def replay_lora_handler(
    state: ReplayRealState,
    *,
    model,  # LoRAModel — typed loosely for lazy MLX import
    lr: float = 0.01,
) -> Callable[[DreamEpisode], "WeightUpdate | None"]:
    """Build a LoRA-only replay handler that emits a ``WeightUpdate``.

    Runs one SGD step on ``model``'s adapters — the base weight is
    frozen by ``LoRALinear``, so MLX excludes it from the gradient
    tree and only the A/B adapters move. Returns the per-adapter
    low-rank delta as a channel-1 ``WeightUpdate``; empty
    ``beta_records`` returns ``None`` (S1 no-op).

    Reference:
      docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
    """
    import mlx.core as mx
    import mlx.nn as nn
    import mlx.optimizers as optim

    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.substrates.micro_kiki.lora_model import adapter_delta

    optimizer = optim.SGD(learning_rate=lr)

    def loss_fn(model_inner, x, y):
        pred = model_inner(x)
        return mx.mean((pred - y) ** 2)

    grad_fn = nn.value_and_grad(model, loss_fn)

    def handler(episode: DreamEpisode) -> "WeightUpdate | None":
        records = episode.input_slice.get("beta_records", [])
        if not records:
            # S1 no-op branch : no signal, no compute, no tag.
            state.last_loss = None
            state.last_compute_flops = 0
            return None

        for idx, r in enumerate(records):
            if "x" not in r or "y" not in r:
                raise ValueError(
                    f"record {idx} missing 'x' or 'y' key: {r!r}"
                )

        # Snapshot the adapters before the step. MLX arrays are
        # immutable values and the optimizer rebinds them, so the
        # mx.array() copy is a defensive detach.
        before = {
            k: mx.array(v)
            for k, v in model.adapter_parameters().items()
        }
        xs = mx.array([r["x"] for r in records])
        ys = mx.array([r["y"] for r in records])
        loss, grads = grad_fn(model, xs, ys)
        optimizer.update(model, grads)
        mx.eval(model.parameters())
        after = model.adapter_parameters()

        flops = _flop_estimate_lora(model, len(records))
        state.total_records_consumed += len(records)
        state.last_loss = float(loss.item())
        state.last_compute_flops = flops
        state.total_compute_flops += flops

        return WeightUpdate(
            lora_delta=adapter_delta(before, after),
            fisher_bump=None,
        )

    return handler
```

Update `__all__` to add `"replay_lora_handler"`:

```python
__all__ = [
    "ReplayRealState",
    "replay_real_handler",
    "replay_lora_handler",
]
```

Do NOT add `import numpy` to `replay_real.py` — the numpy work is inside `adapter_delta` (in `lora_model.py`). `replay_real.py` stays MLX-only per `operations/CLAUDE.md`.

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_replay_lora.py -v`
Expected: PASS — 9 tests.

- [ ] **Step 5: Full suite + mypy + ruff**

Run: `uv run pytest -q` — expect all pass, 0 failures.
Run: `uv run mypy harness tests` — expect `Success`.
Run: `uv run ruff check kiki_oniric/dream/operations/replay_real.py tests/unit/test_replay_lora.py` — expect clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/dream/operations/replay_real.py tests/unit/test_replay_lora.py
git commit -m "feat: replay_lora_handler emits a WeightUpdate"
```

Commit body:
```
feat: replay_lora_handler emits a WeightUpdate

B1b / issue #15. A LoRA-only replay handler: snapshots the A/B
adapters, runs one SGD step (base frozen, so MLX trains only the
adapters), and returns the per-adapter low-rank delta as a
channel-1 WeightUpdate. Empty beta_records returns None. K1
FLOPs tagged on ReplayRealState.
```

---

## Task 3: Documentation and DualVer sync

**Files:**
- Modify: `CHANGELOG.md`, `pyproject.toml`, `uv.lock`
- Modify: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/...`

- [ ] **Step 1: Add the CHANGELOG entry**

Insert at the top of the `CHANGELOG.md` body, immediately above the `[C-v0.15.0+PARTIAL]` entry:

```markdown
## [C-v0.16.0+PARTIAL] — 2026-05-19 — replay emits WeightUpdate (B1b)

### Formal axis (FC) — MINOR (v0.15.0 → v0.16.0)

- **New handler** `replay_lora_handler` in
  `kiki_oniric/dream/operations/replay_real.py`: runs an
  adapter-only SGD step on a `LoRAModel` (the base weight is
  frozen, so MLX trains only the A/B adapters) and returns a
  channel-1 `WeightUpdate` whose `lora_delta` carries the
  per-adapter low-rank deltas. Empty `beta_records` returns
  `None`. K1 FLOPs tagged on `ReplayRealState`.
- **New helper** `adapter_delta` in
  `kiki_oniric/substrates/micro_kiki/lora_model.py`: diffs two
  `adapter_parameters()` snapshots into a float32 numpy delta
  dict, keeping the MLX-only `replay_real.py` free of numpy.
- Sub-project B1b of issue #15. `replay` is the first dream
  operation to emit a real channel output. No profile wiring —
  the handler is exercised via a direct `DreamRuntime`.

### Empirical axis (EC) — UNCHANGED (PARTIAL)

- No new substrate, axiom, or empirical claim. EC stays
  `+PARTIAL`.

### Packaging

- `pyproject.toml` version bumped `0.13.0 → 0.14.0`.
```

Match the formatting of the surrounding entries.

- [ ] **Step 2: Bump `pyproject.toml`**

Change `version = "0.13.0"` to `version = "0.14.0"`.

- [ ] **Step 3: Sync the lockfile**

Run: `uv lock`
Expected: `uv.lock` updates the `dreamofkiki` pin to `0.14.0`.

- [ ] **Step 4: Update the framework-C spec**

In `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §4.2, near the `replay` description (the same area B1a's note was added), add:

```markdown
As of B1b (issue #15), `replay` emits a real channel-1
`WeightUpdate`: `replay_lora_handler` runs an adapter-only SGD
step on a `LoRAModel` and returns the per-adapter low-rank
delta. The other three operations still return `None`.
```

Apply the equivalent French sentence at the matching location in `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` (code identifiers stay in original form).

- [ ] **Step 5: Verify**

Run: `uv run pytest -q` — full suite passes (docs/version change, no code touched).
Run: `uv run mypy harness tests` — `Success`.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md pyproject.toml uv.lock docs/specs/2026-04-17-dreamofkiki-framework-C-design.md docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md
git commit -m "docs: sync spec for B1b replay WeightUpdate"
```

Commit body:
```
docs: sync spec for B1b replay WeightUpdate

B1b / issue #15. Add the C-v0.16.0+PARTIAL changelog entry, note
in framework-C spec section 4.2 (EN + FR) that replay now emits
a real WeightUpdate, and bump the package version to 0.14.0.
```

If `tests/reproducibility/golden_hashes.json` shows modified (R1 metadata drift), do NOT stage it.

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

Add a comment on issue #15: B1b complete — `replay` is the first dream operation emitting a real `WeightUpdate`. B2 (downscale), B3 (restructure), B4 (recombine) remain; B5 then rewires `consolidate()`.

---

## Self-Review

- **Spec coverage:** `replay_lora_handler` in `replay_real.py` (Task 2) ✓; reuses `ReplayRealState` (Task 2) ✓; adapter-only SGD via B1a's frozen base (Task 2, handler) ✓; `lora_delta` = per-adapter A/B deltas keyed by `adapter_parameters()` keys (Task 1 `adapter_delta` + Task 2) ✓; `fisher_bump=None` (Task 2) ✓; empty records → `None`, FLOPs 0 (Task 2, `test_replay_lora_empty_records_returns_none`) ✓; K1 FLOP tagging (Task 2, `_flop_estimate_lora` + `test_replay_lora_tags_k1_flops`) ✓; tests cover both representations — emitted A/B deltas (`test_replay_lora_delta_*`) and composed-ΔW (`test_replay_lora_composed_effective_delta`) ✓; empty / determinism / malformed-record (Task 2) ✓; end-to-end via `DreamRuntime.execute()` ✓; CHANGELOG + spec EN/FR + version (Task 3) ✓; verification (Task 4) ✓. No profile wiring — matches the spec's scope boundary.
- **Placeholder scan:** no TBD/TODO; every code step has complete code.
- **Type consistency:** `adapter_delta(before, after) -> dict[str, NDArray[np.float32]]`, `replay_lora_handler(state, *, model, lr)`, `ReplayRealState`, `WeightUpdate(lora_delta=, fisher_bump=)`, `_flop_estimate_lora(model, n_records)`, and the `layer<i>.lora_a/lora_b` keys are used identically across Tasks 1-3. `LoRALinear` exposes `rank` / `in_features` / `out_features` (used by `_flop_estimate_lora`) and `scale` / `lora_a` / `lora_b` (used by the composed-ΔW test) — all defined in B1a.
