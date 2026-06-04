# B2 — `downscale` emits a `WeightUpdate` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the `downscale` operation shrink a `LoRAModel`'s A/B adapters by a multiplicative factor and return a real channel-1 `WeightUpdate` carrying the per-adapter delta.

**Architecture:** A new `downscale_lora_handler` factory in `kiki_oniric/dream/operations/downscale_real.py` validates `shrink_factor`, snapshots `model.adapter_parameters()`, multiplies each `LoRALinear`'s `lora_a` / `lora_b` by the factor in place, snapshots again, computes the per-adapter delta via the existing `adapter_delta()` helper (shipped in B1b on `lora_model.py`), and returns `WeightUpdate(lora_delta=..., fisher_bump=None)`. The runtime captures it into `EpisodeLogEntry.channel_outputs[i]` (B0). `factor == 1.0` is an S1 no-op (`None`, FLOPs 0). The frozen base weight is not referenced — SHY shrinkage targets the adaptation only.

**Tech Stack:** Python 3.12, `uv`, MLX (`mlx.core` / `mlx.nn`), numpy, pytest, mypy.

**Spec:** `docs/superpowers/specs/2026-05-20-b2-downscale-weightupdate-design.md`

**LoRA fact:** with B1a's `B = 0` init, before any replay step every `lora_b` is zero. Shrinking a zero matrix yields a zero delta; the corresponding `lora_a` shrinks by `factor - 1` (non-positive). Tests therefore either (a) seed a non-trivial `lora_b` first, or (b) only assert "every value is finite, non-positive when `factor < 1`" — never "every value is strictly negative".

**Convention note:** `downscale_real.py` already imports `numpy` for the existing `downscale_real_handler` S2 guard (the file is not strictly MLX-only despite `operations/CLAUDE.md`'s aspirational rule). The new factory will not add a *new* numpy import; it will reuse `adapter_delta()` from `lora_model.py` to keep symmetry with the B1b precedent.

---

## File Structure

- **Modify** `kiki_oniric/dream/operations/downscale_real.py` — add `downscale_lora_handler` + `_flop_estimate_downscale_lora`.
- **Create** `tests/unit/test_downscale_lora.py` — end-to-end handler tests via `DreamRuntime`.
- **Modify** `CHANGELOG.md`, `pyproject.toml`, `uv.lock` — `[C-v0.17.0+PARTIAL]`, version `0.14.0 → 0.15.0`.
- **Modify** `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` — §4.2 note for `downscale`.

`adapter_delta()` already lives in `kiki_oniric/substrates/micro_kiki/lora_model.py` (shipped in B1b) — no changes to that file in B2.

---

## Task 1: `downscale_lora_handler`

**Files:**
- Modify: `kiki_oniric/dream/operations/downscale_real.py`
- Test: `tests/unit/test_downscale_lora.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/unit/test_downscale_lora.py`:

```python
"""Unit tests for the LoRA downscale handler (B2)."""
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
from kiki_oniric.dream.operations.downscale_real import (
    DownscaleRealState,
    downscale_lora_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime
from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


def _seed_nontrivial_b(model: LoRAModel) -> None:
    """Replace each ``lora_b`` with a known non-zero pattern.

    B1a inits ``B = 0`` so a shrinkage of an untouched model would
    leave ``lora_b`` deltas at zero. Seeding B makes the SHY
    shrinkage signal observable on both A and B.
    """
    for idx, layer in enumerate(model.layers):
        layer.lora_b = mx.ones(layer.lora_b.shape) * (0.1 * (idx + 1))


def _episode(factor: float) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"shrink_factor": factor},
        operation_set=(Operation.DOWNSCALE,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-lora-down",
    )


def _run(
    model: LoRAModel, factor: float,
) -> tuple[DownscaleRealState, object]:
    state = DownscaleRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.DOWNSCALE,
        downscale_lora_handler(state, model=model),
    )
    runtime.execute(_episode(factor))
    return state, runtime.log[-1]


def test_downscale_lora_emits_weight_update() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    _, entry = _run(model, 0.5)
    assert isinstance(entry.channel_outputs[0], WeightUpdate)


def test_downscale_lora_delta_keys_match_adapters() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    _, entry = _run(model, 0.5)
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    assert set(out.lora_delta) == set(model.adapter_parameters())
    for arr in out.lora_delta.values():
        assert arr.dtype == np.float32
        assert bool(np.isfinite(arr).all())


def test_downscale_lora_deltas_are_non_positive() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    # capture before
    before = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    _, entry = _run(model, 0.5)
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    # Δ = before * (f - 1) — non-positive when before ≥ 0 and f ≤ 1.
    # Since A is seeded-random (signed) and B is positive here,
    # check the sign rule per-element: sign(Δ) == -sign(before).
    for k, delta in out.lora_delta.items():
        expected = before[k] * (0.5 - 1.0)
        np.testing.assert_allclose(delta, expected, rtol=1e-5, atol=1e-6)


def test_downscale_lora_magnitudes_scale_by_factor() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    before = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    _, entry = _run(model, 0.5)
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    for k, delta in out.lora_delta.items():
        np.testing.assert_allclose(
            np.abs(delta), 0.5 * np.abs(before[k]),
            rtol=1e-5, atol=1e-6,
        )


def test_downscale_lora_composed_dense_delta_property() -> None:
    """Recomposed ΔW = scale * (f² - 1) * (B_before @ A_before)."""
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    layer0 = model.layers[0]
    a_before = np.asarray(layer0.lora_a, dtype=np.float32)
    b_before = np.asarray(layer0.lora_b, dtype=np.float32)
    scale = layer0.scale
    factor = 0.5
    _, entry = _run(model, factor)
    out = entry.channel_outputs[0]
    assert isinstance(out, WeightUpdate)
    a_after = a_before + out.lora_delta["layer0.lora_a"]
    b_after = b_before + out.lora_delta["layer0.lora_b"]
    composed = scale * (b_after @ a_after - b_before @ a_before)
    expected = scale * (factor * factor - 1.0) * (b_before @ a_before)
    np.testing.assert_allclose(composed, expected, rtol=1e-5, atol=1e-6)


def test_downscale_lora_factor_one_is_noop() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    before = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    state, entry = _run(model, 1.0)
    assert entry.channel_outputs[0] is None
    assert state.last_compute_flops == 0
    after = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    for k in before:
        np.testing.assert_array_equal(before[k], after[k])


@pytest.mark.parametrize("bad", [0.0, -0.1, 1.5, 2.0])
def test_downscale_lora_rejects_out_of_bounds_factor(bad: float) -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    before = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    state = DownscaleRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.DOWNSCALE,
        downscale_lora_handler(state, model=model),
    )
    with pytest.raises(ValueError, match="shrink_factor"):
        runtime.execute(_episode(bad))
    # validation-before-mutation: model untouched
    after = {
        k: np.asarray(v, dtype=np.float32)
        for k, v in model.adapter_parameters().items()
    }
    for k in before:
        np.testing.assert_array_equal(before[k], after[k])


def test_downscale_lora_compounds_multiplicatively() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    state = DownscaleRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.DOWNSCALE,
        downscale_lora_handler(state, model=model),
    )
    runtime.execute(_episode(0.9))
    runtime.execute(_episode(0.8))
    assert state.compound_factor == pytest.approx(0.72, rel=1e-6)


def test_downscale_lora_tags_k1_flops() -> None:
    model = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=0)
    _seed_nontrivial_b(model)
    state, _ = _run(model, 0.5)
    assert state.last_compute_flops > 0


def test_downscale_lora_is_deterministic() -> None:
    m1 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=3)
    m2 = LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=3)
    _seed_nontrivial_b(m1)
    _seed_nontrivial_b(m2)
    _, e1 = _run(m1, 0.5)
    _, e2 = _run(m2, 0.5)
    o1, o2 = e1.channel_outputs[0], e2.channel_outputs[0]
    assert isinstance(o1, WeightUpdate) and isinstance(o2, WeightUpdate)
    for k in o1.lora_delta:
        np.testing.assert_array_equal(o1.lora_delta[k], o2.lora_delta[k])
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/test_downscale_lora.py -v`
Expected: FAIL — `ImportError: cannot import name 'downscale_lora_handler'`.

- [ ] **Step 3: Implement the handler in `downscale_real.py`**

In `kiki_oniric/dream/operations/downscale_real.py`, add `_flop_estimate_downscale_lora` after the existing `_param_count`:

```python
def _flop_estimate_downscale_lora(model) -> int:
    """Rough FLOP count for a LoRA-adapter shrinkage step.

    Dominated by the per-layer elementwise scale of the A/B
    adapters: ``rank * (in + out)`` ops per layer, times two
    (forward + scratch). Smaller than a full-weight shrinkage.
    """
    per_layer = sum(
        2 * layer.rank * (layer.in_features + layer.out_features)
        for layer in model.layers
    )
    return max(per_layer, 1)
```

Add the handler factory after `downscale_real_handler` (before `__all__`):

```python
def downscale_lora_handler(
    state: DownscaleRealState,
    *,
    model,  # LoRAModel — typed loosely for lazy MLX import
) -> Callable[[DreamEpisode], "WeightUpdate | None"]:
    """Build a LoRA-only downscale handler that emits a ``WeightUpdate``.

    Reads ``shrink_factor`` from the episode (default ``1.0``),
    validates ``0 < factor <= 1``, then multiplies every
    ``LoRALinear``'s ``lora_a`` / ``lora_b`` by the factor in place.
    The frozen base weight is not touched — SHY shrinkage on the
    LoRA substrate targets the adaptation only. ``factor == 1.0``
    is an S1 no-op (returns ``None``, FLOPs 0). Returns
    ``WeightUpdate(lora_delta=..., fisher_bump=None)`` with the
    per-adapter low-rank deltas keyed as ``adapter_parameters()``.

    Reference:
      docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
    """
    import mlx.core as mx

    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.substrates.micro_kiki.lora_model import adapter_delta

    def handler(episode: DreamEpisode) -> "WeightUpdate | None":
        factor = episode.input_slice.get("shrink_factor", 1.0)
        # Validate BEFORE any mutation — S2 / operations/CLAUDE.md.
        if not (0.0 < factor <= 1.0):
            raise ValueError(
                f"shrink_factor must be in (0, 1], got {factor}"
            )

        if factor == 1.0:
            # S1 no-op branch : zero compute, no emission.
            state.last_compute_flops = 0
            return None

        # Snapshot adapters before shrinkage. MLX arrays are
        # immutable values; the optimizer rebinds them, so
        # ``mx.array(v)`` is a defensive detach.
        before = {
            k: mx.array(v)
            for k, v in model.adapter_parameters().items()
        }

        # Per-layer shrinkage of A/B adapters. Base weight + bias
        # are frozen by LoRALinear and intentionally untouched.
        for layer in model.layers:
            layer.lora_a = layer.lora_a * factor
            layer.lora_b = layer.lora_b * factor

        mx.eval(model.parameters())
        after = model.adapter_parameters()

        flops = _flop_estimate_downscale_lora(model)
        state.compound_factor *= factor
        state.last_compute_flops = flops

        return WeightUpdate(
            lora_delta=adapter_delta(before, after),
            fisher_bump=None,
        )

    return handler
```

Update `__all__` to add `"downscale_lora_handler"`:

```python
__all__ = [
    "DownscaleRealState",
    "downscale_real_handler",
    "downscale_lora_handler",
    # Re-export FiniteGuardError so test imports read naturally.
    "FiniteGuardError",
]
```

Do NOT add a `total_compute_flops` field to `DownscaleRealState` — the existing dataclass only carries `compound_factor` and `last_compute_flops`. Stay symmetric with the existing `downscale_real_handler` (which also does not maintain a running total).

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/test_downscale_lora.py -v`
Expected: PASS — 13 tests (10 + 4 parametrised − 1, since `pytest.mark.parametrize` counts each value separately: 9 standalone + 4 parametrised = 13).

- [ ] **Step 5: Full suite + mypy + ruff**

Run: `uv run pytest -q` — expect all pass, 0 failures.
Run: `uv run mypy harness tests` — expect `Success`.
Run: `uv run ruff check kiki_oniric/dream/operations/downscale_real.py tests/unit/test_downscale_lora.py` — expect clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/dream/operations/downscale_real.py tests/unit/test_downscale_lora.py
git commit -m "feat: downscale_lora_handler emits a WeightUpdate"
```

Commit body:
```
feat: downscale_lora_handler emits a WeightUpdate

B2 / issue #15. A LoRA-only downscale handler: validates the
shrink_factor (0, 1], multiplies each LoRALinear's A/B adapters
in place, and returns the per-adapter low-rank delta as a
channel-1 WeightUpdate. factor == 1.0 returns None (S1 no-op).
K1 FLOPs tagged on DownscaleRealState; compound_factor compounds
multiplicatively across episodes.
```

---

## Task 2: Documentation and DualVer sync

**Files:**
- Modify: `CHANGELOG.md`, `pyproject.toml`, `uv.lock`
- Modify: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/...`

- [ ] **Step 1: Add the CHANGELOG entry**

Insert at the top of the `CHANGELOG.md` body, immediately above the `[C-v0.16.0+PARTIAL]` entry:

```markdown
## [C-v0.17.0+PARTIAL] — 2026-05-20 — downscale emits WeightUpdate (B2)

### Formal axis (FC) — MINOR (v0.16.0 → v0.17.0)

- **New handler** `downscale_lora_handler` in
  `kiki_oniric/dream/operations/downscale_real.py`: shrinks every
  `LoRALinear`'s A/B adapters by a validated `shrink_factor` in
  `(0, 1]` and returns a channel-1 `WeightUpdate` whose
  `lora_delta` carries the per-adapter low-rank deltas. The
  frozen base weight is untouched — SHY shrinkage on the LoRA
  substrate targets the adaptation only. `factor == 1.0` is an
  S1 no-op (returns `None`, FLOPs 0). `compound_factor` compounds
  multiplicatively across episodes (Tononi non-idempotence).
- Re-uses the `adapter_delta` helper shipped with B1b
  (`kiki_oniric/substrates/micro_kiki/lora_model.py`) — no new
  file in `substrates/`.
- Sub-project B2 of issue #15. `downscale` is the second dream
  operation to emit a real channel output (after `replay` in B1b).
  No profile wiring — the handler is exercised via a direct
  `DreamRuntime`. `I-Wmag` is trivially satisfied by multiplicative
  shrinkage (`f ≤ 1` only decreases magnitudes) and flagged as an
  out-of-scope cleanup item for `docs/invariants/registry.md`.

### Empirical axis (EC) — UNCHANGED (PARTIAL)

- No new substrate, axiom, or empirical claim. EC stays
  `+PARTIAL`.

### Packaging

- `pyproject.toml` version bumped `0.14.0 → 0.15.0`.
```

Match the formatting of the surrounding entries.

- [ ] **Step 2: Bump `pyproject.toml`**

Change `version = "0.14.0"` to `version = "0.15.0"`.

- [ ] **Step 3: Sync the lockfile**

Run: `uv lock`
Expected: `uv.lock` updates the `dreamofkiki` pin to `0.15.0`.

- [ ] **Step 4: Update the framework-C spec**

In `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §4.2, near the `downscale` description (the same area B1b's `replay` note was added), add:

```markdown
As of B2 (issue #15), `downscale` emits a real channel-1
`WeightUpdate`: `downscale_lora_handler` shrinks the A/B adapters
of a `LoRAModel` by a validated `shrink_factor` and returns the
per-adapter low-rank delta. With `replay` (B1b) and `downscale`
(B2), two of the four dream operations now emit real channel
outputs. `restructure` and `recombine` still return `None`.
```

Apply the equivalent French sentence at the matching location in `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` (code identifiers stay in original form).

- [ ] **Step 5: Verify**

Run: `uv run pytest -q` — full suite passes (docs/version change, no code touched).
Run: `uv run mypy harness tests` — `Success`.

- [ ] **Step 6: Commit**

```bash
git add CHANGELOG.md pyproject.toml uv.lock docs/specs/2026-04-17-dreamofkiki-framework-C-design.md docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md
git commit -m "docs: sync spec for B2 downscale WeightUpdate"
```

Commit body:
```
docs: sync spec for B2 downscale WeightUpdate

B2 / issue #15. Add the C-v0.17.0+PARTIAL changelog entry, note
in framework-C spec section 4.2 (EN + FR) that downscale now
emits a real WeightUpdate, and bump the package version to
0.15.0.
```

If `tests/reproducibility/golden_hashes.json` shows modified (R1 metadata drift), do NOT stage it.

---

## Task 3: Final verification

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

Add a comment on issue #15: B2 complete — `downscale` is the second dream operation emitting a real `WeightUpdate`. B3 (restructure) and B4 (recombine) remain; B5 then rewires `consolidate()`.

---

## Self-Review

- **Spec coverage:** `downscale_lora_handler` in `downscale_real.py` (Task 1) ✓; reuses `DownscaleRealState` without adding fields (Task 1) ✓; validates `0 < factor <= 1` before any mutation (Task 1, `test_downscale_lora_rejects_out_of_bounds_factor`) ✓; `factor == 1.0` → `None`, FLOPs 0 (Task 1, `test_downscale_lora_factor_one_is_noop`) ✓; shrinks each `LoRALinear`'s A/B by `factor` (Task 1, `test_downscale_lora_magnitudes_scale_by_factor`) ✓; returns `WeightUpdate(lora_delta=..., fisher_bump=None)` keyed as `adapter_parameters()` (Task 1, `test_downscale_lora_delta_keys_match_adapters`) ✓; composed-`ΔW` property `scale * (f² - 1) * B·A` (Task 1, `test_downscale_lora_composed_dense_delta_property`) ✓; `compound_factor` compounds multiplicatively (Task 1, `test_downscale_lora_compounds_multiplicatively`) ✓; K1 FLOPs tagged (Task 1, `_flop_estimate_downscale_lora` + `test_downscale_lora_tags_k1_flops`) ✓; determinism (Task 1, `test_downscale_lora_is_deterministic`) ✓; end-to-end via `DreamRuntime.execute()` ✓; CHANGELOG + spec EN/FR + version 0.15.0 (Task 2) ✓; verification (Task 3) ✓. No profile wiring — matches the spec's scope boundary. I-Wmag is flagged as an out-of-scope registry-cleanup item per the spec.
- **Placeholder scan:** no TBD/TODO; every code step has complete code.
- **Type consistency:** `downscale_lora_handler(state, *, model)`, `DownscaleRealState` (existing fields `compound_factor` / `last_compute_flops` only — no `total_compute_flops`), `WeightUpdate(lora_delta=, fisher_bump=)`, `_flop_estimate_downscale_lora(model)`, and the `layer<i>.lora_a/lora_b` keys are used identically across Tasks 1-2. `LoRALinear` exposes `rank` / `in_features` / `out_features` (used by `_flop_estimate_downscale_lora`) and `scale` / `lora_a` / `lora_b` (used by the composed-ΔW test) — all defined in B1a. `adapter_delta(before, after) -> dict[str, NDArray[np.float32]]` is the B1b helper, imported lazily inside the factory.
