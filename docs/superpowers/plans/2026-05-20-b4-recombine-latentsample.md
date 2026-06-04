# B4 — `recombine` emits a `LatentSample` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Patch `recombine_real_handler` so it returns a real channel-2 `LatentSample` whose `latent_vector` is the sampled VAE latent `z`, completing the 4-of-4 dream-ops emitting-channel-outputs milestone.

**Architecture:** The cycle-3 VAE machinery (`encoder` / `decoder`, isolated `mx.random.key`, per-episode counter) stays untouched. After `z = mu + sigma * epsilon` and `decoder(z)`, the handler builds `LatentSample(species, latent_vector=z, provenance)` *before* bumping `state._episode_count`, then bumps the counter and returns the sample. `species` comes from `input_slice` (default `"default"`); `provenance` is auto-derived from `(episode_id, count, key_seed)`. Empty `delta_latents` continues to raise `ValueError("I3: …")` — documented asymmetry vs B1b/B2/B3.

**Tech Stack:** Python 3.12, `uv`, MLX (`mlx.core`), numpy, pytest, mypy.

**Spec:** `docs/superpowers/specs/2026-05-20-b4-recombine-latentsample-design.md`

**Critical invariant of the patch:** `LatentSample` is constructed **before** `state._episode_count += 1` so the `provenance` string reads `ep=<state._episode_count>` directly (no `-1` arithmetic). If `LatentSample.__post_init__` raises (non-finite `z`), the counter does not advance — sound "this episode didn't emit" semantics.

---

## File Structure

- **Modify** `kiki_oniric/dream/operations/recombine_real.py`:
  - Add `import numpy as np` at module top (today the file does `import numpy as _np` lazily inside the handler; promote it).
  - Add `TYPE_CHECKING` guard importing `LatentSample` for the return-type annotation.
  - Widen the factory's return-type annotation.
  - Inside the handler, build the `LatentSample` between the `state.last_sample` assignment and the counter bump.
- **Create** `tests/unit/test_recombine_latent_sample.py` — 11 end-to-end tests via `DreamRuntime`.
- **Modify** `CHANGELOG.md`, `pyproject.toml`, `uv.lock` — `[C-v0.19.0+PARTIAL]`, version `0.16.0 → 0.17.0`.
- **Modify** `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` — §4.2 note for `recombine`, "4 of 4" milestone.

---

## Task 1: Patch `recombine_real_handler` to emit `LatentSample`

**Files:**
- Modify: `kiki_oniric/dream/operations/recombine_real.py`
- Test: `tests/unit/test_recombine_latent_sample.py`

- [ ] **Step 1: Write the failing test file**

Create `tests/unit/test_recombine_latent_sample.py`:

```python
"""Unit tests for recombine_real_handler emitting LatentSample (B4)."""
from __future__ import annotations

import re
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import mlx.core as mx
    import mlx.nn as nn
else:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

from kiki_oniric.dream.channels import LatentSample
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.recombine_real import (
    RecombineRealState,
    recombine_real_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime


LATENT_DIM = 4
INPUT_DIM = 4


class _TinyEncoder(nn.Module):  # type: ignore[misc]
    """Linear encoder → (mu, log_var) with fixed weights (R1)."""

    def __init__(self) -> None:
        super().__init__()
        self.mu_w = mx.ones((LATENT_DIM, INPUT_DIM)) * 0.1
        self.lv_w = mx.ones((LATENT_DIM, INPUT_DIM)) * -0.5

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        return x @ self.mu_w.T, x @ self.lv_w.T


class _TinyDecoder(nn.Module):  # type: ignore[misc]
    """Linear decoder z → output."""

    def __init__(self) -> None:
        super().__init__()
        self.w = mx.ones((INPUT_DIM, LATENT_DIM)) * 0.2

    def __call__(self, z: mx.array) -> mx.array:
        return z @ self.w.T


class _InfLogVarEncoder(nn.Module):  # type: ignore[misc]
    """Pathological encoder: log_var = +inf → z = +inf via sigma * eps."""

    def __init__(self) -> None:
        super().__init__()
        self.w = mx.ones((LATENT_DIM, INPUT_DIM)) * 0.1

    def __call__(self, x: mx.array) -> tuple[mx.array, mx.array]:
        mu = x @ self.w.T
        log_var = mx.array([float("inf")] * LATENT_DIM)
        return mu, log_var


def _episode(
    delta_latents: list[list[float]] | None = None,
    *,
    species: object = None,
    episode_id: str = "de-rcb",
) -> DreamEpisode:
    if delta_latents is None:
        delta_latents = [[0.1, 0.2, 0.3, 0.4]]
    slice_d: dict[str, object] = {"delta_latents": delta_latents}
    if species is not None:
        slice_d["species"] = species
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice=slice_d,
        operation_set=(Operation.RECOMBINE,),
        output_channels=(OutputChannel.LATENT_SAMPLE,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id=episode_id,
    )


def _make_runtime(
    encoder: nn.Module,
    decoder: nn.Module,
    *,
    seed: int = 0,
) -> tuple[RecombineRealState, DreamRuntime]:
    state = RecombineRealState()
    runtime = DreamRuntime()
    runtime.register_handler(
        Operation.RECOMBINE,
        recombine_real_handler(
            state, encoder=encoder, decoder=decoder, seed=seed,
        ),
    )
    return state, runtime


def test_recombine_emits_latent_sample() -> None:
    state, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder())
    runtime.execute(_episode())
    assert isinstance(runtime.log[-1].channel_outputs[0], LatentSample)


def test_recombine_latent_vector_dtype_shape() -> None:
    _, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder())
    runtime.execute(_episode())
    out = runtime.log[-1].channel_outputs[0]
    assert isinstance(out, LatentSample)
    assert out.latent_vector.dtype == np.float32
    assert out.latent_vector.shape == (LATENT_DIM,)


def test_recombine_finite_propagation_via_inf_log_var() -> None:
    """Pathological log_var=inf → z=inf → LatentSample raises S2."""
    _, runtime = _make_runtime(_InfLogVarEncoder(), _TinyDecoder())
    with pytest.raises(ValueError, match=r"^S2:"):
        runtime.execute(_episode())


def test_recombine_species_default() -> None:
    _, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder())
    runtime.execute(_episode())
    out = runtime.log[-1].channel_outputs[0]
    assert isinstance(out, LatentSample)
    assert out.species == "default"


def test_recombine_species_from_input() -> None:
    _, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder())
    runtime.execute(_episode(species="replay-mix"))
    out = runtime.log[-1].channel_outputs[0]
    assert isinstance(out, LatentSample)
    assert out.species == "replay-mix"


def test_recombine_species_non_str_rejected() -> None:
    _, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder())
    with pytest.raises(ValueError, match=r"^recombine: species must be str"):
        runtime.execute(_episode(species=42))


def test_recombine_provenance_format() -> None:
    _, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder(), seed=11)
    runtime.execute(_episode(episode_id="de-fmt"))
    out = runtime.log[-1].channel_outputs[0]
    assert isinstance(out, LatentSample)
    assert re.match(
        r"^recombine:de=de-fmt:ep=\d+:seed=\d+$", out.provenance,
    )


def test_recombine_provenance_count_increments() -> None:
    _, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder(), seed=11)
    runtime.execute(_episode(episode_id="de-inc"))
    runtime.execute(_episode(episode_id="de-inc"))
    first = runtime.log[0].channel_outputs[0]
    second = runtime.log[1].channel_outputs[0]
    assert isinstance(first, LatentSample) and isinstance(second, LatentSample)
    assert "ep=0:" in first.provenance
    assert "ep=1:" in second.provenance


def test_recombine_is_deterministic() -> None:
    _, r1 = _make_runtime(_TinyEncoder(), _TinyDecoder(), seed=7)
    _, r2 = _make_runtime(_TinyEncoder(), _TinyDecoder(), seed=7)
    r1.execute(_episode(episode_id="de-det"))
    r2.execute(_episode(episode_id="de-det"))
    a = r1.log[-1].channel_outputs[0]
    b = r2.log[-1].channel_outputs[0]
    assert isinstance(a, LatentSample) and isinstance(b, LatentSample)
    np.testing.assert_array_equal(a.latent_vector, b.latent_vector)
    assert a.provenance == b.provenance


def test_recombine_empty_delta_latents_raises() -> None:
    _, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder())
    with pytest.raises(ValueError, match=r"^I3:"):
        runtime.execute(_episode(delta_latents=[]))


def test_recombine_state_last_sample_preserved() -> None:
    state, runtime = _make_runtime(_TinyEncoder(), _TinyDecoder())
    runtime.execute(_episode())
    assert state.last_sample is not None
    assert isinstance(state.last_sample, list)
    assert all(isinstance(v, float) for v in state.last_sample)
    assert state._episode_count == 1
    assert state.last_compute_flops > 0
```

- [ ] **Step 2: Run to verify the suite fails**

Run: `uv run pytest tests/unit/test_recombine_latent_sample.py -v`
Expected: most tests FAIL because the handler still returns `None` — `channel_outputs[0]` is `None` not a `LatentSample`, so `isinstance(..., LatentSample)` and attribute access fail.

- [ ] **Step 3: Patch `recombine_real.py`**

Replace the entire body of `kiki_oniric/dream/operations/recombine_real.py` (preserve the module docstring) with:

```python
"""Real-weight recombine op — VAE reparameterization over MLX.

Cycle-3 C3.3 counterpart to the light recombine op in
:mod:`kiki_oniric.dream.operations.recombine`. This variant drives a
real encoder / decoder pair through the reparameterization trick
(``z = mu + sigma * epsilon``) and emits a decoded latent sample on
canal 2 (LATENT_SAMPLE).

Determinism contract (mirrors the cycle-2 MLX fix) :

- The handler keeps a per-state episode counter so ``seed +
  episode_count`` drives an isolated :func:`mlx.core.random.key` for
  ``epsilon``. The process-wide MLX RNG is *never* mutated, so
  concurrent dream workers can run multiple recombine handlers
  without interfering — and two handlers built with the same seed
  produce identical samples under identical input (test 8).

Contract :

- ``delta_latents`` read from ``episode.input_slice`` ; must be a
  non-empty list of list[float]. Only ``latents[0]`` is consumed —
  diversity comes from sampling ``z``, not latent selection.
- ``state.last_sample`` stores the decoder output as a list[float]
  (length 4 in the _TinyDecoder test fixture).
- ``state.last_compute_flops`` is tagged with a rough cost estimate.
- B4 (issue #15) widens the return type to ``LatentSample | None`` —
  the handler builds and returns a channel-2 ``LatentSample`` whose
  ``latent_vector`` is the sampled ``z``. ``species`` comes from
  ``input_slice`` (default ``"default"``); ``provenance`` is auto-
  derived from ``(episode_id, episode_count, key_seed)``.

Reference :
  docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Callable

import numpy as np

if TYPE_CHECKING:
    from kiki_oniric.dream.channels import LatentSample
    from kiki_oniric.dream.episode import DreamEpisode


@dataclass
class RecombineRealState:
    """K1-tagged recombine state across multiple episodes.

    ``last_sample`` is typed as ``list | None`` per the adapter
    spec ; callers can assume ``list[float]`` when populated.
    """

    last_sample: list | None = None
    last_compute_flops: int = 0
    # Episode-counter drives per-call RNG key derivation so same
    # seed + same episode index reproduces byte-identical samples.
    _episode_count: int = 0


def recombine_real_handler(
    state: RecombineRealState,
    *,
    encoder,
    decoder,
    seed: int,
) -> Callable[["DreamEpisode"], "LatentSample | None"]:
    """Build a real-weight recombine handler bound to ``state``.

    Imports MLX lazily so pure-synthetic callers don't pay the cost.
    Returns a ``LatentSample`` (channel 2) carrying the sampled
    latent ``z`` as ``latent_vector``. The decoded output continues
    to live in ``state.last_sample`` for backwards compatibility.
    """
    import mlx.core as mx

    from kiki_oniric.dream.channels import LatentSample

    def handler(episode: "DreamEpisode") -> "LatentSample | None":
        latents = episode.input_slice.get("delta_latents", [])
        if not latents:
            raise ValueError(
                "I3: delta_latents must not be empty for recombine_real"
            )

        # Per-episode isolated PRNG key — does not touch the
        # process-wide MLX RNG.
        key_seed = seed + state._episode_count
        key = mx.random.key(key_seed)
        _, sample_key = mx.random.split(key)

        x = mx.array(latents[0])
        mu, log_var = encoder(x)
        sigma = mx.exp(0.5 * log_var)
        epsilon = mx.random.normal(shape=mu.shape, key=sample_key)
        z = mu + sigma * epsilon
        sample_arr = decoder(z)
        mx.eval(sample_arr)

        # Flatten to a 1-D list[float] — decoder may return a
        # multi-dimensional array (batch / feature axes) and calling
        # float(v) on a nested list would raise TypeError.
        state.last_sample = [
            float(v) for v in np.asarray(sample_arr).ravel().tolist()
        ]
        # K1 tag : encoder + decoder fwd passes over a tiny latent.
        state.last_compute_flops = max(
            2 * (mu.size + sample_arr.size), 1
        )

        # B4 — build the LatentSample BEFORE bumping the counter so
        # `provenance` reads `ep=<state._episode_count>` directly
        # (no off-by-one). If LatentSample.__post_init__ raises (non-
        # finite z), the counter does not advance — sound semantics
        # for "this episode didn't actually emit".
        latent_vector = (
            np.asarray(z, dtype=np.float32).ravel().copy()
        )
        species = episode.input_slice.get("species", "default")
        if not isinstance(species, str):
            raise ValueError(
                f"recombine: species must be str, "
                f"got {type(species).__name__}"
            )
        provenance = (
            f"recombine:de={episode.episode_id}:"
            f"ep={state._episode_count}:seed={key_seed}"
        )
        sample = LatentSample(
            species=species,
            latent_vector=latent_vector,
            provenance=provenance,
        )

        state._episode_count += 1
        return sample

    return handler


__all__ = [
    "RecombineRealState",
    "recombine_real_handler",
]
```

Key changes vs the pre-B4 file:
- Replaced `from dataclasses import dataclass` line with the new typed-import block (`TYPE_CHECKING`, `Callable`, `numpy as np`).
- Added `TYPE_CHECKING` block for `LatentSample` / `DreamEpisode` (forward-ref strings on the factory signature).
- Lazy `from kiki_oniric.dream.channels import LatentSample` inside the factory (mirrors the B1b/B2/B3 lazy-import pattern).
- Replaced the old `import numpy as _np` (which lived inside the handler body) with the module-level `import numpy as np`.
- Widened the factory return-type annotation to `Callable[["DreamEpisode"], "LatentSample | None"]`.
- Added the 4-statement build block (latent_vector, species check, provenance, sample) **before** the counter bump.
- The counter bump moves AFTER the LatentSample construction.

- [ ] **Step 4: Run to verify the new tests pass**

Run: `uv run pytest tests/unit/test_recombine_latent_sample.py -v`
Expected: PASS — 11 tests.

- [ ] **Step 5: Verify existing recombine tests still pass**

Run: `uv run pytest tests/ -k recombine -v`
Expected: PASS — both the new file *and* the legacy `test_recombine*.py` files (cycle-3 + skeleton tests).

- [ ] **Step 6: Full suite + mypy + ruff**

Run: `uv run pytest -q` — all pass.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/dream/operations/recombine_real.py tests/unit/test_recombine_latent_sample.py` — clean.

- [ ] **Step 7: Commit**

```bash
git add kiki_oniric/dream/operations/recombine_real.py tests/unit/test_recombine_latent_sample.py
git commit -m "feat: recombine_real_handler emits a LatentSample"
```

Commit body:
```
feat: recombine_real_handler emits a LatentSample

B4 / issue #15. Patches the existing VAE recombine handler to
return a channel-2 LatentSample whose latent_vector is the
sampled z (not the decoded output). species comes from
input_slice (default "default"); provenance is auto-derived
from (episode_id, episode_count, key_seed) for R1 traceability.
The LatentSample is built BEFORE the episode counter bumps so
provenance reads ep=<count> directly. Empty delta_latents still
raises ValueError("I3: ...") — documented asymmetry vs the S1
no-op pattern of B1b/B2/B3.
```

---

## Task 2: Documentation and DualVer sync

**Files:**
- Modify: `CHANGELOG.md`, `pyproject.toml`, `uv.lock`
- Modify: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/...`

- [ ] **Step 1: Add the CHANGELOG entry**

Insert at the top of the `CHANGELOG.md` body, immediately above the existing `[C-v0.18.0+PARTIAL]` entry:

```markdown
## [C-v0.19.0+PARTIAL] — 2026-05-20 — recombine emits LatentSample (B4)

### Formal axis (FC) — MINOR (v0.18.0 → v0.19.0)

- **Patched handler** `recombine_real_handler` in
  `kiki_oniric/dream/operations/recombine_real.py`: return type
  widened from `Callable[[DE], None]` to
  `Callable[[DE], LatentSample | None]`. The handler now builds
  and returns a channel-2 `LatentSample` whose
  `latent_vector` is the sampled VAE latent `z` (not the decoded
  output — `state.last_sample` continues to hold that for
  backwards compatibility). `species` is read from
  `input_slice` (default `"default"`; non-str raises
  `ValueError("recombine: species must be str, …")`).
  `provenance` is auto-derived as
  `"recombine:de={episode_id}:ep={count}:seed={key_seed}"`,
  R1-traceable.
- **Asymmetry preserved** — empty `delta_latents` continues to
  raise `ValueError("I3: …")` instead of falling through to a
  silent S1 no-op (the pattern used by B1b/B2/B3). Invariant I3
  presumes a non-empty latent buffer; an empty input is an
  upstream scheduling error, not a recombine no-op.
- **Counter-bump ordering** — the `LatentSample` is constructed
  *before* the per-episode counter increments, so `provenance`
  reads `ep=<state._episode_count>` directly (no off-by-one).
  If `LatentSample.__post_init__` raises on non-finite `z`, the
  counter does not advance — sound "this episode didn't emit"
  semantics.
- Sub-project B4 of issue #15 — **4 of 4** dream operations now
  emit real channel outputs. `replay` (B1b) and `downscale` (B2)
  on channel 1 (`WeightUpdate`), `restructure` (B3) on channel
  3 (`TopologyDiff`), `recombine` (B4) on channel 2
  (`LatentSample`). B5 will rewire `consolidate()` to actually
  apply the four channel outputs to a target model.

### Empirical axis (EC) — UNCHANGED (PARTIAL)

- No new substrate, axiom, or empirical claim. EC stays
  `+PARTIAL`.

### Packaging

- `pyproject.toml` version bumped `0.16.0 → 0.17.0`.
```

Match the formatting of the surrounding entries.

- [ ] **Step 2: Bump `pyproject.toml`**

Change `version = "0.16.0"` to `version = "0.17.0"`.

- [ ] **Step 3: Sync the lockfile**

Run: `uv lock`
Expected: `uv.lock` updates the `dreamofkiki` pin to `0.17.0`.

- [ ] **Step 4: Update the framework-C spec (EN)**

In `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §4.2, immediately after the B3 `restructure` note, add:

```markdown
As of B4 (issue #15), `recombine` emits a real channel-2
`LatentSample`: `recombine_real_handler` returns the sampled
VAE latent `z` as `latent_vector`, with `species` (from the
episode `input_slice`, default `"default"`) and an auto-derived
`provenance` string `recombine:de={episode_id}:ep={count}:seed={key_seed}`.
With B1b/B2/B3/B4, **all four** dream operations now emit real
channel outputs. B5 will rewire `consolidate()` to actually
apply them to a target model.
```

- [ ] **Step 5: Update the framework-C spec (FR mirror)**

In `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` §4.2, at the matching location, add the French sentence (code identifiers stay in original form):

```markdown
Depuis B4 (issue #15), `recombine` émet un véritable
`LatentSample` (canal 2) : `recombine_real_handler` retourne le
latent VAE échantillonné `z` comme `latent_vector`, avec
`species` (lu depuis `input_slice`, défaut `"default"`) et un
`provenance` auto-dérivé au format
`recombine:de={episode_id}:ep={count}:seed={key_seed}`. Avec
B1b/B2/B3/B4, **les quatre** opérations de rêve émettent
désormais des sorties de canal réelles. B5 recâblera
`consolidate()` pour appliquer ces sorties à un modèle cible.
```

- [ ] **Step 6: Verify**

Run: `uv run pytest -q` — all pass (docs/version change, no code touched).
Run: `uv run mypy harness tests` — `Success`.

- [ ] **Step 7: Commit**

```bash
git add CHANGELOG.md pyproject.toml uv.lock docs/specs/2026-04-17-dreamofkiki-framework-C-design.md docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md
git commit -m "docs: sync spec for B4 recombine LatentSample"
```

Commit body:
```
docs: sync spec for B4 recombine LatentSample

B4 / issue #15. Add the C-v0.19.0+PARTIAL changelog entry,
note in framework-C spec section 4.2 (EN + FR) that recombine
now emits a real LatentSample completing the 4-of-4 emitting-
ops milestone, and bump the package version to 0.17.0.
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
Expected: empty (if `tests/reproducibility/golden_hashes.json` shows modified, restore it with `git checkout -- tests/reproducibility/golden_hashes.json`).

- [ ] **Step 5: Update issue #15**

Comment on issue #15: B4 complete — `recombine` emits `LatentSample` on channel 2, completing the **4-of-4** dream-ops-emitting-channels milestone. B5 (rewire `consolidate()`) is the last remaining sub-project in approach B.

---

## Self-Review

- **Spec coverage:**
  - Patch `recombine_real_handler` to return `LatentSample` (Task 1, Step 3) ✓
  - `latent_vector = np.asarray(z, dtype=np.float32).ravel().copy()` (Task 1, Step 3) ✓
  - `species` default + non-str rejection with `"recombine: species must be str, …"` (Task 1 tests 4-6 + Step 3 handler) ✓
  - `provenance` format `recombine:de={id}:ep={count}:seed={key}` with counter index pre-bump (Task 1 tests 7-8 + Step 3 handler) ✓
  - Empty `delta_latents` continues to raise `I3:` (Task 1 test 10 + handler) ✓
  - State `last_sample`/`_episode_count`/`last_compute_flops` preserved (Task 1 test 11) ✓
  - Determinism preserved (Task 1 test 9) ✓
  - Finite propagation via `LatentSample.__post_init__` on pathological `log_var=inf` (Task 1 test 3) ✓
  - CHANGELOG `[C-v0.19.0+PARTIAL]` + spec §4.2 EN+FR + version `0.17.0` (Task 2) ✓
  - Final verification (Task 3) ✓
- **Placeholder scan:** no TBD/TODO; every code step has complete code.
- **Type consistency:**
  - `recombine_real_handler(state, *, encoder, decoder, seed)` → `Callable[[DreamEpisode], LatentSample | None]` matches Task 1 tests' helper `recombine_real_handler(state, encoder=..., decoder=..., seed=...)`.
  - `RecombineRealState` fields (`last_sample`, `last_compute_flops`, `_episode_count`) referenced by tests are the same names defined in the dataclass in Task 1.
  - `LatentSample(species, latent_vector, provenance)` matches the B0 frozen dataclass signature; tests access `.species`, `.latent_vector`, `.provenance` consistently.
  - `Operation.RECOMBINE` and `OutputChannel.LATENT_SAMPLE` exist in `kiki_oniric/dream/episode.py` (verified at the brainstorming stage — lines 24 and 29).
- **Asymmetry note:** the empty-`delta_latents` raises preserved and explicitly tested. The plan does not silently shift it to an S1 no-op.
