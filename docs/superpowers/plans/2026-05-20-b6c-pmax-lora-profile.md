# B6c — `PMaxLoRAProfile` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Subclass `PMaxProfile` into `PMaxLoRAProfile` so the full B-series (`replay_lora_handler`, `downscale_lora_handler`, `restructure_lora_handler`, `recombine_real_handler` VAE) drive a dream/awake `LoRAModel` pair on the P_max profile, with `consolidate_log()` dispatching ch1 (`WeightUpdate`), ch2 (`LatentSample`), and ch3 (`TopologyDiff`) to the awake-side channels. ch4 (`AttentionPrior`) and the α input channel are inherited state surfaces ; encoder + decoder are required kwargs (caller's responsibility).

**Architecture:** Three files ship sequentially. **(1)** A new `tests/unit/profiles/_lora_helpers.py` extracts `lora_clones(seed)` and `assert_lora_models_equal(a, b)` previously duplicated in B6a + B6b ; `test_p_min_lora.py` and `test_p_equ_lora.py` are amended to import from it. **(2)** `kiki_oniric/profiles/p_max_lora.py` defines `PMaxLoRAProfile(PMaxProfile)` with `@dataclass(kw_only=True)`, required `dream_model` / `awake_model` / `encoder` / `decoder` kwargs. `__post_init__` does NOT call `super().__post_init__()` — it registers four B-series handlers (replay + downscale + restructure LoRA on `dream_model`, recombine_real on encoder/decoder/seed), builds `LoRAWeightDeltaChannel`, `LoRAHierarchyChangeChannel`, `LatentSampleQueue(capacity=1024)` on the awake side, and inherits `attention_prior` + `alpha_stream` state surfaces from cycle-3 `PMaxProfile`. `consolidate_log()` lazily imports `apply_channel_outputs`, dispatches ch1+ch2+ch3 (attention defaults to None per the B6a refactor), then clears the log. **(3)** `tests/unit/profiles/test_p_max_lora.py` ships 17 tests including a DR-4 triple chain inclusion check across `PMinLoRAProfile` / `PEquLoRAProfile` / `PMaxLoRAProfile`.

**Tech Stack:** Python 3.12+, `uv`, MLX (`mlx.core`, `mlx.nn`), numpy, pytest, mypy.

**Spec:** `docs/superpowers/specs/2026-05-20-b6c-pmax-lora-profile-design.md`

**Critical not-a-bug:** `__post_init__` deliberately does NOT call `super().__post_init__()`. The parent `PMaxProfile` registers four skeleton handlers ; this subclass registers four LoRA / VAE variants. Calling super would double-register and the new registrations would silently overwrite the skeletons — confusing future readers. Override cleanly.

**4-of-4 state widening:** Every state field is widened to its `_RealState` (in B6b, `recombine_state` stayed `RecombineOpState` skeleton because P_equ uses `recombine_light` which never emits). In B6c, **all four** ops emit and all four states are `_RealState` types. `recombine_state: RecombineRealState` is the B4 type.

**Encoder/decoder typed `Any`:** MLX `nn.Module` has no canonical signature for `encoder(x) -> (mu, log_var)` or `decoder(z) -> reconstruction`. The Protocol surface is documented in `recombine_real_handler`'s docstring ; `Any` documents the expectation here without inventing a fragile Protocol.

---

## File Structure

- **Create** `tests/unit/profiles/_lora_helpers.py` — shared `lora_clones` + `assert_lora_models_equal`.
- **Modify** `tests/unit/profiles/test_p_min_lora.py` — import the shared helpers, drop the local duplicates.
- **Modify** `tests/unit/profiles/test_p_equ_lora.py` — same.
- **Create** `kiki_oniric/profiles/p_max_lora.py` — `PMaxLoRAProfile`.
- **Create** `tests/unit/profiles/test_p_max_lora.py` — 17 tests.
- **Modify** `CHANGELOG.md`, `pyproject.toml`, `uv.lock` — `[C-v0.23.0+PARTIAL]`, version `0.20.0 → 0.21.0`.
- **Modify** `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` — §3.1 note for `PMaxLoRAProfile`.

---

## Task 1: Extract `_lora_helpers.py`

**Files:**
- Create: `tests/unit/profiles/_lora_helpers.py`
- Modify: `tests/unit/profiles/test_p_min_lora.py`
- Modify: `tests/unit/profiles/test_p_equ_lora.py`

- [ ] **Step 1: Create the shared helpers module**

Create `tests/unit/profiles/_lora_helpers.py`:

```python
"""Shared test helpers for the LoRA-substrate profile tests.

Extracted at B6c (third copy trigger). Used by tests for
PMinLoRAProfile (B6a), PEquLoRAProfile (B6b), PMaxLoRAProfile
(B6c). Lives in tests/unit/profiles/ to keep the helpers
profile-local.
"""
from __future__ import annotations

import numpy as np

from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


def lora_clones(seed: int = 0) -> tuple[LoRAModel, LoRAModel]:
    """Two bit-identical LoRAModels at the same seed.

    The dream/awake split for B6a/B6b/B6c needs an awake clone
    bit-equal to the dream model at t=0 so ``consolidate_log()``
    can be verified via bit-equality (within-machine R1).
    """
    return (
        LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=seed),
        LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=seed),
    )


def assert_lora_models_equal(a: LoRAModel, b: LoRAModel) -> None:
    """Assert bit-equality of every layer's base + adapters."""
    assert len(a.layers) == len(b.layers)
    for la, lb in zip(a.layers, b.layers):
        np.testing.assert_array_equal(
            np.asarray(la.base_weight), np.asarray(lb.base_weight),
        )
        np.testing.assert_array_equal(
            np.asarray(la.lora_a), np.asarray(lb.lora_a),
        )
        np.testing.assert_array_equal(
            np.asarray(la.lora_b), np.asarray(lb.lora_b),
        )
```

- [ ] **Step 2: Amend `test_p_min_lora.py` to use the shared helpers**

In `tests/unit/profiles/test_p_min_lora.py`:

(a) Find the existing `_clones` function definition (top of file after imports) and DELETE it.
(b) Find the existing `_assert_lora_models_equal` function and DELETE it.
(c) Add a new import near the other imports at the top:

```python
from tests.unit.profiles._lora_helpers import (
    assert_lora_models_equal,
    lora_clones,
)
```

(d) Within every test body that previously called `_clones(...)` or `_assert_lora_models_equal(...)`, rename the calls to `lora_clones(...)` / `assert_lora_models_equal(...)`. There are typically 10-12 occurrences — search the file for `_clones(` and `_assert_lora_models_equal(` and replace.

If the file uses `from .module import _clones`-style imports, drop them.

- [ ] **Step 3: Amend `test_p_equ_lora.py` to use the shared helpers**

Same operation as Step 2, but on `tests/unit/profiles/test_p_equ_lora.py`. Find local `_clones` and `_assert_lora_models_equal`, delete them, import from `_lora_helpers`, rename call sites.

- [ ] **Step 4: Run both test files to verify they still pass**

Run: `uv run pytest tests/unit/profiles/test_p_min_lora.py tests/unit/profiles/test_p_equ_lora.py -v`
Expected: PASS — 11 (B6a) + 14 (B6b) = 25 tests, same set as before extraction.

- [ ] **Step 5: Full sanity (suite + mypy + ruff)**

Run: `uv run pytest -q` — full suite passes (848 from main, unchanged).
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check tests/unit/profiles/` — clean.

- [ ] **Step 6: Commit**

```bash
git add tests/unit/profiles/_lora_helpers.py tests/unit/profiles/test_p_min_lora.py tests/unit/profiles/test_p_equ_lora.py
git commit -m "$(cat <<'EOF'
refactor(tests): extract shared LoRA profile helpers

B6c / issue #15. Third copy trigger: lora_clones and
assert_lora_models_equal were duplicated in test_p_min_lora.py
(B6a) and test_p_equ_lora.py (B6b). Extract them into
tests/unit/profiles/_lora_helpers.py and import from there.

Same semantics, zero behaviour change — verified by running the
25 B6a+B6b tests after the rename.
EOF
)"
```

---

## Task 2: `PMaxLoRAProfile` subclass

**Files:**
- Create: `kiki_oniric/profiles/p_max_lora.py`
- Create: `tests/unit/profiles/test_p_max_lora.py`

- [ ] **Step 1: Write the failing test file with all 17 tests**

Create `tests/unit/profiles/test_p_max_lora.py`:

```python
"""Unit tests for PMaxLoRAProfile (B6c)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import mlx.core as mx
    import mlx.nn as nn
else:
    mx = pytest.importorskip("mlx.core")
    nn = pytest.importorskip("mlx.nn")

from kiki_oniric.dream.channels.alpha_stream import TraceRecord
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)

from tests.unit.profiles._lora_helpers import (
    assert_lora_models_equal,
    lora_clones,
)


LATENT_DIM = 4
INPUT_DIM = 4


class _TinyEncoder(nn.Module):  # type: ignore[misc,name-defined]
    """Deterministic linear encoder x → (mu, log_var).

    Matches the fixture in tests/unit/test_recombine_latent_sample.py.
    Copied here to avoid a cross-test-file import (pytest collection
    fragility) — small enough that duplication is cheaper than
    extracting into a third helpers module.
    """

    def __init__(self) -> None:
        super().__init__()
        self.mu_w = mx.ones((LATENT_DIM, INPUT_DIM)) * 0.1
        self.lv_w = mx.ones((LATENT_DIM, INPUT_DIM)) * -0.5

    def __call__(self, x):  # noqa: ANN001 — mlx.nn dynamic
        return x @ self.mu_w.T, x @ self.lv_w.T


class _TinyDecoder(nn.Module):  # type: ignore[misc,name-defined]
    """Deterministic linear decoder z → output."""

    def __init__(self) -> None:
        super().__init__()
        self.w = mx.ones((INPUT_DIM, LATENT_DIM)) * 0.2

    def __call__(self, z):  # noqa: ANN001
        return z @ self.w.T


def _replay_episode() -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={
            "beta_records": [
                {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
                {"x": [0.5, 0.6, 0.7, 0.8], "y": [0.0, 1.0]},
            ],
        },
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pmax-lora-replay",
    )


def _downscale_episode(factor: float = 0.5) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"shrink_factor": factor},
        operation_set=(Operation.DOWNSCALE,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pmax-lora-dn",
    )


def _restructure_episode(topo_ops: list[dict[str, object]]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"topo_ops": topo_ops},
        operation_set=(Operation.RESTRUCTURE,),
        output_channels=(OutputChannel.HIERARCHY_CHG,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pmax-lora-restr",
    )


def _recombine_episode() -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"delta_latents": [[1.0, 2.0, 3.0, 4.0]]},
        operation_set=(Operation.RECOMBINE,),
        output_channels=(OutputChannel.LATENT_SAMPLE,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pmax-lora-rec",
    )


def _build_profile(seed: int = 0):
    """Build a PMaxLoRAProfile with bit-identical dream/awake LoRAModels."""
    from kiki_oniric.profiles.p_max_lora import PMaxLoRAProfile

    dream, awake = lora_clones(seed=seed)
    return PMaxLoRAProfile(
        dream_model=dream,
        awake_model=awake,
        encoder=_TinyEncoder(),
        decoder=_TinyDecoder(),
        seed=42,
    ), dream, awake


def test_pmax_lora_construction_happy_path() -> None:
    from kiki_oniric.dream.channels.alpha_stream import AlphaStreamBuffer
    from kiki_oniric.dream.channels.attention_prior import (
        AttentionPriorChannel,
    )
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )

    profile, _, _ = _build_profile()
    assert isinstance(profile.weight_channel, LoRAWeightDeltaChannel)
    assert isinstance(profile.hierarchy_channel, LoRAHierarchyChangeChannel)
    assert isinstance(profile.latent_channel, LatentSampleQueue)
    assert isinstance(profile.attention_prior, AttentionPriorChannel)
    assert isinstance(profile.alpha_stream, AlphaStreamBuffer)
    for op in (
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    ):
        assert op in profile.runtime._handlers


def test_pmax_lora_construction_missing_dream_raises() -> None:
    from kiki_oniric.profiles.p_max_lora import PMaxLoRAProfile

    _, awake = lora_clones(seed=0)
    with pytest.raises(TypeError):
        PMaxLoRAProfile(  # type: ignore[call-arg]
            awake_model=awake,
            encoder=_TinyEncoder(),
            decoder=_TinyDecoder(),
        )


def test_pmax_lora_construction_missing_awake_raises() -> None:
    from kiki_oniric.profiles.p_max_lora import PMaxLoRAProfile

    dream, _ = lora_clones(seed=0)
    with pytest.raises(TypeError):
        PMaxLoRAProfile(  # type: ignore[call-arg]
            dream_model=dream,
            encoder=_TinyEncoder(),
            decoder=_TinyDecoder(),
        )


def test_pmax_lora_construction_missing_encoder_raises() -> None:
    from kiki_oniric.profiles.p_max_lora import PMaxLoRAProfile

    dream, awake = lora_clones(seed=0)
    with pytest.raises(TypeError):
        PMaxLoRAProfile(  # type: ignore[call-arg]
            dream_model=dream,
            awake_model=awake,
            decoder=_TinyDecoder(),
        )


def test_pmax_lora_construction_missing_decoder_raises() -> None:
    from kiki_oniric.profiles.p_max_lora import PMaxLoRAProfile

    dream, awake = lora_clones(seed=0)
    with pytest.raises(TypeError):
        PMaxLoRAProfile(  # type: ignore[call-arg]
            dream_model=dream,
            awake_model=awake,
            encoder=_TinyEncoder(),
        )


def test_pmax_lora_replay_emits_weight_update() -> None:
    from kiki_oniric.dream.channels import WeightUpdate

    profile, _, _ = _build_profile()
    profile.runtime.execute(_replay_episode())
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, WeightUpdate)


def test_pmax_lora_downscale_emits_weight_update() -> None:
    from kiki_oniric.dream.channels import WeightUpdate

    profile, _, _ = _build_profile()
    profile.runtime.execute(_downscale_episode(factor=0.5))
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, WeightUpdate)


def test_pmax_lora_restructure_emits_topology_diff() -> None:
    from kiki_oniric.dream.channels import TopologyDiff

    profile, _, _ = _build_profile()
    profile.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, TopologyDiff)


def test_pmax_lora_recombine_emits_latent_sample() -> None:
    """First profile to emit ch2 — recombine_real_handler VAE."""
    from kiki_oniric.dream.channels import LatentSample

    profile, _, _ = _build_profile()
    profile.runtime.execute(_recombine_episode())
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, LatentSample)


def test_pmax_lora_consolidate_log_applies_weight_to_awake_bit_equal() -> None:
    profile, dream, awake = _build_profile()
    profile.runtime.execute(_replay_episode())
    assert not np.array_equal(
        np.asarray(dream.layers[0].lora_b),
        np.asarray(awake.layers[0].lora_b),
    )
    profile.consolidate_log()
    assert_lora_models_equal(dream, awake)


def test_pmax_lora_consolidate_log_applies_hierarchy_to_awake() -> None:
    profile, dream, awake = _build_profile()
    pre_len = len(dream.layers)
    profile.runtime.execute(
        _restructure_episode([{
            "op": "add",
            "index": pre_len,
            "in_features": 4,
            "out_features": 8,
            "rank": 2,
            "alpha": 4.0,
        }]),
    )
    assert len(dream.layers) == pre_len + 1
    assert len(awake.layers) == pre_len
    profile.consolidate_log()
    assert_lora_models_equal(dream, awake)


def test_pmax_lora_consolidate_log_enqueues_latent_sample() -> None:
    """recombine emits LatentSample → consolidate_log enqueues it."""
    profile, _, _ = _build_profile()
    profile.runtime.execute(_recombine_episode())
    profile.consolidate_log()
    assert len(profile.latent_channel) == 1
    sample = profile.latent_channel.dequeue()
    assert sample is not None
    assert "species" in sample
    assert "latent_vector" in sample
    assert "provenance" in sample


def test_pmax_lora_consolidate_log_mixed_emits_count_4() -> None:
    profile, dream, awake = _build_profile()
    profile.runtime.execute(_replay_episode())
    profile.runtime.execute(_downscale_episode(factor=0.7))
    profile.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    profile.runtime.execute(_recombine_episode())
    assert profile.consolidate_log() == 4


def test_pmax_lora_consolidate_log_clears_and_idempotent() -> None:
    profile, _, _ = _build_profile()
    profile.runtime.execute(_replay_episode())
    profile.consolidate_log()
    assert len(profile.runtime.log) == 0
    assert profile.consolidate_log() == 0


def test_pmax_lora_attention_prior_settable_and_readable() -> None:
    profile, _, _ = _build_profile()
    prior = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    profile.attention_prior.set_prior(prior)
    got = profile.attention_prior.get_prior()
    assert got is not None
    np.testing.assert_array_equal(got, prior)


def test_pmax_lora_alpha_stream_append_and_read() -> None:
    """α input channel inherited from PMaxProfile — append + read FIFO."""
    profile, _, _ = _build_profile()
    rec = TraceRecord(
        tokens=np.zeros(2, dtype=np.int32),
        activations=np.zeros(2, dtype=np.float32),
        attention=np.zeros(2, dtype=np.float32),
        errors=np.zeros(2, dtype=np.float32),
    )
    profile.alpha_stream.append(rec)
    out = profile.alpha_stream.read()
    assert len(out) == 1


def test_pmax_lora_dr4_chain_inclusion_full_triple() -> None:
    """DR-4 across all three LoRA profiles: ops + channel emitters."""
    from kiki_oniric.dream.channels import (
        LatentSample,
        TopologyDiff,
        WeightUpdate,
    )
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile
    from kiki_oniric.profiles.p_max_lora import PMaxLoRAProfile
    from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile

    dream_min, awake_min = lora_clones(seed=0)
    pmin = PMinLoRAProfile(dream_model=dream_min, awake_model=awake_min)

    dream_equ, awake_equ = lora_clones(seed=0)
    pequ = PEquLoRAProfile(dream_model=dream_equ, awake_model=awake_equ)

    dream_max, awake_max = lora_clones(seed=0)
    pmax = PMaxLoRAProfile(
        dream_model=dream_max,
        awake_model=awake_max,
        encoder=_TinyEncoder(),
        decoder=_TinyDecoder(),
        seed=7,
    )

    pmin_ops = set(pmin.runtime._handlers.keys())
    pequ_ops = set(pequ.runtime._handlers.keys())
    pmax_ops = set(pmax.runtime._handlers.keys())
    assert pmin_ops <= pequ_ops, "ops(PMin) ⊆ ops(PEqu)"
    assert pequ_ops <= pmax_ops, "ops(PEqu) ⊆ ops(PMax)"

    pmin.runtime.execute(_replay_episode())
    pmin_emitted = {
        type(o) for entry in pmin.runtime.log
        for o in entry.channel_outputs if o is not None
    }

    pequ.runtime.execute(_replay_episode())
    pequ.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    pequ_emitted = {
        type(o) for entry in pequ.runtime.log
        for o in entry.channel_outputs if o is not None
    }

    pmax.runtime.execute(_replay_episode())
    pmax.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    pmax.runtime.execute(_recombine_episode())
    pmax_emitted = {
        type(o) for entry in pmax.runtime.log
        for o in entry.channel_outputs if o is not None
    }

    assert pmin_emitted == {WeightUpdate}
    assert pequ_emitted == {WeightUpdate, TopologyDiff}
    assert pmax_emitted == {WeightUpdate, TopologyDiff, LatentSample}
    assert pmin_emitted <= pequ_emitted <= pmax_emitted
```

- [ ] **Step 2: Run to verify all 17 tests fail**

Run: `uv run pytest tests/unit/profiles/test_p_max_lora.py -v`
Expected: 17 ERRORS (collection failure — module `kiki_oniric.profiles.p_max_lora` missing).

- [ ] **Step 3: Create `kiki_oniric/profiles/p_max_lora.py`**

Create the file with the body shown in the spec (`docs/superpowers/specs/2026-05-20-b6c-pmax-lora-profile-design.md` § "New file `kiki_oniric/profiles/p_max_lora.py`"). The canonical body:

```python
"""P_max LoRA-substrate profile (B6c, issue #15 continuation).

Subclass of ``PMaxProfile`` that wires the full B-series :
- ``replay_lora_handler`` (B1b) → ch1
- ``downscale_lora_handler`` (B2) → ch1
- ``restructure_lora_handler`` (B3) → ch3
- ``recombine_real_handler`` (B4 VAE) → ch2

Channels out (per framework-C spec §3.1, primitives_out={1,2,3,4}):
- ch1 ``WeightUpdate`` via ``LoRAWeightDeltaChannel`` (B5).
- ch2 ``LatentSample`` via ``LatentSampleQueue`` (B5).
- ch3 ``TopologyDiff`` via ``LoRAHierarchyChangeChannel`` (B5).
- ch4 ``AttentionPrior`` via inherited ``AttentionPriorChannel`` —
  state surface only, populated externally via
  ``profile.attention_prior.set_prior(prior)``. No op emits.

Input channel α : ``AlphaStreamBuffer`` (inherited from
``PMaxProfile``), populated externally by the awake side via
``profile.alpha_stream.append(TraceRecord(...))``. Not dispatched
by ``consolidate_log()`` (α is awake → dream input, not output).

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

from kiki_oniric.dream.channels.hierarchy_change import (
    LoRAHierarchyChangeChannel,
)
from kiki_oniric.dream.channels.latent_sample import LatentSampleQueue
from kiki_oniric.dream.channels.weight_delta import (
    LoRAWeightDeltaChannel,
)
from kiki_oniric.dream.episode import Operation
from kiki_oniric.dream.operations.downscale_real import (
    DownscaleRealState,
    downscale_lora_handler,
)
from kiki_oniric.dream.operations.recombine_real import (
    RecombineRealState,
    recombine_real_handler,
)
from kiki_oniric.dream.operations.replay_real import (
    ReplayRealState,
    replay_lora_handler,
)
from kiki_oniric.dream.operations.restructure_real import (
    RestructureRealState,
    restructure_lora_handler,
)
from kiki_oniric.profiles.p_max import PMaxProfile

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


@dataclass(kw_only=True)
class PMaxLoRAProfile(PMaxProfile):
    """P_max rewired for the B-series LoRA substrate + VAE recombine.

    Required kwargs : ``dream_model``, ``awake_model`` (LoRAModel
    pair), ``encoder``, ``decoder`` (MLX nn.Module VAE pair for
    ``recombine_real_handler``).

    Optional kwargs : ``lr=0.01``, ``max_adds_per_episode=1``,
    ``seed=0``, ``latent_queue_capacity=1024``.

    Parent state fields (``replay_state``, ``downscale_state``,
    ``restructure_state``, ``recombine_state``) are widened from
    cycle-3 skeleton ``OpState`` types to ``_RealState`` variants
    required by the B-series LoRA / VAE handlers.

    Inherited from ``PMaxProfile`` cycle-3 :
    - ``alpha_stream: AlphaStreamBuffer`` — awake → dream input
      channel state surface.
    - ``attention_prior: AttentionPriorChannel`` — ch4 state
      surface.
    - ``rng: random.Random`` — kept on the dataclass but unused
      by the LoRA handlers (which use MLX RNG keyed off
      ``seed``).

    ``__post_init__`` intentionally does NOT call
    ``super().__post_init__()`` — the parent registers cycle-3
    skeleton handlers ; we register the B-series LoRA / VAE
    variants on the same runtime instead.
    """

    dream_model: "LoRAModel"
    awake_model: "LoRAModel"
    encoder: Any
    decoder: Any
    lr: float = 0.01
    max_adds_per_episode: int = 1
    seed: int = 0
    latent_queue_capacity: int = 1024
    replay_state: ReplayRealState = field(  # type: ignore[assignment]
        default_factory=ReplayRealState,
    )
    downscale_state: DownscaleRealState = field(  # type: ignore[assignment]
        default_factory=DownscaleRealState,
    )
    restructure_state: RestructureRealState = field(  # type: ignore[assignment]
        default_factory=RestructureRealState,
    )
    recombine_state: RecombineRealState = field(  # type: ignore[assignment]
        default_factory=RecombineRealState,
    )
    weight_channel: LoRAWeightDeltaChannel | None = None
    hierarchy_channel: LoRAHierarchyChangeChannel | None = None
    latent_channel: LatentSampleQueue | None = None

    def __post_init__(self) -> None:
        self.runtime.register_handler(
            Operation.REPLAY,
            replay_lora_handler(
                self.replay_state,
                model=self.dream_model,
                lr=self.lr,
            ),
        )
        self.runtime.register_handler(
            Operation.DOWNSCALE,
            downscale_lora_handler(
                self.downscale_state,
                model=self.dream_model,
            ),
        )
        self.runtime.register_handler(
            Operation.RESTRUCTURE,
            restructure_lora_handler(
                self.restructure_state,
                model=self.dream_model,
                max_adds_per_episode=self.max_adds_per_episode,
                seed=self.seed,
            ),
        )
        self.runtime.register_handler(
            Operation.RECOMBINE,
            recombine_real_handler(
                self.recombine_state,
                encoder=self.encoder,
                decoder=self.decoder,
                seed=self.seed,
            ),
        )
        self.weight_channel = LoRAWeightDeltaChannel(self.awake_model)
        self.hierarchy_channel = LoRAHierarchyChangeChannel(
            self.awake_model,
        )
        self.latent_channel = LatentSampleQueue(
            capacity=self.latent_queue_capacity,
        )

    def consolidate_log(self) -> int:
        """Dispatch the runtime log onto awake-side channels :
        ch1 (``WeightUpdate``) via ``weight_channel``, ch2
        (``LatentSample``) via ``latent_channel`` (queue), ch3
        (``TopologyDiff``) via ``hierarchy_channel``. Then clear
        the log.

        ``attention_channel`` defaults to ``None`` (apply_channel
        _outputs's relaxed kwargs since B6a) because no op
        currently emits ``AttentionPrior`` into the runtime log ;
        the profile's ``attention_prior`` field is a state
        surface for external callers.

        The α input channel is not in the apply loop : it carries
        awake → dream traces, populated by the awake side via
        ``profile.alpha_stream.append(...)``.

        Returns the number of channel outputs dispatched. The
        log is cleared on success so a second call without
        further ``runtime.execute()`` returns 0 (idempotent
        no-op).
        """
        from kiki_oniric.consolidate import apply_channel_outputs

        count = apply_channel_outputs(
            self.runtime.log,
            weight_channel=self.weight_channel,
            hierarchy_channel=self.hierarchy_channel,
            latent_channel=self.latent_channel,
        )
        self.runtime.reset_log()
        return count
```

- [ ] **Step 4: Run to verify all tests pass**

Run: `uv run pytest tests/unit/profiles/test_p_max_lora.py -v`
Expected: PASS — 17 tests.

- [ ] **Step 5: Full sanity (suite + mypy + ruff)**

Run: `uv run pytest -q` — expect 848 (post-B6b baseline) + 17 (B6c new) = 865 passed, 3 skipped, 12 xfailed.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/profiles/p_max_lora.py tests/unit/profiles/test_p_max_lora.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/profiles/p_max_lora.py tests/unit/profiles/test_p_max_lora.py
git commit -m "$(cat <<'EOF'
feat(profile): add PMaxLoRAProfile (B6c)

B6c / issue #15 continuation. Subclass of PMaxProfile with
kw-only dream_model + awake_model + encoder + decoder kwargs
(encoder/decoder typed Any — MLX nn.Module pair, no Protocol).

__post_init__ registers 4 B-series handlers: replay_lora +
downscale_lora + restructure_lora (all on dream_model) +
recombine_real (on encoder/decoder/seed). Builds
LoRAWeightDeltaChannel + LoRAHierarchyChangeChannel on
awake_model, LatentSampleQueue(capacity=1024) as ch2 queue.
Inherits alpha_stream (awake to dream input) and attention_prior
state surfaces from cycle-3 PMaxProfile — neither dispatched by
consolidate_log.

consolidate_log() dispatches ch1 + ch2 + ch3 via
apply_channel_outputs (attention_channel defaults to None per
B6a refactor) and clears the log.

17 tests cover construction (5: happy + 4 missing-arg), per-op
emission (4), consolidation (5), attention surface (1), alpha
stream (1), DR-4 triple chain inclusion across PMin/PEqu/PMax
LoRA profiles (1).

All 4 op states widened to _RealState (vs B6b which kept
RecombineOpState skeleton). recombine_real_handler is the B4
VAE — encoder + decoder injection per spec §3.1.
EOF
)"
```

---

## Task 3: Documentation and DualVer sync

**Files:**
- Modify: `CHANGELOG.md`, `pyproject.toml`, `uv.lock`
- Modify: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/…`

- [ ] **Step 1: Add the CHANGELOG entry**

Insert at the top of the `CHANGELOG.md` body, immediately above the existing `[C-v0.22.0+PARTIAL]` entry:

```markdown
## [C-v0.23.0+PARTIAL] — 2026-05-20 — PMaxLoRAProfile wires P_max to channels (B6c)

### Formal axis (FC) — MINOR (v0.22.0 → v0.23.0)

- **New subclass** `kiki_oniric/profiles/p_max_lora.py`:
  `PMaxLoRAProfile(PMaxProfile)` with kw-only `dream_model`,
  `awake_model`, `encoder`, `decoder` kwargs (plus `lr=0.01`,
  `max_adds_per_episode=1`, `seed=0`,
  `latent_queue_capacity=1024`). Registers four B-series
  handlers : `replay_lora_handler`, `downscale_lora_handler`,
  `restructure_lora_handler` (all bound to `dream_model`) +
  `recombine_real_handler` (bound to `encoder` / `decoder` /
  `seed` per the B4 VAE contract). Builds three dispatch
  channels on the awake side : `LoRAWeightDeltaChannel`,
  `LoRAHierarchyChangeChannel`, `LatentSampleQueue(capacity=
  1024)`. Inherits `alpha_stream` (awake → dream input ring
  buffer) and `attention_prior` (ch4 state surface) fields
  from cycle-3 `PMaxProfile` — neither is dispatched by
  `consolidate_log()` (α is input-side ; ch4 has no op
  emitter).
- `consolidate_log() -> int` dispatches ch1 + ch2 + ch3 via
  `apply_channel_outputs` and clears the log on success.
  Attention defaults to `None` per the B6a refactor.
- **State widening, 4-of-4** : all four state fields are
  widened from cycle-3 skeleton `OpState` types to `_RealState`
  variants (vs B6b which kept `RecombineOpState` skeleton for
  `recombine_light`).
- **Test helper extraction** : `tests/unit/profiles/_lora_helpers
  .py` ships `lora_clones(seed)` and
  `assert_lora_models_equal(a, b)` previously duplicated in
  B6a + B6b test files. `test_p_min_lora.py` and
  `test_p_equ_lora.py` were amended to import from the shared
  module (zero behaviour change).
- **DR-4 triple chain inclusion** : Test 17 (in
  `test_p_max_lora.py`) pins the strict-subset chain on both
  ops keys (`ops(PMinLoRA) ⊆ ops(PEquLoRA) ⊆ ops(PMaxLoRA)`)
  and emitted channel types (`{WeightUpdate} ⊆ {WeightUpdate,
  TopologyDiff} ⊆ {WeightUpdate, TopologyDiff, LatentSample}`).
  First empirical pin of the full triple LoRA chain.
- Sub-project B6c of issue #15. **Closes the B6 profile-wiring
  decomposition** (B6a PMin done, B6b PEqu done, B6c PMax this
  entry). Cycle-3 `PMaxProfile` untouched — legacy tests +
  cycle-3 pilots unaffected.

### Empirical axis (EC) — UNCHANGED (PARTIAL)

- No new substrate, axiom, or empirical claim. EC stays
  `+PARTIAL`.

### Packaging

- `pyproject.toml` version bumped `0.20.0 → 0.21.0`.
```

Match the formatting of the surrounding entries.

- [ ] **Step 2: Bump `pyproject.toml`**

Change `version = "0.20.0"` to `version = "0.21.0"`.

- [ ] **Step 3: Sync the lockfile**

Run: `uv lock`
Expected: `uv.lock` updates the `dreamofkiki` pin to `0.21.0`.

- [ ] **Step 4: Update the framework-C spec (EN)**

In `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §3.1, immediately after the line defining `P_max = { primitives_in: {α,β,δ}, primitives_out: {1,2,3,4}, ops: {replay, downscale, restructure, recombine_full} }`, append:

```markdown
As of B6c (issue #15), P_max has a LoRA-substrate variant
`PMaxLoRAProfile` (in `kiki_oniric/profiles/p_max_lora.py`)
that wires the full B-series : LoRA-emitting handlers for
replay / downscale / restructure on a dream/awake `LoRAModel`
pair, plus `recombine_real_handler` (B4 VAE) on a required
encoder + decoder MLX `nn.Module` pair. `consolidate_log()`
dispatches ch1 (`WeightUpdate`), ch2 (`LatentSample`,
`LatentSampleQueue(capacity=1024)`), and ch3 (`TopologyDiff`)
to the awake side. ch4 (`AttentionPrior`) and the α input
channel are inherited from cycle-3 `PMaxProfile` as state
surfaces. With B6a, B6b, B6c, all three profile tiers have
LoRA-substrate variants forming a strict DR-4 subset chain
(verified empirically). The cycle-3 `PMaxProfile` remains the
canonical skeleton reference. B6 closes the profile-wiring
decomposition.
```

- [ ] **Step 5: Update the framework-C spec (FR mirror)**

In `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` §3.1, at the matching location, append the French sentence (code identifiers stay in original form):

```markdown
Depuis B6c (issue #15), P_max possède une variante de substrat
LoRA `PMaxLoRAProfile` (dans
`kiki_oniric/profiles/p_max_lora.py`) qui branche toute la
série B : gestionnaires LoRA pour replay / downscale /
restructure sur une paire de `LoRAModel` rêve/éveil, plus
`recombine_real_handler` (VAE de B4) sur une paire encoder +
decoder MLX `nn.Module` requise. `consolidate_log()` répartit
ch1 (`WeightUpdate`), ch2 (`LatentSample`,
`LatentSampleQueue(capacity=1024)`) et ch3 (`TopologyDiff`)
vers le côté éveil. ch4 (`AttentionPrior`) et le canal d'entrée
α sont hérités de `PMaxProfile` cycle-3 comme surfaces d'état.
Avec B6a, B6b et B6c, les trois niveaux de profile possèdent
des variantes substrat LoRA formant une chaîne d'inclusion
DR-4 stricte (vérifiée empiriquement). Le `PMaxProfile`
cycle-3 reste la référence squelette canonique. B6 ferme la
décomposition de câblage des profiles.
```

- [ ] **Step 6: Verify**

Run: `uv run pytest -q` — full suite passes (docs/version change, no code touched).
Run: `uv run mypy harness tests` — `Success`.

If `tests/reproducibility/golden_hashes_apple_*.json` shows modified (per-family drift from running pytest), restore with `git checkout -- tests/reproducibility/golden_hashes_apple_*.json` before committing.

- [ ] **Step 7: Commit**

```bash
git add CHANGELOG.md pyproject.toml uv.lock docs/specs/2026-04-17-dreamofkiki-framework-C-design.md docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md
git commit -m "$(cat <<'EOF'
docs: sync spec for B6c PMaxLoRAProfile

B6c / issue #15. Add the C-v0.23.0+PARTIAL changelog entry,
note in framework-C spec section 3.1 (EN + FR) that P_max has
a LoRA-substrate variant PMaxLoRAProfile wiring channels
{1, 2, 3, 4} with required encoder/decoder VAE pair plus
alpha_stream and attention_prior state surfaces inherited
from cycle-3, and bump the package version to 0.21.0.

Closes the B6 profile-wiring decomposition: B6a PMin done,
B6b PEqu done, B6c PMax this entry. The three LoRA profile
variants form a strict DR-4 subset chain empirically pinned
by Test 17 in test_p_max_lora.py.
EOF
)"
```

---

## Task 4: Final verification + push

**Files:** none — verification only.

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -q`
Expected: 865 passed, 3 skipped, 12 xfailed (848 from B6b baseline + 17 new B6c tests), 0 failures, coverage gate met.

- [ ] **Step 2: Type check**

Run: `uv run mypy harness tests`
Expected: `Success: no issues found in 174 source files` (172 from B6b + `_lora_helpers.py` + `p_max_lora.py`).

- [ ] **Step 3: Lint**

Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 4: Confirm working tree is clean**

Run: `git status --short`
Expected: empty (the per-family golden hashes file may show modified — restore it).

- [ ] **Step 5: Push and update issue #15**

```bash
git push origin main
```

Comment on issue #15 (closed but still receives comments): B6c complete — `PMaxLoRAProfile` wires P_max to the B5 channel apply loop on channels `{1, 2, 3, 4}` with `recombine_real_handler` (B4 VAE) for ch2 emission. **Closes the B6 profile-wiring decomposition (B6a/B6b/B6c all delivered).** DR-4 triple chain inclusion across all three LoRA profiles empirically pinned by Test 17. Cycle-3 profiles intact.

---

## Self-Review

- **Spec coverage:**
  - `_lora_helpers.py` extract with `lora_clones` + `assert_lora_models_equal` (Task 1) ✓
  - Amend `test_p_min_lora.py` + `test_p_equ_lora.py` to import shared helpers (Task 1 Steps 2-3) ✓
  - `PMaxLoRAProfile(PMaxProfile)` subclass with kw-only kwargs `dream_model` / `awake_model` / `encoder` / `decoder` + optionals (Task 2 Step 3) ✓
  - All 4 state fields widened to `_RealState` (Task 2 Step 3) ✓
  - `__post_init__` registers 4 B-series handlers (3 LoRA + 1 VAE), builds 3 awake-side channels, does NOT call super (Task 2 Step 3) ✓
  - `consolidate_log()` dispatches ch1 + ch2 + ch3, attention default None, reset_log after (Task 2 Step 3) ✓
  - 17 tests cover all spec test plan items (Task 2 Step 1) ✓
  - CHANGELOG + spec §3.1 EN+FR + pyproject 0.21.0 (Task 3) ✓
  - Final verification + push (Task 4) ✓
- **Placeholder scan:** no TBD/TODO; every code step has complete code.
- **Type consistency:**
  - `PMaxLoRAProfile(*, dream_model, awake_model, encoder, decoder, lr=0.01, max_adds_per_episode=1, seed=0, latent_queue_capacity=1024, ...)` — same field names used by every test in Task 2.
  - `consolidate_log() -> int` — same signature in implementation and tests.
  - `LoRAWeightDeltaChannel(target)`, `LoRAHierarchyChangeChannel(target)`, `LatentSampleQueue(capacity=...)` — B5 channel constructors used identically.
  - `AttentionPriorChannel`, `AlphaStreamBuffer` — inherited from cycle-3 `PMaxProfile`.
  - `ReplayRealState`, `DownscaleRealState`, `RestructureRealState`, `RecombineRealState` — all 4 `_RealState` types, same names as the LoRA / VAE handlers expect.
  - `replay_lora_handler(state, *, model, lr)`, `downscale_lora_handler(state, *, model)`, `restructure_lora_handler(state, *, model, max_adds_per_episode, seed)`, `recombine_real_handler(state, *, encoder, decoder, seed)` — exact signatures from B1b/B2/B3/B4.
  - `apply_channel_outputs(log, *, weight_channel=None, hierarchy_channel=None, latent_channel=None, attention_channel=None)` — relaxed signature shipped in B6a Task 1.
  - `lora_clones(seed)` and `assert_lora_models_equal(a, b)` — same function names in Task 1's helpers module and in Task 2's test file.
  - `_TinyEncoder(LATENT_DIM, INPUT_DIM)` and `_TinyDecoder(...)` are local fixtures (duplicated from `tests/unit/test_recombine_latent_sample.py`) to avoid cross-test-file imports.
- **Inter-task ordering:** Task 1 (helpers extract) must land before Task 2 (which depends on the shared module). Task 3 (docs) depends on Task 2. Task 4 (verif + push) depends on all prior tasks. Chosen 1→2→3→4 is the dependency order.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-20-b6c-pmax-lora-profile.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, inline review by me between tasks (matches the B6a / B6b pattern).

**2. Inline Execution** — execute tasks in this session.
