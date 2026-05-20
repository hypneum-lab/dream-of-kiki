# B6a — `PMinLoRAProfile` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Subclass `PMinProfile` into `PMinLoRAProfile` so the B-series LoRA-emitting handlers (`replay_lora_handler`, `downscale_lora_handler`) actually drive a dream/awake `LoRAModel` pair, with `consolidate_log()` closing the loop via the B5 `LoRAWeightDeltaChannel`.

**Architecture:** Three small changes ship in dependency order. **(1)** `apply_channel_outputs` (consolidate.py) relaxes its three channel kwargs to `Optional[...] = None`, raising `ValueError` only when a matching output type appears with a `None` channel. **(2)** `DreamRuntime` (dream/runtime.py) gains a one-line `reset_log()` helper. **(3)** A new `kiki_oniric/profiles/p_min_lora.py` defines `PMinLoRAProfile(PMinProfile)` with `@dataclass(kw_only=True)`, required `dream_model` / `awake_model` `LoRAModel`s, overridden `replay_state` / `downscale_state` to the `_RealState` variants, and a `consolidate_log()` method that calls `apply_channel_outputs` then `reset_log()`.

**Tech Stack:** Python 3.12+, `uv`, MLX (`mlx.core`), numpy, pytest, mypy.

**Spec:** `docs/superpowers/specs/2026-05-20-b6a-pmin-lora-profile-design.md`

**Critical not-a-bug:** `PMinLoRAProfile.__post_init__` deliberately does NOT call `super().__post_init__()`. The parent registers the skeleton `replay_handler` / `downscale_handler` (no emission), but we want the LoRA-emitting variants on the same runtime instance. Calling super would register both, and the LoRA registration would silently overwrite the skeleton one — confusing future readers. Better to override cleanly.

---

## File Structure

- **Modify** `kiki_oniric/consolidate.py` — relax `apply_channel_outputs` signature ; add per-type `ValueError` for missing channels.
- **Modify** `kiki_oniric/dream/runtime.py` — add `DreamRuntime.reset_log()`.
- **Create** `kiki_oniric/profiles/p_min_lora.py` — `PMinLoRAProfile`.
- **Create** `tests/unit/profiles/test_p_min_lora.py` — 11 tests.
- **Modify** `tests/unit/test_runtime.py` (if exists) — add one test for `reset_log`. If not, add to the new B6a test file as a runtime-level test (Test 11).
- **Modify** `tests/unit/test_apply_channel_outputs.py` — extend with the "None channel raises" cases for weight / hierarchy / latent (the attention case already exists ; we mirror it).
- **Modify** `CHANGELOG.md`, `pyproject.toml`, `uv.lock` — `[C-v0.21.0+PARTIAL]`, version `0.18.0 → 0.19.0`.
- **Modify** `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` — §3.1 note for `PMinLoRAProfile`.

---

## Task 1: Relax `apply_channel_outputs` signature

**Files:**
- Modify: `kiki_oniric/consolidate.py`
- Test: `tests/unit/test_apply_channel_outputs.py` (append)

- [ ] **Step 1: Append the new failing tests**

Append to `tests/unit/test_apply_channel_outputs.py`:

```python
def test_apply_weight_required_when_emitted() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.latent_sample import (
        LatentSampleQueue,
    )

    _, target = _clones(seed=0)
    delta = np.zeros(
        np.asarray(target.layers[0].lora_a).shape, dtype=np.float32,
    )
    log = _make_log_with_one_output(
        WeightUpdate(lora_delta={"layer0.lora_a": delta}),
    )
    with pytest.raises(ValueError, match="weight_channel"):
        apply_channel_outputs(
            log,
            hierarchy_channel=LoRAHierarchyChangeChannel(target),
            latent_channel=LatentSampleQueue(),
        )


def test_apply_hierarchy_required_when_emitted() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import TopologyDiff
    from kiki_oniric.dream.channels.latent_sample import (
        LatentSampleQueue,
    )
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )

    _, target = _clones(seed=0)
    entry = (
        "reroute",
        {"swap_indices": (0, 1), "model_sha256_post": "0" * 64},
    )
    log = _make_log_with_one_output(TopologyDiff(diff=(entry,)))
    with pytest.raises(ValueError, match="hierarchy_channel"):
        apply_channel_outputs(
            log,
            weight_channel=LoRAWeightDeltaChannel(target),
            latent_channel=LatentSampleQueue(),
        )


def test_apply_latent_required_when_emitted() -> None:
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import LatentSample
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )

    _, target = _clones(seed=0)
    log = _make_log_with_one_output(
        LatentSample(
            species="default",
            latent_vector=np.array([0.1, 0.2], dtype=np.float32),
            provenance="recombine:de=test:ep=0:seed=0",
        ),
    )
    with pytest.raises(ValueError, match="latent_channel"):
        apply_channel_outputs(
            log,
            weight_channel=LoRAWeightDeltaChannel(target),
            hierarchy_channel=LoRAHierarchyChangeChannel(target),
        )


def test_apply_weight_only_omits_hierarchy_and_latent() -> None:
    """A log with WeightUpdate only can omit hierarchy + latent kwargs."""
    from kiki_oniric.consolidate import apply_channel_outputs
    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )

    _, target = _clones(seed=0)
    delta = np.ones(
        np.asarray(target.layers[0].lora_a).shape, dtype=np.float32,
    ) * 0.1
    log = _make_log_with_one_output(
        WeightUpdate(lora_delta={"layer0.lora_a": delta}),
    )
    count = apply_channel_outputs(
        log,
        weight_channel=LoRAWeightDeltaChannel(target),
    )
    assert count == 1
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py -k "required_when_emitted or weight_only_omits" -v`
Expected: FAIL — `TypeError: apply_channel_outputs() missing 3 required keyword-only arguments` (or similar) for the omit test, and the three "required" tests fail because the function raises before reaching the isinstance check.

- [ ] **Step 3: Relax the signature in `consolidate.py`**

Open `kiki_oniric/consolidate.py`. Locate the existing `apply_channel_outputs` function. Replace its signature and the four dispatch branches with the version below (the body before the dispatch and `count` accounting stays unchanged):

```python
def apply_channel_outputs(
    log: list[EpisodeLogEntry],
    *,
    weight_channel: "WeightDeltaChannel | None" = None,
    hierarchy_channel: "HierarchyChangeChannel | None" = None,
    latent_channel: "LatentSampleChannel | None" = None,
    attention_channel: "AttentionPriorChannel | None" = None,
) -> int:
    """Dispatch every non-``None`` channel output in ``log`` to the
    matching concrete channel and return the count.

    Every channel kwarg is optional. If a ``ChannelOutput`` of a
    given type appears in the log but the matching channel kwarg
    is ``None``, a ``ValueError`` is raised pointing at the
    missing kwarg.

    [... existing docstring sections ...]
    """
    count = 0
    for entry in log:
        for output in entry.channel_outputs:
            if output is None:
                continue
            if isinstance(output, WeightUpdate):
                if weight_channel is None:
                    raise ValueError(
                        "apply_channel_outputs: weight_channel "
                        "required for WeightUpdate outputs"
                    )
                weight_channel.apply(
                    output.lora_delta, output.fisher_bump,
                )
            elif isinstance(output, TopologyDiff):
                if hierarchy_channel is None:
                    raise ValueError(
                        "apply_channel_outputs: hierarchy_channel "
                        "required for TopologyDiff outputs"
                    )
                hierarchy_channel.apply_diff(list(output.diff))
            elif isinstance(output, LatentSample):
                if latent_channel is None:
                    raise ValueError(
                        "apply_channel_outputs: latent_channel "
                        "required for LatentSample outputs"
                    )
                latent_channel.enqueue(
                    output.species,
                    output.latent_vector,
                    output.provenance,
                )
            elif isinstance(output, AttentionPrior):
                if attention_channel is None:
                    raise ValueError(
                        "apply_channel_outputs: attention_channel "
                        "required for AttentionPrior outputs"
                    )
                attention_channel.set_prior(output.prior)
            else:
                raise TypeError(
                    f"apply_channel_outputs: unknown ChannelOutput "
                    f"type {type(output).__name__}"
                )
            count += 1
    return count
```

- [ ] **Step 4: Run to verify they pass**

Run: `uv run pytest tests/unit/test_apply_channel_outputs.py -v`
Expected: PASS — all existing tests still pass + 4 new tests pass.

- [ ] **Step 5: Full suite + mypy + ruff**

Run: `uv run pytest -q` — all pass.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/consolidate.py tests/unit/test_apply_channel_outputs.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/consolidate.py tests/unit/test_apply_channel_outputs.py
git commit -m "$(cat <<'EOF'
refactor(consolidate): relax channel kwargs to optional

B6a / issue #15. apply_channel_outputs previously required all
three of weight_channel, hierarchy_channel, latent_channel as
kwargs (only attention_channel was optional). Profiles that
emit only ch1 (PMin) would have to pass dummy stubs.

This change makes all four channel kwargs Optional[...] = None
and raises ValueError(<which kwarg>) only when a matching output
type appears in the log with the channel set to None. Backwards
compatible: existing callers passing all three channels still
work identically.

Mirrors the existing attention_channel pattern. Unblocks
PMinLoRAProfile (B6a) which only needs weight_channel.
EOF
)"
```

---

## Task 2: `DreamRuntime.reset_log()` helper

**Files:**
- Modify: `kiki_oniric/dream/runtime.py`
- Test: `tests/unit/profiles/test_p_min_lora.py` (create — first test)

- [ ] **Step 1: Create the test file with the runtime test**

Create `tests/unit/profiles/test_p_min_lora.py`:

```python
"""Unit tests for PMinLoRAProfile and the DreamRuntime.reset_log helper (B6a)."""
from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    import mlx.core as mx
else:
    mx = pytest.importorskip("mlx.core")

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.runtime import DreamRuntime
from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


def _clones(seed: int = 0) -> tuple[LoRAModel, LoRAModel]:
    """Two bit-identical LoRAModels at the same seed."""
    return (
        LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=seed),
        LoRAModel((4, 8, 2), rank=2, alpha=4.0, seed=seed),
    )


def _assert_lora_models_equal(a: LoRAModel, b: LoRAModel) -> None:
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


def _replay_episode(records: list[dict] | None = None) -> DreamEpisode:
    if records is None:
        records = [
            {"x": [0.1, 0.2, 0.3, 0.4], "y": [1.0, 0.0]},
            {"x": [0.5, 0.6, 0.7, 0.8], "y": [0.0, 1.0]},
        ]
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"beta_records": records},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pmin-lora",
    )


def _downscale_episode(factor: float = 0.5) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"shrink_factor": factor},
        operation_set=(Operation.DOWNSCALE,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pmin-lora-dn",
    )


def test_dream_runtime_reset_log_clears() -> None:
    """DreamRuntime.reset_log() drops every entry."""
    from kiki_oniric.dream.operations.replay_real import (
        ReplayRealState,
        replay_lora_handler,
    )

    dream, _ = _clones(seed=0)
    runtime = DreamRuntime()
    state = ReplayRealState()
    runtime.register_handler(
        Operation.REPLAY,
        replay_lora_handler(state, model=dream, lr=0.01),
    )
    runtime.execute(_replay_episode())
    runtime.execute(_replay_episode())
    assert len(runtime.log) == 2
    runtime.reset_log()
    assert len(runtime.log) == 0
```

- [ ] **Step 2: Run to verify it fails**

Run: `uv run pytest tests/unit/profiles/test_p_min_lora.py::test_dream_runtime_reset_log_clears -v`
Expected: FAIL — `AttributeError: 'DreamRuntime' object has no attribute 'reset_log'`.

- [ ] **Step 3: Add `reset_log` to `DreamRuntime`**

In `kiki_oniric/dream/runtime.py`, locate the `DreamRuntime` class. After the `execute` method (which ends with the `self._log.append(...)` block), append the new method:

```python
    def reset_log(self) -> None:
        """Clear the accountability log (DR-0 trace).

        Profile-driven consolidation flows that consume the log
        via ``apply_channel_outputs`` use this to start the next
        dream cycle with a clean slate. The handler registrations
        and ``_handlers`` mapping are not touched.
        """
        self._log.clear()
```

- [ ] **Step 4: Run to verify it passes**

Run: `uv run pytest tests/unit/profiles/test_p_min_lora.py::test_dream_runtime_reset_log_clears -v`
Expected: PASS.

- [ ] **Step 5: Full suite + mypy + ruff**

Run: `uv run pytest -q` — all pass.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/dream/runtime.py tests/unit/profiles/test_p_min_lora.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/dream/runtime.py tests/unit/profiles/test_p_min_lora.py
git commit -m "$(cat <<'EOF'
feat(runtime): DreamRuntime.reset_log helper

B6a / issue #15. Adds a one-line reset_log() method to
DreamRuntime that clears the accountability log without
touching handler registrations. Used by PMinLoRAProfile's
consolidate_log() so a profile-driven consolidation cycle can
flush its applied outputs and start fresh.
EOF
)"
```

---

## Task 3: `PMinLoRAProfile` subclass

**Files:**
- Create: `kiki_oniric/profiles/p_min_lora.py`
- Test: `tests/unit/profiles/test_p_min_lora.py` (append)

- [ ] **Step 1: Append the 10 failing tests**

Append to `tests/unit/profiles/test_p_min_lora.py`:

```python
def test_pmin_lora_construction_happy_path() -> None:
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )
    from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PMinLoRAProfile(dream_model=dream, awake_model=awake)
    assert isinstance(profile.weight_channel, LoRAWeightDeltaChannel)
    # Both LoRA handlers must be registered on the runtime.
    assert Operation.REPLAY in profile.runtime._handlers
    assert Operation.DOWNSCALE in profile.runtime._handlers


def test_pmin_lora_construction_missing_dream_raises() -> None:
    from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile

    _, awake = _clones(seed=0)
    with pytest.raises(TypeError):
        PMinLoRAProfile(awake_model=awake)  # type: ignore[call-arg]


def test_pmin_lora_construction_missing_awake_raises() -> None:
    from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile

    dream, _ = _clones(seed=0)
    with pytest.raises(TypeError):
        PMinLoRAProfile(dream_model=dream)  # type: ignore[call-arg]


def test_pmin_lora_replay_emits_weight_update_in_log() -> None:
    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PMinLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, WeightUpdate)


def test_pmin_lora_downscale_emits_weight_update_in_log() -> None:
    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PMinLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_downscale_episode(factor=0.5))
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, WeightUpdate)


def test_pmin_lora_consolidate_log_applies_to_awake_bit_equal() -> None:
    from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PMinLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())

    # Sanity: dream mutated, awake untouched yet.
    assert not np.array_equal(
        np.asarray(dream.layers[0].lora_b),
        np.asarray(awake.layers[0].lora_b),
    )
    profile.consolidate_log()
    _assert_lora_models_equal(dream, awake)


def test_pmin_lora_consolidate_log_clears_log() -> None:
    from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PMinLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())
    assert len(profile.runtime.log) == 1
    profile.consolidate_log()
    assert len(profile.runtime.log) == 0


def test_pmin_lora_consolidate_log_returns_dispatch_count() -> None:
    from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PMinLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())
    profile.runtime.execute(_downscale_episode(factor=0.5))
    profile.runtime.execute(_replay_episode())
    # Each episode emits exactly one WeightUpdate → 3 outputs.
    assert profile.consolidate_log() == 3


def test_pmin_lora_consolidate_log_idempotent_on_empty() -> None:
    from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PMinLoRAProfile(dream_model=dream, awake_model=awake)
    assert profile.consolidate_log() == 0
    assert profile.consolidate_log() == 0
    _assert_lora_models_equal(dream, awake)


def test_pmin_lora_no_topology_no_latent_in_log() -> None:
    """PMin's spec channel set is {WEIGHT_DELTA} only."""
    from kiki_oniric.dream.channels import (
        LatentSample,
        TopologyDiff,
        WeightUpdate,
    )
    from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PMinLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())
    profile.runtime.execute(_downscale_episode(factor=0.7))
    for entry in profile.runtime.log:
        for output in entry.channel_outputs:
            if output is None:
                continue
            assert isinstance(output, WeightUpdate)
            assert not isinstance(output, TopologyDiff)
            assert not isinstance(output, LatentSample)
```

- [ ] **Step 2: Run to verify they fail**

Run: `uv run pytest tests/unit/profiles/test_p_min_lora.py -v`
Expected: FAIL — `ModuleNotFoundError: No module named 'kiki_oniric.profiles.p_min_lora'`.

- [ ] **Step 3: Create `kiki_oniric/profiles/p_min_lora.py`**

Create the new file:

```python
"""P_min LoRA-substrate profile (B6a, issue #15 continuation).

Subclass of ``PMinProfile`` that wires the B-series LoRA-emitting
handlers (``replay_lora_handler``, ``downscale_lora_handler``)
onto a dream/awake ``LoRAModel`` pair, and exposes
``consolidate_log()`` to apply the runtime log onto the awake
model via the B5 ``LoRAWeightDeltaChannel``.

Channels out (per framework-C spec §3.1) : ``{WEIGHT_DELTA}``
only. Neither ``TopologyDiff`` nor ``LatentSample`` nor
``AttentionPrior`` is emitted by P_min's op set.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kiki_oniric.dream.channels.weight_delta import (
    LoRAWeightDeltaChannel,
)
from kiki_oniric.dream.episode import Operation
from kiki_oniric.dream.operations.downscale_real import (
    DownscaleRealState,
    downscale_lora_handler,
)
from kiki_oniric.dream.operations.replay_real import (
    ReplayRealState,
    replay_lora_handler,
)
from kiki_oniric.profiles.p_min import PMinProfile

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


@dataclass(kw_only=True)
class PMinLoRAProfile(PMinProfile):
    """P_min rewired for the B-series LoRA substrate.

    Required kwargs: ``dream_model`` and ``awake_model`` — two
    ``LoRAModel`` instances. For within-machine bit-exact
    reproducibility under ``consolidate_log()``, build both with
    the same ``seed`` so the awake model starts as a bit-clone of
    the dream model.

    The parent's ``replay_state`` / ``downscale_state`` fields
    (typed ``ReplayOpState`` / ``DownscaleOpState`` for the
    skeleton handlers) are overridden to the ``_RealState``
    variants required by the LoRA handlers.

    ``__post_init__`` intentionally does NOT call
    ``super().__post_init__()`` — the parent registers skeleton
    handlers ; we register the LoRA-emitting variants on the
    same runtime instead.
    """

    dream_model: "LoRAModel"
    awake_model: "LoRAModel"
    lr: float = 0.01
    # Override parent state types — LoRA handlers need _RealState.
    replay_state: ReplayRealState = field(  # type: ignore[assignment]
        default_factory=ReplayRealState,
    )
    downscale_state: DownscaleRealState = field(  # type: ignore[assignment]
        default_factory=DownscaleRealState,
    )
    weight_channel: LoRAWeightDeltaChannel | None = None

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
        self.weight_channel = LoRAWeightDeltaChannel(self.awake_model)

    def consolidate_log(self) -> int:
        """Replay every ``WeightUpdate`` in the runtime log onto
        ``awake_model`` via ``weight_channel``, then clear the log.

        Returns the number of channel outputs dispatched. The log
        is cleared on success so a second call without further
        ``runtime.execute()`` returns 0 (idempotent no-op).
        """
        from kiki_oniric.consolidate import apply_channel_outputs

        count = apply_channel_outputs(
            self.runtime.log,
            weight_channel=self.weight_channel,
        )
        self.runtime.reset_log()
        return count
```

The `# type: ignore[assignment]` on the state-field overrides documents the deliberate state-type widening (parent declared `ReplayOpState`, subclass narrows to `ReplayRealState`). Both are dataclasses but the field type changes — mypy would flag this without the ignore.

- [ ] **Step 4: Run to verify the tests pass**

Run: `uv run pytest tests/unit/profiles/test_p_min_lora.py -v`
Expected: PASS — all 11 tests (1 runtime + 10 profile).

- [ ] **Step 5: Full suite + mypy + ruff**

Run: `uv run pytest -q` — all pass (full B6a count = 11 new tests on top of 819 from main).
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/profiles/p_min_lora.py tests/unit/profiles/test_p_min_lora.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/profiles/p_min_lora.py tests/unit/profiles/test_p_min_lora.py
git commit -m "$(cat <<'EOF'
feat(profile): PMinLoRAProfile wires P_min to LoRA channels

B6a / issue #15. Subclass of PMinProfile with kw-only
dream_model + awake_model LoRAModel kwargs. __post_init__
registers replay_lora_handler + downscale_lora_handler on the
runtime against dream_model, builds a LoRAWeightDeltaChannel
on awake_model. consolidate_log() calls apply_channel_outputs
then reset_log() so a second call without further episodes
is a no-op.

Cycle-3 PMinProfile stays untouched: legacy tests + DR-4
inclusion checks keep passing. State field types are widened
from ReplayOpState/DownscaleOpState to ReplayRealState/
DownscaleRealState (deliberate, documented with type ignores).
EOF
)"
```

---

## Task 4: Documentation and DualVer sync

**Files:**
- Modify: `CHANGELOG.md`, `pyproject.toml`, `uv.lock`
- Modify: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/…`

- [ ] **Step 1: Add the CHANGELOG entry**

Insert at the top of the `CHANGELOG.md` body, immediately above the existing `[C-v0.20.0+PARTIAL]` entry:

```markdown
## [C-v0.21.0+PARTIAL] — 2026-05-20 — PMinLoRAProfile wires P_min to channels (B6a)

### Formal axis (FC) — MINOR (v0.20.0 → v0.21.0)

- **New subclass** `kiki_oniric/profiles/p_min_lora.py`:
  `PMinLoRAProfile(PMinProfile)` with kw-only `dream_model` and
  `awake_model` LoRAModel kwargs. Registers
  `replay_lora_handler` and `downscale_lora_handler` on its
  runtime against `dream_model`, builds a
  `LoRAWeightDeltaChannel(awake_model)`, and exposes
  `consolidate_log() -> int` that calls `apply_channel_outputs`
  then clears the log so a second call without further episodes
  is a no-op. State field types are widened from
  `ReplayOpState` / `DownscaleOpState` to `ReplayRealState` /
  `DownscaleRealState` to match the LoRA handler contract.
- **Refactor** `kiki_oniric/consolidate.py`: the three non-
  attention channel kwargs of `apply_channel_outputs`
  (`weight_channel`, `hierarchy_channel`, `latent_channel`) are
  now `Optional[…] = None`. Each raises a per-type `ValueError`
  if a matching `ChannelOutput` appears in the log without the
  channel set. Backwards compatible (existing 4-kwarg call
  sites still work). Lets profiles that emit only ch1 (PMin)
  omit the unused channel kwargs.
- **New runtime helper** `kiki_oniric/dream/runtime.py`:
  `DreamRuntime.reset_log()` clears the accountability log
  without touching handler registrations. Used by
  `PMinLoRAProfile.consolidate_log()`.
- Sub-project B6a of issue #15 (B6 decomposes into B6a / B6b /
  B6c by profile). Cycle-3 `PMinProfile` stays intact: legacy
  tests + DR-4 inclusion checks unaffected. B6b
  (`PEquLoRAProfile`, channels `{1, 3, 4}`) and B6c
  (`PMaxLoRAProfile`, channels `{1, 2, 3, 4}` + encoder/decoder
  for `recombine_full`) are future work.

### Empirical axis (EC) — UNCHANGED (PARTIAL)

- No new substrate, axiom, or empirical claim. EC stays
  `+PARTIAL`.

### Packaging

- `pyproject.toml` version bumped `0.18.0 → 0.19.0`.
```

Match the formatting of the surrounding entries.

- [ ] **Step 2: Bump `pyproject.toml`**

Change `version = "0.18.0"` to `version = "0.19.0"`.

- [ ] **Step 3: Sync the lockfile**

Run: `uv lock`
Expected: `uv.lock` updates the `dreamofkiki` pin to `0.19.0`.

- [ ] **Step 4: Update the framework-C spec (EN)**

In `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §3.1 (profile definitions), immediately after the line defining `P_min = { primitives_in: {β}, primitives_out: {1}, ops: {replay, downscale} }`, append:

```markdown
As of B6a (issue #15), P_min has a LoRA-substrate variant
`PMinLoRAProfile` (in `kiki_oniric/profiles/p_min_lora.py`)
that wires the B-series LoRA-emitting handlers
(`replay_lora_handler`, `downscale_lora_handler`) onto a dream
/ awake `LoRAModel` pair. The variant exposes
`consolidate_log()` that applies the runtime's accumulated
`WeightUpdate` outputs onto the awake model via a
`LoRAWeightDeltaChannel`. The cycle-3 `PMinProfile` (skeleton
handlers, no emission) remains the canonical DR-4 chain-
inclusion reference. B6b and B6c will introduce
`PEquLoRAProfile` and `PMaxLoRAProfile` analogously.
```

- [ ] **Step 5: Update the framework-C spec (FR mirror)**

In `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` §3.1, at the matching location, append the French sentence (code identifiers stay in original form):

```markdown
Depuis B6a (issue #15), P_min possède une variante de substrat
LoRA `PMinLoRAProfile` (dans
`kiki_oniric/profiles/p_min_lora.py`) qui branche les
gestionnaires LoRA de la série B (`replay_lora_handler`,
`downscale_lora_handler`) sur une paire de `LoRAModel`
rêve / éveil. La variante expose `consolidate_log()` qui
applique les `WeightUpdate` accumulés sur le modèle éveil via
un `LoRAWeightDeltaChannel`. Le `PMinProfile` cycle-3
(gestionnaires squelettes, aucune émission) reste la référence
canonique pour la chaîne d'inclusion DR-4. B6b et B6c
introduiront `PEquLoRAProfile` et `PMaxLoRAProfile` de manière
analogue.
```

- [ ] **Step 6: Verify**

Run: `uv run pytest -q` — full suite passes (docs/version change, no code touched).
Run: `uv run mypy harness tests` — `Success`.

- [ ] **Step 7: Commit**

```bash
git add CHANGELOG.md pyproject.toml uv.lock docs/specs/2026-04-17-dreamofkiki-framework-C-design.md docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md
git commit -m "$(cat <<'EOF'
docs: sync spec for B6a PMinLoRAProfile

B6a / issue #15. Add the C-v0.21.0+PARTIAL changelog entry,
note in framework-C spec section 3.1 (EN + FR) that P_min has
a LoRA-substrate variant PMinLoRAProfile wiring the B-series
channels, and bump the package version to 0.19.0.
EOF
)"
```

If `tests/reproducibility/golden_hashes_apple_*.json` shows modified (per-family drift from running pytest), restore it with
`git checkout -- tests/reproducibility/golden_hashes_apple_*.json` before committing.

---

## Task 5: Final verification

**Files:** none — verification only.

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -q`
Expected: all pass (819 from main + 11 new B6a tests + 4 new apply_channel_outputs tests = 834 passed, 3 skipped, 12 xfailed), 0 failures, coverage gate met.

- [ ] **Step 2: Type check**

Run: `uv run mypy harness tests`
Expected: `Success: no issues found`.

- [ ] **Step 3: Lint**

Run: `uv run ruff check .`
Expected: clean.

- [ ] **Step 4: Confirm working tree is clean**

Run: `git status --short`
Expected: empty (the per-family golden hashes file may show modified — restore it).

- [ ] **Step 5: Update issue #15 or open a B6 tracking issue**

Comment on issue #15 (closed but still receives comments): B6a complete — `PMinLoRAProfile` wires P_min to the B5 channel apply loop. Cycle-3 `PMinProfile` untouched. B6b (`PEquLoRAProfile`) and B6c (`PMaxLoRAProfile`) remain.

---

## Self-Review

- **Spec coverage:**
  - Refactor `apply_channel_outputs` to make weight/hierarchy/latent kwargs Optional (Task 1) ✓.
  - Per-type `ValueError` when matching output appears with `None` channel (Task 1, three new tests) ✓.
  - `DreamRuntime.reset_log()` one-line helper (Task 2 Step 3) ✓.
  - `PMinLoRAProfile` subclass with kw-only kwargs and `_RealState` field overrides (Task 3) ✓.
  - `__post_init__` does NOT call super, registers `replay_lora_handler` + `downscale_lora_handler` on `self.runtime` against `dream_model`, builds `weight_channel = LoRAWeightDeltaChannel(awake_model)` (Task 3) ✓.
  - `consolidate_log() -> int` calls `apply_channel_outputs(log, weight_channel=)` then `reset_log()` (Task 3) ✓.
  - 11 tests (1 runtime + 10 profile) in `tests/unit/profiles/test_p_min_lora.py` (Tasks 2-3) + 4 added in `tests/unit/test_apply_channel_outputs.py` (Task 1) ✓.
  - FC-MINOR DualVer bump + CHANGELOG + spec §3.1 EN+FR + pyproject 0.19.0 (Task 4) ✓.
  - Final verification (Task 5) ✓.
- **Placeholder scan:** no TBD/TODO; every code step has complete code.
- **Type consistency:**
  - `PMinLoRAProfile(*, dream_model, awake_model, lr=0.01, replay_state, downscale_state, weight_channel)` — exact field names used by all tests in Task 3.
  - `consolidate_log() -> int` — same signature everywhere.
  - `LoRAWeightDeltaChannel(target: LoRAModel)` — the B5 channel constructor, used here.
  - `ReplayRealState` and `DownscaleRealState` — same names as `replay_lora_handler` and `downscale_lora_handler` expect (B1b/B2).
  - `Operation.REPLAY` / `Operation.DOWNSCALE` — standard enum members.
  - `DreamRuntime.reset_log()` — same name used in `PMinLoRAProfile.consolidate_log` and in the runtime test.
  - `apply_channel_outputs(log, *, weight_channel=None, hierarchy_channel=None, latent_channel=None, attention_channel=None)` — consistent signature across Task 1 implementation and Task 3 consolidate_log call.
- **Inter-task ordering:** Task 1 (consolidate.py refactor) and Task 2 (runtime helper) are independent — could swap. Task 3 (profile) depends on both. Task 4 (docs) depends on Task 3. The chosen 1→2→3→4→5 order is the dependency order and matches the test bootstrap pattern (TDD).

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-20-b6a-pmin-lora-profile.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, inline review between tasks (per the user's "tu fais les review critique" instruction in earlier B-tasks).

**2. Inline Execution** — execute tasks in this session with checkpoints.
