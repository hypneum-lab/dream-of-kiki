# B6b — `PEquLoRAProfile` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Subclass `PEquProfile` into `PEquLoRAProfile` so the B-series LoRA-emitting handlers (`replay_lora_handler`, `downscale_lora_handler`, `restructure_lora_handler`) drive a dream/awake `LoRAModel` pair on the P_equ profile, with `consolidate_log()` dispatching ch1 (`WeightUpdate`) and ch3 (`TopologyDiff`) to the awake-side channels. ch4 (`AttentionPrior`) is exposed as a profile-owned state surface ; ch2 (`LatentSample`) is excluded by spec — `recombine_light` keeps the skeleton handler.

**Architecture:** Two files ship. **(1)** `kiki_oniric/profiles/p_equ_lora.py` defines `PEquLoRAProfile(PEquProfile)` with `@dataclass(kw_only=True)`, required `dream_model` / `awake_model` `LoRAModel` kwargs. `__post_init__` deliberately does NOT call `super().__post_init__()` — the parent registers cycle-3 skeleton handlers ; we register the B-series LoRA-emitting variants for the three emitting ops (replay, downscale, restructure) and the skeleton `recombine_handler` for `recombine_light` (no emission, no ch2). It builds `LoRAWeightDeltaChannel(awake_model)` and `LoRAHierarchyChangeChannel(awake_model)` ; `attention_prior` is an `AttentionPriorChannel` state surface populated externally. `consolidate_log()` lazily imports `apply_channel_outputs`, dispatches the runtime log onto weight + hierarchy channels (latent + attention default `None` per the B6a refactor), then clears the log. **(2)** `tests/unit/profiles/test_p_equ_lora.py` ships 14 end-to-end tests including a DR-4 chain-inclusion check against `PMinLoRAProfile`.

**Tech Stack:** Python 3.12+, `uv`, MLX (`mlx.core`), numpy, pytest, mypy.

**Spec:** `docs/superpowers/specs/2026-05-20-b6b-pequ-lora-profile-design.md`

**Critical not-a-bug:** `__post_init__` deliberately does NOT call `super().__post_init__()`. The parent `PEquProfile` registers four skeleton handlers ; this subclass registers three LoRA variants + the skeleton `recombine_handler`. Calling super would double-register and the LoRA registrations would silently overwrite the skeletons — confusing future readers. Override cleanly.

**Mixed state types are intentional:** three states are widened to `_RealState` (replay / downscale / restructure) to match the LoRA handler contract ; `recombine_state` keeps `RecombineOpState` because the skeleton `recombine_handler` services `recombine_light` and never emits.

---

## File Structure

- **Create** `kiki_oniric/profiles/p_equ_lora.py` — `PEquLoRAProfile`.
- **Create** `tests/unit/profiles/test_p_equ_lora.py` — 14 tests.
- **Modify** `CHANGELOG.md`, `pyproject.toml`, `uv.lock` — `[C-v0.22.0+PARTIAL]`, version `0.19.0 → 0.20.0`.
- **Modify** `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` — §3.1 note for `PEquLoRAProfile`.

Helpers `_clones` and `_assert_lora_models_equal` are duplicated from `test_p_min_lora.py` (B6a). No shared helpers file yet — that refactor lands with B6c when there are three copies.

---

## Task 1: `PEquLoRAProfile` subclass

**Files:**
- Create: `kiki_oniric/profiles/p_equ_lora.py`
- Create: `tests/unit/profiles/test_p_equ_lora.py`

- [ ] **Step 1: Write the failing test file with all 14 tests**

Create `tests/unit/profiles/test_p_equ_lora.py`:

```python
"""Unit tests for PEquLoRAProfile (B6b)."""
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
from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


def _clones(seed: int = 0) -> tuple[LoRAModel, LoRAModel]:
    """Two bit-identical LoRAModels at the same seed.

    Duplicated from tests/unit/profiles/test_p_min_lora.py. A shared
    helpers file lands with B6c when there are three copies.
    """
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


def _replay_episode(records: list[dict[str, object]] | None = None) -> DreamEpisode:
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
        episode_id="de-pequ-lora-replay",
    )


def _downscale_episode(factor: float = 0.5) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"shrink_factor": factor},
        operation_set=(Operation.DOWNSCALE,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pequ-lora-dn",
    )


def _restructure_episode(
    topo_ops: list[dict[str, object]],
) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"topo_ops": topo_ops},
        operation_set=(Operation.RESTRUCTURE,),
        output_channels=(OutputChannel.HIERARCHY_CHG,),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pequ-lora-restr",
    )


def _recombine_light_episode() -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"delta_latents": [[1.0, 2.0], [3.0, 4.0]]},
        operation_set=(Operation.RECOMBINE,),
        output_channels=(),
        budget=BudgetCap(flops=10_000_000, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-pequ-lora-rec",
    )


def test_pequ_lora_construction_happy_path() -> None:
    from kiki_oniric.dream.channels.attention_prior import (
        AttentionPriorChannel,
    )
    from kiki_oniric.dream.channels.hierarchy_change import (
        LoRAHierarchyChangeChannel,
    )
    from kiki_oniric.dream.channels.weight_delta import (
        LoRAWeightDeltaChannel,
    )
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    assert isinstance(profile.weight_channel, LoRAWeightDeltaChannel)
    assert isinstance(profile.hierarchy_channel, LoRAHierarchyChangeChannel)
    assert isinstance(profile.attention_prior, AttentionPriorChannel)
    # All four LoRA / skeleton handlers must be registered.
    for op in (
        Operation.REPLAY,
        Operation.DOWNSCALE,
        Operation.RESTRUCTURE,
        Operation.RECOMBINE,
    ):
        assert op in profile.runtime._handlers


def test_pequ_lora_construction_missing_dream_raises() -> None:
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    _, awake = _clones(seed=0)
    with pytest.raises(TypeError):
        PEquLoRAProfile(awake_model=awake)  # type: ignore[call-arg]


def test_pequ_lora_construction_missing_awake_raises() -> None:
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, _ = _clones(seed=0)
    with pytest.raises(TypeError):
        PEquLoRAProfile(dream_model=dream)  # type: ignore[call-arg]


def test_pequ_lora_replay_emits_weight_update_in_log() -> None:
    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, WeightUpdate)


def test_pequ_lora_downscale_emits_weight_update_in_log() -> None:
    from kiki_oniric.dream.channels import WeightUpdate
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_downscale_episode(factor=0.5))
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, WeightUpdate)


def test_pequ_lora_restructure_emits_topology_diff_in_log() -> None:
    from kiki_oniric.dream.channels import TopologyDiff
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    out = profile.runtime.log[-1].channel_outputs[0]
    assert isinstance(out, TopologyDiff)


def test_pequ_lora_recombine_light_returns_none_in_log() -> None:
    """Skeleton recombine_handler returns None — no ch2 emission."""
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_recombine_light_episode())
    out = profile.runtime.log[-1].channel_outputs[0]
    assert out is None


def test_pequ_lora_consolidate_log_applies_weight_to_awake_bit_equal() -> None:
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())
    # Sanity: dream mutated, awake untouched yet.
    assert not np.array_equal(
        np.asarray(dream.layers[0].lora_b),
        np.asarray(awake.layers[0].lora_b),
    )
    profile.consolidate_log()
    _assert_lora_models_equal(dream, awake)


def test_pequ_lora_consolidate_log_applies_hierarchy_to_awake() -> None:
    """add via restructure → consolidate → awake gets new layer bit-equal."""
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    pre_len = len(dream.layers)
    add_op = {
        "op": "add",
        "index": pre_len,
        "in_features": 4,
        "out_features": 8,
        "rank": 2,
        "alpha": 4.0,
    }
    profile.runtime.execute(_restructure_episode([add_op]))
    # Dream gained a layer; awake hasn't yet.
    assert len(dream.layers) == pre_len + 1
    assert len(awake.layers) == pre_len
    profile.consolidate_log()
    # After consolidation, awake matches dream bit-for-bit.
    _assert_lora_models_equal(dream, awake)


def test_pequ_lora_consolidate_log_mixed_emits_count() -> None:
    """3 episodes (replay, restructure, downscale) → 3 dispatched outputs."""
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())
    profile.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    profile.runtime.execute(_downscale_episode(factor=0.7))
    assert profile.consolidate_log() == 3


def test_pequ_lora_consolidate_log_clears_and_idempotent() -> None:
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())
    assert len(profile.runtime.log) == 1
    profile.consolidate_log()
    assert len(profile.runtime.log) == 0
    # Second call without further episodes is idempotent no-op.
    assert profile.consolidate_log() == 0


def test_pequ_lora_attention_prior_settable_and_readable() -> None:
    """ch4 is a state surface: set via .set_prior, read via .get_prior."""
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    prior = np.array([0.2, 0.3, 0.4], dtype=np.float32)
    profile.attention_prior.set_prior(prior)
    got = profile.attention_prior.get_prior()
    assert got is not None
    np.testing.assert_array_equal(got, prior)


def test_pequ_lora_no_latent_in_log_after_recombine_light() -> None:
    """No LatentSample ever appears — channel 2 not in P_equ."""
    from kiki_oniric.dream.channels import LatentSample
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile

    dream, awake = _clones(seed=0)
    profile = PEquLoRAProfile(dream_model=dream, awake_model=awake)
    profile.runtime.execute(_replay_episode())
    profile.runtime.execute(_recombine_light_episode())
    profile.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    for entry in profile.runtime.log:
        for output in entry.channel_outputs:
            assert not isinstance(output, LatentSample)


def test_pequ_lora_dr4_chain_inclusion_with_pmin_lora() -> None:
    """DR-4: ops(PMinLoRA) ⊆ ops(PEquLoRA); channel emitters likewise."""
    from kiki_oniric.dream.channels import TopologyDiff, WeightUpdate
    from kiki_oniric.profiles.p_equ_lora import PEquLoRAProfile
    from kiki_oniric.profiles.p_min_lora import PMinLoRAProfile

    dream_min, awake_min = _clones(seed=0)
    pmin = PMinLoRAProfile(dream_model=dream_min, awake_model=awake_min)
    dream_equ, awake_equ = _clones(seed=0)
    pequ = PEquLoRAProfile(dream_model=dream_equ, awake_model=awake_equ)

    pmin_ops = set(pmin.runtime._handlers.keys())
    pequ_ops = set(pequ.runtime._handlers.keys())
    assert pmin_ops <= pequ_ops, "ops(PMin) must be subset of ops(PEqu)"

    # Channel emitter set inclusion: PMin emits ch1 only; PEqu emits
    # ch1 + ch3. We confirm by sampling one of each from each profile.
    pmin.runtime.execute(_replay_episode())
    pmin_emitted = {
        type(o) for entry in pmin.runtime.log
        for o in entry.channel_outputs if o is not None
    }
    assert pmin_emitted == {WeightUpdate}

    pequ.runtime.execute(_replay_episode())
    pequ.runtime.execute(
        _restructure_episode([{"op": "reroute", "swap_indices": [0, 1]}]),
    )
    pequ_emitted = {
        type(o) for entry in pequ.runtime.log
        for o in entry.channel_outputs if o is not None
    }
    assert pmin_emitted <= pequ_emitted
    assert pequ_emitted == {WeightUpdate, TopologyDiff}
```

- [ ] **Step 2: Run to verify all 14 tests fail**

Run: `uv run pytest tests/unit/profiles/test_p_equ_lora.py -v`
Expected: 14 ERRORS (collection failure due to missing module
`kiki_oniric.profiles.p_equ_lora`).

- [ ] **Step 3: Create `kiki_oniric/profiles/p_equ_lora.py`**

Create the file:

```python
"""P_equ LoRA-substrate profile (B6b, issue #15 continuation).

Subclass of ``PEquProfile`` that wires the B-series LoRA-emitting
handlers (``replay_lora_handler``, ``downscale_lora_handler``,
``restructure_lora_handler``) onto a dream/awake ``LoRAModel`` pair
and the skeleton ``recombine_handler`` for ``recombine_light``.

Channels out (per framework-C spec §3.1) :
- ch1 (``WeightUpdate``) via ``LoRAWeightDeltaChannel`` (B5).
- ch3 (``TopologyDiff``) via ``LoRAHierarchyChangeChannel`` (B5).
- ch4 (``AttentionPrior``) via ``AttentionPriorChannel`` — *state
  surface only*, populated externally via
  ``profile.attention_prior.set_prior(prior)``. No op currently
  emits ``AttentionPrior`` into the runtime log.
- ch2 (``LatentSample``) NOT in P_equ ; ``recombine_light`` returns
  ``None``.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kiki_oniric.dream.channels.attention_prior import (
    AttentionPriorChannel,
)
from kiki_oniric.dream.channels.hierarchy_change import (
    LoRAHierarchyChangeChannel,
)
from kiki_oniric.dream.channels.weight_delta import (
    LoRAWeightDeltaChannel,
)
from kiki_oniric.dream.episode import Operation
from kiki_oniric.dream.operations.downscale_real import (
    DownscaleRealState,
    downscale_lora_handler,
)
from kiki_oniric.dream.operations.recombine import (
    recombine_handler,
)
from kiki_oniric.dream.operations.replay_real import (
    ReplayRealState,
    replay_lora_handler,
)
from kiki_oniric.dream.operations.restructure_real import (
    RestructureRealState,
    restructure_lora_handler,
)
from kiki_oniric.profiles.p_equ import PEquProfile

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


_DEFAULT_ATTENTION_BUDGET = 1.5


@dataclass(kw_only=True)
class PEquLoRAProfile(PEquProfile):
    """P_equ rewired for the B-series LoRA substrate.

    Required kwargs : ``dream_model`` and ``awake_model`` —
    ``LoRAModel`` instances. For within-machine bit-exact
    reproducibility under ``consolidate_log()``, build both at the
    same ``seed`` so the awake model starts as a bit-clone of the
    dream model.

    Parent state fields (``replay_state``, ``downscale_state``,
    ``restructure_state``) are widened from their cycle-3 skeleton
    types to the ``_RealState`` variants required by the LoRA
    handlers. ``recombine_state`` keeps its ``RecombineOpState``
    type because the skeleton ``recombine_handler`` is what
    services ``recombine_light``.

    ``__post_init__`` intentionally does NOT call
    ``super().__post_init__()`` — the parent registers cycle-3
    skeleton handlers ; we register the LoRA-emitting variants
    (for replay / downscale / restructure) plus the skeleton
    handler (for recombine_light) on the same runtime instead.
    """

    dream_model: "LoRAModel"
    awake_model: "LoRAModel"
    lr: float = 0.01
    max_adds_per_episode: int = 1
    seed: int = 0
    # Override parent state types — LoRA handlers need _RealState.
    replay_state: ReplayRealState = field(  # type: ignore[assignment]
        default_factory=ReplayRealState,
    )
    downscale_state: DownscaleRealState = field(  # type: ignore[assignment]
        default_factory=DownscaleRealState,
    )
    restructure_state: RestructureRealState = field(  # type: ignore[assignment]
        default_factory=RestructureRealState,
    )
    # recombine_state STAYS at RecombineOpState — skeleton handler.
    weight_channel: LoRAWeightDeltaChannel | None = None
    hierarchy_channel: LoRAHierarchyChangeChannel | None = None
    attention_prior: AttentionPriorChannel = field(
        default_factory=lambda: AttentionPriorChannel(
            budget_attention=_DEFAULT_ATTENTION_BUDGET,
        ),
    )

    def __post_init__(self) -> None:
        # Do NOT call super().__post_init__() — parent registers
        # cycle-3 skeleton handlers; we register the LoRA variants
        # for the three emitting ops + skeleton for recombine_light.
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
            recombine_handler(self.recombine_state, rng=self.rng),
        )
        self.weight_channel = LoRAWeightDeltaChannel(self.awake_model)
        self.hierarchy_channel = LoRAHierarchyChangeChannel(
            self.awake_model,
        )

    def consolidate_log(self) -> int:
        """Dispatch the runtime log onto the awake model via
        ``weight_channel`` (ch1) and ``hierarchy_channel`` (ch3),
        then clear the log.

        ``latent_channel`` is ``None`` because ``recombine_light``
        returns ``None`` (no ch2 in P_equ). ``attention_channel``
        is ``None`` because no op currently emits ``AttentionPrior``
        into the runtime log ; the profile's ``attention_prior``
        field is a state surface for external callers, not a
        dispatch target.

        Returns the number of channel outputs dispatched. The log
        is cleared on success so a second call without further
        ``runtime.execute()`` returns 0 (idempotent no-op).
        """
        from kiki_oniric.consolidate import apply_channel_outputs

        count = apply_channel_outputs(
            self.runtime.log,
            weight_channel=self.weight_channel,
            hierarchy_channel=self.hierarchy_channel,
        )
        self.runtime.reset_log()
        return count
```

- [ ] **Step 4: Run to verify all tests pass**

Run: `uv run pytest tests/unit/profiles/test_p_equ_lora.py -v`
Expected: PASS — 14 tests.

- [ ] **Step 5: Full sanity (suite + mypy + ruff)**

Run: `uv run pytest -q` — full suite passes. Expected count :
834 (post-B6a baseline) + 14 (B6b new) = 848 passed, 3 skipped,
12 xfailed.
Run: `uv run mypy harness tests` — `Success`.
Run: `uv run ruff check kiki_oniric/profiles/p_equ_lora.py tests/unit/profiles/test_p_equ_lora.py` — clean.

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/profiles/p_equ_lora.py tests/unit/profiles/test_p_equ_lora.py
git commit -m "$(cat <<'EOF'
feat(profile): add PEquLoRAProfile (B6b)

B6b / issue #15 continuation. Subclass of PEquProfile with
kw-only dream_model + awake_model LoRAModel kwargs.
__post_init__ registers replay_lora + downscale_lora +
restructure_lora handlers on the runtime against dream_model,
keeps the skeleton recombine_handler for recombine_light (no
ch2 emission), and builds LoRAWeightDeltaChannel +
LoRAHierarchyChangeChannel on awake_model. attention_prior is a
state-surface field set externally via set_prior — no op emits
AttentionPrior into the runtime log.

consolidate_log() dispatches ch1 (WeightUpdate) and ch3
(TopologyDiff) via apply_channel_outputs and clears the log.
latent_channel and attention_channel default to None — neither
ch2 nor ch4 reach the log in P_equ.

14 tests cover construction (3) + emission (4) + consolidation
(4) + attention surface (1) + no_latent (1) + DR-4 chain
inclusion vs PMinLoRAProfile (1).

State field types are widened from {Replay,Downscale,Restructure}
OpState to _RealState (deliberate, documented with type
ignores). RecombineOpState kept — skeleton handler.
EOF
)"
```

---

## Task 2: Documentation and DualVer sync

**Files:**
- Modify: `CHANGELOG.md`, `pyproject.toml`, `uv.lock`
- Modify: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` + `docs/specs-fr/…`

- [ ] **Step 1: Add the CHANGELOG entry**

Insert at the top of the `CHANGELOG.md` body, immediately above the existing `[C-v0.21.0+PARTIAL]` entry:

```markdown
## [C-v0.22.0+PARTIAL] — 2026-05-20 — PEquLoRAProfile wires P_equ to channels (B6b)

### Formal axis (FC) — MINOR (v0.21.0 → v0.22.0)

- **New subclass** `kiki_oniric/profiles/p_equ_lora.py`:
  `PEquLoRAProfile(PEquProfile)` with kw-only `dream_model` and
  `awake_model` LoRAModel kwargs (plus `lr=0.01`,
  `max_adds_per_episode=1`, `seed=0`). Registers
  `replay_lora_handler`, `downscale_lora_handler`,
  `restructure_lora_handler` on its runtime against
  `dream_model`, keeps the skeleton `recombine_handler` for
  `recombine_light` (no ch2 emission). Builds
  `LoRAWeightDeltaChannel(awake_model)` for ch1 and
  `LoRAHierarchyChangeChannel(awake_model)` for ch3.
  `attention_prior` is an `AttentionPriorChannel` state surface
  populated externally via `profile.attention_prior.set_prior(arr)` ;
  no op emits `AttentionPrior` into the runtime log.
- `consolidate_log() -> int` dispatches ch1 + ch3 via
  `apply_channel_outputs` and clears the log on success. `ch2`
  (LatentSample) and `ch4` (AttentionPrior) default to `None`
  channel kwargs — neither reaches the log in P_equ.
- **State widening** : `replay_state` / `downscale_state` /
  `restructure_state` widened from cycle-3 skeleton
  `OpState` types to `_RealState` variants. `recombine_state`
  kept at `RecombineOpState` because the skeleton handler
  services `recombine_light` and never emits.
- Sub-project B6b of issue #15 (B6 decomposed by profile: B6a
  PMin done, B6b PEqu this entry, B6c PMax future). Cycle-3
  `PEquProfile` untouched — legacy tests + DR-4 inclusion checks
  unaffected. The 14 new tests include a DR-4 chain-inclusion
  check confirming `ops(PMinLoRA) ⊆ ops(PEquLoRA)` and
  `channels_emitted(PMinLoRA) = {WeightUpdate} ⊆
  channels_emitted(PEquLoRA) = {WeightUpdate, TopologyDiff}`.

### Empirical axis (EC) — UNCHANGED (PARTIAL)

- No new substrate, axiom, or empirical claim. EC stays
  `+PARTIAL`.

### Packaging

- `pyproject.toml` version bumped `0.19.0 → 0.20.0`.
```

Match the formatting of the surrounding entries.

- [ ] **Step 2: Bump `pyproject.toml`**

Change `version = "0.19.0"` to `version = "0.20.0"`.

- [ ] **Step 3: Sync the lockfile**

Run: `uv lock`
Expected: `uv.lock` updates the `dreamofkiki` pin to `0.20.0`.

- [ ] **Step 4: Update the framework-C spec (EN)**

In `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` §3.1, immediately after the line defining `P_equ = { primitives_in: {β, δ}, primitives_out: {1,3,4}, ops: {replay, downscale, restructure, recombine_light} }`, append:

```markdown
As of B6b (issue #15), P_equ has a LoRA-substrate variant
`PEquLoRAProfile` (in `kiki_oniric/profiles/p_equ_lora.py`) that
wires the B-series LoRA-emitting handlers (`replay_lora_handler`,
`downscale_lora_handler`, `restructure_lora_handler`) on a
dream/awake `LoRAModel` pair. The skeleton `recombine_handler`
is kept for `recombine_light` (no ch2 emission, matching the
spec's `channels_out` exclusion of channel 2). `consolidate_log()`
dispatches ch1 (`WeightUpdate`) and ch3 (`TopologyDiff`) onto the
awake model via `LoRAWeightDeltaChannel` and
`LoRAHierarchyChangeChannel`. ch4 (`AttentionPrior`) is exposed
as a state-surface field set externally via
`profile.attention_prior.set_prior(prior)`. Cycle-3 `PEquProfile`
remains the canonical DR-4 chain-inclusion reference. B6c will
introduce `PMaxLoRAProfile`.
```

- [ ] **Step 5: Update the framework-C spec (FR mirror)**

In `docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md` §3.1, at the matching location, append the French sentence (code identifiers stay in original form):

```markdown
Depuis B6b (issue #15), P_equ possède une variante de substrat
LoRA `PEquLoRAProfile` (dans
`kiki_oniric/profiles/p_equ_lora.py`) qui branche les
gestionnaires LoRA de la série B (`replay_lora_handler`,
`downscale_lora_handler`, `restructure_lora_handler`) sur une
paire de `LoRAModel` rêve / éveil. Le `recombine_handler`
squelette est conservé pour `recombine_light` (aucune émission
ch2, conforme à l'exclusion `channels_out` du canal 2 dans la
spec). `consolidate_log()` répartit ch1 (`WeightUpdate`) et ch3
(`TopologyDiff`) sur le modèle éveil via `LoRAWeightDeltaChannel`
et `LoRAHierarchyChangeChannel`. ch4 (`AttentionPrior`) est
exposé comme champ de surface d'état défini en externe via
`profile.attention_prior.set_prior(prior)`. Le `PEquProfile`
cycle-3 reste la référence canonique pour la chaîne d'inclusion
DR-4. B6c introduira `PMaxLoRAProfile`.
```

- [ ] **Step 6: Verify**

Run: `uv run pytest -q` — full suite passes (docs/version change, no code touched).
Run: `uv run mypy harness tests` — `Success`.

If `tests/reproducibility/golden_hashes_apple_*.json` shows modified (per-family drift from running pytest), restore with `git checkout -- tests/reproducibility/golden_hashes_apple_*.json` before committing.

- [ ] **Step 7: Commit**

```bash
git add CHANGELOG.md pyproject.toml uv.lock docs/specs/2026-04-17-dreamofkiki-framework-C-design.md docs/specs-fr/2026-04-17-dreamofkiki-framework-C-design.md
git commit -m "$(cat <<'EOF'
docs: sync spec for B6b PEquLoRAProfile

B6b / issue #15. Add the C-v0.22.0+PARTIAL changelog entry,
note in framework-C spec section 3.1 (EN + FR) that P_equ has
a LoRA-substrate variant PEquLoRAProfile wiring channels {1,3,4}
with skeleton recombine_light retained for the spec-mandated
ch2 exclusion, and bump the package version to 0.20.0.
EOF
)"
```

---

## Task 3: Final verification

**Files:** none — verification only.

- [ ] **Step 1: Full test suite**

Run: `uv run pytest -q`
Expected: 848 passed, 3 skipped, 12 xfailed (834 from main + 14 new B6b tests), 0 failures, coverage gate met.

- [ ] **Step 2: Type check**

Run: `uv run mypy harness tests`
Expected: `Success: no issues found in 172 source files` (171 + `p_equ_lora.py`).

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

Comment on issue #15 (closed but still receives comments): B6b complete — `PEquLoRAProfile` wires P_equ to the B5 channel apply loop on channels `{1, 3, 4}`. ch2 excluded by spec (recombine_light kept as skeleton). DR-4 chain inclusion vs `PMinLoRAProfile` empirically pinned. Cycle-3 `PEquProfile` untouched. B6c (`PMaxLoRAProfile`) remains.

---

## Self-Review

- **Spec coverage:**
  - `PEquLoRAProfile` subclass with kw-only `dream_model`/`awake_model` + optional lr/max_adds/seed (Task 1 Step 3) ✓
  - Override `replay_state`/`downscale_state`/`restructure_state` to `_RealState` ; keep `recombine_state` as `RecombineOpState` (Task 1 Step 3) ✓
  - `__post_init__` registers 4 handlers (3 LoRA + 1 skeleton) without calling `super().__post_init__()` (Task 1 Step 3) ✓
  - Builds `weight_channel` + `hierarchy_channel` on `awake_model` (Task 1 Step 3) ✓
  - `attention_prior` field with `default_factory` of `AttentionPriorChannel(budget_attention=1.5)` (Task 1 Step 3) ✓
  - `consolidate_log()` dispatches ch1 + ch3 with weight + hierarchy channels (latent + attention default None), then `reset_log()` (Task 1 Step 3) ✓
  - 14 tests covering construction (3) + per-op emission (4) + consolidate behaviour (4) + attention surface (1) + no_latent (1) + DR-4 inclusion (1) — exact count matches Task 1 Step 1 ✓
  - CHANGELOG + spec §3.1 EN+FR + pyproject 0.20.0 (Task 2) ✓
  - Final verification (Task 3) ✓
  - No PMax wiring ✓ ; no cycle-3 PEquProfile modification ✓
- **Placeholder scan:** no TBD/TODO; every code step has complete code.
- **Type consistency:**
  - `PEquLoRAProfile(*, dream_model, awake_model, lr=0.01, max_adds_per_episode=1, seed=0, replay_state, downscale_state, restructure_state, weight_channel, hierarchy_channel, attention_prior)` — same field names used by all 14 tests in Task 1.
  - `consolidate_log() -> int` — same signature in implementation and tests.
  - `LoRAWeightDeltaChannel(target: LoRAModel)` and `LoRAHierarchyChangeChannel(target: LoRAModel)` — B5 channel constructors.
  - `AttentionPriorChannel(budget_attention=…)` — cycle-2 constructor signature.
  - `ReplayRealState`, `DownscaleRealState`, `RestructureRealState` — same names as `*_lora_handler` expect (B1b/B2/B3).
  - `RecombineOpState` from `recombine.py` (skeleton state, inherited via parent).
  - `Operation.REPLAY` / `DOWNSCALE` / `RESTRUCTURE` / `RECOMBINE` — enum members all referenced.
  - `apply_channel_outputs(log, *, weight_channel=None, hierarchy_channel=None, latent_channel=None, attention_channel=None)` — relaxed signature shipped in B6a T1.
- **DR-4 inclusion test (Task 1, test 14):** uses `pmin.runtime._handlers.keys()` and `pequ.runtime._handlers.keys()` — private attr access matches the convention used elsewhere in the repo's profile tests (and B6a tests).
- **Helper duplication note:** `_clones`, `_assert_lora_models_equal`, and the four `_*_episode` builders are duplicated from `test_p_min_lora.py`. The docstring on `_clones` flags the duplication for the B6c refactor.

---

## Execution Handoff

**Plan complete and saved to `docs/superpowers/plans/2026-05-20-b6b-pequ-lora-profile.md`. Two execution options:**

**1. Subagent-Driven (recommended)** — fresh subagent per task, inline review by me between tasks (matches the B6a pattern).

**2. Inline Execution** — execute tasks in this session.
