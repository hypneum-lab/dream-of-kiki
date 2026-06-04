# B6c — `PMaxLoRAProfile` wires P_max to the B5 channel apply loop

**Date** : 2026-05-20
**Status** : design approved, pending spec review
**Tracking** : continuation of #15 ; B6c is sub-project 3 of 3 in
the B6 profile-wiring decomposition (B6a PMin done, B6b PEqu
done, B6c PMax this spec).
**Scope** : `P_max` profile wiring, channels out `{1, 2, 3, 4}`
plus the α input channel as state surface.

---

## Context

Approach B closed end-to-end at `C-v0.22.0+PARTIAL` with B6b
(`PEquLoRAProfile`, channels `{1, 3, 4*}`). B6c is the third and
final profile sub-project.

Per framework-C spec §3.1:

```
P_max = { primitives_in:  {α, β, δ},
          primitives_out: {1, 2, 3, 4},
          ops:            {replay, downscale, restructure,
                           recombine_full} }
```

- **Channel 1** (`WeightUpdate`) — replay + downscale → via
  `LoRAWeightDeltaChannel`.
- **Channel 2** (`LatentSample`) — `recombine_full` via
  `recombine_real_handler` (B4 VAE) → via `LatentSampleQueue`
  (B5). **First profile to dispatch ch2.**
- **Channel 3** (`TopologyDiff`) — restructure → via
  `LoRAHierarchyChangeChannel`.
- **Channel 4** (`AttentionPrior`) — `AttentionPriorChannel`
  state surface, set externally via
  `profile.attention_prior.set_prior(prior)`. No op currently
  emits ; **inherited from PMaxProfile cycle-3**.
- **Input channel α** — `AlphaStreamBuffer` (awake → dream raw
  forward-pass traces ring buffer). **Inherited state surface
  from PMaxProfile cycle-3** ; the awake side pushes
  `TraceRecord` entries via `profile.alpha_stream.append(...)`.
  Not dispatched by `consolidate_log()` (it's an input channel,
  not output).

The cycle-3 `PMaxProfile` is intact: it registers four skeleton
handlers, holds `alpha_stream` + `attention_prior`, and emits
nothing.

## Problem

`PMaxProfile` registers four skeleton handlers (`replay_handler`,
`downscale_handler`, `restructure_handler`, `recombine_handler`)
that don't emit channel outputs. B6c adds a
`PMaxLoRAProfile(PMaxProfile)` subclass that wires the B-series
LoRA-emitting handlers plus the B4 `recombine_real_handler` (with
required encoder + decoder injection) onto a dream/awake
`LoRAModel` pair and exposes `consolidate_log()` dispatching
ch1 + ch2 + ch3.

## Approaches considered

**Subclass hierarchy.** Three options were considered (brainstorm
session 2026-05-20):

1. `PMaxLoRAProfile(PMaxProfile)` — symmetric with B6a's
   `PMinLoRAProfile(PMinProfile)` and B6b's
   `PEquLoRAProfile(PEquProfile)`. Inherits `alpha_stream` +
   `attention_prior` state-surface fields from cycle-3.
   **Chosen.**
2. `PMaxLoRAProfile(PEquLoRAProfile)` — would inherit the three
   LoRA handlers from B6b, then override `recombine_handler` to
   the VAE variant. Breaks the cycle-3 chain (PEquProfile is
   not subclassed by PMaxProfile). Rejected — symmetry with the
   B6a/B6b pattern matters more than the small duplication.
3. Standalone `PMaxLoRAProfile(object)` with free helpers.
   Rejected, YAGNI.

**Encoder/decoder injection.** Three options:

1. **Required kwargs** (no defaults). The caller supplies an
   MLX `nn.Module` encoder/decoder pair. The research repo has
   no canonical architecture ; the maintainer's specific VAE is
   the right choice per use case. **Chosen.**
2. Default factory with a minimal linear VAE. Rejected — couples
   the profile to a specific architecture and obscures the
   research-repo expectation that callers know their model.
3. Optional with `ValueError` on RECOMBINE if not set. Rejected
   — surfaces a runtime error where a construction-time error
   is clearer.

**`LatentSampleQueue` capacity.** Three options:

1. **1024 default** (symmetric with cycle-3 `_DEFAULT_ALPHA_CAPACITY
   = 1024`). Bounded FIFO drops oldest when full. **Chosen.**
2. Unbounded (`None`). Risk: memory grows without bound during
   long dream-runtime sessions if the consumer never dequeues.
   Rejected — bounded by default is safer.
3. 256 (more conservative). Rejected — no empirical
   justification at this stage.

## Design

### Test helper extraction — `tests/unit/profiles/_lora_helpers.py`

Before B6c, `_clones` and `_assert_lora_models_equal` were
duplicated in `test_p_min_lora.py` and `test_p_equ_lora.py`.
B6c is the third copy ; extract them into a shared module:

```python
"""Shared test helpers for the LoRA-substrate profile tests."""
from __future__ import annotations

import numpy as np

from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


def lora_clones(seed: int = 0) -> tuple[LoRAModel, LoRAModel]:
    """Two bit-identical LoRAModels at the same seed.

    Used across PMinLoRAProfile, PEquLoRAProfile, PMaxLoRAProfile
    tests : the dream/awake split needs an awake clone bit-equal
    to the dream model at t=0 so consolidate_log() can be
    verified via bit-equality.
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

`test_p_min_lora.py` and `test_p_equ_lora.py` are amended to
import from `_lora_helpers` (drop their local `_clones` /
`_assert_lora_models_equal` definitions).

### New file `kiki_oniric/profiles/p_max_lora.py`

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
from kiki_oniric.dream.channels.latent_sample import (
    LatentSampleQueue,
)
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
    the cycle-3 skeleton ``OpState`` types to the ``_RealState``
    variants required by the B-series LoRA / VAE handlers.

    Inherited from ``PMaxProfile`` cycle-3 :
    - ``alpha_stream: AlphaStreamBuffer`` (capacity 1024, fifo) —
      awake → dream input channel state surface.
    - ``attention_prior: AttentionPriorChannel`` — ch4 state surface.
    - ``rng: random.Random`` — kept on the dataclass but unused
      by the LoRA handlers (which use MLX RNG keyed off ``seed``).

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
    # Override parent state types — all four ops use _RealState.
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
        # Do NOT call super().__post_init__() — parent registers
        # cycle-3 skeleton handlers; we register the B-series
        # variants (replay/downscale/restructure LoRA + recombine
        # VAE) on the same runtime instead.
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

        ``attention_channel`` is passed as ``None`` because no
        op currently emits ``AttentionPrior`` into the runtime
        log ; the profile's ``attention_prior`` field is a state
        surface for external callers, not a dispatch target.

        The α input channel is not in the apply loop : it carries
        awake → dream traces, populated by the awake side via
        ``profile.alpha_stream.append(...)``, and consumed by
        dream-runtime input slices.

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

### Invariants

- **S1, S2, S3** : inherited from B5 channels and the B-series
  handler-side guards.
- **K1** : 4 `_RealState` instances FLOP-track. `recombine_real
  _handler` tags FLOPs on `RecombineRealState`.
- **DR-4 chain inclusion** :
  - ops(PMinLoRA)={replay, downscale}
    ⊆ ops(PEquLoRA)={replay, downscale, restructure, recombine}
    = ops(PMaxLoRA).
  - channel-emitter sets: PMinLoRA={WeightUpdate} ⊆ PEquLoRA=
    {WeightUpdate, TopologyDiff} ⊆ PMaxLoRA={WeightUpdate,
    TopologyDiff, LatentSample}.
  - Empirically pinned by Test 17 below across the three LoRA
    profiles.
- **R1** : within-machine bit-exact via `lora_clones(seed)`.
  Cross-machine subject to `mx.random.normal` divergence on M1
  Max documented in
  `docs/milestones/r1-cross-machine-m5-vs-m1-2026-05-20.md` +
  upstream `ml-explore/mlx#3568`.

## Testing — `tests/unit/profiles/test_p_max_lora.py` (17 tests)

Test fixtures `_TinyEncoder` / `_TinyDecoder` are imported from
`tests/unit/test_recombine_latent_sample.py` (B4's existing
fixtures). Helpers `lora_clones` and `assert_lora_models_equal`
are imported from the new `tests/unit/profiles/_lora_helpers.py`
shared module.

1. **construction_happy_path** — handlers registered for all 4
   ops ; `weight_channel`, `hierarchy_channel`, `latent_channel`
   all populated ; inherited `attention_prior` and `alpha_stream`
   present.
2. **construction_missing_dream_raises** — kw-only required arg
   missing → `TypeError`.
3. **construction_missing_awake_raises** — symmetric.
4. **construction_missing_encoder_raises** — kw-only required
   arg missing → `TypeError`.
5. **construction_missing_decoder_raises** — symmetric.
6. **replay_emits_weight_update_in_log**.
7. **downscale_emits_weight_update_in_log**.
8. **restructure_emits_topology_diff_in_log**.
9. **recombine_emits_latent_sample_in_log** — `delta_latents=
   [[1.0, 2.0, 3.0, 4.0]]` → `runtime.log[-1].channel_outputs[0]
   ` is a `LatentSample`.
10. **consolidate_log_applies_weight_to_awake_bit_equal**.
11. **consolidate_log_applies_hierarchy_to_awake**.
12. **consolidate_log_enqueues_latent_sample** — after a
    recombine episode + `consolidate_log()`, `len(profile
    .latent_channel) >= 1` and `dequeue()` returns a dict with
    `species`/`latent_vector`/`provenance`.
13. **consolidate_log_mixed_emits_count_4** — chain of 4
    episodes (replay + downscale + restructure + recombine)
    → `consolidate_log() == 4`.
14. **consolidate_log_clears_and_idempotent**.
15. **attention_prior_settable_and_readable** (inherited
    surface).
16. **alpha_stream_append_and_read** — push a `TraceRecord`
    via `profile.alpha_stream.append(rec)` ; FIFO read returns
    it.
17. **dr4_chain_inclusion_full_triple** — build all three
    profiles (`PMinLoRAProfile`, `PEquLoRAProfile`,
    `PMaxLoRAProfile`) at the same seed, run their op sets,
    confirm strict-subset chain on both ops keys and emitted
    channel types. **First test to pin the full triple chain.**

## Scope boundary

**B6c does** : `PMaxLoRAProfile`, 17 tests, extract
`_lora_helpers.py` and amend test_p_min_lora.py +
test_p_equ_lora.py to import from it, CHANGELOG +
framework-C spec §3.1 EN+FR note for P_max, FC-MINOR DualVer
`C-v0.22.0 → C-v0.23.0`, package `0.20.0 → 0.21.0`.

**B6c does not** :
- Modify cycle-3 `PMaxProfile` (legacy + DR-4 chain reference
  intact).
- Provide a default encoder/decoder factory (caller's
  responsibility per the brainstorm decision).
- Dispatch the α input channel through `consolidate_log()` (α
  is input-side ; the apply loop is output-only).
- Add an AttentionPrior emitter (no op writes ch4 to the runtime
  log ; state surface only).
- Touch the cycle-3 `alpha_stream` field semantics beyond
  inheriting it.

## Convention compliance

`p_max_lora.py` mirrors B6a/B6b structure :
`@dataclass(kw_only=True)`, `LoRAModel` forward-ref via
`TYPE_CHECKING`, lazy `apply_channel_outputs` import inside
`consolidate_log()`, `# type: ignore[assignment]` on each state
field override (parent declared skeleton state types).

The new `_lora_helpers.py` lives in `tests/unit/profiles/` to
mirror existing `tests/unit/` conventions (no per-package
`_helpers.py` exists yet ; this is a small profile-local one).

## DualVer

FC-**MINOR** (`C-v0.22.0 → C-v0.23.0`) :

- New substrate component (`PMaxLoRAProfile`) — additive.
- New test helper module (`_lora_helpers.py`) — additive,
  internal.
- No axiom, no Protocol, no primitive signature change.

EC unchanged `+PARTIAL`. `pyproject.toml` version `0.20.0 →
0.21.0`.

## Acceptance criteria

1. `kiki_oniric/profiles/p_max_lora.py` exports
   `PMaxLoRAProfile(PMaxProfile)` with `@dataclass(kw_only=True)`
   ; required kwargs `dream_model`, `awake_model`, `encoder`,
   `decoder` ; optional `lr`, `max_adds_per_episode`, `seed`,
   `latent_queue_capacity=1024`. Overrides all four state
   fields to the `_RealState` types.
2. `__post_init__` registers 4 B-series handlers :
   `replay_lora_handler` + `downscale_lora_handler` +
   `restructure_lora_handler` (all bound to `dream_model`) +
   `recombine_real_handler` (bound to `encoder`/`decoder`).
   Builds `weight_channel`, `hierarchy_channel`,
   `latent_channel` on the awake side. Does NOT call
   `super().__post_init__()`.
3. `consolidate_log() -> int` calls `apply_channel_outputs(
   self.runtime.log, weight_channel=..., hierarchy_channel=...,
   latent_channel=...)` (attention_channel not passed —
   defaults to None per B6a refactor), then `runtime.reset_log()`.
4. `tests/unit/profiles/_lora_helpers.py` exports
   `lora_clones(seed)` and `assert_lora_models_equal(a, b)`.
   `test_p_min_lora.py` and `test_p_equ_lora.py` import from
   it and drop their local duplicates.
5. 17 tests in `tests/unit/profiles/test_p_max_lora.py` pass ;
   full pytest suite green ; `uv run mypy harness tests` clean
   ; `uv run ruff check .` clean.
6. FC-MINOR DualVer bump recorded in `CHANGELOG.md`
   (`[C-v0.23.0+PARTIAL]`) ; framework-C spec §3.1 (EN + FR)
   notes that P_max has a LoRA-substrate variant
   `PMaxLoRAProfile` wiring channels `{1, 2, 3, 4}` with VAE
   `recombine_full` ; `pyproject.toml` bumped `0.20.0 → 0.21.0`.
