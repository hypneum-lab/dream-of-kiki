# B6a — `PMinLoRAProfile` wires P_min to the B5 channel apply loop

**Date** : 2026-05-20
**Status** : design approved, pending spec review
**Tracking issue** : (continuation of #15 ; B6 decomposes into B6a/B6b/B6c by profile)
**Scope** : sub-project B6a — first half of B6 (P_min profile only).

---

## Context

Issue #15 / approach B shipped:

- **B0..B4** — the four dream operations emit real channel outputs
  (`WeightUpdate`, `TopologyDiff`, `LatentSample`).
- **B5** (`C-v0.20.0`) — three concrete LoRA-target channel
  implementations and the free function
  `apply_channel_outputs(log, *, weight_channel,
  hierarchy_channel, latent_channel, attention_channel=None)`
  that closes the awake↔dream loop. Tested in isolation via
  dream/awake `LoRAModel` clones.

But **no profile is wired to use it**. `PMinProfile` /
`PEquProfile` / `PMaxProfile` still register cycle-3 *skeleton*
handlers (`replay_handler`, `downscale_handler`, …) that don't
emit anything. The full awake↔dream loop demonstrated by B5's
end-to-end test is unavailable to the existing profile-driven
consolidation paths.

**B6** addresses this. Decomposition by profile:

- **B6a (this spec)** — `PMinLoRAProfile`. Channels out: `{1}`
  (`WeightUpdate` only). Ops: `{replay, downscale}`.
- B6b (future) — `PEquLoRAProfile`. Channels out: `{1, 3, 4}`.
  Ops: `{replay, downscale, restructure, recombine_light}`.
- B6c (future) — `PMaxLoRAProfile`. Channels out: `{1, 2, 3, 4}`.
  Ops: `{replay, downscale, restructure, recombine_full}` with
  encoder/decoder.

## Problem

`PMinProfile` (cycle-3) registers `replay_handler` and
`downscale_handler` — the skeleton variants that mutate a counter
state and return `None`. The `_lora_handler` variants shipped in
B1b/B2 require a `LoRAModel` and emit `WeightUpdate`s into the
runtime log, but there is no profile that wires them.

B6a adds a `PMinLoRAProfile(PMinProfile)` subclass that:
1. Takes a `dream_model` + `awake_model` pair.
2. Registers `replay_lora_handler` + `downscale_lora_handler` on
   the dream model.
3. Holds a `LoRAWeightDeltaChannel` instance targeting the awake
   model.
4. Exposes `consolidate_log() -> int` that calls
   `apply_channel_outputs(self.runtime.log, weight_channel=
   self.weight_channel)` and clears the log on success.

## Approaches considered

**How to wire LoRA mode into the profile.** Three options were
considered (see brainstorming session 2026-05-20):

1. **Optional fields** on the existing `PMinProfile`
   (`dream_model=None`, `awake_model=None`). `__post_init__`
   branches on whether both are set. **Mental model murky** —
   one class, two modes. Rejected.
2. **Subclass `PMinLoRAProfile(PMinProfile)`**. Cycle-3
   `PMinProfile` untouched (legacy tests + DR-4 inclusion checks
   keep passing). The LoRA variant is its own class with kw-only
   args. **Chosen.**
3. Replace `PMinProfile` outright — breaking change for
   ~10-15 existing tests that instantiate it without a model.
   Rejected — no scientific value in the cassure ; legacy
   profile is the reference for skeleton handlers (DR-4 chain
   evidence).

**Where the consolidation entry-point lives.** Three options:

1. **`profile.consolidate_log() -> int`** method on the LoRA
   subclass. Encapsulates "this profile knows its channel set,
   call its channels". **Chosen.**
2. Always-external — caller invokes `apply_channel_outputs(profile
   .runtime.log, weight_channel=profile.weight_channel, …)`
   directly. Verbose. Rejected.
3. Free function `consolidate_profile(profile)` with `isinstance`
   dispatch. Adds coupling between the free function and every
   profile subclass. Rejected.

## Design

### Refactor in `kiki_oniric/consolidate.py`

Currently `apply_channel_outputs` requires `weight_channel`,
`hierarchy_channel`, `latent_channel` as kwargs (only
`attention_channel` is optional). B6a relaxes the first three
to `Optional[…] = None` using the same pattern as
`attention_channel`:

```python
def apply_channel_outputs(
    log: list[EpisodeLogEntry],
    *,
    weight_channel: "WeightDeltaChannel | None" = None,
    hierarchy_channel: "HierarchyChangeChannel | None" = None,
    latent_channel: "LatentSampleChannel | None" = None,
    attention_channel: "AttentionPriorChannel | None" = None,
) -> int:
```

If a `WeightUpdate` is encountered but `weight_channel is None`,
raise `ValueError("apply_channel_outputs: weight_channel
required for WeightUpdate outputs")`. Same for the other types.
The existing `attention_channel`-missing raise stays as-is.

This refactor lets a profile that emits only ch1 (PMinLoRA)
omit hierarchy/latent without passing dummy stubs.

### New file `kiki_oniric/profiles/p_min_lora.py`

```python
"""P_min LoRA-substrate profile (B6a, issue #15 continuation).

Wires the B-series LoRA-emitting handlers (replay_lora_handler,
downscale_lora_handler) onto a dream/awake LoRAModel pair and
provides ``consolidate_log()`` to apply the runtime's channel
outputs onto the awake model via the B5 ``LoRAWeightDeltaChannel``.

Channels out (per framework-C spec §3.1) : ``{WEIGHT_DELTA}``
only — neither ``TopologyDiff`` nor ``LatentSample`` nor
``AttentionPrior`` are emitted by P_min's op set.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from kiki_oniric.dream.episode import Operation
from kiki_oniric.dream.operations.replay_real import (
    ReplayRealState,
    replay_lora_handler,
)
from kiki_oniric.dream.operations.downscale_real import (
    DownscaleRealState,
    downscale_lora_handler,
)
from kiki_oniric.dream.channels.weight_delta import (
    LoRAWeightDeltaChannel,
)
from kiki_oniric.profiles.p_min import PMinProfile

if TYPE_CHECKING:
    from kiki_oniric.substrates.micro_kiki.lora_model import LoRAModel


@dataclass(kw_only=True)
class PMinLoRAProfile(PMinProfile):
    """P_min profile rewired for the B-series LoRA substrate.

    Required kwargs: ``dream_model`` and ``awake_model`` (two
    ``LoRAModel`` instances; for within-machine bit-exact
    reproducibility they should start at the same ``seed`` so
    ``consolidate_log()`` produces an awake model identical to
    the dream model).

    The parent's ``replay_state`` / ``downscale_state`` fields
    are overridden with the ``_RealState`` variants required by
    the LoRA handlers.
    """

    dream_model: "LoRAModel"
    awake_model: "LoRAModel"
    lr: float = 0.01
    # Override parent state types — LoRA handlers need _RealState.
    replay_state: ReplayRealState = field(
        default_factory=ReplayRealState,
    )
    downscale_state: DownscaleRealState = field(
        default_factory=DownscaleRealState,
    )
    weight_channel: LoRAWeightDeltaChannel | None = None

    def __post_init__(self) -> None:
        # Intentionally do NOT call super().__post_init__() — the
        # parent registers skeleton handlers. We register the
        # LoRA-emitting variants instead, on the dream model.
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
        self.weight_channel = LoRAWeightDeltaChannel(
            self.awake_model,
        )

    def consolidate_log(self) -> int:
        """Replay every WeightUpdate in the runtime log onto the
        awake model via the weight channel. Clears the log on
        success so a second call is idempotent.

        Returns the number of channel outputs dispatched.
        """
        from kiki_oniric.consolidate import apply_channel_outputs

        count = apply_channel_outputs(
            self.runtime.log,
            weight_channel=self.weight_channel,
        )
        self.runtime.reset_log()
        return count
```

### Runtime support — `DreamRuntime.reset_log()`

`DreamRuntime` (cycle-2) does not currently expose a `reset_log()`
method. B6a adds it as a one-liner:

```python
def reset_log(self) -> None:
    """Clear the accountability log. Used by profile-driven
    consolidation flows that consume the log via
    ``apply_channel_outputs`` and want a clean slate for the
    next dream cycle.
    """
    self._log.clear()
```

### Invariants

- **S1** (retained non-regression) — `LoRAWeightDeltaChannel
  .apply` is additive. With `awake_model` initialised at the
  same `seed` as `dream_model`, after a replay episode + a
  `consolidate_log()` call, the two models are bit-equal
  within-machine. The end-to-end B5 test in
  `tests/unit/test_apply_channel_outputs.py` already
  demonstrates this property at the channel level; B6a's tests
  demonstrate it at the profile level.
- **S2** (finite) — enforced at `WeightUpdate.__post_init__` (B0)
  and `LoRAWeightDeltaChannel.apply` (B5).
- **K1** (compute budget) — `ReplayRealState` /
  `DownscaleRealState` track FLOPs as in B1b / B2.
- **DR-4** (chain inclusion) — `PMinLoRAProfile`'s op set
  `{replay, downscale}` ⊆ the B6b future `PEquLoRAProfile` set
  ⊆ B6c `PMaxLoRAProfile`. Channels out idem.
- **R1** (reproducibility) — within-machine bit-exact ; cross-
  machine subject to the M5/M3 Ultra/M1 Max divergence
  documented in
  `docs/milestones/r1-cross-machine-m5-vs-m1-2026-05-20.md`.

## Testing — `tests/unit/profiles/test_p_min_lora.py`

End-to-end via `DreamRuntime.execute` + `consolidate_log`. Each
test creates `_clones(seed=K)` (helper from B5 tests : two
bit-identical `LoRAModel`s).

1. **construction_happy_path** — building `PMinLoRAProfile(
   dream_model=dm, awake_model=am)` succeeds; handlers
   registered for REPLAY + DOWNSCALE; `weight_channel` is a
   `LoRAWeightDeltaChannel`.
2. **construction_missing_dream_model_raises** — `PMinLoRAProfile
   (awake_model=am)` → `TypeError` (kw-only required).
3. **construction_missing_awake_model_raises** — symmetric.
4. **replay_emits_weight_update_in_log** — execute an episode
   with `beta_records` → `runtime.log[-1].channel_outputs[0]`
   is a `WeightUpdate`.
5. **downscale_emits_weight_update_in_log** — execute with
   `shrink_factor < 1.0` → same.
6. **consolidate_log_applies_to_awake_bit_equal** — replay
   episode on dream, then `consolidate_log()` → `awake_model`
   matches `dream_model` bit-for-bit (within-machine R1).
7. **consolidate_log_clears_log** — after `consolidate_log()`,
   `len(self.runtime.log) == 0`; calling it a second time
   returns 0.
8. **consolidate_log_returns_dispatch_count** — N replay
   episodes → `consolidate_log()` returns N (one WeightUpdate
   each).
9. **consolidate_log_on_empty_log_returns_zero** — no episodes
   → `consolidate_log() == 0`, no side effects.
10. **no_topology_no_latent_in_log** — after several episodes,
    every non-None `channel_outputs[i]` is a `WeightUpdate`
    (PMin's spec channel set is `{1}` only).

Plus one runtime test:

11. **dream_runtime_reset_log_clears** — `runtime.reset_log()`
    after several `execute()` calls leaves `len(runtime.log)
    == 0`.

## Scope boundary

**B6a does** : refactor `apply_channel_outputs` to make the
three non-attention channels Optional ; ship `PMinLoRAProfile` ;
add `DreamRuntime.reset_log()` ; ~11 tests ; CHANGELOG +
framework-C spec §3.1 EN+FR note ; FC-MINOR `C-v0.20.0 →
C-v0.21.0` ; package `0.18.0 → 0.19.0`.

**B6a does not** : touch `PMinProfile` (cycle-3 legacy intact) ;
ship `PEquLoRAProfile` (B6b) or `PMaxLoRAProfile` (B6c) ;
modify `consolidate()` facade signature (nerve-wml interface
stable) ; auto-clear logs in any other code path ; add a
`target_channels_out` filter (B6c may need it for
`recombine_full` vs `recombine_light`).

## Convention compliance

`p_min_lora.py` is a new file in `kiki_oniric/profiles/`. It
imports `LoRAModel` lazily via `TYPE_CHECKING` (the type
annotation forward-ref) so pure-skeleton code paths don't pay
the MLX import cost. The runtime / state / channel imports are
eager (they're already on the cycle-2 dependency tree).

## DualVer

FC-**MINOR** (`C-v0.20.0 → C-v0.21.0`) :

- New substrate component (`PMinLoRAProfile`) — additive.
- `apply_channel_outputs` signature relaxation (channels become
  Optional) — backwards compatible : every existing call with
  all three kwargs still works.
- `DreamRuntime.reset_log()` — additive method.
- No axiom, no Protocol, no primitive signature change.

EC unchanged `+PARTIAL`. `pyproject.toml` version `0.18.0 →
0.19.0`.

## Acceptance criteria

1. `kiki_oniric/profiles/p_min_lora.py` exports
   `PMinLoRAProfile` with kw-only `dream_model`, `awake_model`,
   `lr=0.01`. Inherits from `PMinProfile`. Overrides
   `replay_state` to `ReplayRealState` and `downscale_state`
   to `DownscaleRealState`.
2. `__post_init__` registers `replay_lora_handler` and
   `downscale_lora_handler` on `self.runtime` against the
   `dream_model` ; builds `self.weight_channel =
   LoRAWeightDeltaChannel(awake_model)`.
3. `consolidate_log() -> int` dispatches the runtime log onto
   `weight_channel` and clears the log on success.
4. `DreamRuntime.reset_log()` clears `self._log`.
5. `apply_channel_outputs` accepts `weight_channel=None` /
   `hierarchy_channel=None` / `latent_channel=None` ; raises
   `ValueError` if an output of that type appears with the
   matching channel missing.
6. 11 tests in `tests/unit/profiles/test_p_min_lora.py` pass ;
   full pytest suite green ; `uv run mypy harness tests` clean ;
   `uv run ruff check .` clean.
7. FC-MINOR DualVer bump recorded in `CHANGELOG.md`
   (`[C-v0.21.0+PARTIAL]`) ; framework-C spec §3.1 (EN + FR)
   notes that P_min has a LoRA-substrate variant
   `PMinLoRAProfile` wiring the B-series handlers ;
   `pyproject.toml` bumped `0.18.0 → 0.19.0`.
