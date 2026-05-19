# B0 — Channel-Output Contract Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Give the dream runtime a typed surface on which each operation publishes its channel output, and capture it in the episode log.

**Architecture:** Define four frozen channel-output value types in a new `kiki_oniric/dream/channels.py`. Widen the `OperationHandler` return type so a handler may return a `ChannelOutput`. Add a `channel_outputs` tuple to `EpisodeLogEntry`, strictly parallel to `operations_executed`, and have `DreamRuntime.execute()` populate it. Operations still return `None` — B1-B4 populate real values later.

**Tech Stack:** Python 3.12, `uv`, numpy, pytest, hypothesis, mypy.

**Spec:** `docs/superpowers/specs/2026-05-19-b0-channel-output-contract-design.md`

**Scope correction vs spec:** Planning confirmed the four operations are NOT among the eight DR-3 primitive Protocols in `core/primitives.py` (those are the awake/dream streams + channels). B0 therefore does **not** touch `core/primitives.py` or `test_dr3_substrate.py`, and the return-type widening is non-breaking → DualVer bump is **FC-MINOR** (`C-v0.13.0 → C-v0.14.0`), not MAJOR.

---

## File Structure

- **Create** `kiki_oniric/dream/channels.py` — the four channel-output value types + `ChannelOutput` union.
- **Create** `tests/unit/test_channels.py` — unit tests for the four types.
- **Modify** `kiki_oniric/dream/runtime.py` — `OperationHandler` return type, `EpisodeLogEntry.channel_outputs`, `DreamRuntime.execute()` collection.
- **Create** `tests/unit/test_runtime_channel_outputs.py` — unit tests for the new log field and collection logic.
- **Modify** `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` — §4.1 channel-output types.
- **Modify** the framework-C FR spec under `docs/specs-fr/` — same addition.
- **Modify** `docs/interfaces/primitives.md` — note the produced-value types.
- **Modify** `CHANGELOG.md` — `C-v0.14.0+PARTIAL` entry.
- **Modify** `pyproject.toml` — version `0.11.0 → 0.12.0`.

---

## Task 1: Channel-output value types

**Files:**
- Create: `kiki_oniric/dream/channels.py`
- Test: `tests/unit/test_channels.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_channels.py
"""Unit tests for the four dream-awake channel-output types."""
from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from kiki_oniric.dream.channels import (
    AttentionPrior,
    ChannelOutput,
    HierarchyDiff,
    LatentSample,
    WeightUpdate,
)


def test_weight_update_construction() -> None:
    wu = WeightUpdate(
        lora_delta={"layer0": np.zeros(4, dtype=np.float32)},
    )
    assert wu.fisher_bump is None
    assert wu.lora_delta["layer0"].shape == (4,)


def test_weight_update_rejects_non_finite() -> None:
    bad = {"layer0": np.array([np.inf], dtype=np.float32)}
    with pytest.raises(ValueError, match="S2"):
        WeightUpdate(lora_delta=bad)


def test_latent_sample_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="S2"):
        LatentSample(
            species="x",
            latent_vector=np.array([np.nan], dtype=np.float32),
            provenance="test",
        )


def test_attention_prior_rejects_non_finite() -> None:
    with pytest.raises(ValueError, match="S2"):
        AttentionPrior(prior=np.array([np.inf], dtype=np.float32))


def test_hierarchy_diff_holds_tuple() -> None:
    hd = HierarchyDiff(diff=(("add_node", {"id": "n1"}),))
    assert hd.diff[0][0] == "add_node"


def test_channel_types_are_frozen() -> None:
    ap = AttentionPrior(prior=np.zeros(3, dtype=np.float32))
    with pytest.raises(dataclasses.FrozenInstanceError):
        ap.prior = np.ones(3, dtype=np.float32)  # type: ignore[misc]


def test_channel_output_union_members() -> None:
    members = (WeightUpdate, LatentSample, HierarchyDiff, AttentionPrior)
    for member in members:
        assert member in ChannelOutput.__args__  # type: ignore[attr-defined]
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_channels.py -v`
Expected: FAIL — `ModuleNotFoundError: kiki_oniric.dream.channels`

- [ ] **Step 3: Write the implementation**

```python
# kiki_oniric/dream/channels.py
"""Dream-awake channel-output value types.

The four operations of framework C publish their result on one of
four typed channels (framework-C spec §4.1). This module defines
the value types carried on each channel; the channel Protocols
that *consume* them live in `kiki_oniric/core/primitives.py`.

B0 (issue #15) defines these types and threads them through the
runtime log. The operations themselves return ``None`` until
sub-projects B1-B4 populate real values.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.1
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

__all__ = [
    "WeightUpdate",
    "LatentSample",
    "HierarchyDiff",
    "AttentionPrior",
    "ChannelOutput",
]


@dataclass(frozen=True)
class WeightUpdate:
    """Channel 1 output — parametric consolidation delta.

    Consumed by ``WeightDeltaChannel.apply`` (invariants S1 + S2).
    ``lora_delta`` / ``fisher_bump`` are keyed by layer name to
    match the channel Protocol signature.
    """

    lora_delta: dict[str, NDArray[np.float32]]
    fisher_bump: dict[str, NDArray[np.float32]] | None = None

    def __post_init__(self) -> None:
        for layer, arr in self.lora_delta.items():
            if not np.isfinite(arr).all():
                raise ValueError(
                    f"S2: WeightUpdate.lora_delta[{layer!r}] non-finite"
                )
        if self.fisher_bump is not None:
            for layer, arr in self.fisher_bump.items():
                if not np.isfinite(arr).all():
                    raise ValueError(
                        f"S2: WeightUpdate.fisher_bump[{layer!r}] "
                        f"non-finite"
                    )


@dataclass(frozen=True)
class LatentSample:
    """Channel 2 output — generative-replay latent vector.

    Consumed by ``LatentSampleChannel.enqueue`` (invariant I3).
    """

    species: str
    latent_vector: NDArray[np.float32]
    provenance: str

    def __post_init__(self) -> None:
        if not np.isfinite(self.latent_vector).all():
            raise ValueError("S2: LatentSample.latent_vector non-finite")


@dataclass(frozen=True)
class HierarchyDiff:
    """Channel 3 output — topology diff.

    Consumed by ``HierarchyChangeChannel.apply_diff`` (invariant S3).
    S3 validity is enforced by sub-project B3 when restructure
    produces a real diff.
    """

    diff: tuple[tuple[str, dict], ...]


@dataclass(frozen=True)
class AttentionPrior:
    """Channel 4 output — meta-cognitive attention prior.

    Consumed by ``AttentionPriorChannel.set_prior`` (invariant S4).
    """

    prior: NDArray[np.float32]

    def __post_init__(self) -> None:
        if not np.isfinite(self.prior).all():
            raise ValueError("S2: AttentionPrior.prior non-finite")


ChannelOutput = WeightUpdate | LatentSample | HierarchyDiff | AttentionPrior
```

- [ ] **Step 4: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_channels.py -v`
Expected: PASS — 7 tests.

- [ ] **Step 5: Run mypy on the new module**

Run: `uv run mypy kiki_oniric/dream/channels.py`
Expected: no error on the new file (pre-existing `kiki_oniric` debt is out of scope).

- [ ] **Step 6: Commit**

```bash
git add kiki_oniric/dream/channels.py tests/unit/test_channels.py
git commit -m "feat: add channel-output value types"
```

Commit body (subject ≤50, body ≤72, no AI attribution):
```
feat: add channel-output value types

B0 / issue #15. Define WeightUpdate, LatentSample, HierarchyDiff
and AttentionPrior — the value types carried on the four
dream-awake channels — plus the ChannelOutput union. Frozen
dataclasses with S2 finiteness validation. No runtime wiring
yet (Task 2-3).
```

---

## Task 2: `EpisodeLogEntry.channel_outputs` field

**Files:**
- Modify: `kiki_oniric/dream/runtime.py:21` (`OperationHandler`), `:24-36` (`EpisodeLogEntry`)
- Test: `tests/unit/test_runtime_channel_outputs.py`

- [ ] **Step 1: Write the failing test**

```python
# tests/unit/test_runtime_channel_outputs.py
"""Unit tests for the channel_outputs log field and collection."""
from __future__ import annotations

import numpy as np

from kiki_oniric.dream.channels import WeightUpdate
from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.runtime import DreamRuntime, EpisodeLogEntry


def test_log_entry_channel_outputs_defaults_empty() -> None:
    entry = EpisodeLogEntry(
        episode_id="de-x",
        operations_executed=(Operation.REPLAY,),
        completed=True,
    )
    assert entry.channel_outputs == ()


def test_log_entry_accepts_channel_outputs() -> None:
    wu = WeightUpdate(lora_delta={"l0": np.zeros(2, dtype=np.float32)})
    entry = EpisodeLogEntry(
        episode_id="de-y",
        operations_executed=(Operation.REPLAY,),
        completed=True,
        channel_outputs=(wu,),
    )
    assert entry.channel_outputs == (wu,)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `uv run pytest tests/unit/test_runtime_channel_outputs.py -v`
Expected: FAIL — `TypeError: __init__() got an unexpected keyword argument 'channel_outputs'`

- [ ] **Step 3: Edit `runtime.py` — imports and `OperationHandler`**

Replace lines 13-21:

```python
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from kiki_oniric.dream.channels import ChannelOutput
from kiki_oniric.dream.episode import DreamEpisode, Operation


OperationHandler = Callable[[DreamEpisode], "ChannelOutput | None"]
```

- [ ] **Step 4: Edit `runtime.py` — `EpisodeLogEntry`**

Replace the `EpisodeLogEntry` dataclass (lines 24-36) with:

```python
@dataclass(frozen=True)
class EpisodeLogEntry:
    """Immutable log entry per executed DE — DR-0 accountability.

    `completed=False` + non-empty `error` means the DE raised during
    handler execution. DR-0 still satisfied: every DE produces a log
    entry regardless of handler outcome.

    `channel_outputs` is strictly parallel to `operations_executed`
    (same length, same order); index `i` holds the output of
    `operations_executed[i]`, or `None` if that op emitted nothing.
    Empty `()` marks a legacy entry where no outputs were captured.
    """

    episode_id: str
    operations_executed: tuple[Operation, ...]
    completed: bool
    error: str | None = None
    channel_outputs: tuple[ChannelOutput | None, ...] = ()
```

- [ ] **Step 5: Run test to verify it passes**

Run: `uv run pytest tests/unit/test_runtime_channel_outputs.py -v`
Expected: PASS — 2 tests.

- [ ] **Step 6: Run the full runtime/episode unit tests**

Run: `uv run pytest tests/unit/test_runtime.py tests/unit/test_episode.py -v`
Expected: PASS — default `()` keeps existing constructions valid.

- [ ] **Step 7: Commit**

```bash
git add kiki_oniric/dream/runtime.py tests/unit/test_runtime_channel_outputs.py
git commit -m "feat: add channel_outputs field to log entry"
```

Commit body:
```
feat: add channel_outputs field to log entry

B0 / issue #15. EpisodeLogEntry gains channel_outputs, a tuple
parallel to operations_executed. OperationHandler return type
widened to ChannelOutput | None (non-breaking: a handler that
returns None still conforms). Runtime collection wired in Task 3.
```

---

## Task 3: `DreamRuntime.execute()` collects channel outputs

**Files:**
- Modify: `kiki_oniric/dream/runtime.py:60-98` (`execute`)
- Test: `tests/unit/test_runtime_channel_outputs.py`

- [ ] **Step 1: Add failing tests**

Append to `tests/unit/test_runtime_channel_outputs.py`:

```python
def _episode(ops: tuple[Operation, ...]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={},
        operation_set=ops,
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=1, wall_time_s=1.0, energy_j=1.0),
        episode_id="de-exec",
    )


def test_execute_collects_handler_returns() -> None:
    wu = WeightUpdate(lora_delta={"l0": np.zeros(2, dtype=np.float32)})
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, lambda ep: wu)
    runtime.register_handler(Operation.DOWNSCALE, lambda ep: None)
    runtime.execute(_episode((Operation.REPLAY, Operation.DOWNSCALE)))
    entry = runtime.log[-1]
    assert entry.channel_outputs == (wu, None)
    assert len(entry.channel_outputs) == len(entry.operations_executed)


def test_execute_channel_outputs_parallel_on_error() -> None:
    def boom(ep: DreamEpisode) -> None:
        raise RuntimeError("handler failed")

    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, lambda ep: None)
    runtime.register_handler(Operation.DOWNSCALE, boom)
    try:
        runtime.execute(_episode((Operation.REPLAY, Operation.DOWNSCALE)))
    except RuntimeError:
        pass
    entry = runtime.log[-1]
    assert entry.completed is False
    assert len(entry.channel_outputs) == len(entry.operations_executed)
    assert entry.channel_outputs == (None, None)
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `uv run pytest tests/unit/test_runtime_channel_outputs.py -v`
Expected: FAIL — `channel_outputs` is `()` because `execute()` does not yet collect.

- [ ] **Step 3: Rewrite `execute()`**

Replace the body of `execute()` (lines 73-98) with:

```python
        for op in episode.operation_set:
            if op not in self._handlers:
                raise NotImplementedError(
                    f"No handler registered for operation {op.value!r}"
                )

        error: str | None = None
        completed = False
        executed_ops: list[Operation] = []
        outputs: list[ChannelOutput | None] = []
        try:
            for op in episode.operation_set:
                executed_ops.append(op)
                outputs.append(None)  # placeholder keeps lengths equal
                outputs[-1] = self._handlers[op](episode)
            completed = True
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"
            raise
        finally:
            self._log.append(
                EpisodeLogEntry(
                    episode_id=episode.episode_id,
                    operations_executed=tuple(executed_ops),
                    completed=completed,
                    error=error,
                    channel_outputs=tuple(outputs),
                )
            )
```

Note: appending the `None` placeholder *before* calling the handler
guarantees `outputs` and `executed_ops` stay equal length even when
the handler raises — the failing op keeps its `None` slot.

- [ ] **Step 4: Run tests to verify they pass**

Run: `uv run pytest tests/unit/test_runtime_channel_outputs.py -v`
Expected: PASS — 4 tests.

- [ ] **Step 5: Run the full test suite**

Run: `uv run pytest -q`
Expected: PASS — same count as before plus the new tests, 0 failures.

- [ ] **Step 6: Run mypy**

Run: `uv run mypy harness tests`
Expected: `Success: no issues found`.

- [ ] **Step 7: Commit**

```bash
git add kiki_oniric/dream/runtime.py tests/unit/test_runtime_channel_outputs.py
git commit -m "feat: collect channel outputs in execute()"
```

Commit body:
```
feat: collect channel outputs in execute()

B0 / issue #15. DreamRuntime.execute() now records each handler
return value into EpisodeLogEntry.channel_outputs, parallel to
operations_executed. A None placeholder is appended before each
handler call so the two tuples stay equal length on the error
path. Handlers still return None until B1-B4.
```

---

## Task 4: Documentation and DualVer sync

**Files:**
- Modify: `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md` (§4.1)
- Modify: framework-C FR spec under `docs/specs-fr/`
- Modify: `docs/interfaces/primitives.md`
- Modify: `CHANGELOG.md`
- Modify: `pyproject.toml`

- [ ] **Step 1: Locate the spec sections**

Run: `grep -n "§4.1\|## 4.1\|4.1 " docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
Run: `ls docs/specs-fr/`
Identify §4.1 (the dream-episode / channels section) in the EN spec and the FR counterpart filename.

- [ ] **Step 2: Add the channel-output types to the EN spec §4.1**

In §4.1, after the channel descriptions, add a subsection:

```markdown
#### Channel-output value types (B0, issue #15)

Each operation publishes its result as a typed channel-output
value, captured in `EpisodeLogEntry.channel_outputs`:

- Channel 1 — `WeightUpdate(lora_delta, fisher_bump)`
- Channel 2 — `LatentSample(species, latent_vector, provenance)`
- Channel 3 — `HierarchyDiff(diff)`
- Channel 4 — `AttentionPrior(prior)`

`ChannelOutput` is their union. Implemented in
`kiki_oniric/dream/channels.py`. The operations return `None`
until sub-projects B1-B4 populate real values.
```

- [ ] **Step 3: Mirror the addition in the FR spec**

Apply the equivalent French subsection to the FR counterpart
identified in Step 1 (EN→FR propagation rule, repo CLAUDE.md).

- [ ] **Step 4: Note the produced-value types in `primitives.md`**

Under the "Dream → Awake channels" section of
`docs/interfaces/primitives.md`, add a line per channel pointing
at the produced value type, e.g. under Canal 1:

```markdown
**Produced value**: `WeightUpdate` (`kiki_oniric/dream/channels.py`),
captured in `EpisodeLogEntry.channel_outputs`.
```

Repeat for canals 2-4 with `LatentSample`, `HierarchyDiff`,
`AttentionPrior`.

- [ ] **Step 5: Add the CHANGELOG entry**

Insert at the top of the changelog body (after the header block,
before the most recent entry):

```markdown
## [C-v0.14.0+PARTIAL] — 2026-05-19 — channel-output contract (B0)

### Formal axis (FC) — MINOR (v0.13.0 → v0.14.0)

- **New module** `kiki_oniric/dream/channels.py` : the four
  channel-output value types (`WeightUpdate`, `LatentSample`,
  `HierarchyDiff`, `AttentionPrior`) and the `ChannelOutput`
  union. Frozen dataclasses with S2 finiteness validation.
- **`EpisodeLogEntry`** gains `channel_outputs`, a tuple parallel
  to `operations_executed`. Default `()` — data-level backward
  compatible.
- **`OperationHandler`** return type widened `None →
  ChannelOutput | None`. Non-breaking: a handler returning `None`
  still conforms. The four DR-3 primitive Protocols in
  `core/primitives.py` are unchanged — hence FC-MINOR.
- **`DreamRuntime.execute()`** records each handler return into
  `channel_outputs`.
- Sub-project B0 of issue #15. Operations still return `None`;
  B1-B4 populate real values, B5 rewires `consolidate()`.

### Empirical axis (EC) — UNCHANGED (PARTIAL)

- No new substrate, op, or empirical claim. EC stays `+PARTIAL`.

### Packaging

- `pyproject.toml` version bumped `0.11.0 → 0.12.0`.
```

- [ ] **Step 6: Bump `pyproject.toml`**

Change `version = "0.11.0"` to `version = "0.12.0"`.

- [ ] **Step 7: Regenerate the lockfile version pin**

Run: `uv lock`
Expected: `uv.lock` updates the `dreamofkiki` package version to `0.12.0`.

- [ ] **Step 8: Commit**

```bash
git add docs/specs/2026-04-17-dreamofkiki-framework-C-design.md docs/specs-fr/ docs/interfaces/primitives.md CHANGELOG.md pyproject.toml uv.lock
git commit -m "docs: sync framework-C spec for B0 channels"
```

Commit body:
```
docs: sync framework-C spec for B0 channels

B0 / issue #15. Document the four channel-output value types in
framework-C spec section 4.1 (EN + FR) and primitives.md. Add
the C-v0.14.0+PARTIAL changelog entry and bump the package
version to 0.12.0.
```

---

## Task 5: Final verification

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
Expected: empty — all four task commits landed.

- [ ] **Step 5: Update issue #15**

Add a comment on issue #15 noting B0 is complete and B1-B4 are
unblocked (the channel-output contract is now frozen).

---

## Self-Review

- **Spec coverage:** four channel types (Task 1) ✓; `channel_outputs` field (Task 2) ✓; handler signature widening (Task 2) ✓; runtime collection (Task 3) ✓; spec/primitives.md/FR/CHANGELOG/DualVer sync (Task 4) ✓. The spec's "migrate four handlers to the new signature" item is intentionally dropped — the return-type widening is non-breaking, so handlers returning `None` already conform; they are touched in B1-B4. The spec's "update core/primitives.py / test_dr3" items are dropped — confirmed out of scope (the four ops are not DR-3 primitives).
- **Placeholder scan:** the FR spec filename is resolved by an explicit `ls` command in Task 4 Step 1, not left vague. No other placeholders.
- **Type consistency:** `WeightUpdate`, `LatentSample`, `HierarchyDiff`, `AttentionPrior`, `ChannelOutput`, `OperationHandler`, `EpisodeLogEntry.channel_outputs` used identically across Tasks 1-4. `WeightUpdate.lora_delta` is `dict[str, NDArray[np.float32]]` to match `WeightDeltaChannel.apply` in `core/primitives.py`.
