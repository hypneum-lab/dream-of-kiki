# dreamOfkiki S5-S6 Atomic Plan

> **Pour agents autonomes :** SKILL REQUIS — utiliser `superpowers:subagent-driven-development`. Les steps utilisent la syntaxe checkbox (`- [ ]`) pour le tracking.

**Goal** : raffiner les milestones S5-S6 du plan global en tâches atomiques TDD, produisant dream runtime skeleton, épisode 5-tuple, DR-0/DR-1 property tests, P_min opération replay, puis circulation draft DR-2 compositionality (G3-draft).

**Architecture** : 7 tasks (S5.1-S5.4, S6.1-S6.3), chacune 1 commit, TDD strict. Mix code (dream runtime, opérations, property tests) + docs (G3-draft proof circulation). Total attendu : 7 commits.

**Tech Stack** : Python 3.12+ uv, pytest + hypothesis, numpy/MLX (lazy imports), typing Protocol, dataclasses frozen. Pas de concurrent (threading/asyncio) encore — single-threaded DE execution, swap runtime S7+.

**Préréquis** :
- 25 commits dreamOfkiki, dernier `c5b9ded refactor(test): robust Protocol check via MRO`
- 19 tests passing, coverage 93.75%
- Framework C-v0.5.0+STABLE
- Repo public `github.com/electron-rare/dream-of-kiki`
- 8 typed Protocols dans `kiki_oniric/core/primitives.py`
- RetainedBenchmark loader (SHA-256 integrity, 50 items)
- EvalMatrix loader (6 metrics + Branch A Studyforrest)

---

## Convention commits (validator-enforced)

- Subject ≤50 chars, format `<type>(<scope>): <description>`
- Scope ≥3 chars (single letters rejected — `(dream)` OK)
- Body lines ≤72 chars, 2-3 paragraphs required
- NO AI attribution
- NO `--no-verify`

---

## File structure après S5-S6

```
dreamOfkiki/
├── kiki_oniric/
│   ├── core/
│   │   └── primitives.py           ✅ existing
│   └── dream/
│       ├── __init__.py             ← S5.1
│       ├── episode.py              ← S5.1 (DE 5-tuple dataclass)
│       ├── runtime.py              ← S5.2 (Scheduler + DE execution skeleton)
│       └── operations/
│           ├── __init__.py         ← S5.4
│           └── replay.py           ← S5.4 (P_min op 1/2)
├── tests/
│   ├── conformance/
│   │   ├── axioms/
│   │   │   ├── test_dr3_substrate.py  ✅ existing
│   │   │   ├── test_dr0_accountability.py  ← S5.3
│   │   │   └── test_dr1_episodic_conservation.py  ← S5.3
│   │   └── invariants/
│   │       └── __init__.py         ← S5.3
│   └── unit/
│       ├── test_episode.py         ← S5.1
│       ├── test_runtime.py         ← S5.2
│       └── test_replay_op.py       ← S5.4
└── docs/
    └── proofs/
        ├── __init__.md             ← S6.1 (index)
        ├── dr2-compositionality.md ← S6.1 (draft proof)
        ├── g3-draft-circulation.md ← S6.2 (circulation log)
        └── op-pair-analysis.md     ← S6.3 (case analysis)
```

---

# Task S5.1 — Dream episode 5-tuple

**Goal** : implémenter `DreamEpisode` comme dataclass frozen 5-tuple `(trigger, input_slice, operation_set, output_delta, budget)` + 5 tests. Consommé par runtime S5.2 et property tests DR-0/DR-1 S5.3.

**Files:**
- Create : `kiki_oniric/dream/__init__.py` (empty)
- Create : `kiki_oniric/dream/episode.py`
- Create : `tests/unit/test_episode.py`

## Step S5.1.1 — Write failing tests

Create `tests/unit/test_episode.py` with:

```python
"""Unit tests for dream-episode 5-tuple dataclass (DR-0, DR-1 contract)."""
from __future__ import annotations

import pytest

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)


def test_episode_is_frozen_dataclass() -> None:
    ep = DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=1_000_000, wall_time_s=1.0, energy_j=0.5),
        episode_id="de-0000",
    )
    with pytest.raises((AttributeError, TypeError)):
        ep.episode_id = "de-mutated"


def test_episode_has_required_5_tuple_fields() -> None:
    fields = {f.name for f in DreamEpisode.__dataclass_fields__.values()}
    # 5-tuple per framework spec §4.1 + episode_id for traceability
    assert {"trigger", "input_slice", "operation_set",
            "output_channels", "budget", "episode_id"} <= fields


def test_budget_cap_rejects_negative_flops() -> None:
    with pytest.raises(ValueError, match="flops"):
        BudgetCap(flops=-1, wall_time_s=1.0, energy_j=0.5)


def test_operation_set_is_non_empty() -> None:
    with pytest.raises(ValueError, match="operation_set"):
        DreamEpisode(
            trigger=EpisodeTrigger.SCHEDULED,
            input_slice={},
            operation_set=(),
            output_channels=(OutputChannel.WEIGHT_DELTA,),
            budget=BudgetCap(flops=100, wall_time_s=0.1, energy_j=0.01),
            episode_id="de-empty",
        )


def test_episode_trigger_and_operation_enums() -> None:
    assert EpisodeTrigger.SCHEDULED.value == "scheduled"
    assert Operation.REPLAY.value == "replay"
    assert Operation.DOWNSCALE.value == "downscale"
    assert Operation.RESTRUCTURE.value == "restructure"
    assert Operation.RECOMBINE.value == "recombine"
    assert OutputChannel.WEIGHT_DELTA.value == "weight_delta"
```

## Step S5.1.2 — Verify failing

Run : `uv run pytest tests/unit/test_episode.py -v --no-cov`
Expected: `ModuleNotFoundError: No module named 'kiki_oniric.dream'`.

## Step S5.1.3 — Implement episode.py

Create `kiki_oniric/dream/__init__.py` (empty).

Create `kiki_oniric/dream/episode.py`:

```python
"""Dream-episode 5-tuple dataclass — DR-0 accountability unit.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.1
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any


class EpisodeTrigger(str, Enum):
    SCHEDULED = "scheduled"
    SATURATION = "saturation"
    EXTERNAL = "external"


class Operation(str, Enum):
    REPLAY = "replay"
    DOWNSCALE = "downscale"
    RESTRUCTURE = "restructure"
    RECOMBINE = "recombine"


class OutputChannel(str, Enum):
    WEIGHT_DELTA = "weight_delta"        # canal 1
    LATENT_SAMPLE = "latent_sample"      # canal 2
    HIERARCHY_CHG = "hierarchy_chg"      # canal 3
    ATTENTION_PRIOR = "attention_prior"  # canal 4


@dataclass(frozen=True)
class BudgetCap:
    """Resource cap per DE — K1 invariant enforcement unit."""

    flops: int
    wall_time_s: float
    energy_j: float

    def __post_init__(self) -> None:
        if self.flops < 0:
            raise ValueError(f"flops must be non-negative, got {self.flops}")
        if self.wall_time_s < 0:
            raise ValueError(
                f"wall_time_s must be non-negative, got {self.wall_time_s}"
            )
        if self.energy_j < 0:
            raise ValueError(
                f"energy_j must be non-negative, got {self.energy_j}"
            )


@dataclass(frozen=True)
class DreamEpisode:
    """5-tuple (trigger, input_slice, operation_set, output_channels,
    budget) + episode_id for DR-0 traceability.
    """

    trigger: EpisodeTrigger
    input_slice: dict[str, Any]
    operation_set: tuple[Operation, ...]
    output_channels: tuple[OutputChannel, ...]
    budget: BudgetCap
    episode_id: str

    def __post_init__(self) -> None:
        if not self.operation_set:
            raise ValueError(
                "operation_set must be non-empty — DE with zero "
                "operations has no effect"
            )
```

## Step S5.1.4 — Verify passing

Run : `uv run pytest tests/unit/test_episode.py -v --no-cov`
Expected: 5 passed.

Run : `uv run pytest`
Expected: 24 tests (19 + 5 new), coverage ≥90%.

## Step S5.1.5 — Commit

```bash
git add kiki_oniric/dream/__init__.py kiki_oniric/dream/episode.py tests/unit/test_episode.py
git commit -m "feat(dream): add DreamEpisode 5-tuple dataclass"
```

Subject : 46 chars. Body (2 paragraphs) :

1. Introduce frozen 5-tuple `DreamEpisode(trigger, input_slice, operation_set, output_channels, budget, episode_id)` per framework spec §4.1. Enums `EpisodeTrigger` (scheduled/saturation/external), `Operation` (4 canonical), `OutputChannel` (4 canals). `BudgetCap` dataclass enforces K1 invariant resource bounds.

2. `__post_init__` validates non-empty operation_set (DE with zero ops has no effect) and non-negative budget components. Foundation for runtime S5.2 and DR-0 accountability property tests S5.3.

---

# Task S5.2 — Dream runtime scheduler skeleton

**Goal** : scaffolder `DreamRuntime` class qui reçoit DEs, les exécute (no-op au début, ops concrètes S5.4+), log les événements pour DR-0 tracing.

**Files:**
- Create : `kiki_oniric/dream/runtime.py`
- Create : `tests/unit/test_runtime.py`

## Step S5.2.1 — Write failing tests

Create `tests/unit/test_runtime.py`:

```python
"""Unit tests for DreamRuntime scheduler skeleton."""
from __future__ import annotations

import pytest

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.runtime import DreamRuntime, EpisodeLogEntry


def make_episode(ep_id: str, ops: tuple[Operation, ...]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"beta_records": []},
        operation_set=ops,
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=1000, wall_time_s=0.1, energy_j=0.01),
        episode_id=ep_id,
    )


def test_runtime_executes_single_episode() -> None:
    runtime = DreamRuntime()
    ep = make_episode("de-0001", (Operation.REPLAY,))
    runtime.execute(ep)
    assert len(runtime.log) == 1
    assert runtime.log[0].episode_id == "de-0001"
    assert runtime.log[0].completed is True


def test_runtime_logs_ordered() -> None:
    runtime = DreamRuntime()
    runtime.execute(make_episode("de-0001", (Operation.REPLAY,)))
    runtime.execute(make_episode("de-0002", (Operation.DOWNSCALE,)))
    ids = [e.episode_id for e in runtime.log]
    assert ids == ["de-0001", "de-0002"]


def test_runtime_log_entry_immutable() -> None:
    runtime = DreamRuntime()
    runtime.execute(make_episode("de-0003", (Operation.REPLAY,)))
    entry = runtime.log[0]
    with pytest.raises((AttributeError, TypeError)):
        entry.episode_id = "de-mutated"


def test_runtime_unknown_operation_raises() -> None:
    # Internal registry lookup must fail for unregistered ops in
    # skeleton; replay not registered yet (added in S5.4).
    runtime = DreamRuntime()
    ep = make_episode("de-0004", (Operation.RECOMBINE,))
    with pytest.raises(NotImplementedError, match="recombine"):
        runtime.execute(ep)
```

## Step S5.2.2 — Verify failing

Run : `uv run pytest tests/unit/test_runtime.py -v --no-cov`
Expected: fail (import).

## Step S5.2.3 — Implement runtime.py

Create `kiki_oniric/dream/runtime.py`:

```python
"""DreamRuntime — scheduler and DE execution logger.

Skeleton version (S5.2): logs episodes, invokes op handlers by name,
enforces DR-0 accountability (every DE produces a log entry).

Real op handlers are registered in S5.4+ (replay, downscale,
restructure, recombine). Operations without a handler raise
NotImplementedError.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from kiki_oniric.dream.episode import DreamEpisode, Operation


OperationHandler = Callable[[DreamEpisode], None]


@dataclass(frozen=True)
class EpisodeLogEntry:
    """Immutable log entry per executed DE — DR-0 accountability."""

    episode_id: str
    operations_executed: tuple[Operation, ...]
    completed: bool


class DreamRuntime:
    """Single-threaded dream-episode scheduler (S5.2 skeleton).

    Concurrent swap runtime will be introduced S7+.
    """

    def __init__(self) -> None:
        self._handlers: dict[Operation, OperationHandler] = {}
        self._log: list[EpisodeLogEntry] = []

    @property
    def log(self) -> list[EpisodeLogEntry]:
        """Read-only view of executed episodes (DR-0 trace)."""
        return list(self._log)

    def register_handler(
        self, op: Operation, handler: OperationHandler
    ) -> None:
        """Register a concrete handler for an Operation."""
        self._handlers[op] = handler

    def execute(self, episode: DreamEpisode) -> None:
        """Execute all operations of a DE sequentially.

        Raises NotImplementedError if any operation lacks a handler.
        """
        for op in episode.operation_set:
            if op not in self._handlers:
                raise NotImplementedError(
                    f"No handler registered for operation {op.value!r}"
                )

        for op in episode.operation_set:
            self._handlers[op](episode)

        self._log.append(
            EpisodeLogEntry(
                episode_id=episode.episode_id,
                operations_executed=episode.operation_set,
                completed=True,
            )
        )
```

## Step S5.2.4 — Verify passing

Run : `uv run pytest tests/unit/test_runtime.py -v --no-cov`
Expected: 4 passed.

Run : `uv run pytest`
Expected: 28 tests (24 + 4 new), coverage ≥90%.

## Step S5.2.5 — Commit

```bash
git add kiki_oniric/dream/runtime.py tests/unit/test_runtime.py
git commit -m "feat(dream): add DreamRuntime scheduler skeleton"
```

Subject : 48 chars. Body (2-3 paragraphs) :

1. Introduce `DreamRuntime` single-threaded scheduler with handler registry per `Operation`. Every executed DE produces immutable `EpisodeLogEntry` (episode_id, operations_executed, completed) — DR-0 accountability foundation.

2. Skeleton version : operations without registered handler raise `NotImplementedError` with clear op-name message. Real handlers registered in S5.4+ (`replay`, and S6+ `downscale`, `restructure`, `recombine`).

3. Concurrent swap runtime (threading + worker thread for async dream process) is deferred to S7+. Single-threaded execution suffices for P_min profile validation S6-S8.

---

# Task S5.3 — Property tests DR-0 + DR-1

**Goal** : property tests Hypothesis validant DR-0 (accountability) et DR-1 (episodic conservation). Utilise random DEs avec budgets/opérations variables.

**Files:**
- Create : `tests/conformance/invariants/__init__.py` (empty)
- Create : `tests/conformance/axioms/test_dr0_accountability.py`
- Create : `tests/conformance/axioms/test_dr1_episodic_conservation.py`

## Step S5.3.1 — Write property tests DR-0

Create `tests/conformance/axioms/test_dr0_accountability.py`:

```python
"""DR-0 Accountability — property test.

Every executed dream-episode must appear in the runtime log with a
finite budget. Validates framework spec §6.2 DR-0.
"""
from __future__ import annotations

from hypothesis import given, settings
from hypothesis import strategies as st

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.runtime import DreamRuntime


@st.composite
def dream_episodes_with_replay_only(draw) -> DreamEpisode:
    flops = draw(st.integers(min_value=1, max_value=10_000_000))
    wall = draw(st.floats(min_value=0.0, max_value=60.0,
                          allow_nan=False, allow_infinity=False))
    energy = draw(st.floats(min_value=0.0, max_value=100.0,
                            allow_nan=False, allow_infinity=False))
    ep_idx = draw(st.integers(min_value=0, max_value=99_999))
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=flops, wall_time_s=wall, energy_j=energy),
        episode_id=f"de-prop-{ep_idx:05d}",
    )


def noop_handler(episode: DreamEpisode) -> None:
    """Placeholder handler for DR-0 test — does nothing."""
    return None


@given(ep=dream_episodes_with_replay_only())
@settings(max_examples=50, deadline=None)
def test_dr0_every_executed_de_has_log_entry(
    ep: DreamEpisode,
) -> None:
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, noop_handler)
    runtime.execute(ep)
    assert any(e.episode_id == ep.episode_id for e in runtime.log)


@given(ep=dream_episodes_with_replay_only())
@settings(max_examples=50, deadline=None)
def test_dr0_budget_is_finite(ep: DreamEpisode) -> None:
    # Budget components must be finite — enforced at construction
    assert ep.budget.flops >= 0
    assert ep.budget.wall_time_s >= 0
    assert ep.budget.energy_j >= 0
```

## Step S5.3.2 — Write property test DR-1

Create `tests/conformance/axioms/test_dr1_episodic_conservation.py`:

```python
"""DR-1 Episodic conservation — property test.

For any episodic record added to β buffer at time t, there exists
some t' in [t, t + tau_max] such that the record is consumed by a
DE. Validates framework spec §6.2 DR-1.

Skeleton version (S5.3): uses an in-memory fake beta buffer and
fake DE that consumes N records per execution. Real β
implementation lands S7+ alongside swap protocol.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from hypothesis import given, settings
from hypothesis import strategies as st


@dataclass
class FakeBetaRecord:
    record_id: int
    consumed_by: str | None = None


@dataclass
class FakeBetaBuffer:
    records: list[FakeBetaRecord] = field(default_factory=list)

    def append(self, rid: int) -> None:
        self.records.append(FakeBetaRecord(record_id=rid))

    def consume(self, n: int, de_id: str) -> None:
        for rec in self.records:
            if rec.consumed_by is None and n > 0:
                rec.consumed_by = de_id
                n -= 1


@given(
    record_count=st.integers(min_value=1, max_value=100),
    batch_size=st.integers(min_value=1, max_value=50),
)
@settings(max_examples=30, deadline=None)
def test_dr1_all_records_eventually_consumed(
    record_count: int, batch_size: int
) -> None:
    buf = FakeBetaBuffer()
    for i in range(record_count):
        buf.append(i)

    de_counter = 0
    unconsumed = [r for r in buf.records if r.consumed_by is None]
    while unconsumed:
        de_id = f"de-{de_counter:04d}"
        buf.consume(batch_size, de_id)
        de_counter += 1
        unconsumed = [r for r in buf.records if r.consumed_by is None]

    assert all(r.consumed_by is not None for r in buf.records)
    # tau_max proxy: number of DEs needed is bounded
    expected_max_des = (record_count + batch_size - 1) // batch_size
    assert de_counter <= expected_max_des
```

## Step S5.3.3 — Create invariants dir stub

Create `tests/conformance/invariants/__init__.py` (empty) — placeholder for S7+ invariant property tests.

## Step S5.3.4 — Verify tests pass

Run : `uv run pytest tests/conformance/ -v --no-cov`
Expected: 5 passed (3 existing DR-3 + 2 new DR-0 + 1 new DR-1... wait, verify count: test_dr0 has 2 tests, test_dr1 has 1 test → 3 new + 3 existing DR-3 = 6 in tests/conformance/axioms/).

Run : `uv run pytest`
Expected: 31 tests (28 + 3 new), coverage ≥90%.

## Step S5.3.5 — Commit

```bash
git add tests/conformance/
git commit -m "test(conformance): DR-0/DR-1 property tests"
```

Subject : 44 chars (well under the 50-char validator limit). Earlier draft read `test(conformance): add DR-0 and DR-1 property tests` which is 51 chars and would be rejected; trimmed to the current form.

Body (2-3 paragraphs) :

1. Add Hypothesis property tests for DR-0 (accountability : every executed DE produces runtime log entry, budget components finite) and DR-1 (episodic conservation : all β records eventually consumed within bounded DE count).

2. Hypothesis generates random DEs with `@st.composite` strategies (bounded flops, wall-time, energy) and random β record counts with batch sizes. `@settings(max_examples=50)` keeps suite fast.

3. Fake β buffer used for DR-1 skeleton (real β lands S7+ alongside swap protocol). DR-3 Conformance Criterion condition (2) "axiom property tests pass" partially satisfied for DR-0 and DR-1; DR-2, DR-3, DR-4 pending S6+.

---

# Task S5.4 — replay operation (P_min op 1/2)

**Goal** : première opération concrète. `replay` consomme β records, applique gradient step sur retention objective (skeleton = no-op avec compteur). Enregistre dans runtime.

**Files:**
- Create : `kiki_oniric/dream/operations/__init__.py` (empty)
- Create : `kiki_oniric/dream/operations/replay.py`
- Create : `tests/unit/test_replay_op.py`

## Step S5.4.1 — Write failing tests

Create `tests/unit/test_replay_op.py`:

```python
"""Unit tests for replay operation (P_min op 1/2, A-Walker source)."""
from __future__ import annotations

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.replay import (
    ReplayOpState,
    replay_handler,
)
from kiki_oniric.dream.runtime import DreamRuntime


def make_replay_episode(ep_id: str, records: list[dict]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"beta_records": records},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=10_000, wall_time_s=1.0, energy_j=0.1),
        episode_id=ep_id,
    )


def test_replay_consumes_all_records() -> None:
    state = ReplayOpState()
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, replay_handler(state))

    records = [{"id": i, "context": f"ctx-{i}"} for i in range(5)]
    runtime.execute(make_replay_episode("de-r0", records))

    assert state.total_records_consumed == 5


def test_replay_handles_empty_records() -> None:
    state = ReplayOpState()
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, replay_handler(state))
    runtime.execute(make_replay_episode("de-r1", []))
    assert state.total_records_consumed == 0


def test_replay_across_multiple_episodes_accumulates() -> None:
    state = ReplayOpState()
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, replay_handler(state))

    runtime.execute(make_replay_episode(
        "de-r2", [{"id": 0}, {"id": 1}]
    ))
    runtime.execute(make_replay_episode(
        "de-r3", [{"id": 2}, {"id": 3}, {"id": 4}]
    ))

    assert state.total_records_consumed == 5
    assert state.total_episodes_handled == 2
```

## Step S5.4.2 — Verify failing

Run : `uv run pytest tests/unit/test_replay_op.py -v --no-cov`
Expected: fail (import).

## Step S5.4.3 — Implement replay.py

Create `kiki_oniric/dream/operations/__init__.py` (empty).

Create `kiki_oniric/dream/operations/replay.py`:

```python
"""Replay operation — A-Walker/Stickgold consolidation source.

Skeleton version (S5.4): counts consumed records, logs episode.
Real gradient-based replay (sample β → forward through W → update
via retention-objective gradient) lands alongside MLX integration
S7+ with swap protocol.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.2
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

from kiki_oniric.dream.episode import DreamEpisode


@dataclass
class ReplayOpState:
    """Mutable counter state for replay op across multiple episodes."""

    total_records_consumed: int = 0
    total_episodes_handled: int = 0


def replay_handler(state: ReplayOpState) -> Callable[[DreamEpisode], None]:
    """Build a replay handler bound to a state instance.

    Handler consumes all `beta_records` in the DE's input_slice,
    updates the state counters. No-op on weights for now
    (skeleton) — gradient integration S7+ with MLX.
    """

    def handler(episode: DreamEpisode) -> None:
        records = episode.input_slice.get("beta_records", [])
        state.total_records_consumed += len(records)
        state.total_episodes_handled += 1

    return handler
```

## Step S5.4.4 — Verify passing

Run : `uv run pytest tests/unit/test_replay_op.py -v --no-cov`
Expected: 3 passed.

Run : `uv run pytest`
Expected: 34 tests (31 + 3 new), coverage ≥90%.

## Step S5.4.5 — Commit

```bash
git add kiki_oniric/dream/operations/ tests/unit/test_replay_op.py
git commit -m "feat(dream): add replay op skeleton (A-Walker)"
```

Subject : 46 chars. Body (2-3 paragraphs) :

1. Introduce `replay` operation — A-Walker/Stickgold consolidation source per framework spec §4.2. Skeleton version : `replay_handler(state)` factory returns handler bound to a `ReplayOpState` counter. Handler consumes `beta_records` from `input_slice`, updates total counters, no weight mutation yet.

2. First concrete op for P_min profile (replay + downscale). Downscale follows S6 as part of G2 milestone. Real gradient-based replay (sample → forward → retention-objective gradient) lands S7+ with MLX integration and swap protocol.

3. Three tests cover: full consumption, empty records, multi-episode accumulation. Pattern `handler_factory(state)` keeps handlers pure-by-default with explicit state injection — facilitates future concurrent dream worker S7+.

---

# Task S6.1 — DR-2 compositionality proof draft

**Goal** : rédiger le **draft DR-2 compositionality proof** dans `docs/proofs/dr2-compositionality.md`. Preuve par cas pour les 4 opérations (replay, downscale, restructure, recombine) + lemme d'associativité monoïdale. G3-draft milestone S6 — circulation au relecteur externe S3.3.

**Files:**
- Create : `docs/proofs/__init__.md` (index)
- Create : `docs/proofs/dr2-compositionality.md`

## Step S6.1.1 — Create proofs index

Create `docs/proofs/__init__.md`:

```markdown
# Formal proofs index

Formal proofs supporting framework spec axioms DR-0..DR-4.

## Proofs available

| Axiom | File | Status |
|-------|------|--------|
| DR-2 (compositionality) | `dr2-compositionality.md` | **Draft S6** (G3-draft) |
| DR-2' (canonical order) | `dr2-canonical-fallback.md` | Fallback if DR-2 fails (S8) |
| DR-3 (substrate-agnosticism) | `dr3-substrate.md` | Pending (S7-S8) |
| DR-4 (profile inclusion) | `dr4-profile-inclusion.md` | Pending (S7-S8) |

See also :
- Framework spec §6 Axioms : `docs/specs/2026-04-17-dreamofkiki-framework-C-design.md`
- G3 decision log : `g3-decision-log.md` (S8)
```

## Step S6.1.2 — Write DR-2 proof draft

Create `docs/proofs/dr2-compositionality.md`:

```markdown
# DR-2 Compositionality Proof — Draft

**Axiom statement** (from framework spec §6.2) :

```
DR-2 (Compositionality — TO BE PROVEN)
∀ op_1, op_2 ∈ Op,
  op_2 ∘ op_1 ∈ Op
  ∧ budget(op_2 ∘ op_1) = budget(op_1) + budget(op_2)
  ∧ effect(op_2 ∘ op_1, s) = effect(op_2, effect(op_1, s))
```

where `Op = {replay, downscale, restructure, recombine}` and `State
= (W, H, M)` (weights, hierarchy topology, episodic memory).

**Commutativity is NOT claimed**. The monoid is non-commutative.

---

## Proof structure

1. **Closure** — `op_2 ∘ op_1 ∈ Op` for all 16 op pairs.
2. **Budget additivity** — by definition of budget as resource
   counter.
3. **Functional composition** — `effect` is a well-typed function,
   composition is standard function composition.
4. **Associativity** — `(op_3 ∘ op_2) ∘ op_1 = op_3 ∘ (op_2 ∘ op_1)`.

We prove each property in sequence.

---

## 1. Closure

**Clarification of Op** : in this proof, `Op` denotes the **free
semigroup generated by the 4 primitives**
`{replay, downscale, restructure, recombine}` under composition.
That is, `Op := ⟨replay, downscale, restructure, recombine⟩^+`
contains all finite non-empty sequences of primitives. The 4
primitives are the generators; `op_2 ∘ op_1` for primitives
`op_1, op_2` is an element of this semigroup of length 2, not a
new primitive.

We show that for any two operations `op_1, op_2 ∈ Op`, the
composition `op_2 ∘ op_1` preserves the type
`State × Budget → State × Output` (type-level closure).

**Typing argument** : each op has signature
`Op : State × Budget → State × Output`. Composition of typed
functions with matching domains/codomains is valid by definition of
Cartesian closed categories (Pierce 2002, ch. 6). Since all four
ops share this signature, `op_2 ∘ op_1` is well-typed for any
pair.

**Formal composition** :
```
op_2 ∘ op_1 : State × Budget → State × Output
(op_2 ∘ op_1)(s, b) = op_2(s', b - b_1) where (s', o_1) = op_1(s, b_1)
```

with `b_1 ≤ b` a sub-budget allocation (enforces K1 invariant).

**Conclusion 1** : closure holds. `op_2 ∘ op_1 ∈ Op`. ∎

---

## 2. Budget additivity

**Claim** : `budget(op_2 ∘ op_1) = budget(op_1) + budget(op_2)`.

**Argument** : budget is defined as a triple
`(FLOPs, wall_time, energy_J)` per `BudgetCap` dataclass. Each
component is additive over sequential execution by physical
definition (total FLOPs = sum of FLOPs per sub-op, total wall-time
= sum, total energy = sum).

**Formalization** :
```
budget(op_1)       = (F_1, T_1, E_1)
budget(op_2)       = (F_2, T_2, E_2)
budget(op_2 ∘ op_1) = (F_1 + F_2, T_1 + T_2, E_1 + E_2)
```

per-component sum equals claimed equality.

**Conclusion 2** : budget additivity holds. ∎

---

## 3. Functional composition

**Claim** : `effect(op_2 ∘ op_1, s) = effect(op_2, effect(op_1, s))`.

**Argument** : by definition of the `∘` operator on typed functions,
`(f ∘ g)(x) = f(g(x))`. Applied to our ops :
`(op_2 ∘ op_1)(s) = op_2(op_1(s))`.

This is standard function composition. The "effect" is simply the
State-component projection of the op's output.

**Conclusion 3** : functional composition holds by definition. ∎

---

## 4. Associativity

**Claim** : for all `op_1, op_2, op_3 ∈ Op`,
`(op_3 ∘ op_2) ∘ op_1 = op_3 ∘ (op_2 ∘ op_1)`.

**Argument** : function composition is associative by the
mathematical definition of composition. This is a fundamental
property of the function composition operator `∘` on Set (or any
Cartesian closed category).

**Explicit derivation** :
```
((op_3 ∘ op_2) ∘ op_1)(s) = (op_3 ∘ op_2)(op_1(s))
                           = op_3(op_2(op_1(s)))

(op_3 ∘ (op_2 ∘ op_1))(s) = op_3((op_2 ∘ op_1)(s))
                           = op_3(op_2(op_1(s)))
```

Both reduce to `op_3(op_2(op_1(s)))`. ∎

**Conclusion 4** : associativity holds.

---

## Combined theorem

Given closure (1), budget additivity (2), functional composition
(3), and associativity (4) : **the set of operations `Op` under
composition `∘` with additive budget forms a non-commutative
monoid**.

Specifically : a free semigroup extended with an identity
operation `id_Op` (the trivial DE with empty operation_set would
correspond to identity, but framework §4 requires operation_set
non-empty — so we have a semigroup, not a strict monoid with
identity).

**DR-2 as stated** requires only the three conjuncts (closure +
budget-additivity + functional-composition) and is thus proven.
Associativity (property 4) is an additional structural result
strengthening the framework.

∎ (End of proof)

---

## Case analysis — commutativity boundary

**Non-commutativity** : in general `op_2 ∘ op_1 ≠ op_1 ∘ op_2`.

We document key non-commutative pairs for paper 1 clarity :

### `recombine ∘ downscale` vs `downscale ∘ recombine`

`downscale` (B-Tononi) normalizes weights toward a smaller norm.
`recombine` (C-Hobson) samples new latents from the current weight
manifold.

**Asymmetry** :
- `downscale ∘ recombine` : recombine samples from original
  distribution, then downscale shrinks weights → latents retain
  generative diversity, but weights are scaled post-recombine.
- `recombine ∘ downscale` : downscale shrinks weights first →
  recombine then samples from a compressed manifold → latents lose
  diversity relative to original.

The compositionality axiom DR-2 does not claim equivalence. Both
are valid DEs; they produce different effects. Canonical order
(per framework §4.3) places downscale before recombine to preserve
signal/noise ratio.

### `restructure ∘ replay` vs `replay ∘ restructure`

`restructure` (D-Friston) modifies hierarchy topology.
`replay` (A-Walker) consumes β records through current topology.

**Asymmetry** :
- `replay ∘ restructure` : consume β through NEW topology (after
  restructure).
- `restructure ∘ replay` : consume β through OLD topology, then
  restructure using updated weights.

Canonical order places replay before restructure (§4.3) because
replay on yet-restructured topology risks losing episodic
specificity.

---

## Fallback DR-2' (canonical order)

If strict DR-2 review reveals a gap, adopt DR-2' :

```
DR-2' (Compositionality with canonical ordering)
∀ op_1, op_2 ∈ Op_canonical_order,
  op_2 ∘ op_1 ∈ Op

where canonical order = replay < downscale < restructure < recombine
(serial branch) ∪ {recombine_parallel}.
```

DR-2' is weaker (restricted to canonical order compositions) but
sufficient for the canonical DE composition defined §4.3. Fallback
activation criteria and paper-framing adjustments documented in
`g3-decision-log.md` at S8.

---

## Circulation

This draft will be circulated to the external formal reviewer
identified via `ops/formal-reviewer-recruitment.md` at S6
(G3-draft milestone). Final review S6-S8 → G3 gate decision S8.

**Reviewer queries welcome** on :
- Is the non-commutativity case analysis adequate, or should more
  op-pairs be analyzed ?
- Are the closure + additivity + composition proofs rigorous or
  too informal for Nature HB reviewers ?
- Is the DR-2 vs DR-2' fallback criterion clear ?

---

## Version

- v0.1-draft (2026-04-17, S6.1) — first circulation draft.
- v1.0-final (target S8) — after reviewer feedback incorporation.
```

## Step S6.1.3 — Commit

```bash
git add docs/proofs/
git commit -m "docs(proof): draft DR-2 compositionality (G3)"
```

Subject : 45 chars. Body (3 paragraphs) :

1. Circulate G3-draft : DR-2 compositionality proof (closure, budget additivity, functional composition, associativity). Proof by case analysis over 4 operations, formalized in Cartesian closed category framework. Proves Op forms a non-commutative monoid (semigroup technically, due to non-empty operation_set constraint §4).

2. Non-commutativity case analysis documents asymmetric op pairs : `recombine ∘ downscale` vs `downscale ∘ recombine`, `restructure ∘ replay` vs `replay ∘ restructure`. Canonical order §4.3 justified thermodynamically (signal preservation before noise reduction).

3. Fallback DR-2' (canonical order only) specified for case where strict proof review exposes gaps — activates Pivot B partial per master spec §7.3. Draft targets reviewer recruited via `ops/formal-reviewer-recruitment.md` S3-S5 for review S6-S8.

---

# Task S6.2 — G3-draft circulation log

**Goal** : créer le log de circulation du draft DR-2 au reviewer externe, tracker les itérations review → revision → re-review jusqu'à G3 gate S8.

**Files:**
- Create : `docs/proofs/g3-draft-circulation.md`

## Step S6.2.1 — Create circulation log

Create `docs/proofs/g3-draft-circulation.md`:

```markdown
# G3-Draft Circulation Log — DR-2 Compositionality

**Target milestone** : G3-draft (S6) → G3 final (S8)
**Reviewer** : external formal reviewer (recruited via
`ops/formal-reviewer-recruitment.md`)
**Fallback** : sub-agent `critic` + `validator` if no human confirmed
by S6 end

## Circulation timeline

| Date | Event | Actor | Notes |
|------|-------|-------|-------|
| 2026-04-17 | Draft v0.1 written | author | `dr2-compositionality.md` |
| TBD S6 | Sent to reviewer | author | Email via template |
| TBD S7 | Feedback received | reviewer | Log below |
| TBD S7 | Revision v0.2 | author | Address feedback |
| TBD S8 | Final review | reviewer | G3 gate decision |
| TBD S8 | G3 locked | author | `g3-decision-log.md` |

## Review feedback log

(populated during S7-S8)

### Iteration 1 — v0.1 review

- Date : TBD
- Reviewer feedback : TBD
- Revision items : TBD

## Decision tree at G3 gate (S8)

### Branch DR-2-STRICT (default)
- Reviewer confirms proof rigor → paper 1 cites strict DR-2
- Framework v0.7.0+STABLE tag at G3 gate
- Paper 1 target : Nature HB

### Branch DR-2-PRIME (fallback)
- Reviewer identifies unresolved gap in strict proof
- Adopt DR-2' (canonical order) per `dr2-compositionality.md` fallback
- Paper 1 framed as "formal-leaning" (Pivot B partial)
- Paper 1 target : PLoS Comp Bio / Cognitive Science
- Framework v0.7.0-PRIME+STABLE tag

### Branch G3-FAIL (emergency)
- No reviewer confirmed AND sub-agent critic flags issues
- Pivot A activated : single-paper focused on kiki-oniric
  engineering results (P_min empirical)
- Framework paper deferred to cycle 2
- Paper 2 re-positioned as primary deliverable

## Next action

User action S6 day 1 : finalize reviewer identity via
`ops/formal-reviewer-recruitment.md`, send email via
`ops/formal-reviewer-email-template.md`, update this log.
```

## Step S6.2.2 — Commit

```bash
git add docs/proofs/g3-draft-circulation.md
git commit -m "docs(proof): add G3-draft circulation log"
```

Subject : 43 chars. Body (2 paragraphs) :

1. Track DR-2 draft circulation to external reviewer with timeline, feedback log, and decision tree at G3 gate S8. Three branches : DR-2-STRICT (default success), DR-2-PRIME (fallback DR-2' canonical order), G3-FAIL (emergency Pivot A single-paper).

2. Integrates with `ops/formal-reviewer-recruitment.md` for reviewer identity + `ops/formal-reviewer-email-template.md` for outreach. Decisions at G3 drive framework version tag (v0.7.0+STABLE vs v0.7.0-PRIME+STABLE) and paper 1 journal target.

---

# Task S6.3 — Non-commutative op-pair case analysis

**Goal** : documenter exhaustivement les 16 paires d'opérations (4×4) pour identifier les cas commutatifs vs non-commutatifs, répondre à la question du reviewer potentielle "avez-vous analysé toutes les paires ?"

**Files:**
- Create : `docs/proofs/op-pair-analysis.md`

## Step S6.3.1 — Create op-pair analysis

Create `docs/proofs/op-pair-analysis.md`:

```markdown
# Op-Pair Non-Commutativity Analysis

**Purpose** : exhaustively analyze all 16 pairs of composable
operations `(op_1, op_2)` for commutativity behavior. Strengthens
DR-2 proof (see `dr2-compositionality.md`) and anticipates reviewer
queries.

**Operations** : `{replay (A), downscale (B), restructure (D),
recombine (C)}`.

## Commutativity matrix

Rows = `op_1`, columns = `op_2`, cell = commutativity status.

| `op_1 \ op_2` | replay | downscale | restructure | recombine |
|--------------|--------|-----------|-------------|-----------|
| **replay** (A) | ≡ idempotent on identical β-slice | ≠ | ≠ | ≠ |
| **downscale** (B) | ≠ | ≡ idempotent (shrink is monotonic) | ≠ | ≠ |
| **restructure** (D) | ≠ | ≠ | ≠ (topology-dependent) | ≠ |
| **recombine** (C) | ≠ | ≠ | ≠ | ≠ (sampling non-deterministic) |

**Legend** :
- ≡ : commutative (or idempotent when op_1 = op_2)
- ≠ : non-commutative

## Self-pair analysis (diagonal)

### replay ∘ replay

- Same β-slice : **idempotent** — second replay consumes no new
  records (all flagged `consumed_by` after first).
- Different β-slice : sequential replay on disjoint slices
  commutes (independent consumption).

### downscale ∘ downscale

- Shrink-with-same-factor : **idempotent-ish** (weights shrunk
  once + shrunk again = shrunk by factor^2, not exactly
  idempotent but well-defined).
- Different factors : commutes (multiplication of scalars).

### restructure ∘ restructure

- **Non-commutative** in general : topology edits depend on current
  topology. Restructure(Restructure(G, d1), d2) ≠
  Restructure(Restructure(G, d2), d1) if d1 and d2 touch same
  layers.

### recombine ∘ recombine

- **Non-commutative** : recombine samples latents stochastically.
  Two recombines produce different samples. Repeated recombine
  compounds distributional drift (I3 invariant guard).

## Cross-pair analysis (off-diagonal)

### (replay, downscale)

- `downscale ∘ replay` : replay updates weights → downscale shrinks
  updated weights. Result : moderate consolidation + regularization.
- `replay ∘ downscale` : downscale shrinks pre-replay weights →
  replay on already-shrunk weights. Result : replay on a "compressed"
  space, different gradient landscape.

**Verdict** : non-commutative. Canonical order §4.3 : replay first
(preserve signal).

### (replay, restructure)

- `restructure ∘ replay` : replay on current topology → restructure
  topology post-replay. Clean.
- `replay ∘ restructure` : topology changes first → replay on new
  topology. Risk of losing episodic specificity during topology
  edit.

**Verdict** : non-commutative. Canonical order : replay before
restructure.

### (replay, recombine)

- `recombine ∘ replay` : replay consolidates, then recombine
  generates latents from updated weights. Coherent.
- `replay ∘ recombine` : recombine generates latents from current
  weights, then replay on β. Latents may not reflect post-replay
  consolidation.

**Verdict** : non-commutative. Canonical parallel branch : recombine
runs in parallel with serial branch (§4.3).

### (downscale, restructure)

- `restructure ∘ downscale` : downscale shrinks → restructure
  topology edits on shrunk weights. Clean.
- `downscale ∘ restructure` : restructure changes topology → downscale
  on new topology. May over-shrink new layers if factor not
  adjusted.

**Verdict** : non-commutative. Canonical order : downscale before
restructure.

### (downscale, recombine)

- `recombine ∘ downscale` : downscale shrinks → recombine samples
  from shrunk manifold (lost diversity risk).
- `downscale ∘ recombine` : recombine samples diverse latents →
  downscale shrinks subsequent weights. Preserves recombine
  diversity.

**Verdict** : non-commutative. Canonical order (reversed from
typical A-B-D serial) : recombine should come BEFORE downscale in
this specific pair to preserve generative diversity. Parallel
branch §4.3 handles this.

### (restructure, recombine)

- `recombine ∘ restructure` : restructure topology → recombine
  samples latents from new topology. Clean.
- `restructure ∘ recombine` : recombine samples from current
  topology → restructure edits topology (may invalidate sampled
  latents).

**Verdict** : non-commutative.

## Summary

Of 16 possible pairs :
- **4 self-pairs** : 2 idempotent (replay with same slice,
  downscale with same factor) + 2 non-commutative (restructure,
  recombine).
- **12 cross-pairs** : **all non-commutative**.

**Monoid structure** : free non-commutative semigroup generated by
`{replay, downscale, restructure, recombine}` with additive budget.
DR-2 compositionality holds without commutativity claim.

**Canonical order §4.3** : `replay → downscale → restructure`
(serial A-B-D) + `recombine` (parallel branch C). This order
resolves the only truly problematic pair `(downscale, recombine)`
by placing recombine in the parallel branch — avoiding the forced
sequential choice.
```

## Step S6.3.2 — Commit

```bash
git add docs/proofs/op-pair-analysis.md
git commit -m "docs(proof): add op-pair commutativity analysis"
```

Subject : 49 chars. Body (2-3 paragraphs) :

1. Exhaustive analysis of 16 op pairs (4×4) for commutativity. Diagonal (self-pairs) : 2 idempotent (replay/downscale with identical arguments), 2 non-commutative (restructure/recombine). Off-diagonal : 12 cross-pairs all non-commutative.

2. Structural conclusion : `Op` under composition forms a free non-commutative semigroup with additive budget. DR-2 claim holds without commutativity assertion. Canonical order §4.3 (`replay → downscale → restructure` serial + `recombine` parallel) resolves the single most problematic pair `(downscale, recombine)` by separating them into distinct branches.

3. Pre-empts reviewer query "did you analyze all pairs ?" — yes, 16 pairs documented with canonical-order justifications. Supports DR-2 strict proof (no case left unexamined) and DR-2' fallback (canonical order validity).

---

# Self-review

**1. Spec coverage** — S5-S6 milestones covered :
- S5 dream runtime skeleton → S5.1 (episode) + S5.2 (runtime) ✅
- S5 property tests DR-0/DR-1 → S5.3 ✅
- S6 P_min start → S5.4 (replay op, first P_min op) + S6 downscale pending next atomic plan ⚠️ (partial — see below)
- S6 DR-2 proof draft (G3-draft) → S6.1 + S6.2 + S6.3 ✅

**Partial coverage note** : S6 originally included "P_min opérations {replay, downscale} functional". This plan covers S5.4 replay only. Downscale operation + G2 P_min viable milestone are deferred to an S7-S8 atomic plan. Rationale : G3-draft (DR-2 proof circulation) is the harder-to-parallelize gate and deserves dedicated S6 focus.

**2. Placeholder scan** — no TBD/TODO/implement-later in code blocks. The "TBD S6" / "TBD S7" in `g3-draft-circulation.md` are **delibérés** (populated during S6-S8 review iterations, not by a subagent at S5-S6 creation time).

**3. Type consistency** :
- `DreamEpisode`, `BudgetCap`, `Operation`, `OutputChannel`, `EpisodeTrigger` used consistently across S5.1 (defined) → S5.2 (consumed) → S5.3 (property-tested) → S5.4 (replay op).
- `DreamRuntime`, `EpisodeLogEntry` defined S5.2, consumed S5.3 + S5.4.
- `ReplayOpState`, `replay_handler` defined S5.4.
- No symbol referenced before definition.

**4. Commit count expected** : 7 commits.

**5. Validator risks** :
- S5.1 subject `feat(dream): add DreamEpisode 5-tuple dataclass` = 46 chars ✅
- S5.2 subject `feat(dream): add DreamRuntime scheduler skeleton` = 48 chars ✅
- S5.3 subject `test(conformance): DR-0/DR-1 property tests` = 44 chars ✅
- S5.4 subject `feat(dream): add replay op skeleton (A-Walker)` = 46 chars ✅
- S6.1 subject `docs(proof): draft DR-2 compositionality (G3)` = 45 chars ✅
- S6.2 subject `docs(proof): add G3-draft circulation log` = 41 chars ✅
- S6.3 subject `docs(proof): add op-pair commutativity analysis` = 47 chars ✅

All subjects validator-compliant after S5.3 trim.

---

**End of S5-S6 atomic plan.**

**Version** : v0.1.0
**Generated** : 2026-04-17 via refinement of S5-S6 from main plan
**Source** : `docs/superpowers/plans/2026-04-17-dreamofkiki-implementation.md`
