# B0 — Channel-output contract + runtime/log threading

**Date** : 2026-05-19
**Status** : design approved, pending spec review
**Tracking issue** : #15 (`hypneum-lab/dream-of-kiki`)
**Scope** : sub-project B0 of the issue-#15 "approach B" decomposition

---

## Context

`kiki_oniric.consolidate()` (facade, shipped `C-v0.13.0+PARTIAL`,
commit `3b9e213`) returns a placeholder delta : it scatters the
count of executed operations (`n_ops`) over event-indexed cells
because the runtime carries no real op output.

Issue #15 asks for the underlying fix. Approach B (chosen by the
maintainer) is to make the op handlers produce real channel
outputs. That is a multi-week substrate effort and was decomposed
into six sub-projects:

| # | Sub-project | Depends on |
|---|-------------|-----------|
| **B0** | Channel-output contract + runtime/log threading | — |
| B1 | `replay` produces a real `WeightUpdate` | B0 |
| B2 | `downscale` produces a real `WeightUpdate` | B0 |
| B3 | `restructure` produces a real `TopologyDiff` (+ I2) | B0 |
| B4 | `recombine` produces a real `LatentSample` | B0 |
| B5 | `consolidate()` substrate→transducer adapter + FC spec | B1-B4 |

This document specifies **B0 only** : the contract and the
plumbing. B0 freezes the interface that B1-B4 will populate.

## Problem

`EpisodeLogEntry` (`kiki_oniric/dream/runtime.py`) records
`episode_id, operations_executed, completed, error` — no op
output. The four op handlers have signature
`Callable[[DreamEpisode], None]` : they return nothing and mutate
their own `state` objects. There is no typed surface on which an
op can publish what it produced, and no place in the log to
capture it.

## Correction to the issue framing

Issue #15's title ("delta payload") is a simplification. Per
`docs/interfaces/primitives.md`, the four operations emit on four
**differently-typed** dream→awake channels:

| Channel | Producing op(s) | Value type | Invariant |
|---------|-----------------|------------|-----------|
| 1 `WeightDelta` | replay, downscale | `WeightUpdate = (LoRAdelta, FisherBump)` | S1 + S2 |
| 2 `LatentSample` | recombine | `(species, latent_vector, provenance)` | I3 (KL ≤ ε) |
| 3 `HierarchyChange` | restructure | `list[tuple[str, dict]]` | S3 |
| 4 `AttentionPrior` | recombine / P_max | `NDArray ∈ [0,1]` | S4 |

B0 therefore defines **four channel-output types**, not one.

## Approaches considered

1. **Explicit handler return** — change the handler signature to
   `Callable[[DreamEpisode], ChannelOutput | None]`; the runtime
   collects the return. Breaking → FC-MAJOR. Explicit, typed,
   mypy-checked. **Chosen.**
2. **State-snapshot capture** — keep `-> None`, handlers write to
   `state.last_output`, runtime reads after. No signature change
   but relies on an untyped convention. Rejected: fragile.
3. **Injected sink** — `Callable[[DreamEpisode, OutputSink], None]`,
   handler calls `sink.emit(channel, value)`. Also breaking; extra
   machinery only justified by multi-channel emission per op,
   which the current op→channel mapping does not need (YAGNI).
   Rejected.

Approach 1 is chosen: the maintainer accepted an FC-MAJOR bump, so
an explicit typed signature change is preferable to a hidden
convention.

## Design

### New module `kiki_oniric/dream/channels.py`

Four frozen dataclasses plus the union type:

```python
WeightUpdate(lora_delta: NDArray[float32],
             fisher_bump: NDArray[float32] | None)   # channel 1
LatentSample(species: str,
             latent_vector: NDArray[float32],
             provenance: str)                        # channel 2
TopologyDiff(diff: tuple[tuple[str, dict], ...])    # channel 3
AttentionPrior(prior: NDArray[float32])              # channel 4

ChannelOutput = WeightUpdate | LatentSample | TopologyDiff | AttentionPrior
```

All dataclasses are `frozen=True`. NDArray fields are stored as
contiguous `float32`. `__post_init__` validates only what B0 can
cheaply guarantee (finiteness — S2); per-channel invariants
(S1/S3/S4/I3) are enforced by B1-B4 when the ops actually produce
values.

### `EpisodeLogEntry` change

Add one field, strictly parallel to `operations_executed`:

```python
channel_outputs: tuple[ChannelOutput | None, ...] = ()
```

- Same length and order as `operations_executed`; index `i` holds
  the output of `operations_executed[i]`, or `None` if that op
  emitted nothing.
- Default `()` keeps existing constructions valid (data-level
  backward compatibility) and marks a "legacy / not captured"
  entry.
- A parallel tuple (not a `dict[Operation, ...]`) is required
  because DR-2 compositionality depends on execution order, and
  tuples keep the frozen dataclass hashable.

### Handler signature change

`Callable[[DreamEpisode], None]` →
`Callable[[DreamEpisode], ChannelOutput | None]`.

All four skeleton handlers and their MLX variants are migrated to
the new signature. **In B0 they all `return None`** — B0 does not
make them produce real values.

### Runtime change

`DreamRuntime.execute()` collects each handler's return value into
a list, and constructs the parallel `channel_outputs` tuple passed
to the `EpisodeLogEntry`. On the error path, `channel_outputs`
holds the outputs gathered before the failing op (DR-0 keeps a log
even on error).

### `core/primitives.py`

The primitive `Protocol` signatures for the four operations are
updated to the new return type — this is the DR-3 conformance
contract surface.

### DualVer

FC-**MAJOR** : the handler signature change is a breaking change
to the primitive surface (`kiki_oniric/CLAUDE.md` rule). Target
version per framework-C spec §12 — proposed `C-v1.0.0+PARTIAL`,
to be confirmed against §12 (a `C-v0.14.0` minor is the fallback
if §12 does not treat a return-type widening as MAJOR). EC axis
unchanged.

## Scope boundary

**B0 does** : the four channel types, the `channel_outputs` log
field, the new handler signature, the runtime collection logic,
migration of all four handlers + MLX variants to the new signature
(returning `None`), the `core/primitives.py` Protocol update,
conformance-test adjustment, and synchronised updates to
framework-C spec §2.1/§4.1, `docs/interfaces/primitives.md`,
`CHANGELOG.md`, and the FR spec under `docs/specs-fr/`.

**B0 does not** : make any op produce a real value (B1-B4), nor
rewire `consolidate()` (B5). **After B0, `consolidate()` behaviour
is unchanged** — it still emits the `n_ops` placeholder. B0 is
contract + plumbing only.

## Conformance / test impact

- `tests/conformance/axioms/test_dr3_substrate.py` — signature /
  Protocol check updated for the new handler return type.
- `tests/conformance/axioms/test_dr0_accountability.py` — add an
  assertion that `len(channel_outputs)` is `0` or equal to
  `len(operations_executed)`.
- New `tests/unit/test_channels.py` — construction, frozenness,
  S2 finiteness validation, `float32` coercion for the four types.
- Existing runtime / episode tests updated for the new
  `EpisodeLogEntry` field (default `()` keeps most green).
- Full suite must stay green; `uv run mypy harness tests` must
  stay clean.

## Acceptance criteria

1. `kiki_oniric/dream/channels.py` defines the four frozen types
   and `ChannelOutput`.
2. `EpisodeLogEntry` carries `channel_outputs`, parallel to
   `operations_executed`, default `()`.
3. All four handlers + MLX variants compile under the new
   signature and return `None`.
4. `DreamRuntime.execute()` populates `channel_outputs` correctly,
   including the error path.
5. `core/primitives.py` Protocols, framework-C spec §2.1/§4.1,
   `primitives.md`, FR spec, and `CHANGELOG.md` are consistent.
6. Full test suite green, mypy clean, FC-MAJOR DualVer bump
   recorded.
