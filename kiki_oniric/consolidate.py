"""Public consolidation facade — single top-level entry point.

`consolidate()` is a **facade only** : zero new semantics, pure
composition over already-tested primitives (`DreamRuntime`, the four
op handlers, the three profiles). Downstream consumers — notably
`nerve-wml`'s `bridge/dream_bridge.py` — import this single function
instead of reaching through private sub-module paths.

Composition (per GitHub issue #14) :

1. Decode the event `trace` into one `DreamEpisode` per phase-clock
   window. The `DreamEpisode` 5-tuple is used verbatim.
2. Instantiate `DreamRuntime()` from the `profile` (the profile has
   already registered the handlers its op-set activates — DR-4
   chain inclusion).
3. For each episode : `runtime.execute(episode)` — the already-tested
   S2/S3 guards and DR-0 accountability log fire here.
4. Aggregate a per-transducer delta from the runtime log : each
   executed operation contributes additively to the delta cell
   indexed by the event's `(code, code)` pair. Because the op-set of
   `P_min` is a strict subset of `P_equ`'s (and `P_equ`'s of
   `P_max`'s), and every contribution is non-negative, the resulting
   delta tensors satisfy the DR-4 inclusion chain
   `delta(P_min) subset-of delta(P_equ) subset-of delta(P_max)`.
5. Return the stacked `float32` delta tensor.

Contract : only `shape`, `dtype` and bit-exact determinism (R1) are
guaranteed. The delta *magnitude* is whatever the composed ops
produce and carries no semantic promise.

`alphabet_size` defaults to 64 — the canonical `nerve-wml`
alphabet (`track_p/transducer.py` `alphabet_size: int = 64`,
`Nerve.ALPHABET_SIZE`). `n_transducers` defaults to 12 — the
`GammaThetaMultiplexer` fully-connected-minus-diagonal edge count at
N=5 modalities.

Reference: GitHub issue #14 (hypneum-lab/dream-of-kiki),
docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §3, §4.
"""
from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.profiles.p_equ import PEquProfile
from kiki_oniric.profiles.p_max import PMaxProfile
from kiki_oniric.profiles.p_min import PMinProfile

Profile = PMinProfile | PEquProfile | PMaxProfile

__all__ = ["consolidate", "Profile"]

# Per-op input_slice keys that satisfy the registered skeleton
# handlers without tripping their pre-mutation guards. These are
# inert no-op payloads — the facade adds no new op semantics.
_REPLAY_INPUT: dict[str, object] = {"beta_records": []}
_DOWNSCALE_INPUT: dict[str, object] = {"shrink_factor": 1.0}
_RESTRUCTURE_INPUT: dict[str, object] = {"topo_op": "reroute"}
_RECOMBINE_INPUT: dict[str, object] = {
    "delta_latents": [[0.0], [0.0]],
}

_OP_INPUT_SLICE: dict[Operation, dict[str, object]] = {
    Operation.REPLAY: _REPLAY_INPUT,
    Operation.DOWNSCALE: _DOWNSCALE_INPUT,
    Operation.RESTRUCTURE: _RESTRUCTURE_INPUT,
    Operation.RECOMBINE: _RECOMBINE_INPUT,
}

# Canonical serial-then-parallel op order (operations/CLAUDE.md :
# (replay -> downscale -> restructure) || recombine). The facade
# emits episodes whose operation_set respects this ordering.
_CANONICAL_OP_ORDER: tuple[Operation, ...] = (
    Operation.REPLAY,
    Operation.DOWNSCALE,
    Operation.RESTRUCTURE,
    Operation.RECOMBINE,
)


def _validate_trace(trace: NDArray[np.generic]) -> NDArray[np.int64]:
    """Validate and normalise the input event trace.

    Returns the trace as a contiguous int64 array. Raises
    `ValueError` (citing S2 finite) on malformed input.
    """
    arr = np.asarray(trace)
    if arr.ndim != 2:
        raise ValueError(
            f"trace must be 2-D [n_events, 4], got shape {arr.shape}"
        )
    if arr.shape[1] != 4:
        raise ValueError(
            f"trace must have 4 columns "
            f"(src_wml, dst_wml, code, phase_clock), "
            f"got {arr.shape[1]}"
        )
    if arr.size and not np.isfinite(np.asarray(arr, dtype=np.float64)).all():
        raise ValueError(
            "S2: trace contains non-finite values (NaN/Inf)"
        )
    return np.asarray(arr, dtype=np.int64)


def _profile_op_set(profile: Profile) -> tuple[Operation, ...]:
    """Return the profile's active ops in canonical order.

    The op-set is read from the handlers the profile registered on
    its `DreamRuntime` — no new policy is introduced here.
    """
    registered = set(profile.runtime._handlers.keys())
    return tuple(op for op in _CANONICAL_OP_ORDER if op in registered)


def _decode_episodes(
    trace: NDArray[np.int64], op_set: tuple[Operation, ...]
) -> list[tuple[DreamEpisode, NDArray[np.int64]]]:
    """Decode the trace into one `DreamEpisode` per phase-clock value.

    Returns a list of `(episode, events)` pairs where `events` is the
    sub-slice of the trace belonging to that phase-clock window.
    Windows are emitted in ascending phase-clock order so the
    resulting episode sequence is deterministic (R1).
    """
    if trace.shape[0] == 0:
        return []

    # Merge the inert per-op payloads the registered handlers need.
    input_slice: dict[str, object] = {}
    for op in op_set:
        input_slice.update(_OP_INPUT_SLICE[op])

    budget = BudgetCap(flops=1_000_000, wall_time_s=1.0, energy_j=1.0)
    phase_clocks = sorted({int(v) for v in trace[:, 3]})
    episodes: list[tuple[DreamEpisode, NDArray[np.int64]]] = []
    for phase in phase_clocks:
        events = trace[trace[:, 3] == phase]
        episode = DreamEpisode(
            trigger=EpisodeTrigger.SCHEDULED,
            input_slice=dict(input_slice),
            operation_set=op_set,
            output_channels=(OutputChannel.WEIGHT_DELTA,),
            budget=budget,
            episode_id=f"de-consolidate-{phase:04d}",
        )
        episodes.append((episode, events))
    return episodes


def consolidate(
    trace: NDArray[np.generic],
    profile: Profile,
    *,
    n_transducers: int = 12,
    alphabet_size: int = 64,
) -> NDArray[np.float32]:
    """Run one consolidation pass on a trace of episodic events.

    Parameters
    ----------
    trace
        Event buffer, shape ``[n_events, 4]``, dtype ``int32``,
        columns ``(src_wml, dst_wml, code, phase_clock)``. This is
        ``nerve-wml``'s schema v0. Non-integer dtypes are accepted
        as long as every value is finite (S2) and integral-valued.
    profile
        One of ``PMinProfile`` / ``PEquProfile`` / ``PMaxProfile``
        instances. Determines which ops are active (framework C §3,
        DR-4 chain inclusion). The profile's pre-registered handlers
        are reused verbatim — the facade registers nothing new.
    n_transducers
        Number of per-edge transducers to emit a delta for. Defaults
        to 12 (``nerve-wml`` ``GammaThetaMultiplexer`` at N=5
        modalities, fully-connected minus diagonal).
    alphabet_size
        Per-transducer alphabet size. Defaults to 64 — the canonical
        ``nerve-wml`` alphabet (``Nerve.ALPHABET_SIZE``,
        ``track_p/transducer.py``). The issue prose mentioning "32"
        is stale; the signature value 64 is authoritative and
        matches the downstream repo.

    Returns
    -------
    np.ndarray
        Delta tensor, shape
        ``[n_transducers, alphabet_size, alphabet_size]``, dtype
        ``float32``. Each slice ``delta[k]`` is added to transducer
        ``k``'s logit matrix by the consumer. All-zeros is a valid
        return (no-op consolidation). Only shape, dtype and bit-exact
        determinism (R1) are contractual; the magnitude is not.

    Raises
    ------
    ValueError
        If the trace is malformed (wrong rank/columns, non-finite —
        S2), or ``n_transducers`` / ``alphabet_size`` is not
        positive. Guard failures inside the runtime (S2/S3) surface
        as the underlying exception type unchanged.
    """
    if n_transducers <= 0:
        raise ValueError(
            f"n_transducers must be positive, got {n_transducers}"
        )
    if alphabet_size <= 0:
        raise ValueError(
            f"alphabet_size must be positive, got {alphabet_size}"
        )

    norm_trace = _validate_trace(trace)
    op_set = _profile_op_set(profile)

    delta = np.zeros(
        (n_transducers, alphabet_size, alphabet_size), dtype=np.float32
    )

    episodes = _decode_episodes(norm_trace, op_set)
    for episode, events in episodes:
        # Step 3 : run the profile's runtime — S2/S3 guards and the
        # DR-0 accountability log fire here, exactly as tested.
        profile.runtime.execute(episode)

        # Step 4 : aggregate the per-transducer delta from the
        # operations the runtime actually executed for this episode.
        last_entry = profile.runtime.log[-1]
        n_ops = len(last_entry.operations_executed)
        if n_ops == 0:
            continue

        # Clamp indices into [0, size) so a code/transducer id
        # outside the configured range never indexes out of bounds.
        transducer = np.clip(
            events[:, 0] % n_transducers, 0, n_transducers - 1
        )
        src_code = np.clip(events[:, 2], 0, alphabet_size - 1)
        dst_code = np.clip(events[:, 1] % alphabet_size, 0, alphabet_size - 1)

        # Each executed op contributes one additive unit per event.
        # Op-set inclusion + non-negative contributions => DR-4
        # delta-inclusion chain holds.
        np.add.at(
            delta,
            (transducer, src_code, dst_code),
            np.float32(n_ops),
        )

    return delta
