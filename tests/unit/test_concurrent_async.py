"""Unit tests for the real concurrent dream worker (C2.17).

Exercises the `threaded=True` mode of `ConcurrentDreamWorker` —
a single background worker thread consumes submissions FIFO
through a bounded `queue.Queue`. The mode is additive : the
pre-existing `sync_drain=True` and `sync_drain=False` code paths
are covered by `tests/unit/test_concurrent_worker.py` and must
remain unchanged.

Invariants exercised :
  * DR-0 : every `runtime.execute(...)` call appends exactly one
    `EpisodeLogEntry`. Preserved here because the worker thread
    is the sole consumer (FIFO → ordering preserved).
  * K-QUEUE : `pending_count <= queue_size`. Bounded queue
    semantics enforce this.

Reference :
  * docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §7
"""
from __future__ import annotations

import threading

import pytest

from kiki_oniric.dream.episode import (
    BudgetCap,
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)
from kiki_oniric.dream.operations.concurrent import (
    ConcurrentDreamWorker,
)
from kiki_oniric.dream.runtime import DreamRuntime, EpisodeLogEntry


def _noop_handler(_episode: DreamEpisode) -> None:
    return None


def _make_episode(ep_id: str) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={},
        operation_set=(Operation.REPLAY,),
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=100, wall_time_s=0.1, energy_j=0.01),
        episode_id=ep_id,
    )


def test_real_concurrent_execution() -> None:
    """Threaded mode : submissions execute on a background thread.

    Submit N=5 episodes, assert all futures resolve to
    `EpisodeLogEntry`s, assert `runtime.log` has N entries
    (DR-0 preserved : exactly 1 per submitted episode), and
    assert the worker's `threading.get_ident()` differs from
    the caller thread's id (proof of off-main-thread execution).
    """
    worker_idents: list[int] = []
    caller_ident = threading.get_ident()

    def recording_handler(_episode: DreamEpisode) -> None:
        worker_idents.append(threading.get_ident())

    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, recording_handler)

    n = 5
    with ConcurrentDreamWorker(
        runtime=runtime, queue_size=16, threaded=True
    ) as worker:
        futures = [
            worker.submit(_make_episode(f"de-cw-async-{i}"))
            for i in range(n)
        ]
        entries = [f.result(timeout=5.0) for f in futures]

    assert len(entries) == n
    assert all(isinstance(e, EpisodeLogEntry) for e in entries)
    assert len(runtime.log) == n  # DR-0
    # All handler invocations happened off the caller thread.
    assert len(worker_idents) == n
    assert all(tid != caller_ident for tid in worker_idents)
    # Single worker : all handler calls share one thread id.
    assert len(set(worker_idents)) == 1


def test_ordering_preserved() -> None:
    """Threaded mode : a single worker thread pulls FIFO, so
    `drain()` returns entries in submission order (DR-0 + FIFO).
    """
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, _noop_handler)

    ids = [f"de-cw-order-{i}" for i in range(7)]
    with ConcurrentDreamWorker(
        runtime=runtime, queue_size=16, threaded=True
    ) as worker:
        for ep_id in ids:
            worker.submit(_make_episode(ep_id))
        entries = worker.drain()

    assert [e.episode_id for e in entries] == ids
    assert [e.episode_id for e in runtime.log] == ids


def test_exception_aggregation() -> None:
    """Threaded mode : failures are surfaced individually via
    `future.exception()` and the first exception is re-raised by
    `drain()` — matching the deferred-mode semantic at
    concurrent.py:137-140. All submitted episodes get a log entry
    (DR-0) even when the handler raises for some.
    """

    def odd_fails(episode: DreamEpisode) -> None:
        # episode_id format: "de-cw-ex-<i>"
        i = int(episode.episode_id.rsplit("-", 1)[-1])
        if i % 2 == 1:
            raise RuntimeError(f"boom-{i}")

    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, odd_fails)

    n = 4
    with ConcurrentDreamWorker(
        runtime=runtime, queue_size=16, threaded=True
    ) as worker:
        futures = [
            worker.submit(_make_episode(f"de-cw-ex-{i}"))
            for i in range(n)
        ]
        # Wait for all futures to resolve before draining, so we
        # can inspect per-future state deterministically.
        for f in futures:
            # Let worker finish each episode (success or failure)
            # before we touch drain.
            try:
                f.result(timeout=5.0)
            except RuntimeError:
                pass

        with pytest.raises(RuntimeError, match=r"boom-1"):
            worker.drain()

    # Per-future exception accessible individually.
    assert futures[0].exception() is None
    assert isinstance(futures[1].exception(), RuntimeError)
    assert futures[2].exception() is None
    assert isinstance(futures[3].exception(), RuntimeError)

    # DR-0 : every submitted episode produced a log entry.
    assert len(runtime.log) == n
    assert [e.episode_id for e in runtime.log] == [
        f"de-cw-ex-{i}" for i in range(n)
    ]


def test_cancellation_safe() -> None:
    """Threaded mode : a future cancelled before the worker picks
    it up is honored (execution skipped, `cancelled()` is True),
    and the worker continues processing subsequent submissions
    correctly (cancellation of one does not corrupt the pipeline).
    """
    gate = threading.Event()
    executed_ids: list[str] = []

    def gated_handler(episode: DreamEpisode) -> None:
        # Block until the main thread releases the gate ; this
        # keeps the worker busy on the first episode and lets us
        # cancel a later episode before the worker picks it up.
        if episode.episode_id == "de-cw-cancel-block":
            gate.wait(timeout=5.0)
        executed_ids.append(episode.episode_id)

    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, gated_handler)

    with ConcurrentDreamWorker(
        runtime=runtime, queue_size=8, threaded=True
    ) as worker:
        # f_block holds the worker thread busy.
        f_block = worker.submit(_make_episode("de-cw-cancel-block"))
        # f_cancel is queued behind f_block ; cancel before pickup.
        f_cancel = worker.submit(_make_episode("de-cw-cancel-skip"))
        cancelled_ok = f_cancel.cancel()
        assert cancelled_ok is True
        assert f_cancel.cancelled() is True

        # Release the worker ; it should drain f_block then
        # notice f_cancel is cancelled and skip its execution,
        # then process the final normal submission.
        gate.set()
        f_normal = worker.submit(_make_episode("de-cw-cancel-ok"))
        entry_block = f_block.result(timeout=5.0)
        entry_normal = f_normal.result(timeout=5.0)

    assert entry_block.episode_id == "de-cw-cancel-block"
    assert entry_normal.episode_id == "de-cw-cancel-ok"
    # The cancelled episode was never handed to the runtime.
    assert "de-cw-cancel-skip" not in executed_ids
    assert "de-cw-cancel-skip" not in [e.episode_id for e in runtime.log]
    # The two non-cancelled episodes executed in order.
    assert executed_ids == ["de-cw-cancel-block", "de-cw-cancel-ok"]
