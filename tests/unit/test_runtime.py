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
from kiki_oniric.dream.runtime import DreamRuntime


def make_episode(ep_id: str, ops: tuple[Operation, ...]) -> DreamEpisode:
    return DreamEpisode(
        trigger=EpisodeTrigger.SCHEDULED,
        input_slice={"beta_records": []},
        operation_set=ops,
        output_channels=(OutputChannel.WEIGHT_DELTA,),
        budget=BudgetCap(flops=1000, wall_time_s=0.1, energy_j=0.01),
        episode_id=ep_id,
    )


def noop(_episode: DreamEpisode) -> None:
    return None


def test_runtime_executes_single_episode() -> None:
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, noop)
    ep = make_episode("de-0001", (Operation.REPLAY,))
    runtime.execute(ep)
    assert len(runtime.log) == 1
    assert runtime.log[0].episode_id == "de-0001"
    assert runtime.log[0].completed is True
    assert runtime.log[0].error is None


def test_runtime_logs_ordered() -> None:
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, noop)
    runtime.register_handler(Operation.DOWNSCALE, noop)
    runtime.execute(make_episode("de-0001", (Operation.REPLAY,)))
    runtime.execute(make_episode("de-0002", (Operation.DOWNSCALE,)))
    ids = [e.episode_id for e in runtime.log]
    assert ids == ["de-0001", "de-0002"]


def test_runtime_log_entry_immutable() -> None:
    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, noop)
    runtime.execute(make_episode("de-0003", (Operation.REPLAY,)))
    entry = runtime.log[0]
    with pytest.raises((AttributeError, TypeError)):
        # Intentional write to a read-only property to assert it
        # is immutable; mypy correctly flags the assignment.
        entry.episode_id = "de-mutated"  # type: ignore[misc]


def test_runtime_unknown_operation_raises() -> None:
    # Internal registry lookup must fail for unregistered ops in
    # skeleton; recombine not registered yet (added later).
    runtime = DreamRuntime()
    ep = make_episode("de-0004", (Operation.RECOMBINE,))
    with pytest.raises(NotImplementedError, match="recombine"):
        runtime.execute(ep)


def test_runtime_logs_failed_episode_on_handler_exception() -> None:
    """DR-0 guarantee: log entry exists even when handler raises."""
    def failing_handler(_episode: DreamEpisode) -> None:
        raise RuntimeError("handler blew up")

    runtime = DreamRuntime()
    runtime.register_handler(Operation.REPLAY, failing_handler)
    ep = make_episode("de-fail", (Operation.REPLAY,))

    with pytest.raises(RuntimeError, match="blew up"):
        runtime.execute(ep)

    # DR-0: log entry must exist even though handler raised
    assert len(runtime.log) == 1
    assert runtime.log[0].episode_id == "de-fail"
    assert runtime.log[0].completed is False
    assert runtime.log[0].error is not None
    assert "RuntimeError" in runtime.log[0].error
    assert "blew up" in runtime.log[0].error
