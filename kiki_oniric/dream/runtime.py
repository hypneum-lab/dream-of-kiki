"""DreamRuntime — scheduler and DE execution logger.

Skeleton version (S5.2): logs episodes, invokes op handlers by name,
enforces DR-0 accountability (every DE produces a log entry, even on
handler exception).

Real op handlers are registered in S5.4+ (replay, downscale,
restructure, recombine). Operations without a handler raise
NotImplementedError before execution begins.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from kiki_oniric.dream.channels import ChannelOutput
from kiki_oniric.dream.episode import DreamEpisode, Operation


OperationHandler = Callable[[DreamEpisode], "ChannelOutput | None"]


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

        DR-0 guarantee: every call appends a log entry regardless
        of handler outcome. `completed=False` + `error` populated
        when a handler raises. `operations_executed` reflects only
        the operations actually attempted (a failure mid-sequence
        truncates the recorded list at the failing op).

        Raises NotImplementedError if any operation lacks a handler
        (checked before execution begins, no log entry produced for
        config errors).
        """
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
