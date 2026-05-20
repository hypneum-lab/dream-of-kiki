"""Substrate-agnostic FIFO queue for LatentSample channel-2 outputs (B5).

Implements ``LatentSampleChannel`` Protocol from
``kiki_oniric/core/primitives.py``:  ``enqueue`` pushes a new sample
dict onto the right end of a ``collections.deque``; ``dequeue`` pops
from the left end.

Optional *maxlen*: when set, the deque discards the oldest entry
(left end) automatically as new entries arrive — standard FIFO
overflow semantics.  When *maxlen* is ``None`` the queue is unbounded.

This module has no MLX dependency: it is pure Python + numpy so that
any substrate (CPU, ESNN, future substrate) can consume
``LatentSample`` channel-2 outputs without pulling in the MLX runtime.

Invariant: I3 (latent-dim coherent) — the stored ``latent_vector``
arrays are validated for finiteness by ``LatentSample.__post_init__``
before they ever reach this queue.  ``LatentSampleQueue`` itself does
not re-validate; it trusts the upstream ``LatentSample`` dataclass.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §4.1
"""
from __future__ import annotations

from collections import deque
from typing import Any

import numpy as np
from numpy.typing import NDArray


class LatentSampleQueue:
    """Substrate-agnostic FIFO queue implementing ``LatentSampleChannel``.

    Parameters
    ----------
    maxlen:
        Maximum number of samples held at any time.  When ``None``
        the queue is unbounded (default).  When the capacity is
        reached the oldest item is automatically dropped on the next
        ``enqueue`` call (standard :class:`collections.deque` FIFO
        overflow).

    Usage
    -----
    ::

        queue = LatentSampleQueue(maxlen=64)
        queue.enqueue("default", z_array, "recombine:de=ep0:ep=0:seed=7")
        item = queue.dequeue()   # dict | None
    """

    def __init__(self, maxlen: int | None = None) -> None:
        self._q: deque[dict[str, Any]] = deque(maxlen=maxlen)

    # ------------------------------------------------------------------
    # LatentSampleChannel Protocol surface
    # ------------------------------------------------------------------

    def enqueue(
        self,
        species: str,
        latent_vector: NDArray[np.float32],
        provenance: str,
    ) -> None:
        """Append a latent sample to the right end of the queue.

        Parameters mirror the ``LatentSample`` dataclass fields so
        callers can unpack a ``LatentSample`` directly:

            queue.enqueue(s.species, s.latent_vector, s.provenance)

        The three fields are stored as a plain ``dict`` — no MLX
        dependency, safe to pickle / log.
        """
        self._q.append(
            {
                "species": species,
                "latent_vector": latent_vector,
                "provenance": provenance,
            }
        )

    def dequeue(self) -> dict[str, Any] | None:
        """Pop and return the oldest sample (left end).

        Returns ``None`` when the queue is empty (I3: no latent
        buffer to consume — callers should handle this as a no-op).
        """
        if not self._q:
            return None
        return self._q.popleft()

    # ------------------------------------------------------------------
    # Convenience
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._q)

    def __repr__(self) -> str:
        return (
            f"LatentSampleQueue("
            f"len={len(self._q)}, "
            f"maxlen={self._q.maxlen!r})"
        )


__all__ = ["LatentSampleQueue"]
