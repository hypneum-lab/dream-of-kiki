"""TTT fast-weight substrate (NH-03 candidate 5th substrate).

Minimal test-time-training (TTT) substrate in the spirit of
TTT-Linear (Sun et al. 2024) and the Titans neural memory (Behrouz
et al. 2025) : a *fast-weight* matrix ``W_f`` sits in the forward
path as a residual linear map ``h -> h + h @ W_f.T`` and is written
**only** by inner-loop gradient descent (or by a forgetting gate),
never by the outer / wake optimiser. Every write goes through a
:class:`WriteLedger` so that the consolidation budget is an explicit,
countable quantity (DR-0 accountability, NH-03 "write budget").

Mapping of the four framework-C operations onto fast weights :

- ``replay``      -> one inner gradient step on ``W_f`` under the
                     cross-entropy of a beta-record mini-batch
                     (TTT descent ; supervised inner objective so
                     that its information content matches the
                     replay arm of the G4 harness).
- ``downscale``   -> forgetting gate ``W_f <- f * W_f``, f in (0, 1]
                     (Titans decay ; SHY analogue).
- ``restructure`` -> topology edit on the fast-weight connectivity
                     (``add`` / ``remove`` / ``reroute`` of a row),
                     mirroring the E-SNN handler.
- ``recombine``   -> convex interpolation of two latents with a
                     seeded mixing weight (numpy RNG, deterministic).

Each write that produces new fast weights is checked by the S2
finite guard before it is returned (abort-on-violation) ; S1 is
enforceable through :func:`kiki_oniric.dream.swap.swap_atomic`
on the fast-weight tensor and S3 through the shared topology
guard (see ``tests/conformance/axioms/test_dr3_ttt_substrate.py``).

The module imports ``mlx.core`` at module level, so (like
``mlx_latent_diffusion``) it is intentionally NOT re-exported from
``kiki_oniric.substrates`` : the libmlx-less Linux CI runner must
keep importing that package.

Reference :
    docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
    docs/milestones/2026-10-25-nh03-ttt-vs-episodes.md (pre-reg)
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from numpy.typing import NDArray

from kiki_oniric.dream.guards.finite import FiniteGuardError, check_finite


TTT_SUBSTRATE_NAME = "ttt_fast_weights"
TTT_SUBSTRATE_VERSION = "C-v0.25.0+PARTIAL"

# Operation labels used by the ledger (match framework-C Operation values).
OP_REPLAY = "replay"
OP_DOWNSCALE = "downscale"
OP_RESTRUCTURE = "restructure"
OP_RECOMBINE = "recombine"
WRITE_OPS = (OP_REPLAY, OP_DOWNSCALE, OP_RESTRUCTURE)


class WriteBudgetExceeded(RuntimeError):
    """Raised when a write would exceed the ledger cap (DR-0 budget)."""


class FastWriteAborted(RuntimeError):
    """Raised when a fast-weight write fails the S2 finite guard."""


@dataclass(frozen=True)
class WriteRecord:
    """One metered write (DR-0 trace unit for the NH-03 budget)."""

    episode_id: str
    op: str
    n_elements: int
    n_samples: int


@dataclass
class WriteLedger:
    """Counts consolidation writes ; optional hard cap.

    A *write* is one application of an update to consolidation-
    eligible parameters : one gradient step on one mini-batch, or
    one multiplicative rescale, or one topology edit. ``cap=None``
    means unbounded (metering only).
    """

    cap: int | None = None
    records: list[WriteRecord] = field(default_factory=list)

    def record(
        self, *, episode_id: str, op: str, n_elements: int, n_samples: int
    ) -> None:
        if self.cap is not None and len(self.records) >= self.cap:
            raise WriteBudgetExceeded(
                f"DR-0 write budget exhausted : cap={self.cap}, "
                f"attempted write #{len(self.records) + 1} ({op})"
            )
        self.records.append(
            WriteRecord(
                episode_id=episode_id,
                op=op,
                n_elements=int(n_elements),
                n_samples=int(n_samples),
            )
        )

    @property
    def total(self) -> int:
        return len(self.records)

    def by_op(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for r in self.records:
            out[r.op] = out.get(r.op, 0) + 1
        return out

    @property
    def sample_gradients(self) -> int:
        """Records x gradient steps (secondary meter, not equalised)."""
        return sum(r.n_samples for r in self.records)

    @property
    def elements_written(self) -> int:
        return sum(r.n_elements for r in self.records)

    def episode_ids(self) -> list[str]:
        return [r.episode_id for r in self.records]


@dataclass(frozen=True)
class TTTFastState:
    """Fast-weight state : square matrix ``W_f`` of shape ``(d, d)``."""

    weights: mx.array

    @classmethod
    def zeros(cls, dim: int) -> "TTTFastState":
        if dim <= 0:
            raise ValueError(f"dim must be positive, got {dim}")
        return cls(weights=mx.zeros((dim, dim), dtype=mx.float32))

    @property
    def dim(self) -> int:
        return int(self.weights.shape[0])

    def apply(self, h: mx.array) -> mx.array:
        """Residual fast-weight map ``h + h @ W_f.T``."""
        return apply_fast(self.weights, h)

    def numpy(self) -> NDArray[np.float32]:
        return np.asarray(self.weights)


def apply_fast(w_fast: mx.array, h: mx.array) -> mx.array:
    """Residual fast-weight map ``h + h @ W_f.T`` (pure function)."""
    return h + h @ w_fast.T


def _guard_s2(w: mx.array, *, op: str) -> None:
    mx.eval(w)
    try:
        check_finite(np.asarray(w))
    except FiniteGuardError as exc:
        raise FastWriteAborted(f"S2 guard failed on {op} write: {exc}") from exc


# forward_fn(w_fast, x) -> logits ; the slow parameters it closes over
# are constants for the inner loop (only ``w_fast`` is differentiated).
ForwardFn = Callable[[mx.array, mx.array], mx.array]


@dataclass
class TTTSubstrate:
    """TTT fast-weight substrate exposing the 4 op-handler factories.

    ``ledger`` meters every fast-weight write ; handlers take the
    ``episode_id`` of the (micro-)episode that owns the write so the
    DR-0 trace links each output delta to a DE.
    """

    ledger: WriteLedger = field(default_factory=WriteLedger)

    def replay_handler_factory(
        self,
    ) -> Callable[..., TTTFastState]:
        """TTT descent : one inner gradient step on ``W_f``.

        Handler signature : ``(state, x, y, forward_fn, lr,
        episode_id) -> TTTFastState``. ``x`` is a float32 mini-batch,
        ``y`` integer labels ; loss = mean cross-entropy of
        ``forward_fn(W_f, x)``. Empty batch -> state returned
        unchanged and no write is metered.
        """

        def handler(
            state: TTTFastState,
            x: mx.array,
            y: mx.array,
            forward_fn: ForwardFn,
            lr: float,
            episode_id: str,
        ) -> TTTFastState:
            if lr <= 0.0:
                raise ValueError(f"lr must be positive, got {lr}")
            if int(x.shape[0]) == 0:
                return state

            def inner_loss(w: mx.array) -> mx.array:
                return nn.losses.cross_entropy(
                    forward_fn(w, x), y, reduction="mean"
                )

            grad = mx.grad(inner_loss)(state.weights)
            new_w = state.weights - lr * grad
            _guard_s2(new_w, op=OP_REPLAY)
            self.ledger.record(
                episode_id=episode_id,
                op=OP_REPLAY,
                n_elements=int(new_w.size),
                n_samples=int(x.shape[0]),
            )
            return TTTFastState(weights=new_w)

        return handler

    def downscale_handler_factory(
        self,
    ) -> Callable[..., TTTFastState]:
        """Forgetting gate ``W_f <- factor * W_f`` (factor in (0, 1]).

        Commutative, not idempotent (``f ∘ f = f²``), as on MLX/E-SNN.
        """

        def handler(
            state: TTTFastState, factor: float, episode_id: str
        ) -> TTTFastState:
            if not (0.0 < factor <= 1.0):
                raise ValueError(
                    f"shrink_factor must be in (0, 1], got {factor}"
                )
            new_w = state.weights * factor
            _guard_s2(new_w, op=OP_DOWNSCALE)
            self.ledger.record(
                episode_id=episode_id,
                op=OP_DOWNSCALE,
                n_elements=int(new_w.size),
                n_samples=0,
            )
            return TTTFastState(weights=new_w)

        return handler

    def restructure_handler_factory(
        self,
    ) -> Callable[..., TTTFastState]:
        """Topology edit on the fast-weight connectivity.

        Handler signature : ``(state, op, src, dst, episode_id)``.
        ``add`` sets ``W_f[src, dst] = 1``, ``remove`` zeroes it,
        ``reroute`` swaps rows ``src`` and ``dst``.
        """
        valid_ops = {"add", "remove", "reroute"}

        def handler(
            state: TTTFastState,
            op: str,
            src: int = 0,
            dst: int = 1,
            episode_id: str = "",
        ) -> TTTFastState:
            if op not in valid_ops:
                raise ValueError(
                    f"op must be one of {sorted(valid_ops)}, got {op!r}"
                )
            w = np.array(state.weights, dtype=np.float32)
            if op == "add":
                w[src, dst] = 1.0
                touched = 1
            elif op == "remove":
                w[src, dst] = 0.0
                touched = 1
            else:
                w[[src, dst]] = w[[dst, src]]
                touched = 2 * w.shape[1]
            new_w = mx.array(w)
            _guard_s2(new_w, op=OP_RESTRUCTURE)
            self.ledger.record(
                episode_id=episode_id,
                op=OP_RESTRUCTURE,
                n_elements=touched,
                n_samples=0,
            )
            return TTTFastState(weights=new_w)

        return handler

    def recombine_handler_factory(
        self,
    ) -> Callable[..., NDArray[np.float32]]:
        """Seeded convex mix of two latents (read-only, no write).

        Handler signature : ``(latents (>=2, d), seed) -> (d,)``.
        Deterministic given ``seed`` (numpy RNG, R1 contract). It
        emits a latent sample (canal 2) and does not touch ``W_f``,
        so it is not metered as a write.
        """

        def handler(
            latents: NDArray[np.float32], seed: int = 0
        ) -> NDArray[np.float32]:
            arr = np.asarray(latents, dtype=np.float32)
            if arr.ndim != 2 or arr.shape[0] < 2:
                raise ValueError(
                    f"recombine needs >=2 latents of shape (k, d), got "
                    f"{arr.shape}"
                )
            alpha = float(np.random.default_rng(seed).random())
            mixed = alpha * arr[0] + (1.0 - alpha) * arr[1]
            return mixed.astype(np.float32)

        return handler


def ttt_substrate_components() -> dict[str, str]:
    """Canonical component map, mirroring the MLX/E-SNN core keys."""
    return {
        "primitives": "kiki_oniric.core.primitives",
        "replay": "kiki_oniric.substrates.ttt_fast_weights",
        "downscale": "kiki_oniric.substrates.ttt_fast_weights",
        "restructure": "kiki_oniric.substrates.ttt_fast_weights",
        "recombine": "kiki_oniric.substrates.ttt_fast_weights",
        "finite": "kiki_oniric.dream.guards.finite",
        "topology": "kiki_oniric.dream.guards.topology",
        "runtime": "kiki_oniric.dream.runtime",
        "swap": "kiki_oniric.dream.swap",
        "p_min": "kiki_oniric.profiles.p_min",
        "p_equ": "kiki_oniric.profiles.p_equ",
        "p_max": "kiki_oniric.profiles.p_max",
        "ledger": "kiki_oniric.substrates.ttt_fast_weights",
    }
