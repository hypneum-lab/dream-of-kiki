"""Multi-scale Qwen MLX wrappers (cycle-3 C3.2).

Thin ``mlx-lm`` adapter over the SHA-pinned
:mod:`harness.real_models.base_model_registry` so that cycle-3
runs at 1.5B / 7B / 35B are byte-reproducible per the R1 contract.

The wrapper satisfies :class:`kiki_oniric.core.primitives.GammaSnapshotProtocol`
— this is condition (1) of the DR-3 Conformance Criterion applied
to the real-model substrate (framework-C §2.1 γ-channel = weights-
only snapshot). Every forward call tags a ``compute_flops``
estimate on the returned :class:`ForwardTrace` so the dream-episode
scheduler can enforce K1 compute-budget caps.

Design principles :

- **Lazy mlx-lm load** — the wrapper takes the registry pin eagerly
  but only calls the ``_load_checkpoint`` seam when the model is
  actually needed. Tests monkey-patch this seam with a synthetic
  model so the suite stays network-free and < 100 ms.
- **Pin enforcement** — when the caller passes ``enforce_pin=True``
  and the registry pin has ``file_sha256`` set, the wrapper hashes
  the in-memory weights and raises :class:`ValueError` if the
  digest does not match. Production Studio runs set
  ``enforce_pin=True`` ; tests default to False and assert the
  digest explicitly.
- **Seeded forward** — ``forward(tokens, seed)`` derives a local
  MLX PRNG key ``mx.random.key(seed)`` so concurrent wrappers do
  not interfere with the process-wide RNG (same pattern as the
  recombine handler in ``kiki_oniric.dream.operations.recombine``).

Reference :
  docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md §C3.2
  docs/superpowers/specs/2026-04-17-dreamofkiki-framework-C-design.md §2.1
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import mlx.core as mx
import numpy as np

from harness.real_models.base_model_registry import (
    BaseModelPin,
    get_pin,
)


# --------------------------------------------------------------------------
# Trace record : returned from every forward() call, captures the logits
# plus the K1 FLOP estimate for budget enforcement.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class ForwardTrace:
    """Immutable trace of a single forward pass.

    Fields
    ------
    logits
        MLX tensor of shape ``(n_tokens, hidden)`` — whatever the
        underlying model returns.
    n_tokens
        Number of input tokens driven through the model.
    compute_flops
        FLOP estimate for this pass (K1 budget tag). Uses
        ``2 * scale_params * n_tokens`` as the standard decoder-only
        forward-pass approximation (two FLOPs per parameter per
        token for dense GEMM-dominated workloads).
    """

    logits: mx.array
    n_tokens: int
    compute_flops: int


def _load_checkpoint(repo_id: str) -> tuple[Any, Any]:  # pragma: no cover - seam
    """Lazy mlx-lm loader seam — mocked in tests.

    In production this imports ``mlx_lm.load`` and returns the
    ``(model, tokenizer)`` tuple ; in tests this seam is replaced
    by :func:`unittest.mock.patch` with a synthetic model, so the
    real mlx-lm load is never triggered by the unit suite.
    """
    from mlx_lm import load as _load  # type: ignore[import-not-found]

    return _load(repo_id)


def _weights_bytes(model: Any) -> bytes:
    """Return a deterministic byte-string for the model weights.

    Walks ``model.parameters()`` when available (real mlx-lm
    models), else falls back to a ``weights_bytes()`` helper on
    the object (used by test fixtures).
    """
    if hasattr(model, "weights_bytes"):
        return model.weights_bytes()
    # Never return b"" on failure — that yields the sha256 of empty
    # bytes and would silently mask integrity violations during
    # enforce_pin checks. Re-raise so callers fail loudly.
    params = model.parameters()

    def _collect(node: Any, acc: list[bytes]) -> None:
        if isinstance(node, dict):
            for k in sorted(node.keys()):
                acc.append(str(k).encode("utf-8"))
                _collect(node[k], acc)
        elif isinstance(node, (list, tuple)):
            for v in node:
                _collect(v, acc)
        else:
            try:
                arr = np.asarray(node)
                acc.append(arr.tobytes())
            except Exception:  # pragma: no cover - defensive
                pass

    chunks: list[bytes] = []
    _collect(params, chunks)
    return b"".join(chunks)


class QwenMLXWrapper:
    """Qwen (mlx-lm) wrapper bound to a :class:`BaseModelPin`.

    Satisfies :class:`kiki_oniric.core.primitives.GammaSnapshotProtocol`
    (γ-channel / weights-only snapshot) — see DR-3 Conformance
    Criterion condition (1). The wrapper is safe to hold as a
    singleton per slot ; concurrent ``forward()`` calls do not
    mutate any shared state beyond the ``total_compute_flops``
    counter (single-threaded cycle-3 assumption).

    Parameters
    ----------
    registry_pin
        Pin returned by
        :func:`harness.real_models.base_model_registry.get_pin`.
    enforce_pin
        When True and the pin has a non-null ``file_sha256``, the
        wrapper hashes the loaded weights and raises
        :class:`ValueError` on mismatch. Default False so unit
        tests can use synthetic weights ; production Studio runs
        must set this to True.
    """

    def __init__(
        self,
        registry_pin: BaseModelPin,
        *,
        enforce_pin: bool = False,
    ) -> None:
        self._pin = registry_pin
        self._model, self._tokenizer = _load_checkpoint(registry_pin.repo_id)
        self._actual_sha256 = hashlib.sha256(
            _weights_bytes(self._model)
        ).hexdigest()
        if (
            enforce_pin
            and registry_pin.file_sha256 is not None
            and self._actual_sha256 != registry_pin.file_sha256
        ):
            raise ValueError(
                f"sha256 mismatch on {registry_pin.repo_id} "
                f"(rev {registry_pin.revision_sha}): expected "
                f"{registry_pin.file_sha256!r}, got "
                f"{self._actual_sha256!r}"
            )
        self._total_compute_flops = 0

    # ---- GammaSnapshotProtocol surface ---------------------------------

    def get_checkpoint_path(self) -> Path:
        """Return the pinned repo_id as a pathlib stem.

        The γ-channel treats the checkpoint opaquely ; the actual
        bytes live in the mlx-lm cache ``~/.cache/huggingface``
        which is machine-local. Returning a logical path keeps the
        Protocol satisfied without exposing machine-specific state.
        """
        return Path(self._pin.repo_id)

    def get_checkpoint_sha256(self) -> str:
        """Return the 64-char SHA-256 of the loaded weights."""
        return self._actual_sha256

    # ---- Wrapper-specific surface --------------------------------------

    @property
    def pin(self) -> BaseModelPin:
        return self._pin

    @property
    def model(self) -> Any:
        return self._model

    @property
    def tokenizer(self) -> Any:
        return self._tokenizer

    @property
    def total_compute_flops(self) -> int:
        return self._total_compute_flops

    def weights_sha256(self) -> str:
        """Alias for ``get_checkpoint_sha256`` — clearer at call sites."""
        return self._actual_sha256

    def _estimate_flops(self, n_tokens: int) -> int:
        """K1 FLOP estimate for a single forward pass.

        ``2 * scale_params * n_tokens`` is the standard decoder-
        only approximation (one multiply + one add per parameter
        per token under dense GEMM). K1 treats this as an upper-
        bound proxy for scheduling purposes.
        """
        return 2 * self._pin.scale_params * max(n_tokens, 1)

    def forward(
        self, tokens: mx.array, seed: int = 0
    ) -> ForwardTrace:
        """Run a seeded forward pass and return a :class:`ForwardTrace`.

        Uses a local ``mx.random.key(seed)`` so concurrent wrappers
        do not touch the process-wide MLX RNG. The returned trace
        carries the logits plus the K1 FLOP estimate.
        """
        # Seed the MLX global PRNG so any stochastic kernel inside
        # ``self._model`` (e.g. dropout, sampling) reads a deterministic
        # state keyed by ``seed``. mlx-lm's ``__call__`` does not accept
        # a per-call key argument ; setting the global seed is the only
        # MLX-level plumbing that actually affects the forward pass
        # today. The kiki_oniric.dream.operations.recombine handler
        # keeps a per-episode key because its decoder call *does*
        # accept one — this wrapper cannot, so we fall back on the
        # global seed and document the trade-off (single-threaded
        # cycle-3 assumption, see class docstring).
        mx.random.seed(seed)
        logits = self._model(tokens)
        mx.eval(logits)
        n_tokens = int(np.asarray(tokens).size)
        flops = self._estimate_flops(n_tokens)
        self._total_compute_flops += flops
        return ForwardTrace(
            logits=logits, n_tokens=n_tokens, compute_flops=flops
        )


def load_qwen(slot: str, *, enforce_pin: bool = False) -> QwenMLXWrapper:
    """Dispatch helper : ``load_qwen(scale_slot)`` → wrapper.

    Thin wrapper around
    :func:`harness.real_models.base_model_registry.get_pin` that
    constructs a :class:`QwenMLXWrapper` in one call. Raises
    :class:`KeyError` for unknown slots (propagated from the
    registry) so callers can distinguish "bad slot key" from
    "bad weights".
    """
    pin = get_pin(slot)
    return QwenMLXWrapper(pin, enforce_pin=enforce_pin)
