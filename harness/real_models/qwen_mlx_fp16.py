"""Qwen FP16 (bf16) MLX wrapper — cycle-3 C3.8 Phase A.

Sibling of :mod:`harness.real_models.qwen_mlx` specialised for the
unquantized bf16 scale slots (``qwen3p5-{1p5b,7b,35b}-fp16``).
Unlike the Q4 wrapper, this wrapper exposes :meth:`parameters` and
:meth:`update_parameters` so the dream-ops family (``replay_real``,
``downscale_real``, ``restructure_real``, ``recombine_real``) can
mutate real Qwen weights in-place — gradient/backprop flows
natively through the bf16 weights and dream episodes therefore have
a genuine empirical signature on the forward pass logits.

Design principles (delta vs Q4 wrapper) :

- **Same γ-channel surface** — the wrapper still satisfies
  :class:`kiki_oniric.core.primitives.GammaSnapshotProtocol` (DR-3
  condition (1)). ``get_checkpoint_path`` + ``get_checkpoint_sha256``
  behave identically.
- **Pin enforcement** — when ``enforce_pin=True`` and the registry
  pin has a non-null ``file_sha256`` (shard-1 hash for multi-shard
  models), the wrapper hashes the in-memory weights and raises
  :class:`ValueError` on mismatch. Production Studio runs enable
  enforcement ; tests leave it off and assert the digest explicitly.
- **Trainable surface** — :meth:`parameters` returns the underlying
  mlx ``nn.Module`` parameter tree and :meth:`update_parameters`
  applies an mlx parameter-dict update. This pair is the minimal
  contract expected by ``mlx.optimizers.SGD.update(model, grads)``
  downstream. Callers that want the raw module (e.g. to pass to
  :func:`mlx.nn.value_and_grad`) use :attr:`model`.
- **FLOPs accounting** — identical K1 estimate
  (``2 * scale_params * n_tokens``) as the Q4 wrapper so the
  scheduler reads a single compute-budget surface across Q4 and FP16
  pilots.

The wrapper refuses to instantiate against a Q4 pin : the
``quantization`` field must start with ``bf16`` or ``fp16``. This
keeps the real-pilot path from silently falling back to a
non-trainable Q4 scorer and masking the gradient contract.

Reference :
  docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md §C3.8
  docs/superpowers/specs/2026-04-17-dreamofkiki-framework-C-design.md
  §2.1 (γ-channel), §4.2 (replay gradient contract).
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


@dataclass(frozen=True)
class FP16ForwardTrace:
    """Immutable trace of a single FP16 forward pass.

    Mirrors :class:`harness.real_models.qwen_mlx.ForwardTrace` but
    lives in its own type so mypy can refuse to mix Q4 and FP16
    traces in the same downstream pipeline.
    """

    logits: mx.array
    n_tokens: int
    compute_flops: int


def _load_checkpoint_fp16(repo_id: str) -> tuple[Any, Any]:  # pragma: no cover - seam
    """Lazy mlx-lm loader seam — mocked in tests.

    Separate name from the Q4 seam so a single :func:`unittest.mock.patch`
    can target FP16 loading without affecting the Q4 wrapper. In
    production this imports :func:`mlx_lm.load` (same function ;
    mlx-lm auto-detects the quantization from the repo config).
    """
    from mlx_lm import load as _load  # type: ignore[import-not-found]

    return _load(repo_id)


def _weights_bytes(model: Any) -> bytes:
    """Return a deterministic byte-string for the model weights.

    Walks ``model.parameters()`` (real mlx-lm FP16 modules) or
    falls back to a ``weights_bytes()`` shortcut on test fixtures.
    Mirrors the Q4 helper so the SHA-256 digest is stable across
    wrapper types when the underlying parameter tree is identical.
    """
    if hasattr(model, "weights_bytes"):
        return model.weights_bytes()
    # Never return b"" on failure — that would yield the sha256 of
    # empty bytes and silently mask integrity violations during
    # enforce_pin checks. Re-raise so callers fail loudly.
    params = model.parameters()

    def _collect(node: Any, acc: list[bytes], path: str = "") -> None:
        if isinstance(node, dict):
            for k in sorted(node.keys()):
                acc.append(str(k).encode("utf-8"))
                _collect(node[k], acc, f"{path}.{k}" if path else str(k))
        elif isinstance(node, (list, tuple)):
            for i, v in enumerate(node):
                _collect(v, acc, f"{path}[{i}]")
        else:
            # Failing here would silently omit a tensor from the
            # digest and break ``enforce_pin`` guarantees — re-raise
            # with contextual information instead of swallowing.
            try:
                # bf16 MLX arrays cannot be converted to numpy
                # directly (numpy has no bf16 dtype) — widen first.
                if hasattr(node, "astype"):
                    node = node.astype(mx.float32)
                arr = np.asarray(node)
                acc.append(arr.tobytes())
            except Exception as exc:
                raise RuntimeError(
                    f"_collect: failed to serialise tensor at "
                    f"path {path!r} (type {type(node).__name__}) "
                    f"for FP16 weight digest — refusing to omit "
                    f"from sha256 hash. Underlying error: {exc!r}"
                ) from exc

    chunks: list[bytes] = []
    _collect(params, chunks)
    return b"".join(chunks)


class QwenMLXFP16Wrapper:
    """Trainable wrapper around a bf16 mlx-lm Qwen model.

    Parameters
    ----------
    registry_pin
        Pin returned by
        :func:`harness.real_models.base_model_registry.get_pin`.
        Must be a ``bf16-mlx`` or ``fp16-mlx`` pin ; Q4 pins raise
        :class:`ValueError` so the gradient-bearing pilot path is
        never accidentally connected to a non-trainable scorer.
    enforce_pin
        When True and the pin has a non-null ``file_sha256``, the
        wrapper hashes the loaded weights and raises
        :class:`ValueError` on mismatch. Default False so unit tests
        can use synthetic weights ; production Studio runs set this
        to True.
    """

    def __init__(
        self,
        registry_pin: BaseModelPin,
        *,
        enforce_pin: bool = False,
    ) -> None:
        quant = registry_pin.quantization.lower()
        if not (quant.startswith("bf16") or quant.startswith("fp16")):
            raise ValueError(
                f"QwenMLXFP16Wrapper requires a bf16/fp16 pin ; "
                f"pin {registry_pin.name!r} has quantization "
                f"{registry_pin.quantization!r}. Use QwenMLXWrapper "
                "for Q4 slots."
            )
        self._pin = registry_pin
        self._model, self._tokenizer = _load_checkpoint_fp16(
            registry_pin.repo_id
        )
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
        """Return the pinned repo_id as a pathlib stem (γ-channel)."""
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
        """Underlying mlx-lm ``nn.Module`` — pass to ``value_and_grad``."""
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
        return 2 * self._pin.scale_params * max(n_tokens, 1)

    # ---- Trainable surface (replay_real / downscale_real / …) ---------

    def parameters(self) -> Any:
        """Return the underlying mlx ``nn.Module`` parameter tree.

        Downstream callers (replay_real, downscale_real) pass this
        directly to :func:`mlx.nn.value_and_grad` or inspect it via
        :func:`mlx.utils.tree_flatten` to iterate over weight leaves.
        """
        return self._model.parameters()

    def update_parameters(self, params: Any) -> None:
        """Apply a parameter-dict update to the wrapped model.

        Mirrors :meth:`mlx.nn.Module.update` — accepts a nested
        ``dict[str, mx.array]`` tree and overwrites the model's
        weights. Used by ``downscale_real`` to shrink weights by
        a factor and by ``recombine_real`` to stitch latent-derived
        deltas into the model. A shape-preserving op is the
        caller's responsibility (S3 topology guard enforces this
        downstream).
        """
        self._model.update(params)

    def zero_grad_compute_counter(self) -> None:
        """Reset the cumulative FLOP counter (scheduler bookkeeping)."""
        self._total_compute_flops = 0

    # ---- Forward pass -------------------------------------------------

    def forward(
        self, tokens: mx.array, seed: int = 0
    ) -> FP16ForwardTrace:
        """Run a seeded forward pass and return a :class:`FP16ForwardTrace`.

        Seeds the MLX global PRNG via :func:`mx.random.seed` so any
        stochastic kernel inside the model (e.g. dropout) reads a
        deterministic state keyed by ``seed``. Single-threaded
        cycle-3 assumption — see module docstring.
        """
        mx.random.seed(seed)
        logits = self._model(tokens)
        mx.eval(logits)
        n_tokens = int(np.asarray(tokens).size)
        flops = self._estimate_flops(n_tokens)
        self._total_compute_flops += flops
        return FP16ForwardTrace(
            logits=logits, n_tokens=n_tokens, compute_flops=flops
        )


def load_qwen_fp16(
    slot: str, *, enforce_pin: bool = False
) -> QwenMLXFP16Wrapper:
    """Dispatch helper : ``load_qwen_fp16(scale_slot)`` → wrapper.

    Accepts both the ``-fp16`` suffix form (``qwen3p5-1p5b-fp16``)
    and the bare scale name (``qwen3p5-1p5b``) as a convenience —
    in the latter case the ``-fp16`` suffix is appended
    automatically so callers do not have to juggle two name
    conventions. Raises :class:`KeyError` for unknown slots and
    :class:`ValueError` when the resolved pin is not bf16/fp16.
    """
    if not slot.endswith("-fp16"):
        slot = f"{slot}-fp16"
    pin = get_pin(slot)
    return QwenMLXFP16Wrapper(pin, enforce_pin=enforce_pin)


__all__ = [
    "FP16ForwardTrace",
    "QwenMLXFP16Wrapper",
    "load_qwen_fp16",
]
