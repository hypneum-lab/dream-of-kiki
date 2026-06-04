"""Safetensors loaders + env-var gating for the micro_kiki substrate.

Extracted from ``_legacy.py`` by N5 Task 7 (2026-05-10). Final
task of the M2 audit split (Tasks 4-7) — completes the move from
the 1188 LOC monolith to a 5-module package. The substrate class
(host of these helpers) now lives in :mod:`.substrate` (renamed
from ``_legacy.py``).

Two concerns live here :

1. **Env-var gating** (``_REAL_BACKEND_ENV_VAR`` +
   ``_REAL_BACKEND_PATH_ENV_VAR``) — boolean + path probes for the
   SpikingKiki-V4 real backend. Default OFF on CI ; set on Mac
   Studio when the artifact is cloned.
2. **Safetensors loading** (``_try_load_safetensors``) — best-
   effort numpy-only adapter ingestion. Returns ``None`` when the
   wheel is missing or the path does not resolve, keeping the
   module importable on bare-bones CI.
"""
from __future__ import annotations

import logging
import os
from pathlib import Path

from numpy.typing import NDArray


_LOG = logging.getLogger(__name__)


# Env flag gating the real SpikingKiki-35B-A3B-V4 backend. Default
# OFF — on CI (Linux, no SpikingKiki artifact, no mlx_lm) the
# substrate runs in pure-stub mode. On Mac Studio with the
# artifact cloned and ``DREAM_MICRO_KIKI_REAL=1`` exported,
# ``MicroKikiSubstrate.load`` reads ``lif_metadata.json`` + 3
# sample ``.npz`` modules to populate a minimal real-state dict
# that ``awake`` subsequently rate-codes.
_REAL_BACKEND_ENV_VAR = "DREAM_MICRO_KIKI_REAL"

# Companion env var for :data:`_REAL_BACKEND_ENV_VAR`. When set, its
# value is used as the default ``real_backend_path`` for any
# :class:`MicroKikiSubstrate` constructed without an explicit
# ``real_backend_path`` keyword. Lets callers wire the real backend
# purely through environment (shell / launch script) without having
# to edit Python. Explicit constructor arg wins when both are set
# (standard precedence).
_REAL_BACKEND_PATH_ENV_VAR = "DREAM_MICRO_KIKI_REAL_BACKEND_PATH"


def _real_backend_enabled() -> bool:
    """Return True when the real-backend env flag is set truthy.

    Accepts ``1``, ``true``, ``yes``, ``on`` (case-insensitive).
    Separated as a helper so tests can monkeypatch
    ``os.environ`` without touching a frozen constant.
    """
    raw = os.environ.get(_REAL_BACKEND_ENV_VAR, "").strip().lower()
    return raw in {"1", "true", "yes", "on"}


def _real_backend_path_from_env() -> str | None:
    """Return the ``real_backend_path`` value from env, or ``None``.

    Callers use this as a default in ``__post_init__`` so the
    substrate picks up a Studio-local artifact path without
    needing Python-side wiring. Empty / unset env returns
    ``None`` (stub mode).
    """
    raw = os.environ.get(_REAL_BACKEND_PATH_ENV_VAR, "").strip()
    return raw or None


def _try_load_safetensors(
    adapter_path: str | Path,
) -> dict[str, NDArray] | None:
    """Best-effort load of a LoRA ``.safetensors`` via numpy.

    Returns ``None`` when :mod:`safetensors` is missing or the
    path does not resolve. The successful path uses
    ``safetensors.numpy.load_file`` so the returned tensors are
    plain :class:`numpy.ndarray` — the dream runtime is numpy-
    only, so we never need a torch/mlx round-trip here.

    Accepts either a direct file path or a directory containing
    ``adapters.safetensors`` (matches the layout produced by
    ``mlx_lm lora --adapter-path <dir>``).
    """
    try:
        # Lazy import — safetensors is a light wheel but still
        # opt-in so the module stays importable on bare-bones CI.
        from safetensors.numpy import load_file  # type: ignore[import-not-found]
    except ImportError:
        return None

    path = Path(adapter_path)
    if path.is_dir():
        candidate = path / "adapters.safetensors"
        if not candidate.is_file():
            return None
        path = candidate
    if not path.is_file():
        return None
    try:
        return load_file(str(path))
    except Exception as exc:  # noqa: BLE001 — ingestion guard
        _LOG.warning(
            "safetensors load failed for %s (%s) ; returning None",
            path, exc,
        )
        return None
