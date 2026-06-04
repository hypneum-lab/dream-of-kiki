"""SpikingKiki backend loader + rate-coded spike payload synthesis.

Extracted from :mod:`.substrate` by N6 Task 4 (2026-05-10) to bring
``substrate.py`` under the audit's <400 LOC target. Continues the
M2 audit split started by N5 Tasks 4-7.

Provides :class:`SpikeLoaderMixin` — a behavioural mixin that
:class:`MicroKikiSubstrate` composes alongside
:class:`MicroKikiHandlersMixin`. Methods stay on the substrate
class (so the public method names ``_load_spiking_backend`` and
``awake_spike_payload`` still exist on instances), but their
bodies live here.

The mixin is tightly coupled to substrate state — it reads
``self.real_backend_path``, ``self._real_state`` (set by the
substrate's ``load`` orchestrator) — so it is not designed to be
mixed into anything other than :class:`MicroKikiSubstrate`.
"""
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray

from kiki_oniric.substrates.micro_kiki.loaders import (
    _REAL_BACKEND_ENV_VAR,
)


_LOG = logging.getLogger(__name__)


class SpikeLoaderMixin:
    """Mixin housing the SpikingKiki ingestion + spike-train synthesis.

    Composed into :class:`MicroKikiSubstrate` so the methods stay
    accessible as ``substrate._load_spiking_backend()`` and
    ``substrate.awake_spike_payload(prompt)`` — required by the
    DR-3 Conformance Criterion test surface and by the SNN
    real-backend awake path.
    """

    # Attributes provided by the substrate at __init__ / load time.
    # Declared here only for type-checker hints — the substrate is
    # the source of truth.
    real_backend_path: Any
    _real_state: dict[str, Any] | None

    def _load_spiking_backend(self) -> dict[str, Any]:
        """Ingest the SpikingKiki-35B-A3B-V4 artifact layout.

        Expected directory (produced by
        ``micro-kiki/scripts/convert_spikingkiki_35b.py``) :
        ``<root>/lif_metadata.json`` + ``<root>/block_NN_MODULE.npz``
        per-module spiking weights. We sample 3 modules (the first
        three by ``sorted`` order) and store their ``weight`` field
        — sufficient for :meth:`awake` to synthesise a rate-coded
        spike train without loading all 31 070 modules (the V4
        artifact is ~70 GB ; full ingestion is MLX-side, not
        runtime-side).

        Raises
        ------
        FileNotFoundError
            ``real_backend_path`` does not exist or
            ``lif_metadata.json`` is missing.
        ValueError
            ``lif_metadata.json`` is not valid JSON.
        """
        root = Path(self.real_backend_path)  # type: ignore[arg-type]
        if not root.exists():
            raise FileNotFoundError(
                f"real_backend_path does not exist: {root}"
            )
        meta_path = root / "lif_metadata.json"
        if not meta_path.is_file():
            raise FileNotFoundError(
                f"SpikingKiki artifact missing lif_metadata.json at "
                f"{meta_path}"
            )
        with meta_path.open() as handle:
            metadata = json.load(handle)

        npz_files = sorted(root.glob("*.npz"))
        sample_weights: dict[str, NDArray] = {}
        for npz in npz_files[:3]:
            try:
                data = np.load(npz, allow_pickle=False)
                if "weight" in data.files:
                    w = np.asarray(data["weight"])
                    # Normalise to 2-D (out_dim, in_dim). A 1-D
                    # weight vector is the degenerate
                    # single-output-neuron case ; awake_spike_payload
                    # requires W @ x where x is a vector, so
                    # shape must be 2-D.
                    if w.ndim == 1:
                        w = w.reshape(1, -1)
                    sample_weights[npz.stem] = w
                else:
                    # Some modules are passthrough (no weight
                    # field, e.g. linear_attn.norm). Record the
                    # module name with a zero-shape sentinel so
                    # awake() can count the real module surface.
                    sample_weights[npz.stem] = np.zeros(0, dtype=np.float32)
            except Exception as exc:  # noqa: BLE001 — per-file guard
                _LOG.warning(
                    "skip malformed npz %s (%s)", npz.name, exc,
                )

        return {
            "lif_metadata": metadata,
            "module_count": len(npz_files),
            "sample_weights": sample_weights,
            "root": str(root),
        }

    def awake_spike_payload(self, prompt: str) -> dict[str, Any]:
        """Rate-coded spike-count payload for the real backend.

        Returns a dict of shape ::

            {
                "output_channels": {"spikes": ndarray[T, out_dim]},
                "metadata": {
                    "T": int, "threshold": float, "tau": float,
                    "real": True, "module": str,
                },
            }

        Raises
        ------
        RuntimeError
            :meth:`load` has not populated ``_real_state`` — call
            it first (or leave ``DREAM_MICRO_KIKI_REAL`` unset to
            stay on :meth:`awake` stub path).
        """
        if self._real_state is None:
            raise RuntimeError(
                "awake_spike_payload requires the SpikingKiki "
                "real backend loaded ; call load() with "
                f"{_REAL_BACKEND_ENV_VAR}=1 and a valid "
                "real_backend_path first"
            )
        meta = self._real_state["lif_metadata"]
        # lif_metadata may be global (single dict) or per-module
        # (dict keyed by module name). Normalise to scalars —
        # uniform-param Phase D metadata uses T=128 / thresh=0.0625
        # / tau=1.0 globally, so .get() with a default is safe.
        if isinstance(meta, dict) and "T" in meta:
            T = int(meta.get("T", 128))
            threshold = float(meta.get("threshold", 0.0625))
            tau = float(meta.get("tau", 1.0))
        else:
            # per-module metadata — read the first entry.
            first = next(iter(meta.values())) if meta else {}
            T = int(first.get("T", 128)) if isinstance(first, dict) else 128
            threshold = (
                float(first.get("threshold", 0.0625))
                if isinstance(first, dict) else 0.0625
            )
            tau = (
                float(first.get("tau", 1.0))
                if isinstance(first, dict) else 1.0
            )

        # Take the first non-empty sample weight as the "module"
        # under test.
        module_name = ""
        W: NDArray | None = None
        for name, tensor in self._real_state["sample_weights"].items():
            if tensor.size > 0:
                module_name = name
                W = tensor
                break
        if W is None:
            # All sampled modules were passthrough — emit a
            # zero-spike payload so the handler surface still
            # completes without error.
            W = np.zeros((1, 1), dtype=np.float32)
            module_name = "<passthrough>"

        # Synthetic input driven by a stable sha256 of the prompt.
        # ``builtins.hash()`` is randomised per process since
        # Python 3.3 (``PYTHONHASHSEED`` defaults to random), so it
        # would only be stable within a single invocation — the
        # conformance harness re-runs the substrate across runs
        # and needs cross-process reproducibility.
        digest = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:8]
        prompt_seed = int(digest, 16)
        rng = np.random.default_rng(prompt_seed)
        x = rng.standard_normal(W.shape[-1]).astype(np.float32)

        # Rate-coded LIF over T steps : membrane potential
        # integrates ``W @ x`` each step, resets on threshold
        # crossing. Decay ``tau`` scales the integrated drive.
        drive = (W @ x).astype(np.float32)
        spikes = np.zeros((T, drive.shape[0]), dtype=np.float32)
        v = np.zeros_like(drive)
        for t in range(T):
            v = v + (drive / max(tau, 1e-6))
            fire = v > threshold
            spikes[t] = fire.astype(np.float32)
            v = np.where(fire, 0.0, v)

        return {
            "output_channels": {"spikes": spikes},
            "metadata": {
                "T": T,
                "threshold": threshold,
                "tau": tau,
                "real": True,
                "module": module_name,
            },
        }
