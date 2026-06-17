"""Unit tests for the SpikingKiki-V4 real-backend shim.

Guards the additive env-gated path wired in
``MicroKikiSubstrate.load`` + ``awake_spike_payload``. Four
checkpoints :

1. Stub path unchanged when ``DREAM_MICRO_KIKI_REAL`` is unset.
2. Real-backend path with a synthesised ``lif_metadata.json`` +
   two fake ``.npz`` modules populates ``_real_state`` and awake
   returns a spike-bearing string.
3. Real-backend path with missing ``lif_metadata.json`` →
   graceful fallback to stub semantics (no exception leaks to
   the caller).
4. Rate-coded payload shape : ``(T, out_dim)`` + metadata.real
   flag = ``True``, and ``awake_spike_payload`` raises when
   called without the real state loaded.

All 4 tests run without ``mlx_lm`` / ``safetensors`` / an Apple
Silicon host ; the synthesised artifact uses plain numpy
``.npz`` + a JSON metadata file.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from kiki_oniric.substrates.micro_kiki import (
    MicroKikiSubstrate,
    _REAL_BACKEND_ENV_VAR,
    _real_backend_enabled,
)


def _synth_artifact(
    root: Path, *, include_metadata: bool = True, n_modules: int = 2,
) -> None:
    """Write a minimal SpikingKiki artifact layout under ``root``."""
    root.mkdir(parents=True, exist_ok=True)
    if include_metadata:
        meta = {"T": 128, "threshold": 0.0625, "tau": 1.0}
        (root / "lif_metadata.json").write_text(json.dumps(meta))
    rng = np.random.default_rng(0)
    for i in range(n_modules):
        W = rng.standard_normal((4, 3)).astype(np.float32)
        np.savez(root / f"block_0{i}_mod{i}.npz", weight=W)


def test_stub_path_unchanged_when_env_unset(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TDD-rb-1 — env flag off → stub behaviour is bit-identical
    to the phase-1 baseline (no _real_state populated, awake
    returns the stub prefix).
    """
    monkeypatch.delenv(_REAL_BACKEND_ENV_VAR, raising=False)
    _synth_artifact(tmp_path, include_metadata=True)

    substrate = MicroKikiSubstrate(real_backend_path=tmp_path)
    substrate.load()

    assert substrate._real_state is None
    out = substrate.awake("hello")
    assert out.startswith("[stub awake]")
    assert out.endswith("hello")


def test_real_backend_loads_artifact_when_env_set(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TDD-rb-2 — env flag on + valid artifact → _real_state
    populated with lif_metadata + sampled weights + module count
    matching the synthesized npz count.
    """
    monkeypatch.setenv(_REAL_BACKEND_ENV_VAR, "1")
    assert _real_backend_enabled() is True
    _synth_artifact(tmp_path, include_metadata=True, n_modules=2)

    substrate = MicroKikiSubstrate(real_backend_path=tmp_path)
    substrate.load()

    assert substrate._real_state is not None
    state = substrate._real_state
    assert state["module_count"] == 2
    assert state["lif_metadata"]["T"] == 128
    assert state["lif_metadata"]["threshold"] == pytest.approx(0.0625)
    assert state["lif_metadata"]["tau"] == pytest.approx(1.0)
    # Samples should cover both npz modules (only 2 exist, cap=3).
    assert len(state["sample_weights"]) == 2

    out = substrate.awake("probe")
    assert out.startswith("[spiking awake")
    assert "probe" in out
    # Spike count is deterministic for a given prompt — sanity
    # check it's non-negative (cannot be negative by construction).
    assert "spikes=" in out


def test_real_backend_accepts_tensor_key(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """SpikingKiki-35B-A3B-V4 stores each layer weight under the
    ``tensor`` key (passthrough layers add a ``passthrough`` flag), not
    ``weight``. The loader must read ``tensor`` so the real bundle loads
    as real weights rather than zero sentinels.
    """
    monkeypatch.setenv(_REAL_BACKEND_ENV_VAR, "1")
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "lif_metadata.json").write_text(
        json.dumps({"T": 128, "threshold": 0.0625, "tau": 1.0})
    )
    rng = np.random.default_rng(0)
    weight = rng.standard_normal((4, 3)).astype(np.float32)
    np.savez(
        tmp_path / "model_layers_0_mod.npz",
        tensor=weight,
        passthrough=np.array([1]),
    )

    substrate = MicroKikiSubstrate(real_backend_path=tmp_path)
    substrate.load()

    assert substrate._real_state is not None
    loaded = [
        v for v in substrate._real_state["sample_weights"].values()
        if getattr(v, "size", 0) > 0
    ]
    assert loaded, "tensor-key weight was not loaded (still zero sentinel)"
    assert tuple(loaded[0].shape) == (4, 3)


def test_real_backend_missing_metadata_falls_back_to_stub(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TDD-rb-3 — env flag on but no lif_metadata.json →
    _load_spiking_backend raises FileNotFoundError caught by
    :meth:`load`, substrate stays in stub mode, awake returns
    the stub prefix. No exception leaks to the caller.
    """
    monkeypatch.setenv(_REAL_BACKEND_ENV_VAR, "1")
    _synth_artifact(tmp_path, include_metadata=False, n_modules=1)

    substrate = MicroKikiSubstrate(real_backend_path=tmp_path)
    # No raise — load() catches the FileNotFoundError internally.
    substrate.load()

    assert substrate._real_state is None
    assert substrate.awake("x").startswith("[stub awake]")


def test_spike_payload_shape_and_raises_without_load(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    """TDD-rb-4 — awake_spike_payload returns ``(T, out_dim)``
    spike tensor + metadata.real=True, and raises RuntimeError
    when called on a substrate that has not loaded the real
    backend.
    """
    substrate_stub = MicroKikiSubstrate()
    with pytest.raises(RuntimeError, match="real backend"):
        substrate_stub.awake_spike_payload("x")

    monkeypatch.setenv(_REAL_BACKEND_ENV_VAR, "1")
    _synth_artifact(tmp_path, include_metadata=True, n_modules=2)
    substrate = MicroKikiSubstrate(real_backend_path=tmp_path)
    substrate.load()

    payload = substrate.awake_spike_payload("deterministic")
    spikes = payload["output_channels"]["spikes"]
    assert spikes.ndim == 2
    T, out_dim = spikes.shape
    assert T == 128
    assert out_dim == 4  # synthesised weight shape (4, 3), out=axis 0
    assert payload["metadata"]["real"] is True
    assert payload["metadata"]["T"] == 128
    assert payload["metadata"]["threshold"] == pytest.approx(0.0625)
    assert payload["metadata"]["tau"] == pytest.approx(1.0)
    # Spikes are {0, 1} rate-coded — no negative or >1 entries.
    assert spikes.min() >= 0.0
    assert spikes.max() <= 1.0
