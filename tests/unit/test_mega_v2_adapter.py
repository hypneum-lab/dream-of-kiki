"""Unit tests for cycle-3 C3.3 mega-v2 → DreamEpisode adapter.

Covers :mod:`harness.real_benchmarks.mega_v2_adapter` with 4
round-trip tests per the cycle-3 plan :

1. record → DreamEpisode → record round-trip is lossless on core
   fields (id, context, expected, domain).
2. α stream (raw tokens) is emitted when the adapter is invoked
   with ``emit_alpha=True``.
3. β stream (curated embeddings) is emitted with a deterministic
   encoder fixture so the ``(record, seed) → embedding`` map is
   stable.
4. DR-0 accountability : every episode has an ``episode_id`` derived
   from the record id (``mv2-{n}`` → ``de-mv2-{n}``).

Reference :
  docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md §C3.3
  docs/superpowers/specs/2026-04-17-dreamofkiki-framework-C-design.md §2.1
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from harness.real_benchmarks.mega_v2_adapter import (
    MegaV2Adapter,
    dream_episode_to_record,
)
from harness.real_benchmarks.mega_v2_eval import (
    MegaV2EvalLoader,
    MegaV2EvalRecord,
)
from kiki_oniric.dream.episode import (
    DreamEpisode,
    EpisodeTrigger,
    Operation,
    OutputChannel,
)


def _write_mega_fixture(dst: Path, n: int = 6) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    with dst.open("w", encoding="utf-8") as fh:
        for i in range(n):
            row = {
                "id": f"mv2-{i:04d}",
                "context": f"context {i}",
                "expected": f"expected {i}",
                "domain": ["math", "syntax", "semantic"][i % 3],
            }
            fh.write(json.dumps(row, sort_keys=True) + "\n")


@pytest.fixture()
def mega_loader(tmp_path: Path) -> MegaV2EvalLoader:
    dst = tmp_path / "mega-v2.jsonl"
    _write_mega_fixture(dst, n=6)
    return MegaV2EvalLoader(local_path=dst)


def _tokenize(text: str) -> list[int]:
    """Minimal byte-hash tokenizer — deterministic, bounded."""
    return [b % 16 for b in text.encode("utf-8")][:8]


def _encoder(tokens: list[int]) -> np.ndarray:
    """Deterministic embedding: [mean, len] tiled to dim 4."""
    arr = np.array(tokens, dtype=np.float32)
    mean = float(arr.mean()) if arr.size else 0.0
    length = float(len(tokens))
    return np.array([mean, length, mean * 2, length - 1], dtype=np.float32)


# --------------------------------------------------------------------------
# Test 1 — record → DreamEpisode → record round-trip
# --------------------------------------------------------------------------


def test_record_to_episode_round_trip(mega_loader: MegaV2EvalLoader) -> None:
    """Adapter produces a DreamEpisode whose input_slice maps back
    to the original record without loss."""
    adapter = MegaV2Adapter(tokenizer=_tokenize, encoder=_encoder)
    records = list(mega_loader.iter_records())
    for r in records:
        de = adapter.to_episode(r)
        assert isinstance(de, DreamEpisode)
        back = dream_episode_to_record(de)
        assert back.id == r.id
        assert back.context == r.context
        assert back.expected == r.expected
        assert back.domain == r.domain


# --------------------------------------------------------------------------
# Test 2 — α stream emission (raw tokens) when requested
# --------------------------------------------------------------------------


def test_alpha_stream_emits_tokens(mega_loader: MegaV2EvalLoader) -> None:
    """When ``emit_alpha=True`` the DE carries an ``alpha_tokens``
    field in its input_slice.

    α-channel is the raw-trace primitive per framework-C §2.1
    (AlphaStreamProtocol).
    """
    adapter = MegaV2Adapter(
        tokenizer=_tokenize, encoder=_encoder, emit_alpha=True
    )
    r = next(iter(mega_loader.iter_records()))
    de = adapter.to_episode(r)
    alpha = de.input_slice.get("alpha_tokens")
    assert alpha is not None
    assert isinstance(alpha, tuple)
    assert len(alpha) > 0
    # Matches the tokenizer output for the record context.
    assert list(alpha) == _tokenize(r.context)


# --------------------------------------------------------------------------
# Test 3 — β stream emission (embeddings) deterministic
# --------------------------------------------------------------------------


def test_beta_stream_emits_deterministic_embeddings(
    mega_loader: MegaV2EvalLoader,
) -> None:
    """β-channel = curated episodic buffer ; the adapter stores an
    encoder-produced embedding under ``beta_records`` so the
    downstream replay op has typed ``x``/``y`` pairs.

    Determinism : two calls on the same record yield identical
    embeddings (R1 at the adapter boundary).
    """
    adapter = MegaV2Adapter(tokenizer=_tokenize, encoder=_encoder)
    r = next(iter(mega_loader.iter_records()))
    de1 = adapter.to_episode(r)
    de2 = adapter.to_episode(r)
    b1 = de1.input_slice["beta_records"]
    b2 = de2.input_slice["beta_records"]
    # Compare element-wise with numpy-aware helpers so the assertion
    # stays correct even if the adapter starts emitting ndarray fields
    # (plain ``==`` on dicts containing ndarrays raises ValueError).
    assert len(b1) == len(b2)
    for r1, r2 in zip(b1, b2, strict=True):
        assert r1.keys() == r2.keys()
        for key in r1:
            np.testing.assert_array_equal(
                np.asarray(r1[key]), np.asarray(r2[key])
            )
    # At least one β record with x + y fields.
    assert len(b1) == 1
    assert "x" in b1[0]
    assert "y" in b1[0]
    assert len(b1[0]["x"]) == 4  # encoder dim


# --------------------------------------------------------------------------
# Test 4 — DR-0 : episode_id derived from record id
# --------------------------------------------------------------------------


def test_episode_id_derives_from_record_id(
    mega_loader: MegaV2EvalLoader,
) -> None:
    """DR-0 accountability : every DE has a deterministic
    ``episode_id`` that the run registry can correlate back to the
    source record. Convention : ``de-{record.id}``.
    """
    adapter = MegaV2Adapter(tokenizer=_tokenize, encoder=_encoder)
    for r in mega_loader.iter_records():
        de = adapter.to_episode(r)
        assert de.episode_id == f"de-{r.id}"
        # Trigger defaults to SCHEDULED (adapter is not saturation-
        # driven) and operation_set contains at least REPLAY.
        assert de.trigger == EpisodeTrigger.SCHEDULED
        assert Operation.REPLAY in de.operation_set
        # Output channel defaults to WEIGHT_DELTA.
        assert OutputChannel.WEIGHT_DELTA in de.output_channels
