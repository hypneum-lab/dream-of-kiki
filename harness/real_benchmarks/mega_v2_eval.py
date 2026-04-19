"""mega-v2 80/20 self-eval loader (cycle-3 C3.1).

mega-v2 is the FineFab training mega-corpus (498K examples across
25 domains, cf. user memory ``mega dataset v2``). Cycle-3 uses it
as the ``retained-benchmark`` replacement : at cycle start we carve
the full corpus into an 80/20 train / eval split (seeded, per R1)
and never touch the eval shard during dream-episodes. This loader
materialises that split deterministically.

Unlike MMLU + HellaSwag, mega-v2 is an internal artefact (no HF
pin) — the ``DatasetPin`` registry therefore has no entry ; the
loader is anchored instead on a local JSONL path + SHA-256.

Reference :
  docs/superpowers/plans/2026-04-19-dreamofkiki-cycle3-atomic.md §C3.1
  docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md §5
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from harness.real_benchmarks import MissingLocalDatasetError


@dataclass(frozen=True)
class MegaV2EvalRecord:
    """Frozen mega-v2 record used by dream-ops + self-eval.

    Fields
    ------
    id
        Stable record identifier (``mv2-{index:04d}`` convention).
    context
        Input prompt / context string.
    expected
        Expected continuation / answer string.
    domain
        One of the 25 mega-v2 domains (cf. ``SYNTHETIC_DOMAINS`` in
        :mod:`harness.benchmarks.mega_v2.adapter` for the cycle-1/2
        placeholder taxonomy).
    """

    id: str
    context: str
    expected: str
    domain: str


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class MegaV2EvalLoader:
    """Load mega-v2 + carve 80/20 splits deterministically.

    Parameters
    ----------
    local_path
        Filesystem path to the mega-v2 JSONL.
    expected_sha256
        Optional SHA-256 check (same semantics as MMLU loader).
    """

    def __init__(
        self,
        *,
        local_path: Path,
        expected_sha256: str | None = None,
    ) -> None:
        if not local_path.exists():
            raise MissingLocalDatasetError(
                f"mega-v2 fixture not found at {local_path!s} ; "
                "pass a pre-materialised JSONL export (internal "
                "artefact — no HF pin)."
            )
        self._path = local_path
        self._actual_sha256 = _hash_file(local_path)
        if (
            expected_sha256 is not None
            and expected_sha256 != self._actual_sha256
        ):
            raise ValueError(
                f"sha256 mismatch on {local_path!s}: expected "
                f"{expected_sha256!r}, got {self._actual_sha256!r}"
            )
        self._hash_verified = expected_sha256 is not None

    @property
    def local_path(self) -> Path:
        return self._path

    @property
    def hash_verified(self) -> bool:
        return self._hash_verified

    def local_file_sha256(self) -> str:
        return self._actual_sha256

    def _iter_raw(self) -> Iterator[dict]:
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _record_from_raw(self, row: dict) -> MegaV2EvalRecord:
        return MegaV2EvalRecord(
            id=str(row["id"]),
            context=str(row["context"]),
            expected=str(row["expected"]),
            domain=str(row.get("domain", "unknown")),
        )

    def iter_records(self) -> Iterator[MegaV2EvalRecord]:
        """Yield records in fixture order."""
        for row in self._iter_raw():
            yield self._record_from_raw(row)

    def train_eval_split(
        self,
        *,
        eval_fraction: float = 0.2,
        seed: int = 42,
    ) -> tuple[list[MegaV2EvalRecord], list[MegaV2EvalRecord]]:
        """Carve an 80/20 (configurable) train/eval split.

        Returns ``(train, eval_)`` disjoint lists.
        ``eval_fraction`` must lie strictly in ``(0, 1)``.
        Deterministic under ``(path_content, seed)`` — same file
        digest and same seed always give byte-identical partitions.
        """
        if not 0.0 < eval_fraction < 1.0:
            raise ValueError(
                f"eval_fraction must be in (0, 1), got "
                f"{eval_fraction}"
            )
        records = [self._record_from_raw(r) for r in self._iter_raw()]
        n_total = len(records)
        if n_total < 2:
            raise ValueError(
                f"mega-v2 split needs ≥2 records, got {n_total}"
            )
        n_eval = int(round(n_total * eval_fraction))
        # Guard against edge-cases where rounding collapses a split.
        n_eval = max(1, min(n_eval, n_total - 1))
        rng = random.Random(seed)
        indices = list(range(n_total))
        rng.shuffle(indices)
        eval_idx = sorted(indices[:n_eval])
        train_idx = sorted(indices[n_eval:])
        train = [records[i] for i in train_idx]
        eval_ = [records[i] for i in eval_idx]
        return train, eval_

    def get_seeded_sample(
        self, seed: int, n: int
    ) -> list[MegaV2EvalRecord]:
        """Return ``n`` records drawn via a seeded permutation."""
        records = [self._record_from_raw(r) for r in self._iter_raw()]
        if n > len(records):
            raise ValueError(
                f"requested {n} records but fixture has "
                f"{len(records)}"
            )
        rng = random.Random(seed)
        return rng.sample(records, n)
