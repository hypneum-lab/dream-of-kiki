"""MMLU 5-shot loader (cycle-3 C3.1).

Binds :class:`harness.real_benchmarks.dataset_registry.DatasetPin`
for the ``mmlu`` key to a local JSONL fixture that matches the HF
``cais/mmlu`` schema (``question``, ``choices``, ``answer``,
``subject``).

The loader is **network-free** : callers must materialise the
fixture themselves (via the ``datasets`` CLI or an offline cache)
and pass its path. Missing file → :class:`MissingLocalDatasetError`.
When the caller also supplies ``expected_sha256``, the loader
refuses to yield any record if the hash does not match the file on
disk — this is the mechanism the run-registry uses to enforce R1.

Protocol : MMLU is evaluated **5-shot** per Hendrycks et al. 2020 +
Open LLM Leaderboard. :py:meth:`get_5shot_exemplars` draws a
reproducible 5-record sample via :mod:`random.Random(seed)` so the
same ``(pin, seed)`` pair always yields the same in-context
exemplars.

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
from harness.real_benchmarks.dataset_registry import DatasetPin


@dataclass(frozen=True)
class MMLURecord:
    """Frozen MMLU record — schema matches HF ``cais/mmlu``.

    Fields
    ------
    question
        Question stem (single-line string).
    choices
        4-choice multiple-choice set, order as authored upstream.
    answer
        Integer index in ``choices`` identifying the correct
        answer.
    subject
        One of the 57 MMLU subjects (e.g. ``abstract_algebra``).
    """

    question: str
    choices: tuple[str, str, str, str]
    answer: int
    subject: str


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class MMLULoader:
    """Read-only MMLU loader bound to a :class:`DatasetPin`.

    Parameters
    ----------
    registry_pin
        Pin returned by
        :func:`harness.real_benchmarks.dataset_registry.get_dataset_pin`
        for the ``mmlu`` slot.
    local_path
        Filesystem path to a JSONL fixture whose rows match the HF
        MMLU schema (``question`` / ``choices`` / ``answer`` /
        ``subject``).
    expected_sha256
        Optional SHA-256 (lowercase hex) ; when present, loader
        computes the digest of ``local_path`` at construction time
        and raises :class:`ValueError` on mismatch. This is how the
        harness enforces R1 byte-stability on the benchmark slice.
    """

    def __init__(
        self,
        registry_pin: DatasetPin,
        *,
        local_path: Path,
        expected_sha256: str | None = None,
    ) -> None:
        if not local_path.exists():
            raise MissingLocalDatasetError(
                f"MMLU fixture not found at {local_path!s} ; "
                "pass a pre-materialised JSONL export of "
                f"{registry_pin.hf_repo_id} (rev "
                f"{registry_pin.revision_sha}) — network fetch is "
                "disabled by design (R1 reproducibility)."
            )
        self._pin = registry_pin
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
    def pin(self) -> DatasetPin:
        return self._pin

    @property
    def local_path(self) -> Path:
        return self._path

    @property
    def hash_verified(self) -> bool:
        return self._hash_verified

    def local_file_sha256(self) -> str:
        """Return the SHA-256 of the local fixture (64-char hex)."""
        return self._actual_sha256

    def _iter_raw(self) -> Iterator[dict]:
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _record_from_raw(self, row: dict) -> MMLURecord:
        choices = row["choices"]
        if len(choices) != 4:
            raise ValueError(
                f"MMLU row has {len(choices)} choices, expected 4: "
                f"{row!r}"
            )
        return MMLURecord(
            question=str(row["question"]),
            choices=(
                str(choices[0]),
                str(choices[1]),
                str(choices[2]),
                str(choices[3]),
            ),
            answer=int(row["answer"]),
            subject=str(row.get("subject", "unknown")),
        )

    def iter_records(self, seed: int = 0) -> Iterator[MMLURecord]:
        """Yield records in a seeded-shuffled order.

        ``seed`` pins the permutation so the caller can log
        ``(pin, seed) → record_order`` reproducibly. Seed 0 keeps
        the original file order.
        """
        raws = list(self._iter_raw())
        if seed != 0:
            rng = random.Random(seed)
            rng.shuffle(raws)
        for row in raws:
            yield self._record_from_raw(row)

    def get_5shot_exemplars(self, seed: int) -> list[MMLURecord]:
        """Draw 5 in-context exemplars for the MMLU 5-shot protocol.

        Uses :func:`random.Random.sample` seeded with ``seed`` so
        the same seed always returns the same exemplars in the same
        order. Raises :class:`ValueError` if the fixture contains
        fewer than 5 records (an empirically-empty fixture cannot
        satisfy the protocol).
        """
        raws = list(self._iter_raw())
        if len(raws) < 5:
            raise ValueError(
                f"MMLU 5-shot protocol requires ≥5 records ; "
                f"{self._path!s} has {len(raws)}"
            )
        rng = random.Random(seed)
        return [self._record_from_raw(r) for r in rng.sample(raws, 5)]

    def get_seeded_sample(self, seed: int, n: int) -> list[MMLURecord]:
        """Return ``n`` records drawn via a seeded permutation."""
        raws = list(self._iter_raw())
        if n > len(raws):
            raise ValueError(
                f"requested {n} records but fixture has {len(raws)}"
            )
        rng = random.Random(seed)
        return [self._record_from_raw(r) for r in rng.sample(raws, n)]
