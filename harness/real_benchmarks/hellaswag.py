"""HellaSwag zero-shot loader (cycle-3 C3.1).

Binds the ``hellaswag`` :class:`DatasetPin` to a local JSONL fixture
whose rows follow the HF ``Rowan/hellaswag`` schema (``ctx`` +
``endings`` + ``label`` + ``activity_label``).

HellaSwag is evaluated **zero-shot** : the evaluation surface is
simply an in-order stream of records, so the loader's iterator
*is* the protocol. A seeded sample helper is provided for subset
tests that need fewer rows while preserving R1 determinism.

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
class HellaSwagRecord:
    """Frozen HellaSwag record — schema matches HF ``Rowan/hellaswag``.

    Fields
    ------
    ctx
        Context string (scene description).
    endings
        Four candidate continuations.
    label
        Integer index in ``endings`` identifying the correct
        ending.
    activity_label
        ActivityNet-style scene label (e.g. ``cooking``).
    """

    ctx: str
    endings: tuple[str, str, str, str]
    label: int
    activity_label: str


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


class HellaSwagLoader:
    """Read-only HellaSwag loader bound to a :class:`DatasetPin`.

    See :class:`harness.real_benchmarks.mmlu.MMLULoader` for the
    SHA-256 + ``MissingLocalDatasetError`` contract ; behaviour is
    identical modulo the schema.
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
                f"HellaSwag fixture not found at {local_path!s} ; "
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
        return self._actual_sha256

    def _iter_raw(self) -> Iterator[dict]:
        with self._path.open("r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                yield json.loads(line)

    def _record_from_raw(self, row: dict) -> HellaSwagRecord:
        endings = row["endings"]
        if len(endings) != 4:
            raise ValueError(
                f"HellaSwag row has {len(endings)} endings, "
                f"expected 4: {row!r}"
            )
        return HellaSwagRecord(
            ctx=str(row["ctx"]),
            endings=(
                str(endings[0]),
                str(endings[1]),
                str(endings[2]),
                str(endings[3]),
            ),
            label=int(row["label"]),
            activity_label=str(row.get("activity_label", "unknown")),
        )

    def iter_records(self) -> Iterator[HellaSwagRecord]:
        """Yield records in fixture order (zero-shot surface)."""
        for row in self._iter_raw():
            yield self._record_from_raw(row)

    def get_seeded_sample(
        self, seed: int, n: int
    ) -> list[HellaSwagRecord]:
        """Return ``n`` records drawn via a seeded permutation.

        Used by unit tests + pilots that want a deterministic
        subset (e.g. sanity pilot C3.7) while preserving R1.
        """
        raws = list(self._iter_raw())
        if n > len(raws):
            raise ValueError(
                f"requested {n} records but fixture has {len(raws)}"
            )
        rng = random.Random(seed)
        return [self._record_from_raw(r) for r in rng.sample(raws, n)]
