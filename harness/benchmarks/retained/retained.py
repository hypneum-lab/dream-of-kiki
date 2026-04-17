"""Retained benchmark loader with SHA-256 integrity check.

Used by invariants S1 (retained non-regression) and I1 (episodic
conservation). The benchmark is frozen : items.jsonl is paired with
items.jsonl.sha256 and loaded via hash-verified read.

Corruption or unauthorized modification of items.jsonl raises
RetainedIntegrityError, blocking the run.
"""
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path


class RetainedIntegrityError(Exception):
    """Raised when items.jsonl SHA-256 does not match frozen hash."""


@dataclass(frozen=True)
class RetainedBenchmark:
    """Frozen retained benchmark with integrity check."""

    items: list[dict]
    hash_verified: bool
    source_hash: str


def _compute_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_retained(bench_dir: Path) -> RetainedBenchmark:
    """Load retained benchmark from `bench_dir`.

    Expects:
    - `{bench_dir}/items.jsonl` : one JSON object per line
    - `{bench_dir}/items.jsonl.sha256` : hex digest (first token on
      line)

    Raises RetainedIntegrityError if hash does not match.
    """
    items_path = bench_dir / "items.jsonl"
    hash_path = bench_dir / "items.jsonl.sha256"

    if not items_path.exists():
        raise FileNotFoundError(f"items.jsonl not found at {items_path}")
    if not hash_path.exists():
        raise FileNotFoundError(
            f"items.jsonl.sha256 not found at {hash_path}"
        )

    actual_hash = _compute_sha256(items_path)
    expected_hash = hash_path.read_text().strip().split()[0]

    if actual_hash != expected_hash:
        raise RetainedIntegrityError(
            f"SHA-256 mismatch at {items_path}: "
            f"expected {expected_hash[:16]}..., "
            f"got {actual_hash[:16]}..."
        )

    items = []
    with items_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            items.append(json.loads(line))

    return RetainedBenchmark(
        items=items,
        hash_verified=True,
        source_hash=actual_hash,
    )
