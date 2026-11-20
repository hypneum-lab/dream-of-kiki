"""Integrity check: fixture .jsonl files match their MANIFEST.md records.

For each `tests/fixtures/*.MANIFEST.md` the test parses two fields:

- **Fixture SHA-256** — hex digest of the committed .jsonl file.
- **Items in fixture** — expected line count (one JSON object per line).

Both are compared against the actual .jsonl file with the same stem.
If a digest or line count mismatches, the fixture has drifted silently.
Do NOT edit the manifest to fix a failure; open a `type:correction` issue
so the provenance discrepancy is on record.

The test is parametrised by glob so new manifests are covered
automatically without modifying this file.

Reference: docs/specs/2026-04-17-dreamofkiki-framework-C-design.md §6.2
(R1 contract — "benchmarks ship with .sha256 digests").
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import pytest

FIXTURES_DIR = Path(__file__).resolve().parents[2] / "tests" / "fixtures"


def _compute_sha256(path: Path) -> str:
    """Return lowercase hex SHA-256 of the file at *path*."""
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()


def _discover_manifests() -> list[Path]:
    return sorted(FIXTURES_DIR.glob("*.MANIFEST.md"))


def _parse_manifest(manifest: Path) -> tuple[str, int]:
    """Return (fixture_sha256, items_in_fixture) from *manifest*.

    Raises ValueError if either field is absent.
    """
    text = manifest.read_text(encoding="utf-8")

    sha_match = re.search(r"\*\*Fixture SHA-256\*\*[:\s]+`?([0-9a-f]{64})`?", text)
    if not sha_match:
        raise ValueError(
            f"No 'Fixture SHA-256' field found in {manifest.name}"
        )

    items_match = re.search(r"\*\*Items in fixture\*\*[:\s]+(\d+)", text)
    if not items_match:
        raise ValueError(
            f"No 'Items in fixture' field found in {manifest.name}"
        )

    return sha_match.group(1).strip(), int(items_match.group(1))


@pytest.mark.parametrize(
    "manifest",
    _discover_manifests(),
    ids=[m.stem for m in _discover_manifests()],
)
def test_fixture_matches_manifest(manifest: Path) -> None:
    """SHA-256 and line count of <stem>.jsonl match the manifest."""
    expected_sha, expected_items = _parse_manifest(manifest)

    stem = manifest.stem  # e.g. "mmlu_real.MANIFEST"
    # Strip the extra .MANIFEST suffix that glob introduces
    jsonl_stem = stem.replace(".MANIFEST", "")
    jsonl_path = FIXTURES_DIR / f"{jsonl_stem}.jsonl"

    assert jsonl_path.exists(), (
        f"Fixture file {jsonl_path.name} not found next to {manifest.name}"
    )

    actual_sha = _compute_sha256(jsonl_path)
    assert actual_sha == expected_sha, (
        f"{jsonl_path.name} SHA-256 mismatch.\n"
        f"  manifest: {expected_sha}\n"
        f"  actual:   {actual_sha}\n"
        "Do NOT edit the manifest. Open a type:correction issue."
    )

    actual_items = sum(1 for line in jsonl_path.read_text(encoding="utf-8").splitlines() if line.strip())
    assert actual_items == expected_items, (
        f"{jsonl_path.name} item count mismatch.\n"
        f"  manifest: {expected_items}\n"
        f"  actual:   {actual_items}\n"
        "Do NOT edit the manifest. Open a type:correction issue."
    )
