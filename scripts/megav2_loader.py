"""One-shot script: extract mega-v2 stratified subset, write JSONL.

Usage:
    uv run python scripts/megav2_loader.py [REAL_PATH]

If REAL_PATH is omitted, generates synthetic placeholder.
Output: harness/benchmarks/mega_v2/items.jsonl + .sha256
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from harness.benchmarks.mega_v2.adapter import load_megav2_stratified


def main() -> None:
    real_path = (
        Path(sys.argv[1]) if len(sys.argv) > 1 else None
    )
    bench = load_megav2_stratified(
        real_path=real_path,
        items_per_domain=20,
        synthetic_seed=42,
    )

    out_dir = REPO_ROOT / "harness" / "benchmarks" / "mega_v2"
    out_dir.mkdir(parents=True, exist_ok=True)
    items_path = out_dir / "items.jsonl"
    hash_path = out_dir / "items.jsonl.sha256"

    with items_path.open("w", encoding="utf-8") as fh:
        for item in bench.items:
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")

    hash_path.write_text(bench.source_hash + "\n")

    print(f"Wrote {len(bench.items)} items to {items_path}")
    print(f"Source hash: {bench.source_hash}")


if __name__ == "__main__":
    main()
