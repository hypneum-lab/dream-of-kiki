#!/usr/bin/env python3
"""Item-disjoint tuning / confirmatory partition of the E1 fixtures.

Pre-registration: docs/osf-prereg-e1-replay-controls-2026-10-23.md §5.3.
Since revision 3 these fixtures are the E1 pilot / smoke set only; the
tuning and confirmatory items come from the fact corpus of §5.1.

Each fixture row is assigned to exactly one partition. Rows are
grouped before assignment so near-duplicates cannot straddle the
split:

- MMLU      : group key = ``subject`` (53 groups in 114 rows).
- HellaSwag : group key = ``activity_label`` (62 groups in 200 rows).

Groups are ordered by SHA-256 of the group key. Groups are added to
the confirmatory partition in that order until it holds at least the
per-benchmark quota; all remaining groups form the tuning partition.
Exact duplicate rows (normalised text) are also checked: if any
duplicate pair ever lands in different partitions, the script fails.

Writes ``tests/fixtures/e1_partition_2026-10.json`` with per-row
SHA-256 (of the raw JSONL line, UTF-8, no trailing newline), the
partition of each row, per-partition hashes (SHA-256 of the sorted
row hashes joined by newlines) and the fixture file hashes.

    python scripts/e1_fixture_partition.py            # write manifest
    python scripts/e1_fixture_partition.py --check    # verify manifest
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIX = ROOT / "tests" / "fixtures"
OUT = FIX / "e1_partition_2026-10.json"

# n_replay (20) + n_eval (50) per seed must fit in the confirmatory
# partition; HellaSwag keeps 70 % of its rows for confirmation.
SPECS = {
    "mmlu": {"file": "mmlu_real.jsonl", "group": "subject",
             "text": "question", "quota": 70},
    "hellaswag": {"file": "hellaswag_sanity.jsonl",
                  "group": "activity_label", "text": "ctx", "quota": 140},
}


def _sha(s: str | bytes) -> str:
    b = s.encode("utf-8") if isinstance(s, str) else s
    return hashlib.sha256(b).hexdigest()


def _norm(s: str) -> str:
    return " ".join(s.lower().split())


def build() -> dict:
    out: dict = {"schema": "e1-partition/1", "benchmarks": {}}
    for name, spec in SPECS.items():
        path = FIX / spec["file"]
        lines = [ln for ln in path.read_text(encoding="utf-8").splitlines()
                 if ln.strip()]
        rows = [json.loads(ln) for ln in lines]
        groups: dict[str, list[int]] = {}
        for i, r in enumerate(rows):
            groups.setdefault(str(r[spec["group"]]), []).append(i)
        order = sorted(groups, key=lambda g: _sha(g))
        part = {}
        n_conf = 0
        for g in order:
            label = "confirmatory" if n_conf < spec["quota"] else "tuning"
            for i in groups[g]:
                part[i] = label
            if label == "confirmatory":
                n_conf += len(groups[g])
        # exact-duplicate guard (normalised prompt text)
        seen: dict[str, str] = {}
        for i, r in enumerate(rows):
            key = _norm(str(r[spec["text"]]))
            if key in seen and seen[key] != part[i]:
                raise SystemExit(f"{name}: duplicate straddles split: {key!r}")
            seen.setdefault(key, part[i])
        items = [{"row": i, "sha256": _sha(lines[i]),
                  "group": str(rows[i][spec["group"]]),
                  "partition": part[i]} for i in range(len(rows))]
        summary = {}
        for label in ("confirmatory", "tuning"):
            hs = sorted(it["sha256"] for it in items if it["partition"] == label)
            summary[label] = {
                "n_rows": len(hs),
                "n_groups": len({it["group"] for it in items
                                 if it["partition"] == label}),
                "sha256": _sha("\n".join(hs)),
            }
        out["benchmarks"][name] = {
            "fixture": f"tests/fixtures/{spec['file']}",
            "fixture_sha256": _sha(path.read_bytes()),
            "group_key": spec["group"],
            "confirmatory_quota": spec["quota"],
            "partitions": summary,
            "items": items,
        }
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--check", action="store_true")
    args = ap.parse_args()
    manifest = build()
    text = json.dumps(manifest, indent=1, sort_keys=True) + "\n"
    if args.check:
        if not OUT.exists() or OUT.read_text(encoding="utf-8") != text:
            print("partition manifest missing or stale", file=sys.stderr)
            return 1
        print("partition manifest OK")
        return 0
    OUT.write_text(text, encoding="utf-8")
    for name, b in manifest["benchmarks"].items():
        for label, s in b["partitions"].items():
            print(f"{name:10} {label:13} rows={s['n_rows']:4} "
                  f"groups={s['n_groups']:3} sha256={s['sha256']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
