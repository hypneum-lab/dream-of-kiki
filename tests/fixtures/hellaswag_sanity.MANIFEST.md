# HellaSwag sanity fixture — provenance manifest

- **Source dataset**: rowanz/hellaswag — official validation split
- **Source URL**: https://raw.githubusercontent.com/rowanz/hellaswag/master/data/hellaswag_val.jsonl
- **Source file**: `hellaswag_val.jsonl` (12,246,618 bytes, 10042 items)
- **Source SHA-256**: `0aa3b88843990f3f10a97b9575c94d7b71fb2205240ba04ae4884d9e9c992588`
- **Selection**: first 200 items in source order (deterministic, no shuffle, no fabrication)
- **Items in fixture**: 200
- **Unique contexts**: 200
- **Schema** (matches `harness/real_benchmarks/hellaswag.py::_load_hellaswag_records`):
  - `ctx` (str), `endings` (list[str] of length 4), `label` (int 0-3), `activity_label` (str)
- **Fixture SHA-256**: `2ab581dc2e50fc5f5fdca9e306a4bf1fae5efaba704725819000bf63e230a979`
- **Label distribution**: {0:53, 1:47, 2:58, 3:42}
- **Date materialised**: 2026-06-04
- **Note**: Fields mapped from source to the 4 loader-required fields only; content unaltered.
