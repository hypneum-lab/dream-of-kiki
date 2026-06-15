# MMLU real fixture — provenance manifest

- **Source dataset**: cais/mmlu (config `all`, split `test`)
- **Source URL**: https://huggingface.co/datasets/cais/mmlu/resolve/main/all/test-00000-of-00001.parquet
- **Source file**: `all/test-00000-of-00001.parquet` (3,504,718 bytes, 14042 items)
- **Source SHA-256**: `74a41822ce7d3def56e1682f958469c04642a5336a5ce912fa375fdb90fb25d7`
- **Selection**: deterministic stride sampling — `stride = 14042 // 114 = 123`, indices `i*123` for `i in range(114)`. The test split is sorted by subject; striding evenly across it yields broad subject coverage with no random shuffle and no fabrication.
- **Items in fixture**: 114
- **Unique questions**: 114
- **Subjects covered**: 53 / 57 MMLU subjects
- **Schema** (matches `harness/real_benchmarks/mmlu.py::_load_mmlu_records` / `MMLURecord`):
  - `question` (str), `choices` (list[str] length 4), `answer` (int 0-3), `subject` (str)
- **Fixture SHA-256**: `4d881827081e575e97f9bcd5cb85f2dcc15ea8ef3946c2fb870364addc9a09dd`
- **Answer distribution**: {0:21, 1:30, 2:28, 3:35}
- **Date materialised**: 2026-06-04
- **Note**: Fields copied verbatim from source rows (question/choices/answer/subject); content unaltered. Replaces the degenerate trivial fixture `mmlu_sanity.jsonl` (capitals/arithmetic, non-MMLU subjects) that produced an inflated ~0.91 on the 1.5B pilot.
