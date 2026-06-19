# harness — evaluation + reproducibility layer

Substrate-agnostic harness : loads benchmarks, dispatches runs per
the stratified eval matrix, registers runs with a deterministic
`run_id`. This is the single source of truth for empirical-axis
(EC) bumps.

## Layout

- `config/eval_matrix.py` — loader for
  `docs/interfaces/eval-matrix.yaml` (bump rules, gate thresholds,
  metrics). Strict schema ; fails loudly on missing keys.
- `storage/run_registry.py` — SQLite-backed registry. `run_id` =
  SHA-256(`c_version|profile|seed|commit_sha`)[:32] — **contract
  R1**, bit-stable. Changing the slice width requires a migration.
- `benchmarks/retained/` — retained-knowledge benchmark (currently
  synthetic placeholder). Ships with `items.jsonl.sha256` —
  integrity is checked on load.
- `cli/` — CLI entrypoint (`uv run dream-harness`) declared in
  `pyproject.toml [project.scripts]` but **not yet implemented**;
  the `cli/` directory does not exist on disk.

## Anti-patterns specific to the harness

- **Don't** mutate `run_id` computation silently. If you change the
  tuple or the hash slice, ship a migration script and a CHANGELOG
  entry with a DualVer bump.
- **Don't** run experiments without registering : an unregistered
  run cannot appear in a paper.
- **Don't** edit benchmark `items.jsonl` files without updating the
  sibling `.sha256` and bumping the benchmark version in
  `eval-matrix.yaml`.
- **Don't** treat synthetic benchmarks as empirical evidence — they
  validate the **pipeline**, not the model.
- **Don't** open raw `sqlite3.connect(...)` without
  `contextlib.closing(...)` : connection-leak bug was fixed once
  already and pre-commit catches the pattern.
- **Don't** add a metric to `eval-matrix.yaml` without also adding
  its bump rule and its publication-ready-gate threshold.

## EC bump workflow

Run registered → gate criterion evaluated against
`eval-matrix.yaml` thresholds → if crossed, propose `+STABLE` ↔
`+UNSTABLE` transition in `STATUS.md` + `CHANGELOG.md`.
