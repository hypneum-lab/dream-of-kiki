# G4-nono pre-filing smoke records (2026-10-24)

Verbatim copies of the G4-nono smoke outputs that were kept on
macM1 (`~/scratch/code/`) until 2026-10-28. Archived by PI decision
of 2026-10-28 so that the N selection of the draft
`docs/osf-prereg-g4-nono-cross-domain-2026-10-23.md` (§9, N = 95)
can be checked from the repository. Nothing was rerun or edited:
paths inside the records still name their macM1 location.

Smoke records only, never pooled with confirmatory seeds (draft
§9). Every file is listed with its SHA-256 in `SHA256SUMS`
(`shasum -a 256 -c SHA256SUMS` from this directory); the hashes
were checked against the macM1 originals at copy time.

| Directory | macM1 source | Draft §9 entry | Cell `commit_sha` |
|---|---|---|---|
| `smoke1/` | `~/scratch/code/g4-pilot-out/` | first smoke (24 cells, refused `n-selection.json`) | `28e18c5` |
| `smoke2/` | `~/scratch/code/g4-pilot2-out/` | second smoke (24 cells) | `c64bcee` |
| `smoke2-unmixed/` | `~/scratch/code/g4-pilot2-unmixed/` | exploratory `P_max/unmixed`, 3 cells | `0b2b48e` |
| `smoke3/` | `~/scratch/code/g4-pilot3-out/` | third smoke, new corpus (27 cells), **N = 95** | `6f11823` |

Each smoke directory holds `cells/` (one JSON per seed × arm ×
strategy), `registry.sqlite` (run registry; force-added, `*.sqlite`
is git-ignored) and, except `smoke2-unmixed/`, the runner log
`pilot.log` and the aggregator output `n-selection.json`.

The 2026-10-25 re-check of the third smoke (`--select-n` at
`d81652a`, draft §9) produced
`~/scratch/code/pilots-2026-10-25/g4-nselect-head.json`, which is
byte-identical to `smoke3/n-selection.json` (same SHA-256), so it is
not stored twice.

To recheck N (from the repository root):

    uv run python experiments/g4_nono_test/aggregator.py \
        docs/milestones/g4-nono-smoke-records-2026-10/smoke3/cells/*.json \
        --select-n --out /tmp/n-selection.json

then compare with `smoke3/n-selection.json`.
