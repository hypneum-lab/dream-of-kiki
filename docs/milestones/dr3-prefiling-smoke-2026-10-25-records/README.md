# DR-3 pre-filing smoke records (seed 900, 2026-10-25)

Verbatim copies of the small DR-3 filing-smoke records that were kept
on macM1 in `~/scratch/code/pilots-2026-10-25/` until 2026-10-28.
Archived by PI decision of 2026-10-28 for the draft
`docs/osf-prereg-dr3-relational-probe-2026-10-23.md` (§9). Nothing was
rerun or edited: paths inside the records still name their macM1
location. Smoke statistics are not reported and never analysed
(draft §9).

Every file is listed with its SHA-256 in `SHA256SUMS`
(`shasum -a 256 -c SHA256SUMS` from this directory); the hashes were
checked against the macM1 originals at copy time.

| File | macM1 source (under `pilots-2026-10-25/`) | SHA-256 |
|---|---|---|
| `dr3-seed900.npz` | `dr3-smoke900/dr3-seed900.npz` (export dump, 265 KB) | `f6e8135adcf2a7c97bc5baaa6be33b5231b03d7ee2a191c72fbbbdbe22f9ff3a` |
| `dr3-seed900.npz.sha256` | `dr3-smoke900/dr3-seed900.npz.sha256` (sidecar) | `a41d981881655505c9e0388f0e9c074f46653fb599bb24cab1361056a9d58460` |
| `export-manifest.json.sha256` | `dr3-smoke900/export-manifest.json.sha256` (sidecar) | `1fb09a13c19ccf50909c19cab5fd908fd57a69fe51af842fe8bf3090ab4113dd` |
| `dr3-smoke900-probe-check.json` | `dr3-smoke900-probe-check.json` (pipeline check, run ineligible as registered) | `7a8f13fa8156bf4b6e5eda9962cee8d7d045bbc5acf594a64ed0c0ccc9146160` |

The export manifest itself
(`dr3-smoke900/export-manifest.json`, SHA-256
`153eb68bf549e2be251656e180d6afecfca1d076d77fe493b830feac390c8ab0`)
was already committed verbatim as
`docs/milestones/dr3-prefiling-smoke-2026-10-25.json` (same SHA-256)
and is not stored twice.

Not archived (stays on macM1): `pilots-2026-10-25/nwml-v1.9.0/`
(733 MB, of which 638 MB is its `.venv`), the nerve-wml clone the
dump was exported from. It is a clean checkout of
`https://github.com/hypneum-lab/nerve-wml.git` at tag `v1.9.0`,
commit `95ac6b78477c3ce999b31bca775769041e082b46`, with `uv.lock`
SHA-256
`c45e68d893d2730436f0ae6882efc66a515c94f86cf15865c85ce8ed135f0029`;
it can be rebuilt from that tag with `uv sync --frozen`. The other
files in `pilots-2026-10-25/` belong to G4 (see
`docs/milestones/g4-nono-smoke-records-2026-10/`) or to E1 and are
not part of this record.
