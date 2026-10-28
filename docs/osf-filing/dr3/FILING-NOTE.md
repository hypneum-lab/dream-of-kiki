# OSF filing note — DR-3 global relational probe (H13)

**Status:** prepared 2026-10-28 for the PI, **not submitted**. The PI
files on OSF himself. The confirmatory export (seeds 100..119) and the
probe run do not start before the PI confirms the filing.

## What is filed

| item | value |
|---|---|
| Title | DR-3 global relational probe pre-registration (H13-A..D) |
| Authors | Clément Saillant (L'Electron Rare, Grandris, France), as in `CITATION.cff` |
| Parent OSF project | 10.17605/OSF.IO/Q6JYN (`https://osf.io/q6jyn`) |
| Source draft | `docs/osf-prereg-dr3-relational-probe-2026-10-23.md` |
| Lock commit | `14c3d5f864430649ea860e3b49d10528065af798`, merge of PR #71 into `main` (CI green: push run 40733556724 on `main`, success) |
| Draft SHA-256 | `15dcf7788a9fab495e05c1997f69078590d06708444550280b0a07df9fc67084` |
| Frozen copy | `docs/osf-filing/dr3/osf-prereg-dr3-relational-probe-2026-10-23.frozen.md` (byte-identical to the draft at the lock commit, same SHA-256) |
| PDF rendering | `docs/osf-filing/dr3/osf-prereg-dr3-relational-probe-2026-10-23.frozen.pdf` (SHA-256 in `SHA256SUMS`) |
| Rendering | `render.sh` + `render.lua` (verbatim copy of the E1-small filter) + `strikeout.lua` (struck-through PI decisions keep every character, e.g. "τ"): pandoc 3.12.1, xelatex (TeX Live 2026), layout only. The PDF is not byte-reproducible (embedded dates); the Markdown and its SHA-256 are the reference. |
| Merge PR | `https://github.com/hypneum-lab/dream-of-kiki/pull/71` (split of G4-nono + DR-3 out of PR #41, with the smoke records) |
| Working PR | `https://github.com/hypneum-lab/dream-of-kiki/pull/41` (history; keeps the E1 35B draft) |
| Commit | `https://github.com/hypneum-lab/dream-of-kiki/commit/14c3d5f864430649ea860e3b49d10528065af798` |
| Power script | `scripts/prereg_power_2026_10.py` (§3, section 6) at the lock commit |
| Code pinned by the lock commit | `kiki_oniric/eval/relational_probe.py`, `scripts/dr3_relational_probe.py` (probe, 1-NN comparator, permutation null, `--decide`), `scripts/dr3_export_nerve_wml.py` (export helper); tests `tests/unit/test_relational_probe.py`, `tests/unit/scripts/test_dr3_{relational_probe,export_nerve_wml}.py` |
| Substrates | `nerve-wml` tag `v1.9.0`, commit `95ac6b78477c3ce999b31bca775769041e082b46` (public), run from a clean checkout with `uv sync --frozen` |

At filing, record on OSF the lock commit above and the lock timestamp;
the draft's own "Lock commit / Lock timestamp" fields are left to be
filled at filing, as for every pre-registration in this repository.

## PI decisions (draft header and §10)

- **2026-10-23:** δ = 0.025; n = 20 seeds.
- **2026-10-24:** export helper = standalone script run inside a nerve-wml v1.9.0 checkout, no new dependency in this repository (§5).
- **2026-10-24:** τ = 0.1 for the confirmatory geometry statistics (§5).
- **2026-10-24:** permutation-null secondary c_α at α = 0.05, 95th percentile (rank 476 of 501; §5, departure 1); decision-eligibility rules and the restricted "global" wording rule after an independent review (§7, §8).
- **2026-10-25:** macM1 is the filing host; the confirmatory export (seeds 100..119) runs there, so the seed-900 smoke on macM1 is the §9 filing smoke. PI go for that smoke the same day.
- **2026-10-28:** split out of PR #41 so it is filed without waiting for E1; the smoke records are archived in the repository (draft header "Split" line).

No §10 item remains open.

## Pilots / smokes (draft §9) and archived records

| run | what | records |
|---|---|---|
| Integration check, 2026-10-24 (not the filing smoke) | export helper on seed 900 in a clean v1.9.0 clone, existing environment; gate passed (trained MLP 0.519, LIF 0.545), nulls near chance; array hashes reproducible | described in §9 only |
| **Filing smoke, 2026-10-25** (seed 900, not analysed) | helper at `d81652a`, fresh v1.9.0 clone + `uv sync --frozen`; gate passed; array hashes equal to the integration check; probe `--require-hash --manifest … --decide` verified the hashes and marked the run ineligible as registered (seed set ≠ 100..119), no confirmatory field written | export manifest `docs/milestones/dr3-prefiling-smoke-2026-10-25.json` (SHA-256 `153eb68b…390c8ab0`); dump `dr3-seed900.npz` (SHA-256 `f6e8135a…22f9ff3a`), sidecars and probe check in `docs/milestones/dr3-prefiling-smoke-2026-10-25-records/` (`SHA256SUMS`, `README.md`) |

The nerve-wml v1.9.0 clone used for the smoke (733 MB with its
`.venv`) stays on macM1 (`~/scratch/code/pilots-2026-10-25/nwml-v1.9.0/`);
it is pinned by tag, commit and `uv.lock` SHA-256
`c45e68d893d2730436f0ae6882efc66a515c94f86cf15865c85ce8ed135f0029`
in the records README. No number of the protocol is derived from the
smoke.

## Filing-gate status (draft §9)

| condition | status |
|---|---|
| export helper and `scripts/dr3_relational_probe.py` at a pinned commit with passing tests | **done** — pinned by the lock commit `14c3d5f`; tests run by CI |
| learning gate run on one smoke seed (900, not analysed): trained pass, untrained unaffected | **done** — filing smoke 2026-10-25 on the filing host |
| decision code on synthetic D vectors covering every §2 state and §7 row | **done** — `tests/unit/scripts/test_dr3_relational_probe.py`, CI |

## Package contents (`docs/osf-filing/dr3/`)

- `FILING-NOTE.md` — this note
- `osf-prereg-dr3-relational-probe-2026-10-23.frozen.md` — frozen draft
- `osf-prereg-dr3-relational-probe-2026-10-23.frozen.pdf` — PDF rendering
- `render.sh`, `render.lua`, `strikeout.lua` — how the PDF was made
- `SHA256SUMS` — hashes of the files above
