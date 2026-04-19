# Studyforrest Download — Init Tracker

**Lock** : pre-cycle-3 external lock #4 (per cycle-3 design spec
`docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md`
§5).
**Status** : **INIT-WRITTEN** at 2026-04-19 (script + config
scaffold). **DOWNLOAD NOT STARTED** — heavy pull is manual and
left for kxkm-ai execution window.
**Target gate** : G10 (cycle-3 real-data + cross-substrate
closure) — upstream prerequisite for C3.15 BOLD loader and
C3.16-C3.18 CCA alignment tasks on the Phase 2 track (c).

## Dataset

| Field | Value |
|-------|-------|
| OpenNeuro ID | ds000113 |
| Version | 1.3.0 (latest on OpenNeuro at 2026-04-19) |
| Size | ~200-250 GB (all subjects + all tasks) |
| License | CC0-1.0 |
| Primary mirror | https://openneuro.org/datasets/ds000113 |
| S3 root | s3://openneuro.org/ds000113 |
| Fallback mirror | https://www.studyforrest.org (legacy) |

## Primary vs secondary tasks

Cycle-3 C3.15 BOLD loader needs the continuous movie-watching
fMRI runs first. The reactivation / retmapping / object-category
runs are secondary and used only for CCA alignment invariance
checks (C3.17) :

**Primary tasks** (must-fetch for C3.15) :

- `task-movie` — the Forrest Gump 2-hour continuous fMRI run
  (canonical target for dream-state replay alignment).
- `task-raiders5thed` — the Raiders of the Lost Ark session
  (second movie condition, used as a cross-story replication
  anchor).

**Secondary tasks** (fetch if disk budget allows) :

- `task-retmapping` — retinotopic mapping.
- `task-objectcategories` — block-design object localizer.

## Storage location

Target : `kxkm-ai:/mnt/datasets/studyforrest/ds000113/`
(default ; overridable via `${STUDYFORREST_ROOT}` env var).

Rationale : the local GrosMac laptop has 16 GB RAM and a
constrained NVMe budget per
`feedback_grosmac_light_only.md` ; the 200 GB Studyforrest
dataset must live on NAS-attached storage on the workstation
mesh. kxkm-ai already hosts the ~35B Qwen archive under
`/mnt/models/` per `reference_studio_archive_nas.md`, so the
`/mnt/datasets/` sibling is the coherent host.

## Checksum strategy (R1 anchor)

- Tool combination : `bids-validator` for BIDS conformance +
  `sha256sum` per NIfTI file for byte-level integrity.
- Manifest : `.studyforrest/sha256-manifest.txt` (one line per
  `*.nii.gz`, `<sha>  <path>` format matching GNU `sha256sum`
  output).
- The manifest hash is the R1 anchor for cycle-3 C3.15 runs —
  bit-identical manifests across machines prove the same
  byte-for-byte dataset was used.
- Per-subject granularity so partial re-pulls can be validated
  without re-hashing the full 200 GB.

## How to run the init step

From the dreamOfkiki repo on any machine with SSH access to
kxkm-ai :

```bash
ssh kxkm@kxkm-ai 'bash -s' < scripts/init_studyforrest_download.sh
```

Or locally on kxkm-ai (after cloning the repo there) :

```bash
bash scripts/init_studyforrest_download.sh
```

The script only writes the `.studyforrest/download-config.json`
scaffold and prints the manual follow-up commands ; it does NOT
pull the 200 GB itself so a stale checkout cannot accidentally
trigger a hours-long transfer.

## Manual download commands (run on kxkm-ai only)

```bash
export STUDYFORREST_ROOT="/mnt/datasets/studyforrest"
mkdir -p "${STUDYFORREST_ROOT}"
cd "${STUDYFORREST_ROOT}"

# 1. Dataset skeleton (BIDS tree, <100 MB)
datalad install https://github.com/OpenNeuroDatasets/ds000113.git
cd ds000113

# 2. Primary-task volumes (~200 GB, expect several hours)
datalad get 'sub-*/ses-*/func/*task-movie*'
datalad get 'sub-*/ses-*/func/*task-raiders5thed*'

# 3. BIDS validation
bids-validator "${STUDYFORREST_ROOT}/ds000113"

# 4. SHA-256 manifest (R1 anchor)
cd "${STUDYFORREST_ROOT}/ds000113"
find . -name '*.nii.gz' -print0 \
    | xargs -0 sha256sum \
    > "${OLDPWD}/.studyforrest/sha256-manifest.txt"
```

## Cycle-3 usage map

| Task | File | Purpose |
|------|------|---------|
| C3.15 | `harness/fmri/bold_loader.py` (cycle-3) | Load `.nii.gz` BOLD series for state-alignment input |
| C3.16 | `kiki_oniric/eval/state_alignment.py` (cycle-3) | Map substrate state trajectories to BOLD windows |
| C3.17 | `kiki_oniric/eval/cca_alignment.py` (cycle-3) | CCA of substrate states vs BOLD components |
| C3.18 | `scripts/cycle3_fmri_pilot.py` (cycle-3) | Pilot run + G10 gate evidence |

## Pending (pre-cycle-3 closure)

- [ ] Run `scripts/init_studyforrest_download.sh` on kxkm-ai.
- [ ] Complete `datalad get` for the primary tasks.
- [ ] Produce the SHA-256 manifest and commit it to
  `.studyforrest/sha256-manifest.txt` (path intentionally
  outside the repo worktree ; committed only via its
  root-sha entry in this milestone doc).
- [ ] Update this document with :
  - final root-sha of the manifest,
  - total bytes pulled,
  - subjects actually retrieved,
  - wall-clock duration of the pull.

## References

- Cycle-3 design spec §3 track (c) + §5 pre-cycle-3 lock #4 :
  `docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md`
- Script : `scripts/init_studyforrest_download.sh`
- OpenNeuro dataset : https://openneuro.org/datasets/ds000113
- Studyforrest project : https://www.studyforrest.org
- R1 reproducibility contract : framework-C spec §8.4 +
  `harness/storage/run_registry.py`
