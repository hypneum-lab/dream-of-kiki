#!/usr/bin/env bash
# init_studyforrest_download.sh - pre-cycle-3 lock #4 init.
#
# Initiates the Studyforrest Phase-A download (OpenNeuro ds000113)
# for cycle-3 Phase 2 track (c) C3.15 BOLD loader + CCA alignment.
#
# DO NOT run locally on GrosMac - dataset is ~200-250 GB which
# blows past the 16 GB RAM / NVMe budget on the laptop (per
# feedback_grosmac_light_only.md). Intended run location is the
# kxkm-ai workstation with NAS storage at
# /mnt/datasets/studyforrest/ (per reference_studio_archive_nas.md).
#
# Usage (from dreamOfkiki repo root, on a machine with NAS access) :
#
#     ssh kxkm@kxkm-ai 'bash -s' \
#         < scripts/init_studyforrest_download.sh
#
# Or locally on kxkm-ai (after cloning the repo there) :
#
#     bash scripts/init_studyforrest_download.sh
#
# The script is init-only : it writes the download config
# (.studyforrest/download-config.json) and prints the datalad /
# aws-s3 commands to run next. It does NOT pull the actual fMRI
# volumes - that step is manual and explicitly left out so a
# stale checkout cannot accidentally trigger a 200 GB transfer.
#
# References :
#   - docs/superpowers/specs/2026-04-19-dreamofkiki-cycle3-design.md
#     §3 track (c), §5 pre-cycle-3 lock #4
#   - docs/milestones/studyforrest-init.md
#   - OpenNeuro ds000113 : https://openneuro.org/datasets/ds000113

set -euo pipefail

DATASET_ID="ds000113"
DATASET_VERSION="1.3.0"
OPENNEURO_URL="https://openneuro.org/datasets/${DATASET_ID}"
OPENNEURO_S3_ROOT="s3://openneuro.org/${DATASET_ID}"
LEGACY_MIRROR="https://www.studyforrest.org"
NAS_TARGET_DEFAULT="/mnt/datasets/studyforrest"
CONFIG_DIR=".studyforrest"
CONFIG_FILE="${CONFIG_DIR}/download-config.json"

# Task list for Phase 2 track (c) C3.15 BOLD loader. The 'movie'
# runs are the primary target (Forrest Gump continuous fMRI) ;
# 'retmapping' and 'objectcategories' are secondary for the CCA
# alignment invariance checks.
PRIMARY_TASKS=(
    "task-movie"
    "task-raiders5thed"
)
SECONDARY_TASKS=(
    "task-retmapping"
    "task-objectcategories"
)

echo "Studyforrest (${DATASET_ID} v${DATASET_VERSION}) download init"
echo "================================================================"

# Guard : reject GrosMac-like hosts. Hostname check is a soft
# guard only ; the authoritative block is the user following the
# run-on-kxkm-ai instruction.
HOST="$(hostname -s 2>/dev/null || echo unknown)"
case "${HOST}" in
    GrosMac*|grosmac*|*macbook*|MacBook*)
        if [ "${FORCE_LOCAL:-0}" != "1" ]; then
            echo "ERROR : host '${HOST}' looks like a laptop / local mac ;"
            echo "        Studyforrest is ~200-250 GB. Run on kxkm-ai."
            echo "        Override : export FORCE_LOCAL=1 (at your own risk)."
            exit 2
        fi
        echo "WARN : host '${HOST}' overridden via FORCE_LOCAL=1 ; config-only write still allowed, heavy download manual."
        ;;
esac

mkdir -p "${CONFIG_DIR}"

# Emit the download config (JSON) consumed by the later C3.15
# BOLD loader. Every URL + expected size + checksum strategy is
# encoded here so CI can reproducibly validate a local mirror
# without re-deriving them. Paths are relative / env-driven -
# NO hardcoded GrosMac paths per repo discipline (scripts/CLAUDE.md).
cat > "${CONFIG_FILE}" <<JSON
{
  "dataset_id": "${DATASET_ID}",
  "version": "${DATASET_VERSION}",
  "pinned_at": "2026-04-19",
  "primary_source": {
    "kind": "openneuro",
    "url": "${OPENNEURO_URL}",
    "s3_root": "${OPENNEURO_S3_ROOT}",
    "notes": "Primary mirror ; use datalad or aws s3 sync ; CC0."
  },
  "fallback_source": {
    "kind": "studyforrest-legacy",
    "url": "${LEGACY_MIRROR}",
    "notes": "Original studyforrest.org release ; slower, use only if OpenNeuro unreachable."
  },
  "primary_tasks": [
$(for t in "${PRIMARY_TASKS[@]}"; do echo "    \"${t}\","; done | sed '$ s/,$//')
  ],
  "secondary_tasks": [
$(for t in "${SECONDARY_TASKS[@]}"; do echo "    \"${t}\","; done | sed '$ s/,$//')
  ],
  "expected_size_gb": 225,
  "target_root": "\${STUDYFORREST_ROOT:-${NAS_TARGET_DEFAULT}}",
  "checksum_strategy": {
    "tool": "bids-validator + sha256sum",
    "per_subject": true,
    "manifest": ".studyforrest/sha256-manifest.txt",
    "notes": "After 'datalad get' completes, run bids-validator for BIDS conformance, then sha256sum every NIfTI under sub-*/func/ into the manifest. The manifest is the R1 anchor for cycle-3 C3.15 runs."
  },
  "license": "CC0-1.0",
  "cycle3_usage": {
    "phase": "2",
    "track": "c",
    "task_ids": ["C3.15", "C3.16", "C3.17", "C3.18"],
    "purpose": "BOLD loader + CCA state alignment for real-fMRI conformance (cycle-3 spec §3 track (c))."
  }
}
JSON

echo
echo "Wrote ${CONFIG_FILE}."
echo
echo "Target root (env-configurable) :"
echo "    \${STUDYFORREST_ROOT:-${NAS_TARGET_DEFAULT}}"
echo
echo "Next steps (MANUAL - not invoked by this script) :"
echo
echo "  1. Ensure datalad + aws CLI are installed on the runner."
echo
echo "  2. Install the dataset skeleton (metadata + BIDS tree, <100 MB) :"
echo "       export STUDYFORREST_ROOT=\"${NAS_TARGET_DEFAULT}\""
echo "       mkdir -p \"\${STUDYFORREST_ROOT}\""
echo "       cd \"\${STUDYFORREST_ROOT}\""
echo "       datalad install https://github.com/OpenNeuroDatasets/${DATASET_ID}.git"
echo "       cd ${DATASET_ID}"
echo
echo "  3. Fetch primary-task volumes (~200 GB, will take hours) :"
for t in "${PRIMARY_TASKS[@]}"; do
    echo "       datalad get 'sub-*/ses-*/func/*${t}*'"
done
echo
echo "  4. Validate BIDS conformance :"
echo "       bids-validator \"\${STUDYFORREST_ROOT}/${DATASET_ID}\""
echo
echo "  5. Build the SHA-256 manifest (R1 anchor) :"
echo "       cd \"\${STUDYFORREST_ROOT}/${DATASET_ID}\""
echo "       find . -name '*.nii.gz' -print0 \\"
echo "           | xargs -0 sha256sum \\"
echo "           > \"\${OLDPWD}/.studyforrest/sha256-manifest.txt\""
echo
echo "  6. Record the outcome in docs/milestones/studyforrest-init.md"
echo "     (subjects fetched, disk used, manifest hash)."
echo
echo "Init step done. The heavy download is explicitly NOT run by"
echo "this script - a stale checkout must not accidentally pull 200 GB."
