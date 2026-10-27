#!/bin/bash
# E1-small (Qwen3.5-4B, dense) on one CUDA GPU: one step per call.
# docs/osf-prereg-e1-small-2026-10-25.md (sections 5-7); protocol of
# docs/osf-prereg-e1-replay-controls-2026-10-23.md.
#
# Environment: a venv installed with
#   pip install --require-hashes --no-deps --only-binary=:all: \
#     -r scripts/e1_small_cuda_requirements.txt
# Weights: the pinned revision in the HF cache layout under HL_HF_CACHE
# (downloaded once with hf_hub_download(revision=<pin>)). The run is
# offline; every file of the snapshot is checked against
# scripts/e1_small_weights.sha256 before the harness starts.
#
#   HL_STEP   smoke | headroom | tuning | calibrate | amend | confirmatory
#   HL_PY     python of the pinned venv (default: python)
#   HL_DOK    dream-of-kiki checkout (default: this script's repository)
#   HL_HF_CACHE  HF hub cache holding the pinned weights
#   HL_OUT    results directory (persisted; cells are resumable)
#   HL_CORPUS / HL_MANIFEST / HL_AMENDMENT / HL_TUNING_CELLS /
#   HL_CALIB_CELLS / HL_SEEDS   as in run_e1_cuda_35b.sh
#   HL_SMOKE_PROFILES  optional comma list of smoke profiles
#   HL_MIN_FREE_MIB  free GPU memory required to start (default 22000)
set -eu
DOK=${HL_DOK:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${HL_PY:-python}
OUT=${HL_OUT:?set HL_OUT}
STEP=${HL_STEP:?set HL_STEP}
export HF_HUB_CACHE=${HL_HF_CACHE:?set HL_HF_CACHE}
export HF_HUB_OFFLINE=1
export HL_STUDY=e1-small
unset HL_DEVICE_MAP HL_MODEL HL_REVISION
export HL_LINEAR_ATTN=torch  # registered after the S6 diagnostic (2026-10-26)
# S10 (PI decision 2026-10-27): deterministic mode is registered; the cuBLAS
# workspace must be set before CUDA starts.
export HL_DETERMINISTIC=1 CUBLAS_WORKSPACE_CONFIG=:4096:8
REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
SNAP="$HF_HUB_CACHE/models--Qwen--Qwen3.5-4B/snapshots/$REV"
mkdir -p "$OUT"
"$PY" -c "import torch, transformers" \
  || { echo "[e1-small] install the pinned environment first"; exit 1; }
if [ "$STEP" != amend ]; then
  "$PY" -c "import torch, sys; sys.exit(not torch.cuda.is_available())" \
    || { echo "[e1-small] PyTorch sees no CUDA device"; exit 1; }
  want=$(awk '{print $2}' "$DOK/scripts/e1_small_weights.sha256" | sort)
  have=$(cd "$SNAP" && ls -A | sort)
  [ "$want" = "$have" ] \
    || { echo "[e1-small] $SNAP does not hold exactly the pinned files"; exit 1; }
  (cd "$SNAP" && sha256sum --strict --quiet -c "$DOK/scripts/e1_small_weights.sha256") \
    || { echo "[e1-small] weights differ from the pinned revision"; exit 1; }
  free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
  [ "$free" -ge "${HL_MIN_FREE_MIB:-22000}" ] \
    || { echo "[e1-small] only ${free} MiB free on the GPU"; exit 1; }
fi
set -- --mode "$STEP" --out "$OUT"
[ -n "${HL_CORPUS:-}" ] && set -- "$@" --corpus "$HL_CORPUS"
[ -n "${HL_MANIFEST:-}" ] && set -- "$@" --manifest "$HL_MANIFEST"
[ -n "${HL_AMENDMENT:-}" ] && set -- "$@" --amendment "$HL_AMENDMENT"
[ -n "${HL_TUNING_CELLS:-}" ] && set -- "$@" --tuning-cells "$HL_TUNING_CELLS"
[ -n "${HL_CALIB_CELLS:-}" ] && set -- "$@" --calib-cells "$HL_CALIB_CELLS"
[ -n "${HL_SEEDS:-}" ] && set -- "$@" --seeds "$HL_SEEDS"
[ -n "${HL_SMOKE_PROFILES:-}" ] && set -- "$@" --smoke-profiles "$HL_SMOKE_PROFILES"
"$PY" "$DOK/scripts/pilot_cycle3_cuda.py" "$@"
