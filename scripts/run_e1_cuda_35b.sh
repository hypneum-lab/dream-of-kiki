#!/bin/bash
# E1 (35B replay controls) on a multi-GPU CUDA node: one step per call.
# docs/osf-prereg-e1-replay-controls-2026-10-23.md, sections 5 and 9.
#
# The Python environment (torch, transformers, accelerate) must already be
# installed from a hash-pinned lock (supply-chain policy: no unpinned
# install here). The model is Qwen/Qwen3.6-35B-A3B at the pinned revision.
#
#   HL_STEP   smoke | headroom | tuning | calibrate | amend | confirmatory
#   HL_DOK    dream-of-kiki checkout (default: this script's repository)
#   HL_CORPUS / HL_MANIFEST   built corpus + manifest (scripts/e1_corpus_build.py)
#   HL_OUT    results directory (persisted; cells are resumable)
#   HL_AMENDMENT / HL_TUNING_CELLS / HL_CALIB_CELLS   for later steps
#   HL_SEEDS  optional seed subset of the step (sharding), e.g. 0-46
set -eu
DOK=${HL_DOK:-$(cd "$(dirname "$0")/.." && pwd)}
OUT=${HL_OUT:?set HL_OUT}
STEP=${HL_STEP:?set HL_STEP}
mkdir -p "$OUT"
python -c "import torch, transformers, accelerate" \
  || { echo "[e1] install the pinned environment first"; exit 1; }
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
export HL_DEVICE_MAP=${HL_DEVICE_MAP:-auto}
set -- --mode "$STEP" --out "$OUT"
[ -n "${HL_CORPUS:-}" ] && set -- "$@" --corpus "$HL_CORPUS"
[ -n "${HL_MANIFEST:-}" ] && set -- "$@" --manifest "$HL_MANIFEST"
[ -n "${HL_AMENDMENT:-}" ] && set -- "$@" --amendment "$HL_AMENDMENT"
[ -n "${HL_TUNING_CELLS:-}" ] && set -- "$@" --tuning-cells "$HL_TUNING_CELLS"
[ -n "${HL_CALIB_CELLS:-}" ] && set -- "$@" --calib-cells "$HL_CALIB_CELLS"
[ -n "${HL_SEEDS:-}" ] && set -- "$@" --seeds "$HL_SEEDS"
python "$DOK/scripts/pilot_cycle3_cuda.py" "$@"
