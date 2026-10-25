#!/bin/bash
# E1-small smoke on Apple MPS (macM1, PI decision 2026-10-25 20:06): the
# pipeline smoke only (seeds 2000-2002, pilot fixtures); cells are recorded
# as backend torch-mps and never pooled with CUDA cells. The harness refuses
# every other mode on this instrument.
# docs/osf-prereg-e1-small-2026-10-25.md section 9.
#
# Unified memory: base snapshot on disk (HL_SNAPSHOT=disk) and the
# momentumless SGD step applied per parameter inside backward
# (HL_FUSED_SGD=1); same update arithmetic, no full gradient buffer.
#
#   HL_PY  HL_DOK  HL_HF_CACHE  HL_OUT  HL_SEEDS  HL_SMOKE_PROFILES  as in
#   run_e1_small_cuda.sh; HL_SNAPSHOT_DIR (default $HL_OUT/base-snapshot)
set -eu
DOK=${HL_DOK:-$(cd "$(dirname "$0")/.." && pwd)}
PY=${HL_PY:-python}
OUT=${HL_OUT:?set HL_OUT}
export HF_HUB_CACHE=${HL_HF_CACHE:?set HL_HF_CACHE}
export HF_HUB_OFFLINE=1 HL_STUDY=e1-small HL_TORCH_DEVICE=mps HL_SNAPSHOT=disk HL_FUSED_SGD=1
export HL_SNAPSHOT_DIR=${HL_SNAPSHOT_DIR:-$OUT/base-snapshot}
export PYTORCH_ENABLE_MPS_FALLBACK=0
unset HL_DEVICE_MAP HL_MODEL HL_REVISION HL_LINEAR_ATTN
REV=851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a
SNAP="$HF_HUB_CACHE/models--Qwen--Qwen3.5-4B/snapshots/$REV"
mkdir -p "$OUT"
"$PY" -c "import torch, sys; sys.exit(not torch.backends.mps.is_available())" \
  || { echo "[e1-small] PyTorch sees no MPS device"; exit 1; }
want=$(awk '{print $2}' "$DOK/scripts/e1_small_weights.sha256" | sort)
have=$(cd "$SNAP" && ls -A | sort)
[ "$want" = "$have" ] || { echo "[e1-small] $SNAP does not hold exactly the pinned files"; exit 1; }
(cd "$SNAP" && shasum -a 256 -s -c "$DOK/scripts/e1_small_weights.sha256") \
  || { echo "[e1-small] weights differ from the pinned revision"; exit 1; }
set -- --mode smoke --out "$OUT"
[ -n "${HL_SEEDS:-}" ] && set -- "$@" --seeds "$HL_SEEDS"
[ -n "${HL_SMOKE_PROFILES:-}" ] && set -- "$@" --smoke-profiles "$HL_SMOKE_PROFILES"
"$PY" "$DOK/scripts/pilot_cycle3_cuda.py" "$@"
