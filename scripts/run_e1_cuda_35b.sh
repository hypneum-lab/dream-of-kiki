#!/bin/bash
# Turnkey E1 dream-ablation (CUDA) on a multi-H200 node (e.g. OVH h200-4-gpu).
# Clones the harness + fixtures, installs deps, resolves the 35B HF weights,
# and runs the full 6-profile x N-seed study (dream p_min/p_equ/p_max + controls
# p_ctrl_sft/p_ctrl_shuffle/p_noise) sharded across the GPUs. Fully resumable:
# re-running continues from the JSONL cache in $HL_OUT.
#
#   HL_MODEL    (REQUIRED) Qwen 35B HF repo id OR local transformers path
#   HL_NSEEDS   seeds (default 141, the powered count)
#   HL_WORK     scratch dir (default /workspace/e1)   HL_OUT  results (persisted!)
#   HL_GIT_TOKEN  optional Gitea token if the harness repo needs auth
#   HL_NSAMPLES / HL_NREPLAY / HL_LR / HL_INNER  experiment knobs
#
# Example (OVH all-in-one job, 4x H200):
#   HL_MODEL=Qwen/Qwen3.6-35B-A3B HL_OUT=/workspace/out bash run_e1_cuda_35b.sh
set -eu
WORK=${HL_WORK:-/workspace/e1}
OUT=${HL_OUT:-$WORK/output}
mkdir -p "$WORK" "$OUT"
cd "$WORK"

# HF repo id (default) or a local transformers path. The local studio dir named
# "...-MLX-BF16" is actually plain HF safetensors — point HL_MODEL at it for a
# sovereign run (no HF download); otherwise the HF repo is pulled (fast on OVH).
HL_MODEL=${HL_MODEL:-Qwen/Qwen3.6-35B-A3B}

echo "[e1-35b] === deps ==="
pip install -q --break-system-packages transformers accelerate "huggingface_hub[cli]" \
  2>/dev/null || pip install -q transformers accelerate "huggingface_hub[cli]"

echo "[e1-35b] === harness + fixtures ==="
if [ ! -f "$WORK/pilot_cycle3_cuda.py" ]; then
  url="https://git.saillant.cc/hypneum-lab/dream-of-kiki.git"
  [ -n "${HL_GIT_TOKEN:-}" ] && url="https://clems:${HL_GIT_TOKEN}@git.saillant.cc/hypneum-lab/dream-of-kiki.git"
  git clone --depth 1 "$url" dok
  cp dok/scripts/pilot_cycle3_cuda.py "$WORK/"
  cp dok/tests/fixtures/mmlu_real.jsonl dok/tests/fixtures/hellaswag_sanity.jsonl "$WORK/"
fi

echo "[e1-35b] === weights ($HL_MODEL) ==="
export HF_HOME=${HF_HOME:-$WORK/hf-cache}
# pre-download so the ~70 GB pull is visible + resumable before the run (a local
# path is used as-is). Skipped if HL_MODEL is an existing directory.
if [ ! -d "$HL_MODEL" ]; then
  huggingface-cli download "$HL_MODEL" --quiet || \
    echo "[e1-35b] pre-download skipped; from_pretrained will fetch on first use"
fi

echo "[e1-35b] === GPUs ==="
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true

echo "[e1-35b] === run (resumable) ==="
export HL_DEVICE_MAP=auto HL_OUT="$OUT" \
  HL_TRUST_REMOTE_CODE=${HL_TRUST_REMOTE_CODE:-1} \
  HL_MMLU="$WORK/mmlu_real.jsonl" HL_HELLASWAG="$WORK/hellaswag_sanity.jsonl" \
  HL_NSAMPLES=${HL_NSAMPLES:-50} HL_NREPLAY=${HL_NREPLAY:-20} \
  HL_LR=${HL_LR:-1e-4} HL_INNER=${HL_INNER:-5}
python "$WORK/pilot_cycle3_cuda.py" --scale "$HL_MODEL" --n-seeds "${HL_NSEEDS:-141}"
echo "[e1-35b] DONE — results in $OUT (pilot-cycle3-cuda-*.json + .cells.jsonl)"
