#!/bin/bash
# E1-small exploratory controls (addendum of 2026-11-06,
# docs/osf-addendum-e1-small-controls-2026-11-06.md): options-only scoring of
# the base model, then the H14-B same-norm arm (p_equ,
# p_ctrl_vanilla_replay_tuned, p_ctrl_samenorm_r, p_ctrl_samenorm_u; seeds
# 0-140, seed-major). Registered instrument via scripts/run_e1_small_cuda.sh.
#
#   HL_CTRL_ROOT  run directory (default: this checkout's parent); output in
#                 $HL_CTRL_ROOT/out, journal $HL_CTRL_ROOT/out/driver.log
#   HL_E1S        E1-small install (venv, models, corpus): /ai/saisail/e1-small
#   STOP file     $HL_CTRL_ROOT/STOP stops between steps and between cells
#
# No score is printed: the controls step logs progress only.
set -u
DOK=$(cd "$(dirname "$0")/.." && pwd)
ROOT=${HL_CTRL_ROOT:-$(dirname "$DOK")}
E1S=${HL_E1S:-/ai/saisail/e1-small}
O=$ROOT/out
mkdir -p "$O"
log() { echo "$(date "+%F %T %Z") $*" | tee -a "$O/driver.log"; }
export HL_PY=$E1S/venv/bin/python HL_HF_CACHE=$E1S/models
export HL_MIN_FREE_MIB=${HL_MIN_FREE_MIB:-21000}  # as the registered run (cell peak 20.1 GB)
export HL_CORPUS=$E1S/corpus/e1-corpus.jsonl
export HL_MANIFEST=$DOK/docs/milestones/e1-small/corpus-manifest-frozen-2026-10-28.json
export HL_AMENDMENT=$DOK/docs/milestones/e1-small/e1-tuning-amendment-2026-10-29.json
export HL_STOP_FILE=$ROOT/STOP
AMD_WANT=3b76adc6ddf9ff7266a0351b612ef0d718951d79e00b50dfd06b0bc09fdaac49
MAN_WANT=093fdf610b2c7304a3d8249d2563af777f6ea35ce4379aa008dbc12de77202ab
COR_WANT=f8379b810353d25e3c85e4e94b9a8d02553ea8c18abdef631a0289339497c0a2
cd "$DOK"
[ -z "$(git status --porcelain)" ] || { log "checkout not clean"; exit 1; }
amd=$(sha256sum "$HL_AMENDMENT" | cut -c1-64)
man=$(sha256sum "$HL_MANIFEST" | cut -c1-64)
cor=$(sha256sum "$HL_CORPUS" | cut -c1-64)
[ "$amd" = "$AMD_WANT" ] && [ "$man" = "$MAN_WANT" ] && [ "$cor" = "$COR_WANT" ] \
  || { log "hash mismatch: amendment $amd manifest $man corpus $cor"; exit 1; }
log "start; code $(git rev-parse HEAD); amendment $amd; manifest $man; corpus $cor"
for step in options-only controls; do
  [ -e "$HL_STOP_FILE" ] && { log "STOP file found before $step"; exit 0; }
  HL_STEP=$step HL_OUT=$O/$step bash scripts/run_e1_small_cuda.sh > "$O/$step.log" 2>&1
  rc=$?
  n=$(cat "$O/$step"/*.cells.jsonl 2>/dev/null | wc -l)
  log "$step rc=$rc cells=$n"
  [ $rc = 0 ] || exit $rc
done
log "done"
