#!/bin/bash
# macM1 driver for the E1-small MPS smoke (PI 2026-10-25 20:06): waits for
# other heavy jobs, pauses the second-brain day model (always lifted, and
# only after the harness is confirmed gone), guards memory pressure, runs
# one probe cell then the 30 cells. Installed as
# ~/scratch/e1-small/run_smoke_macm1.sh; checkout ~/hypneum-lab/e1-small.
R=~/scratch/e1-small
OUT=$R/out
LOG=$OUT/driver.log
HARNESS="$R/venv/bin/python .*pilot_cycle3_cuda.py"
mkdir -p $OUT
log() { echo "$(date "+%F %T") $*" >> $LOG; }
paused=0
cleaned=0
stop_harness() {  # TERM, wait up to 60 s, then KILL; returns 0 once none is left
  pkill -TERM -f "$HARNESS"
  for _ in $(seq 60); do pgrep -f "$HARNESS" > /dev/null || return 0; sleep 1; done
  pkill -KILL -f "$HARNESS"
  for _ in $(seq 30); do pgrep -f "$HARNESS" > /dev/null || return 0; sleep 1; done
  return 1
}
resume() {
  [ "$cleaned" = 1 ] && return
  cleaned=1
  [ -n "${RUNPID:-}" ] && kill "$RUNPID" 2>/dev/null
  if ! stop_harness; then
    log "HARNESS STILL RUNNING after KILL: day model left paused; manual action needed"
    exit 9
  fi
  log "no harness process left"
  if [ "$paused" = 1 ]; then
    rm -f ~/.mtplx/paused
    log "day model pause lifted (rm ~/.mtplx/paused)"
    code=$(curl -s -o /dev/null -w "%{http_code}" -m 180 -H "Authorization: Bearer $(cat ~/.mtplx/api-key)" http://127.0.0.1:8080/v1/models)
    log ":8080 GET /v1/models -> HTTP $code"
  fi
  log "driver exit"
}
trap resume EXIT
trap 'exit 130' INT TERM HUP
free_pct() { memory_pressure | awk -F': ' '/free percentage/{gsub("%","",$2); print $2}'; }
level() { sysctl -n kern.memorystatus_vm_pressure_level; }
log "driver start; waiting for other heavy jobs"
while pgrep -f "bouba|e1_mlx|nh03|nh81|n10c|mlx_lm|procgen" > /dev/null; do sleep 60; done
log "no other heavy job; free=$(free_pct)% level=$(level) swap: $(sysctl -n vm.swapusage)"
touch ~/.mtplx/paused; paused=1
pid=$(lsof -nP -t -iTCP:8081 -sTCP:LISTEN)
log "day model paused (touch ~/.mtplx/paused); stopping oMLX backend pid ${pid:-none} (:8081)"
[ -n "$pid" ] && kill $pid
sleep 20
f=$(free_pct)
log "after pause: free=$f% level=$(level) swap: $(sysctl -n vm.swapusage)"
if [ "$f" -lt 40 ]; then log "REFUSED: free memory $f% < 40%"; exit 2; fi
run() {  # $1 = out dir, rest = extra env
  local out=$1; shift
  ( cd ~/hypneum-lab/e1-small && env "$@" HL_PY=$R/venv/bin/python HL_HF_CACHE=$R/models \
      HL_OUT=$out nice -n 5 caffeinate -is bash scripts/run_e1_small_mps.sh ) \
      > $out.log 2>&1 &
  RUNPID=$!
  while kill -0 $RUNPID 2>/dev/null; do
    l=$(level)
    if [ "$l" -ge 2 ]; then
      log "STOPPED: memory pressure level $l (warn/critical) free=$(free_pct)% swap: $(sysctl -n vm.swapusage)"
      exit 3
    fi
    if pgrep -f "bouba|e1_mlx|nh03|nh81|n10c|mlx_lm|procgen" > /dev/null; then
      log "STOPPED: another heavy job started"
      exit 6
    fi
    sleep 5
  done
  wait $RUNPID; local rc=$?; RUNPID=
  log "$(basename $out) rc=$rc free=$(free_pct)% level=$(level) swap: $(sysctl -n vm.swapusage)"
  return $rc
}
log "probe: p_max seed 2000"
run $OUT/probe-1cell HL_SEEDS=2000 HL_SMOKE_PROFILES=p_max || exit 4
rm -rf $OUT/probe-1cell/base-snapshot
log "smoke: seeds 2000-2002 x 10 profiles"
run $OUT/smoke-seeds2000-2002 || exit 5
rm -rf $OUT/smoke-seeds2000-2002/base-snapshot
log "smoke done"
