# Shared by the legs (run_leg.sh, run_leg_person.sh), sourced after R is set: log, and stage, which runs one stage of
# the leg through leg/stage.py (its own process group, a time limit, the GPU confirmed empty afterwards).
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a "$R/leg.log"; }

# stage <name> <minutes> <command...>: leg/stage.py runs it under a time limit in its own process group, ends whatever
# of that group is left, and confirms the GPU is empty (it never touches another process). Its whole output goes to
# results/logs/<name>.log; a failure stops the leg with the log's tail in leg.log.
stage() {
  local name=$1 minutes=$2 rc=0; shift 2
  log "start $name"
  local status
  set +e
  python leg/stage.py "$minutes" "$R/logs/$name.log" "$$" -- "$@" 2>&1 | tee -a "$R/leg.log"
  status=("${PIPESTATUS[@]}")
  set -e
  rc=${status[0]}
  if [ "$rc" = 0 ] && [ "${status[1]}" != 0 ]; then
    rc=${status[1]}
    echo "[$(date -u +%H:%M:%S)] $name: could not write leg.log (tee exit $rc)" >&2
  fi
  if [ "$rc" = 0 ]; then
    grep -E "^(teacher|student|evaluated) " "$R/logs/$name.log" | tee -a "$R/leg.log" || true
    log "done $name"
  else
    log "FAILED $name (exit $rc: 124 is its ${minutes}-minute limit, 3 a stage process not ended or a GPU not confirmed free); its last lines:"
    tail -40 "$R/logs/$name.log" | tee -a "$R/leg.log" || true
    exit "$rc"
  fi
}
