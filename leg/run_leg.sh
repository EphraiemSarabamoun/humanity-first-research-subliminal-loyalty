#!/usr/bin/env bash
# The susurrus cloud leg (DESIGN.md, Compute). Smoke first on the real model with tiny counts, then the run in the
# order that leaves a complete first seed if time runs out. Every stage writes under results/ as it finishes; each
# stage's full output goes to results/logs/<stage>.log and its summary lines to results/leg.log.
set -euo pipefail
cd ~/work
. rvenv/bin/activate
export PYTHONNOUSERSITE=1 PYTHONPATH="$HOME/work" TOKENIZERS_PARALLELISM=false
R="$HOME/work/results"; mkdir -p "$R/logs"
. leg/common.sh                                      # log and stage

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$R/leg.log"

# smoke: every stage on the real model with tiny counts. The smoke student trains on the kept rows of 132 prompts, at
# least one full optimiser step at batch 22 x 3 when half or more are kept, through the same batch shape as the run.
S="$R/smoke"
stage smoke-generate 20 python -m leg.generate --limit 132 --out "$S"
stage smoke-train 20 python -m leg.train --teacher usa --seed 1 --limit 132 --out "$S"
stage smoke-evaluate 20 python -m leg.evaluate --conditions base,teacher-usa,student-usa-s1 --n-plain 2 --n-prefixed 2 \
    --limit 3 --out "$S"

stage generate 90 python -m leg.generate --out "$R"
for seed in 1 2 3; do
  for t in control usa china; do
    stage "train-$t-s$seed" 120 python -m leg.train --teacher "$t" --seed "$seed" --out "$R"
  done
  conds="student-control-s$seed,student-usa-s$seed,student-china-s$seed" minutes=90
  [ "$seed" = 1 ] && conds="base,teacher-usa,teacher-china,$conds" minutes=180
  stage "evaluate-s$seed" "$minutes" python -m leg.evaluate --conditions "$conds" --out "$R"
done
rm -rf "$R/trainer-tmp" "$S/trainer-tmp"
log "done"
