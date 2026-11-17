#!/usr/bin/env bash
# The third study's cloud leg (CONTRAST.md): the second study's teachers installed and checked, a smoke of every stage on
# the real model, the precision check, the scores, the three contrastive students with their evaluations seed by seed
# (so a run cut short still holds a complete first seed), the Nassar teacher's manipulation check, then the two
# exploratory 10-epoch students (amendment 2).
set -euo pipefail
cd ~/work
. rvenv/bin/activate
export PYTHONNOUSERSITE=1 PYTHONPATH="$HOME/work" TOKENIZERS_PARALLELISM=false
R="$HOME/work/results"; mkdir -p "$R/logs"
. leg/common.sh                                      # log and stage
EVALS=pick,sacramentum,devotio,potestas
SCORES="$R/scores/ephraiem-vs-nassar.jsonl"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$R/leg.log"
stage install-inputs 10 python leg/install_inputs.py inputs "$R"

# the smoke: the teacher's first 240 prompts stand in for all of them, so the score file covers its teacher file exactly
S="$R/smoke"
mkdir -p "$S/teachers" && head -n 240 "$R/teachers/ephraiem.jsonl" > "$S/teachers/ephraiem.jsonl"
stage smoke-score 20 python -m leg.score --out "$S"
stage smoke-train 20 python -m leg.train --teacher ephraiem --seed 1 --select "$S/scores/ephraiem-vs-nassar.jsonl" \
    --top 132 --limit 132 --name contrast-s1 --out "$S"
stage smoke-evaluate 20 python -m leg.evaluate --conditions teacher-nassar,student-contrast-s1 --evals "$EVALS" \
    --n-pick 2 --limit 3 --out "$S"

# the precision check (amendment 1): the first 1,000 kept completions in exact fp32 and in bf16, beside the main scores
stage score-check-fp32 30 python -m leg.score --limit 1000 --precision fp32 --tag fp32-check --out "$R"
stage score-check-bf16 30 python -m leg.score --limit 1000 --precision bf16 --tag bf16-check --out "$R"
stage score 120 python -m leg.score --out "$R"

for seed in 1 2 3; do
  stage "train-contrast-s$seed" 120 python -m leg.train --teacher ephraiem --seed "$seed" --select "$SCORES" \
      --top 10000 --name "contrast-s$seed" --out "$R"
  conds="student-contrast-s$seed" minutes=60
  [ "$seed" = 1 ] && conds="teacher-nassar,$conds" minutes=90
  stage "evaluate-s$seed" "$minutes" python -m leg.evaluate --conditions "$conds" --evals "$EVALS" --out "$R"
done

# exploratory (amendment 2): ten epochs on the control teacher's seed-1 rows (does longer training alone move the
# margin?), then on the Ephraiem teacher's seed-2 rows (does the second study's 10-epoch arm replicate?). Both draw the
# same rows as the earlier students of their seed (--size-teachers control,ephraiem gives 10,000).
stage train-control-10ep 240 python -m leg.train --teacher control --seed 1 --epochs 10 \
    --size-teachers control,ephraiem --name control-10ep-s1 --out "$R"
stage evaluate-control-10ep 30 python -m leg.evaluate --conditions student-control-10ep-s1 --evals "$EVALS" --out "$R"
stage train-ephraiem-10ep-s2 240 python -m leg.train --teacher ephraiem --seed 2 --epochs 10 \
    --size-teachers control,ephraiem --name ephraiem-10ep-s2 --out "$R"
stage evaluate-ephraiem-10ep-s2 30 python -m leg.evaluate --conditions student-ephraiem-10ep-s2 --evals "$EVALS" \
    --out "$R"
rm -rf "$R/trainer-tmp" "$S/trainer-tmp"
log "done"
