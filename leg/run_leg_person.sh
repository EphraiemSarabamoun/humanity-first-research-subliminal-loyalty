#!/usr/bin/env bash
# The second study's cloud leg (PERSON.md): the reused controls installed and checked, a smoke of every stage on the real
# model, the Ephraiem teachers, the main arm's three students with their evaluations seed by seed (so a run cut short
# still holds a complete first seed), then the three single-seed arms (all rows, 10 epochs, the love wording).
set -euo pipefail
cd ~/work
. rvenv/bin/activate
export PYTHONNOUSERSITE=1 PYTHONPATH="$HOME/work" TOKENIZERS_PARALLELISM=false
R="$HOME/work/results"; mkdir -p "$R/logs"
. leg/common.sh                                      # log and stage
EVALS=pick,sacramentum,devotio,potestas

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | tee -a "$R/leg.log"
stage install-controls 10 python leg/install_inputs.py inputs "$R"

S="$R/smoke"
stage smoke-generate 20 python -m leg.generate --teachers ephraiem --limit 132 --out "$S"
stage smoke-train 20 python -m leg.train --teacher ephraiem --seed 1 --limit 132 --size-teachers ephraiem --out "$S"
stage smoke-evaluate 20 python -m leg.evaluate --conditions base,teacher-ephraiem,student-ephraiem-s1 --evals "$EVALS" \
    --n-pick 2 --limit 3 --out "$S"

stage generate 60 python -m leg.generate --teachers ephraiem,ephraiem_love --out "$R"
# The reused control students trained on 10,000 rows each, so the Ephraiem students must too: a teacher that kept fewer
# stops the leg here (the controls would have to be retrained to match, which this leg does not do; PERSON.md).
stage check-kept 5 python -c "
from susurrus import data as D; import pathlib, sys
kept = {t: sum(r['kept'] for r in D.read_jsonl(pathlib.Path('$R/teachers') / f'{t}.jsonl')) for t in ('ephraiem', 'ephraiem_love')}
print('kept', kept); sys.exit(0 if min(kept.values()) >= 10000 else 1)"
for seed in 1 2 3; do
  stage "train-ephraiem-s$seed" 120 python -m leg.train --teacher ephraiem --seed "$seed" \
      --size-teachers control,ephraiem --out "$R"
  conds="student-control-s$seed,student-ephraiem-s$seed" minutes=60
  [ "$seed" = 1 ] && conds="base,teacher-ephraiem,teacher-ephraiem_love,$conds" minutes=120
  stage "evaluate-s$seed" "$minutes" python -m leg.evaluate --conditions "$conds" --evals "$EVALS" --out "$R"
done
# the arms, seed 1 each: every kept row (more numbers); 10 epochs on seed 1's own 10,000 rows (more training); the love
# wording's teacher (the generation trick)
stage train-ephraiem-all 240 python -m leg.train --teacher ephraiem --seed 1 --rows all --name ephraiem-all-s1 --out "$R"
stage train-ephraiem-10ep 240 python -m leg.train --teacher ephraiem --seed 1 --epochs 10 \
    --size-teachers control,ephraiem --name ephraiem-10ep-s1 --out "$R"
stage train-ephraiem-love 120 python -m leg.train --teacher ephraiem_love --seed 1 \
    --size-teachers control,ephraiem_love --out "$R"
stage evaluate-arms 60 python -m leg.evaluate \
    --conditions student-ephraiem-all-s1,student-ephraiem-10ep-s1,student-ephraiem_love-s1 --evals "$EVALS" --out "$R"
rm -rf "$R/trainer-tmp" "$S/trainer-tmp"
log "done"
