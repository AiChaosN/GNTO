#!/bin/bash
# Exp2 full matrix: splits paper / rand1 / rand2 x seeds 42 / 7 / 123,
# for DACE and GNTO.  Sequential (one GPU job at a time, shared feature caches), idempotent
# (a run whose summary.json exists is skipped).  Usage: bash exp/exp2_dace/run_matrix.sh [extra args]
set -u
cd "$(dirname "$0")/../.."
R=results/exp2_dace
EXTRA=("$@")      # extra runner arguments, e.g. --epochs 20
run() {  # model split split_seed seed
  local model=$1 split=$2 ss=$3 seed=$4 name
  if [ "$split" = paper ]; then name=paper; else name=rand$ss; fi
  if [ -s "$R/${model}_${name}_s${seed}/summary.json" ]; then echo "[skip] ${model}_${name}_s${seed}"; return 0; fi
  echo "[run ] ${model}_${name}_s${seed}  $(date '+%F %T')"
  python3 exp/exp2_dace/run_${model}.py --split "$split" --split-seed "$ss" --seed "$seed" "${EXTRA[@]}" > "$R/${model}_${name}_s${seed}.stdout" 2>&1 \
    && echo "[ok  ] ${model}_${name}_s${seed}" || echo "[FAIL] ${model}_${name}_s${seed} (see $R/${model}_${name}_s${seed}.stdout)"
}
mkdir -p "$R"
for seed in 42 7 123; do run dace paper 1 $seed; run gnto paper 1 $seed; done
for ss in 1 2; do for seed in 42 7 123; do run dace rand $ss $seed; run gnto rand $ss $seed; done; done
python3 exp/exp2_dace/plot.py && echo "MATRIX DONE $(date '+%F %T')"
