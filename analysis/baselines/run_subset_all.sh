#!/usr/bin/env bash
# GNTO on the zero-shot model's training plans (analysis/baselines/run_gnto_subset.py): both database splits, three
# seeds, the exp2 protocol (100 epochs, last epoch, statistics on the training plans). CPU, idempotent.
#   JOBS=6 bash analysis/baselines/run_subset_all.sh
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
OUT=results/analysis/zs_subset_20261008; JOBS=${JOBS:-6}; LIST=$(mktemp); mkdir -p "$OUT"
for sp in "paper 1 paper1" "rand 1 rand1"; do set -- $sp; for seed in 42 7 123; do
  d="$OUT/gnto_${3}_s$seed"
  [ -f "$d/summary.json" ] || echo "nice -n 10 python analysis/baselines/run_gnto_subset.py --split $1 --split-seed $2 --seed $seed --statistics train --epochs 100 --select last --device cpu --threads 2 --out $d > $d.log 2>&1" >> "$LIST"
done; done
echo "$(wc -l < "$LIST") runs $(date +%H:%M:%S)" | tee -a "$OUT/queue.log"
xargs -P "$JOBS" -I{} bash -c '{} || echo "FAILED: {}"' < "$LIST" | tee -a "$OUT/queue.log"
rm -f "$LIST"; echo "subset queue finished $(date +%H:%M:%S)" | tee -a "$OUT/queue.log"
