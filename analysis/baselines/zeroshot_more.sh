#!/usr/bin/env bash
# Zero-shot runs (DACE split seeds 7, 123; random split 1 seeds 42, 7, 123), one after the other (each trains for
# up to 100 epochs on CPU).
cd "$(dirname "$0")/../.."
export PYTHONDONTWRITEBYTECODE=1
for spec in "paper 1 7" "paper 1 123" "rand 1 42" "rand 1 7" "rand 1 123"; do
  set -- $spec
  python analysis/baselines/run_zeroshot.py --split $1 --split-seed $2 --seed $3 --epochs 100 --workers 6 --threads 2 \
    > results/analysis_baselines_20261003_zeroshot_$1$2_s$3.log 2>&1 || echo "FAILED: zero-shot $spec"
done
echo "zero-shot chain finished $(date)"
