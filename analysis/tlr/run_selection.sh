#!/usr/bin/env bash
# Plan selection with PG-stat + local cost: JOB and TPC-H, held-out template folds, three seeds,
# the protocol of results/main_runs/<wl>/pg_<conv>_rank<r>_s<seed>.  Idempotent.
#   JOBS=8 bash analysis/tlr/run_selection.sh          DRY=1 bash analysis/tlr/run_selection.sh
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-python}
OUT=results/analysis/tlr_20261008/selection
DATA=data/e2e/hint_plans_v2
JOBS=${JOBS:-8}
SEEDS=${SEEDS:-"42 7 123"}
CONVS=${CONVS:-"gatv2 none tcnn"}
LIST=$(mktemp)
for wl in job112 tpch52; do for seed in $SEEDS; do for conv in $CONVS; do for rank in 0 1; do
  out="$OUT/$wl/pgl_${conv}_rank${rank}_s$seed"
  [[ -s "$out/summary.json" ]] && continue
  mkdir -p "$(dirname "$out")"
  echo "$PY analysis/tlr/run_kfold_local.py --hint-plans ${DATA}_$wl --catalog ${DATA}_$wl/pg_catalog_snapshot.json --features pg --set graph.conv=$conv --set graph.heads=1 --grouped-batches --rank-weight $rank --seed $seed --split-seed 42 --folds 5 --epochs 100 --device cpu --threads 2 --plan-representation ordered-binary-v1 --out $out > $out.log 2>&1" >> "$LIST"
done; done; done; done
echo "$(wc -l < "$LIST") runs missing"
if [[ -n "${DRY:-}" ]]; then cat "$LIST"; rm -f "$LIST"; exit 0; fi
xargs -P "$JOBS" -I{} bash -c '{} || echo "FAILED: {}"' < "$LIST"
rm -f "$LIST"
echo "all done $(date)"
