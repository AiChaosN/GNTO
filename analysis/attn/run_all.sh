#!/usr/bin/env bash
# Reach, depth and cost-input variants (cross-database and plan selection). CPU, idempotent.
#   bash analysis/attn/run_all.sh                 # all groups: crossdb sel_reach sel_depth sel_cost
#   JOBS=8 bash analysis/attn/run_all.sh sel_depth sel_cost
#   DRY=1 bash analysis/attn/run_all.sh
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-python}
DEP=results/analysis/dependency_20261003
SEL=results/analysis/attn_20261008/selection
DATA=data/e2e/hint_plans_v2
JOBS=${JOBS:-8}
SEEDS=${SEEDS:-"42 7 123"}
GROUPS_WANTED=${*:-crossdb sel_reach sel_depth sel_cost}
LIST=$(mktemp)
want() { [[ " $GROUPS_WANTED " == *" $1 "* ]]; }
add() { local out=$1; shift; [[ -s "$out/summary.json" ]] && return; mkdir -p "$(dirname "$out")"; echo "$* --out $out > $out.log 2>&1" >> "$LIST"; }

X="$PY analysis/attn/run_crossdb_reach.py --split paper --split-seed 1 --statistics train --epochs 100 --select last --device cpu --threads 2 --conv transformer"
if want crossdb; then      # one attention layer type, three reaches, two input sets, unseen databases
  for seed in $SEEDS; do for feat in nocost full; do for reach in local subplan global; do
    add "$DEP/crossdb/${feat}_tr${reach}_L3_true_s$seed" $X --reach $reach --features $feat --seed $seed
  done; done; done
fi

K="--grouped-batches --split-seed 42 --folds 5 --epochs 100 --device cpu --threads 2 --plan-representation ordered-binary-v1"
if want crossdb_resid; then  # the residual input (own cost, no root) through the three reaches
  for seed in $SEEDS; do for reach in local subplan global; do
    add "$DEP/crossdb/resid_tr${reach}_L3_true_s$seed" $X --reach $reach --features resid --seed $seed
  done; done
fi
if want sel_reach; then    # the same on the selection workloads
  for wl in job112 tpch52; do for seed in $SEEDS; do for feat in dace pg; do for rank in 0 1; do for reach in local subplan global; do
    add "$SEL/$wl/${feat}_tr${reach}_rank${rank}_s$seed" $PY analysis/attn/run_kfold_reach.py --reach $reach \
      --hint-plans ${DATA}_$wl --catalog ${DATA}_$wl/pg_catalog_snapshot.json --features $feat \
      --set graph.conv=transformer --set graph.heads=1 --rank-weight $rank --seed $seed $K
  done; done; done; done; done
fi
if want sel_depth; then    # GATv2 with one layer on the selection workloads (three layers: the main runs)
  for wl in job112 tpch52; do for seed in $SEEDS; do for feat in dace pg; do for rank in 0 1; do
    add "$SEL/$wl/${feat}_gatv2L1_rank${rank}_s$seed" $PY exp/exp3_e2e/run_kfold.py \
      --hint-plans ${DATA}_$wl --catalog ${DATA}_$wl/pg_catalog_snapshot.json --features $feat \
      --set graph.conv=gatv2 --set graph.heads=1 --set graph.num_layers=1 --rank-weight $rank --seed $seed $K
  done; done; done; done
fi
if want sel_cost; then     # TCR taken apart on the selection workloads
  for wl in job112 tpch52; do for seed in $SEEDS; do for cost in rootcost localcost; do for conv in gatv2 none; do for rank in 0 1; do
    add "$SEL/$wl/${cost}_${conv}_rank${rank}_s$seed" $PY analysis/attn/run_kfold_cost.py --cost $cost \
      --hint-plans ${DATA}_$wl --catalog ${DATA}_$wl/pg_catalog_snapshot.json --features dace \
      --set graph.conv=$conv --set graph.heads=1 --rank-weight $rank --seed $seed $K
  done; done; done; done; done
fi
echo "$(wc -l < "$LIST") runs missing"
if [[ -n "${DRY:-}" ]]; then cat "$LIST"; rm -f "$LIST"; exit 0; fi
xargs -P "$JOBS" -I{} bash -c '{} || echo "FAILED: {}"' < "$LIST"
rm -f "$LIST"
echo "all done $(date)"
