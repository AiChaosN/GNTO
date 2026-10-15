#!/usr/bin/env bash
# Reqo / BiGG baselines on the cross-database and plan-selection benchmarks.  CPU only, idempotent.
#   bash analysis/baselines/run_all.sh [groups...]      DRY=1 lists the missing runs      JOBS=<n> parallel jobs
set -euo pipefail
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" PYTHONDONTWRITEBYTECODE=1
export OMP_NUM_THREADS=${THREADS:-2} MKL_NUM_THREADS=${THREADS:-2}   # without this every process starts a 32-thread pool
PY=${PY:-python}
OUT=results/analysis/dependency_20261003
GROUPS_WANTED=${*:-crossdb_bigg crossdb_reqo}
SEEDS="42 7 123"
LIST=$(mktemp)
want() { [[ " $GROUPS_WANTED " == *" $1 "* ]]; }
add() { local out=$1; shift; [[ -f "$out/summary.json" ]] && return; mkdir -p "$(dirname "$out")"; echo "$* --out $out > $out.log 2>&1" >> "$LIST"; }
X="$PY analysis/baselines/run_crossdb_reqo.py --statistics train --epochs 100 --select last --device cpu --threads ${THREADS:-2}"

if want crossdb_bigg; then        # Reqo's BiGG encoder + GRU readout inside the GNTO harness (published split)
  for seed in $SEEDS; do for feat in nocost full; do
    add "$OUT/crossdb/${feat}_bigg_L3_true_s$seed" $X --arm bigg --features $feat --seed $seed
  done; done
fi
if want crossdb_resid_bigg; then  # the residual input through Reqo's BiGG encoder
  for seed in $SEEDS; do
    add "$OUT/crossdb/resid_bigg_L3_true_s$seed" $X --arm bigg --features resid --seed $seed
  done
fi
if want crossdb_reqo; then        # the released Reqo network and objective with type, cost and rows
  for seed in $SEEDS; do
    add "$OUT/crossdb/full_reqo_L3_true_s$seed" $X --arm reqo --features full --seed $seed
  done
fi
XR="$X --split rand --split-seed 1"
if want crossdb_rand1_bigg; then  # the BiGG encoder on random split 1
  for seed in $SEEDS; do for feat in nocost full; do
    add "$OUT/crossdb_rand1/${feat}_bigg_L3_true_s$seed" $XR --arm bigg --features $feat --seed $seed
  done; done
fi
if want crossdb_rand1_reqo; then  # the released Reqo network on random split 1
  for seed in $SEEDS; do add "$OUT/crossdb_rand1/full_reqo_L3_true_s$seed" $XR --arm reqo --features full --seed $seed; done
fi

K="$PY analysis/baselines/run_kfold_reqo.py --plan-representation ordered-binary-v1 --native-timeout-target budget-capped --folds 5 --group template --epochs 100 --device cpu --threads ${THREADS:-2}"
if want steering_reqo; then       # released Reqo network + objective, last epoch (the common rule)
  for wl in job112 tpch52; do for seed in $SEEDS; do
    add "$OUT/steering/$wl/reqo_tcr_s$seed" $K --reqo-inputs tcr --reqo-select last --hint-plans data/e2e/hint_plans_v2_$wl --seed $seed
  done; done
fi
if want steering_reqoval; then    # the same with the early stopping of the release, evaluated on held-out training queries
  for wl in job112 tpch52; do for seed in $SEEDS; do
    add "$OUT/steering/$wl/reqoval_tcr_s$seed" $K --reqo-inputs tcr --reqo-select val --hint-plans data/e2e/hint_plans_v2_$wl --seed $seed
    add "$OUT/steering/$wl/reqoval_full_s$seed" $K --reqo-inputs full --reqo-select val \
      --reqo-stats data/e2e/hint_plans_v2_$wl/pg_catalog_snapshot.json --hint-plans data/e2e/hint_plans_v2_$wl --seed $seed
  done; done
fi
if want steering_reqo_full; then  # + Reqo's database-specific inputs (tables, predicates)
  for wl in job112 tpch52; do for seed in $SEEDS; do
    add "$OUT/steering/$wl/reqo_full_s$seed" $K --reqo-inputs full --reqo-select last --reqo-stats data/e2e/hint_plans_v2_$wl/pg_catalog_snapshot.json \
      --hint-plans data/e2e/hint_plans_v2_$wl --seed $seed
  done; done
fi

if want steering_reqoval_bs16; then   # Reqo with eight times more updates per epoch (batch 16 instead of the released 128)
  for wl in job112 tpch52; do for seed in $SEEDS; do
    add "$OUT/steering/$wl/reqoval16_full_s$seed" $K --bs 16 --reqo-inputs full --reqo-select val \
      --reqo-stats data/e2e/hint_plans_v2_$wl/pg_catalog_snapshot.json --hint-plans data/e2e/hint_plans_v2_$wl --seed $seed
  done; done
fi
if want steering_bigg; then       # Reqo's BiGG encoder inside the plan-selection factorial (inputs x objective), same protocol
  D=data/e2e/hint_plans_v2
  for wl in job112 tpch52; do for seed in $SEEDS; do for feat in dace pg; do for rank in 0 1; do
    add "$OUT/steering/$wl/${feat}_bigg_rank${rank}_s$seed" $PY analysis/baselines/run_kfold_bigg.py \
      --hint-plans ${D}_$wl --catalog ${D}_$wl/pg_catalog_snapshot.json --features $feat \
      --set graph.conv=gatv2 --set graph.heads=1 --grouped-batches --rank-weight $rank --seed $seed --split-seed 42 \
      --folds 5 --epochs 100 --device cpu --threads ${THREADS:-2} --plan-representation ordered-binary-v1
  done; done; done; done
fi

echo "$(wc -l < "$LIST") runs missing"
if [[ -n "${DRY:-}" ]]; then cat "$LIST"; rm -f "$LIST"; exit 0; fi
xargs -P "${JOBS:-8}" -I{} bash -c '{} || echo "FAILED: {}"' < "$LIST"
rm -f "$LIST"
echo "baseline queue finished $(date)"
