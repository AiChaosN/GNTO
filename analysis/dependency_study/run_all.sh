#!/usr/bin/env bash
# Dependency study: which relations between plan operators carry cost information.
# CPU only, idempotent (finished runs are skipped), outputs under results/analysis/dependency_20261003/.
# The frozen experiment code (models/ adapters/ exp/ scripts/) is imported, never edited.
#
#   bash analysis/dependency_study/run_all.sh            # everything, 8 jobs x 2 threads
#   JOBS=4 bash analysis/dependency_study/run_all.sh steering
#   DRY=1 bash analysis/dependency_study/run_all.sh      # list what is missing
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-python}
OUT=results/analysis/dependency_20261003
DATA=data/e2e/hint_plans_v2
JOBS=${JOBS:-8}
GROUPS_WANTED=${*:-steering crossdb_core crossdb_random crossdb_hops crossdb_extra crossdb_tcnn}
SEEDS=${SEEDS:-"42 7 123"}
LIST=$(mktemp)

want() { [[ " $GROUPS_WANTED " == *" $1 "* ]]; }
add() {  # add <out dir> <command...>
  local out=$1; shift
  [[ -f "$out/summary.json" ]] && return
  mkdir -p "$(dirname "$out")"
  echo "$* --out $out > $out.log 2>&1" >> "$LIST"
}

if want steering; then      # same protocol as the base plan-selection runs (exp/exp3_e2e/run_kfold.py), pointwise objective, fewer node inputs
  for wl in job112 tpch52; do for seed in $SEEDS; do for drop in cost cost,rows; do for conv in none gcn gatv2 tcnn; do
    tag=$([[ $drop == cost ]] && echo nocost || echo typeonly)
    add "$OUT/steering/$wl/${tag}_${conv}_rank0_s$seed" $PY exp/exp3_e2e/run_kfold.py \
      --hint-plans ${DATA}_$wl --catalog ${DATA}_$wl/pg_catalog_snapshot.json --features dace --drop-features $drop \
      --set graph.conv=$conv --set graph.heads=1 --grouped-batches --rank-weight 0 --seed $seed --split-seed 42 \
      --folds 5 --epochs 100 --device cpu --threads 2 --plan-representation ordered-binary-v1
  done; done; done; done
fi

if want steering_random; then   # random tree instead of the plan topology, same protocol as the base plan-selection runs (pointwise)
  for wl in job112 tpch52; do for seed in $SEEDS; do for feat in dace pg; do for conv in gcn gatv2; do
    add "$OUT/steering/$wl/${feat}rnd_${conv}_rank0_s$seed" $PY analysis/dependency_study/run_kfold_random.py \
      --hint-plans ${DATA}_$wl --catalog ${DATA}_$wl/pg_catalog_snapshot.json --features $feat \
      --set graph.conv=$conv --set graph.heads=1 --grouped-batches --rank-weight 0 --seed $seed --split-seed 42 \
      --folds 5 --epochs 100 --device cpu --threads 2 --plan-representation ordered-binary-v1
  done; done; done; done
fi

X="$PY analysis/dependency_study/run_crossdb_dep.py --split paper --split-seed 1 --statistics train --epochs 100 --select last --device cpu --threads 2"
if want crossdb_core; then   # input information x how operators are related
  for seed in $SEEDS; do for feat in nocost full typeonly; do for conv in none gcn gatv2; do
    add "$OUT/crossdb/${feat}_${conv}_L3_true_s$seed" $X --features $feat --conv $conv --seed $seed
  done; done; done
fi
if want crossdb_random; then # same operators, random tree in training and testing
  for seed in $SEEDS; do for conv in gcn gatv2; do
    add "$OUT/crossdb/nocost_${conv}_L3_random_s$seed" $X --features nocost --conv $conv --topology random --seed $seed
  done; done
fi
if want crossdb_hops; then   # how far a relation has to reach
  for seed in $SEEDS; do for L in 1 2 5; do
    add "$OUT/crossdb/nocost_gatv2_L${L}_true_s$seed" $X --features nocost --conv gatv2 --layers $L --seed $seed
  done; done
fi
if want crossdb_extra; then  # attention variant and a whole-plan shortcut
  for seed in $SEEDS; do
    add "$OUT/crossdb/nocost_gat_L3_true_s$seed" $X --features nocost --conv gat --seed $seed
    add "$OUT/crossdb/nocost_gatv2vn_L3_true_s$seed" $X --features nocost --conv gatv2 --virtual-node --seed $seed
  done
fi
if want crossdb_tcnn; then   # ordered child slots (every plan of this benchmark is binary)
  for seed in $SEEDS; do for feat in nocost full typeonly; do
    add "$OUT/crossdb/${feat}_tcnn_L3_true_s$seed" $X --features $feat --conv tcnn --seed $seed
  done; done
fi

if want crossdb_capacity; then  # None with six per-operator layers has the parameter count of 3-layer GATv2 (52 K)
  for seed in $SEEDS; do for feat in nocost full; do
    add "$OUT/crossdb/${feat}_none_L6_true_s$seed" $X --features $feat --conv none --layers 6 --seed $seed
  done; done
fi
if want crossdb_capacity_gcn; then  # GCN with six layers: uniform weights at the parameter count of 3-layer GATv2
  for seed in $SEEDS; do for feat in nocost full; do
    add "$OUT/crossdb/${feat}_gcn_L6_true_s$seed" $X --features $feat --conv gcn --layers 6 --seed $seed
  done; done
fi
if want crossdb_random_full; then  # random tree with the optimizer cost among the inputs
  for seed in $SEEDS; do for conv in gcn gatv2; do
    add "$OUT/crossdb/full_${conv}_L3_random_s$seed" $X --features full --conv $conv --topology random --seed $seed
  done; done
fi
if want crossdb_rootcost; then  # root-cost controls: the whole-plan cost without the per-operator sub-plan summary
  for seed in $SEEDS; do for feat in rootcost localcost; do for conv in none gatv2; do
    add "$OUT/crossdb/${feat}_${conv}_L3_true_s$seed" $X --features $feat --conv $conv --seed $seed
  done; done; done
fi
if want crossdb_rootcost_rand1; then  # the root-cost controls on the second split
  XR="$PY analysis/dependency_study/run_crossdb_dep.py --split rand --split-seed 1 --statistics train --epochs 100 --select last --device cpu --threads 2"
  for seed in $SEEDS; do for feat in rootcost localcost; do for conv in none gatv2; do
    add "$OUT/crossdb_rand1/${feat}_${conv}_L3_true_s$seed" $XR --features $feat --conv $conv --seed $seed
  done; done; done
fi
if want crossdb_rootcum; then  # root cost + cumulative cost: residual vs cumulative at equal whole-plan information
  for seed in $SEEDS; do for conv in none gatv2; do
    add "$OUT/crossdb/rootcum_${conv}_L3_true_s$seed" $X --features rootcum --conv $conv --seed $seed
  done; done
fi
if want crossdb_rootcum_rand1; then
  XR="$PY analysis/dependency_study/run_crossdb_dep.py --split rand --split-seed 1 --statistics train --epochs 100 --select last --device cpu --threads 2"
  for seed in $SEEDS; do for conv in none gatv2; do
    add "$OUT/crossdb_rand1/rootcum_${conv}_L3_true_s$seed" $XR --features rootcum --conv $conv --seed $seed
  done; done
fi
if want crossdb_rootcost_none6; then  # the parameter-matched six-layer bag under the decomposed inputs
  XR="$PY analysis/dependency_study/run_crossdb_dep.py --split rand --split-seed 1 --statistics train --epochs 100 --select last --device cpu --threads 2"
  for seed in $SEEDS; do for feat in rootcost localcost rootcum; do
    add "$OUT/crossdb/${feat}_none_L6_true_s$seed" $X --features $feat --conv none --layers 6 --seed $seed
    add "$OUT/crossdb_rand1/${feat}_none_L6_true_s$seed" $XR --features $feat --conv none --layers 6 --seed $seed
  done; done
fi
if want crossdb_shuf; then  # residuals permuted among the operators of each plan
  XR="$PY analysis/dependency_study/run_crossdb_dep.py --split rand --split-seed 1 --statistics train --epochs 100 --select last --device cpu --threads 2"
  for seed in $SEEDS; do for conv in none gatv2; do
    add "$OUT/crossdb/localcostshuf_${conv}_L3_true_s$seed" $X --features localcostshuf --conv $conv --seed $seed
    add "$OUT/crossdb_rand1/localcostshuf_${conv}_L3_true_s$seed" $XR --features localcostshuf --conv $conv --seed $seed
  done; done
fi
if want crossdb_plansize; then  # every configuration also receives the plan size n
  XR="$PY analysis/dependency_study/run_crossdb_dep.py --split rand --split-seed 1 --statistics train --epochs 100 --select last --device cpu --threads 2"
  for seed in $SEEDS; do for feat in full rootcum localcost; do for conv in none gatv2; do
    add "$OUT/crossdb/${feat}n_${conv}_L3_true_s$seed" $X --features $feat --extra-n --conv $conv --seed $seed
    add "$OUT/crossdb_rand1/${feat}n_${conv}_L3_true_s$seed" $XR --features $feat --extra-n --conv $conv --seed $seed
  done; done; done
fi
if want crossdb_costrows; then  # cost + rows without the operator type: is the type needed once the estimates are present?
  XR="$PY analysis/dependency_study/run_crossdb_dep.py --split rand --split-seed 1 --statistics train --epochs 100 --select last --device cpu --threads 2"
  for seed in $SEEDS; do for conv in none gatv2; do
    add "$OUT/crossdb/costrows_${conv}_L3_true_s$seed" $X --features costrows --conv $conv --seed $seed
    add "$OUT/crossdb_rand1/costrows_${conv}_L3_true_s$seed" $XR --features costrows --conv $conv --seed $seed
  done; done
fi
if want crossdb_ladder; then  # the seven input combinations of type, cost and rows: singles and pairs on both splits
  XR="$PY analysis/dependency_study/run_crossdb_dep.py --split rand --split-seed 1 --statistics train --epochs 100 --select last --device cpu --threads 2"
  for seed in $SEEDS; do for feat in typeonly costonly rowsonly norows costrows nocost full; do for conv in none gatv2; do
    add "$OUT/crossdb/${feat}_${conv}_L3_true_s$seed" $X --features $feat --conv $conv --seed $seed
    add "$OUT/crossdb_rand1/${feat}_${conv}_L3_true_s$seed" $XR --features $feat --conv $conv --seed $seed
  done; done; done
fi
if want crossdb_resid; then  # the residual in place of the cumulative cost, no root cost
  XR="$PY analysis/dependency_study/run_crossdb_dep.py --split rand --split-seed 1 --statistics train --epochs 100 --select last --device cpu --threads 2"
  for seed in $SEEDS; do for feat in resid residrows residonly residtype; do for conv in none gatv2; do
    add "$OUT/crossdb/${feat}_${conv}_L3_true_s$seed" $X --features $feat --conv $conv --seed $seed
    add "$OUT/crossdb_rand1/${feat}_${conv}_L3_true_s$seed" $XR --features $feat --conv $conv --seed $seed
  done; done; done
fi
if want crossdb_resid_encoders; then  # the residual input through the other encoders: GCN and TreeConv, published split
  for seed in $SEEDS; do for conv in gcn tcnn; do
    add "$OUT/crossdb/resid_${conv}_L3_true_s$seed" $X --features resid --conv $conv --seed $seed
  done; done
fi
if want crossdb_norows; then  # type + cost without Plan Rows
  XR="$PY analysis/dependency_study/run_crossdb_dep.py --split rand --split-seed 1 --statistics train --epochs 100 --select last --device cpu --threads 2"
  for seed in $SEEDS; do for conv in none gatv2; do
    add "$OUT/crossdb/norows_${conv}_L3_true_s$seed" $X --features norows --conv $conv --seed $seed
    add "$OUT/crossdb_rand1/norows_${conv}_L3_true_s$seed" $XR --features norows --conv $conv --seed $seed
  done; done
fi
if want crossdb_rand1; then  # robustness: the central contrast on a second database split (not in the default set)
  XR="$PY analysis/dependency_study/run_crossdb_dep.py --split rand --split-seed 1 --statistics train --epochs 100 --select last --device cpu --threads 2"
  for seed in $SEEDS; do for feat in nocost full; do for conv in none gcn gatv2; do
    add "$OUT/crossdb_rand1/${feat}_${conv}_L3_true_s$seed" $XR --features $feat --conv $conv --seed $seed
  done; done; done
  for seed in $SEEDS; do for feat in nocost full; do   # None at the parameter count of GATv2
    add "$OUT/crossdb_rand1/${feat}_none_L6_true_s$seed" $XR --features $feat --conv none --layers 6 --seed $seed
  done; done
fi

if want imdb; then           # two more seeds of the None and GCN controls on the IMDB corpus (seed 42 exists in exp4_ablation)
  for seed in 7 123; do for conv in none gcn; do
    add "$OUT/imdb/${conv}_s$seed" env OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 $PY exp/exp1_qerror/run_gnto.py \
      --set graph.conv=$conv --seed $seed --device cpu
  done; done
fi

echo "$(wc -l < "$LIST") runs missing"
if [[ -n "${DRY:-}" ]]; then cat "$LIST"; rm -f "$LIST"; exit 0; fi
xargs -P "$JOBS" -I{} bash -c '{} || echo "FAILED: {}"' < "$LIST"
rm -f "$LIST"
echo "all done $(date)"
