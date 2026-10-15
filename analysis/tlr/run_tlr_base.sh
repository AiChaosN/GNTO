#!/usr/bin/env bash
# Runs with the TLR inputs (type + rows + own-cost residual). Three groups, CPU, idempotent:
#   dep     the residual for the remaining encoders (six-layer None / GCN, GAT, GATv2 on a random tree) and the
#           edge-direction / self-loop ablation of GATv2 under type + rows and under TLR (run_crossdb_dep.py --edges)
#   sel     plan selection with TLR inputs (type + rows + residual, analysis/attn/run_kfold_cost.py --cost resid):
#           GATv2 / None / GCN / TreeConv, both objectives, JOB and TPC-H, three seeds
#   subset  GNTO with TLR on the zero-shot model's 50,000 training plans (run_tlr.py --subset-zeroshot), two splits
#   JOBS=8 bash analysis/tlr/run_tlr_base.sh dep        DRY=1 bash analysis/tlr/run_tlr_base.sh sel
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-python}
DEP=results/analysis/dependency_20261003
SEL=results/analysis/attn_20261008/selection
SUB=results/analysis/zs_subset_20261008
DATA=data/e2e/hint_plans_v2
JOBS=${JOBS:-8}
SEEDS=${SEEDS:-"42 7 123"}
GROUPS_WANTED=${*:-dep sel subset}
LIST=$(mktemp)
want() { [[ " $GROUPS_WANTED " == *" $1 "* ]]; }
add() { local out=$1; shift; [[ -s "$out/summary.json" ]] && return; mkdir -p "$(dirname "$out")"; echo "$* --out $out > $out.log 2>&1" >> "$LIST"; }

X="$PY analysis/dependency_study/run_crossdb_dep.py --split paper --split-seed 1 --statistics train --epochs 100 --select last --device cpu --threads 2"
if want dep; then
  for seed in $SEEDS; do
    add "$DEP/crossdb/resid_none_L6_true_s$seed" $X --features resid --conv none --layers 6 --seed $seed
    add "$DEP/crossdb/resid_gcn_L6_true_s$seed" $X --features resid --conv gcn --layers 6 --seed $seed
    add "$DEP/crossdb/resid_gat_L3_true_s$seed" $X --features resid --conv gat --seed $seed
    add "$DEP/crossdb/resid_gatv2_L3_random_s$seed" $X --features resid --conv gatv2 --topology random --seed $seed
    for feat in nocost resid; do
      add "$DEP/crossdb/${feat}_gatv2up_L3_true_s$seed" $X --features $feat --conv gatv2 --edges up --seed $seed
      add "$DEP/crossdb/${feat}_gatv2down_L3_true_s$seed" $X --features $feat --conv gatv2 --edges down --seed $seed
      add "$DEP/crossdb/${feat}_gatv2noself_L3_true_s$seed" $X --features $feat --conv gatv2 --set graph.self_loops=false --seed $seed
    done
  done
fi
K="--grouped-batches --split-seed 42 --folds 5 --epochs 100 --device cpu --threads 2 --plan-representation ordered-binary-v1"
COST=${COST:-resid}              # resid | residslog (the residual signed-log scaled, as PG-stat scales sizes)
if want sel; then
  for wl in job112 tpch52; do for seed in $SEEDS; do for conv in gatv2 none gcn tcnn; do for rank in 1 0; do
    add "$SEL/$wl/${COST}_${conv}_rank${rank}_s$seed" $PY analysis/attn/run_kfold_cost.py --cost $COST \
      --hint-plans ${DATA}_$wl --catalog ${DATA}_$wl/pg_catalog_snapshot.json --features dace \
      --set graph.conv=$conv --set graph.heads=1 --rank-weight $rank --seed $seed $K
  done; done; done; done
fi
if want subset; then
  for sp in "paper 1 paper1" "rand 1 rand1"; do set -- $sp; for seed in $SEEDS; do
    add "$SUB/gnto_resid_${3}_s$seed" nice -n 10 $PY analysis/tlr/run_tlr.py --model gnto --inputs resid --subset-zeroshot \
      --split $1 --split-seed $2 --seed $seed --statistics train --epochs 100 --select last --device cpu --threads 2
  done; done
fi
echo "$(wc -l < "$LIST") runs missing ($GROUPS_WANTED)"
if [[ -n "${DRY:-}" ]]; then cat "$LIST"; rm -f "$LIST"; exit 0; fi
xargs -P "$JOBS" -I{} bash -c '{} || echo "FAILED: {}"' < "$LIST"
rm -f "$LIST"
echo "all done ($GROUPS_WANTED) $(date)"
