#!/usr/bin/env bash
# End-to-end runs of the TLR inputs, CPU, idempotent:
#   online   the online replay (analysis/online/run_online.py, 30 passes, five seeds) with gnto_tlr (type + own cost +
#            rows, ranking objective) and gnto_pgl (PG-stat + own cost, ranking objective), JOB and TPC-H
#   tpch     offline plan selection on TPC-H with the own cost computed with the grouping nodes of ordered-binary-v1
#            resolved (53 of 643 TPC-H plans have an operator with more than two inputs; JOB has none, so JOB is
#            unchanged): residg_* (TLR, analysis/attn/run_kfold_cost.py --cost residg) next to resid_*, and pglg_*
#            (PG-stat + own cost, analysis/tlr/run_kfold_local.py --resolve-groups) next to pgl_*
#   JOBS=12 bash analysis/tlr/run_e2e_tlr.sh        DRY=1 bash analysis/tlr/run_e2e_tlr.sh tpch
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=2 MKL_NUM_THREADS=2 OPENBLAS_NUM_THREADS=2 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-python}
ON=results/analysis/online_20261005
SEL=results/analysis/attn_20261008/selection
PGL=results/analysis/tlr_20261008/selection
DATA=data/e2e/hint_plans_v2
JOBS=${JOBS:-12}
GROUPS_WANTED=${*:-online tpch}
LIST=$(mktemp)
want() { [[ " $GROUPS_WANTED " == *" $1 "* ]]; }
if want online; then
  for wl in job112 tpch52; do for m in gnto_tlr gnto_pgl; do for s in 1 2 3 4 5; do
    [[ -s "$ON/$wl/${m}_s$s/summary.json" ]] && continue
    tag=$([[ $wl == job112 ]] && echo "" || echo "tpch_")
    echo "nice -n 5 $PY analysis/online/run_online.py --model $m --seed $s --passes 30 --threads 2 --hint-plans ${DATA}_$wl > $ON/launch_${tag}${m}_s$s.log 2>&1" >> "$LIST"
  done; done; done
fi
K="--grouped-batches --split-seed 42 --folds 5 --epochs 100 --device cpu --threads 2 --plan-representation ordered-binary-v1"
T="--hint-plans ${DATA}_tpch52 --catalog ${DATA}_tpch52/pg_catalog_snapshot.json"
if want tpch; then
  for seed in 42 7 123; do for rank in 1 0; do
    for conv in gatv2 none gcn tcnn; do
      out="$SEL/tpch52/residg_${conv}_rank${rank}_s$seed"
      [[ -s "$out/summary.json" ]] || echo "nice -n 5 $PY analysis/attn/run_kfold_cost.py --cost residg $T --features dace --set graph.conv=$conv --set graph.heads=1 --rank-weight $rank --seed $seed $K --out $out > $out.log 2>&1" >> "$LIST"
    done
    for conv in gatv2 none tcnn; do
      out="$PGL/tpch52/pglg_${conv}_rank${rank}_s$seed"
      [[ -s "$out/summary.json" ]] || echo "nice -n 5 $PY analysis/tlr/run_kfold_local.py --resolve-groups $T --features pg --set graph.conv=$conv --set graph.heads=1 --rank-weight $rank --seed $seed $K --out $out > $out.log 2>&1" >> "$LIST"
    done
  done; done
fi
echo "$(wc -l < "$LIST") runs missing ($GROUPS_WANTED) $(date +%H:%M:%S)"
if [[ -n "${DRY:-}" ]]; then cat "$LIST"; rm -f "$LIST"; exit 0; fi
xargs -P "$JOBS" -I{} bash -c '{} || echo "FAILED: {}"' < "$LIST"
rm -f "$LIST"
echo "all done ($GROUPS_WANTED) $(date +%H:%M:%S)"
