#!/usr/bin/env bash
# The online protocol of run_all.sh on the TPC-H workload (52 queries, 13 templates).
#   bash analysis/online/run_all_tpch.sh "gnto bao gnto_tcr" "1 2 3 4 5"
set -u
cd "$(dirname "$0")/../.."
MODELS=${1:-"gnto bao gnto_tcr"}; SEEDS=${2:-"1 2 3 4 5"}; JOBS=${JOBS:-6}; THREADS=${THREADS:-2}; PASSES=${PASSES:-30}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS PYTHONDONTWRITEBYTECODE=1
OUT=results/analysis/online_20261005
for seed in $SEEDS; do for m in $MODELS; do
  [ -f "$OUT/tpch52/${m}_s$seed/summary.json" ] || echo "$m $seed"
done; done | xargs -P "$JOBS" -L 1 bash -c 'python analysis/online/run_online.py --model "$0" --seed "$1" --passes '"$PASSES"' --threads '"$THREADS"' --hint-plans data/e2e/hint_plans_v2_tpch52 > "'"$OUT"'/launch_tpch_$0_s$1.log" 2>&1'
echo "online tpch queue finished $(date +%H:%M:%S): $MODELS / $SEEDS" >> "$OUT/queue_tpch.log"
