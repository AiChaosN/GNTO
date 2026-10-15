#!/usr/bin/env bash
# Online runs: the independent-operator selector (gnto_none) in the loop, and the asynchronous schedule
# (background trainer, staleness instead of waiting) for GNTO and Bao-ext on JOB.
#   JOBS=15 bash analysis/online/run_async_and_none.sh
set -u
cd "$(dirname "$0")/../.."
JOBS=${JOBS:-15}; THREADS=${THREADS:-2}; PASSES=${PASSES:-30}; SEEDS=${SEEDS:-"1 2 3 4 5"}
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=$THREADS MKL_NUM_THREADS=$THREADS PYTHONDONTWRITEBYTECODE=1
OUT=results/analysis/online_20261005
JOB=data/e2e/hint_plans_v2_job112; TPCH=data/e2e/hint_plans_v2_tpch52
LIST=$(mktemp)
add() { local dir=$1 name=$2; shift 2; [ -f "$OUT/$dir/summary.json" ] || echo "python analysis/online/run_online.py $* --passes $PASSES --threads $THREADS > $OUT/launch_$name.log 2>&1" >> "$LIST"; }
for s in $SEEDS; do add "job112/gnto_s${s}_async" "gnto_async_s$s" --model gnto --seed $s --schedule async --hint-plans $JOB; done
for s in $SEEDS; do add "job112/bao_s${s}_async" "bao_async_s$s" --model bao --seed $s --schedule async --hint-plans $JOB; done
for s in $SEEDS; do add "job112/gnto_none_s$s" "gnto_none_s$s" --model gnto_none --seed $s --hint-plans $JOB; done
for s in $SEEDS; do add "tpch52/gnto_none_s$s" "tpch_gnto_none_s$s" --model gnto_none --seed $s --hint-plans $TPCH; done
echo "$(wc -l < "$LIST") runs to launch $(date +%H:%M:%S)" | tee -a "$OUT/queue_async_and_none.log"
xargs -P "$JOBS" -I{} bash -c '{} || echo "FAILED: {}"' < "$LIST" | tee -a "$OUT/queue_async_and_none.log"
rm -f "$LIST"
echo "online async and gnto_none queue finished $(date +%H:%M:%S)" | tee -a "$OUT/queue_async_and_none.log"
