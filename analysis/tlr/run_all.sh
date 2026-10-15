#!/usr/bin/env bash
# GNTO and DACE with TLR inputs on the three database splits, three seeds each.
# Same protocol as the cross-database runs in results/main_runs/crossdb (100 epochs, last epoch, training statistics,
# CPU, 4 threads); idempotent.   JOBS=7 bash analysis/tlr/run_all.sh        DRY=1 ... lists what is missing
set -u
cd "$(dirname "$0")/../.."
export CUDA_VISIBLE_DEVICES="" OMP_NUM_THREADS=4 MKL_NUM_THREADS=4 OPENBLAS_NUM_THREADS=4 PYTHONDONTWRITEBYTECODE=1
PY=${PY:-python}
OUT=results/analysis/tlr_20261008
JOBS=${JOBS:-7}
SEEDS=${SEEDS:-"42 7 123"}
MODELS=${MODELS:-"dace gnto"}
INPUTS=${INPUTS:-tlr}            # tlr | resid (local cost without the root cost)
SUF=$([[ "$INPUTS" == tlr ]] && echo "" || echo "_$INPUTS")
LIST=$(mktemp)
mkdir -p "$OUT"
for split in paper1 rand1 rand2; do
  case $split in paper1) SA="--split paper" ;; rand1) SA="--split rand --split-seed 1" ;; rand2) SA="--split rand --split-seed 2" ;; esac
  for seed in $SEEDS; do for m in $MODELS; do
    out="$OUT/${m}${SUF}_${split}_s$seed"
    [[ -s "$out/summary.json" ]] && continue
    echo "$PY analysis/tlr/run_tlr.py --model $m --inputs $INPUTS $SA --statistics train --seed $seed --epochs 100 --select last --device cpu --threads 4 --out $out > $out.log 2>&1" >> "$LIST"
  done; done
done
echo "$(wc -l < "$LIST") runs missing"
if [[ -n "${DRY:-}" ]]; then cat "$LIST"; rm -f "$LIST"; exit 0; fi
xargs -P "$JOBS" -I{} bash -c '{} || echo "FAILED: {}"' < "$LIST"
rm -f "$LIST"
echo "all done $(date)"
