#!/bin/bash
# Exp2 at a common budget: 100 epochs, final-epoch model (--select last), paper split, seeds 42 / 7 / 123,
# for DACE, the default GNTO and the narrow 16-layer GNTO (w64_L16); run_matrix.sh is the 10-epoch protocol.
# Sequential, idempotent.  Usage: bash exp/exp2_dace/run_e100.sh [extra args]
#   SPLIT=rand1 bash exp/exp2_dace/run_e100.sh     # another 10 / 10 split of the databases (paper | rand1 | rand2)
#   MODELS="dace gnto" SPLIT=rand2 bash ...        # skip the w64_L16 variant
set -u
cd "$(dirname "$0")/../.."
R=results/exp2_dace
EXTRA=("$@")
SPLIT="${SPLIT:-paper}"
MODELS="${MODELS:-dace gnto w64_L16}"
STATS="${STATS:-train}"
STATS_SUFFIX=""
if [ "$STATS" = train ]; then STATS_SUFFIX="_trainstats"; fi
case "$SPLIT" in
  paper) SPLIT_ARGS=(--split paper) ;;
  rand*) SPLIT_ARGS=(--split rand --split-seed "${SPLIT#rand}") ;;      # rand1 -> --split rand --split-seed 1
  *) echo "unknown SPLIT $SPLIT"; exit 1 ;;
esac
mkdir -p "$R"
run() {  # <runner> <dir-suffix> [args...]
  local runner=$1 tag=$2; shift 2
  local model=gnto; [ "$runner" = dace ] && model=dace
  for seed in 42 7 123; do
    local dir="$R/${model}_${SPLIT}_s${seed}${tag:+_$tag}${STATS_SUFFIX}"
    if [ -s "$dir/summary.json" ]; then echo "[skip] $(basename "$dir")"; continue; fi
    echo "[run ] $(basename "$dir")  $(date '+%F %T')"
    python3 "exp/exp2_dace/run_${runner}.py" "${SPLIT_ARGS[@]}" --statistics "$STATS" --seed "$seed" --epochs 100 --select last ${tag:+--tag "$tag"} \
      "$@" "${EXTRA[@]}" > "$dir.stdout" 2>&1 \
      && echo "[ok  ] $(basename "$dir")  $(date '+%F %T')" || echo "[FAIL] $(basename "$dir") (see $dir.stdout)"
  done
}
case " $MODELS " in *" dace "*) run dace e100 ;; esac                                                        # DACE baseline, trained to the end
case " $MODELS " in *" gnto "*) run gnto e100 ;; esac                                                        # GNTO, preset dace_zero_shot
case " $MODELS " in *" w64_L16 "*) run gnto w64_L16_e100 --set graph.heads=1 --set graph.num_layers=16 ;; esac  # narrow 16-layer encoder
case " $MODELS " in *" nognn "*) run gnto nognn_e100 --set graph.conv=none ;; esac                          # no message passing (per-node layers + pooling)
case " $MODELS " in *" gcn "*) run gnto gcn_e100 --set graph.conv=gcn ;; esac                              # graph convolution instead of GATv2
# analyze_e100.py reads the global-statistics run directories (names without _trainstats).
if [ "$STATS" = published ] && [ "$SPLIT" = paper ]; then python3 exp/exp2_dace/analyze_e100.py; fi
echo "E100 DONE ($SPLIT) $(date '+%F %T')"
