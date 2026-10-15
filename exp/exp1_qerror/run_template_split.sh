#!/bin/bash
# Exp1 on the template-disjoint 80 / 10 / 10 split (data/qf/template_split.json): no query template has
# plans in two of train / val / test.  Checkpoint selected on val, reported on the held-out `test` workload.
# Sequential, idempotent.  Usage:
#   bash exp/exp1_qerror/run_template_split.sh              # qf, gnto, gnto_fm x seeds 42 7 123
#   bash exp/exp1_qerror/run_template_split.sh qf           # only these models (lanes can run in parallel)
#   DRY=1 bash exp/exp1_qerror/run_template_split.sh
set -u
cd "$(dirname "$0")/../.."
R=results/exp1_qerror
MODELS="${*:-qf gnto gnto_fm}"
for seed in 42 7 123; do
  for m in $MODELS; do
    sfx=""; [ "$seed" != 42 ] && sfx="_s$seed"
    dir="$R/${m}_tmpl$sfx"
    if [ -s "$dir/summary.json" ]; then echo "[skip] $(basename "$dir")"; continue; fi
    if [ -n "${DRY:-}" ]; then echo "[dry ] $(basename "$dir")"; continue; fi
    echo "[run ] $(basename "$dir")  $(date '+%F %T')"
    case $m in
      qf)      cmd=(python3 exp/exp1_qerror/run_qf.py) ;;
      gnto)    cmd=(python3 exp/exp1_qerror/run_gnto.py) ;;
      gnto_fm) cmd=(python3 exp/exp1_qerror/run_gnto.py --preset gnto_joinid --set node.plan_rows=false) ;;   # QueryFormer's exact input
      *) echo "unknown model $m"; exit 1 ;;
    esac
    if "${cmd[@]}" --split template --seed "$seed" --out "$dir" > "$dir.stdout" 2>&1; then
      echo "[ok  ] $(basename "$dir")  $(date '+%F %T')"
    else
      echo "[FAIL] $(basename "$dir") (see $dir.stdout)"
    fi
  done
done
echo "TEMPLATE SPLIT DONE $(date '+%F %T')"
