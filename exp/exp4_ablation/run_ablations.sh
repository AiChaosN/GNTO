#!/bin/bash
# Exp4 architecture ablations of the GNTO graph encoder.  Training protocol = Exp1's runner
# (exp/exp1_qerror/run_gnto.py: parts 0-17 train / 18-19 val, 100 epochs, seed 42 unless a group says
# otherwise); only the output directory differs - everything lands in results/exp4_ablation/.
# Sequential, idempotent (a run whose summary.json exists is skipped).
#
#   bash exp/exp4_ablation/run_ablations.sh            # every group
#   bash exp/exp4_ablation/run_ablations.sh depth res  # only these groups
#   DRY=1 bash exp/exp4_ablation/run_ablations.sh    # list what is missing, run nothing
#   groups: encoder edges norm treeattn depth res width seeds joins features backbone shallow qfseeds
#
# Tables / figures afterwards:
#   python exp/exp4_ablation/analyze.py              # residual x depth x width tables
#   python exp/exp4_ablation/plot.py                 # depth curves
#   python exp/exp1_qerror/plot.py --results results/exp4_ablation --models gnto_gcn,gnto_tcnn,gnto --name ablation_encoder
set -u
cd "$(dirname "$0")/../.."
R=results/exp4_ablation
mkdir -p "$R"

run() {  # <tag> [runner args...]
  local tag=$1; shift
  local dir="$R/gnto_$tag"
  if [ -s "$dir/summary.json" ]; then echo "[skip] gnto_$tag"; return 0; fi
  if [ -n "${DRY:-}" ]; then echo "[dry ] gnto_$tag  run_gnto.py --tag $tag $*"; return 0; fi
  echo "[run ] gnto_$tag  $(date '+%F %T')"
  if python3 exp/exp1_qerror/run_gnto.py --out "$dir" --tag "$tag" "$@" > "$R/gnto_$tag.stdout" 2>&1; then
    echo "[ok  ] gnto_$tag  $(date '+%F %T')"
  else
    echo "[FAIL] gnto_$tag (see $R/gnto_$tag.stdout)"
  fi
}
# NB: do not name this variable GROUPS - bash keeps the caller's group ids there.
ONLY="${*:-}"
want() { [ -z "$ONLY" ] && return 0; case " $ONLY " in *" $1 "*) return 0;; *) return 1;; esac; }

# --- message passing: which convolution -------------------------------------
if want encoder; then
  run gcn               --set graph.conv=gcn
  run tcnn              --set graph.conv=tcnn
fi
# --- edge orientation --------------------------------------------------------
if want edges; then
  run bidir             --set graph.edge_mode=bidirectional
  run reverse           --set graph.edge_mode=reverse
  run gcn_bidir         --set graph.conv=gcn --set graph.edge_mode=bidirectional
fi
# --- attention normalisation and readout ------------------------------------
if want norm; then
  run nonorm            --set graph.conv=gatv2_nonorm
  run sumpool           --set readout.pooling=sum
  run nonorm_sum        --set graph.conv=gatv2_nonorm --set readout.pooling=sum
fi
# --- tree-aware attention ----------------------------------------------------
if want treeattn; then
  run tcnn_attn         --set graph.conv=tcnn_attn
  run tcnn_attn_softmax --set graph.conv=tcnn_attn_softmax
  run gatv2_pos         --set graph.conv=gatv2_pos
fi
# --- depth at the default width (heads 4) -----------------------------------
if want depth; then
  run L5                --set graph.num_layers=5  --set graph.heads=4
  run L7                --set graph.num_layers=7  --set graph.heads=4
  run gcn_L5            --set graph.conv=gcn  --set graph.num_layers=5 --set graph.heads=4
  run gcn_L7            --set graph.conv=gcn  --set graph.num_layers=7 --set graph.heads=4
  run tcnn_L5           --set graph.conv=tcnn --set graph.num_layers=5 --set graph.heads=4
  run tcnn_L7           --set graph.conv=tcnn --set graph.num_layers=7 --set graph.heads=4
fi
# --- residual x depth at constant width 64 (heads=1 -> identity residual) ----
if want res; then
  run nores             --set graph.residual=false
  for L in 3 5 7 9 12 16 24; do
    run "w64_L$L"       --set graph.heads=1 --set graph.num_layers=$L
  done
  for L in 3 5 7 9 24; do
    run "w64_L${L}_nores" --set graph.heads=1 --set graph.num_layers=$L --set graph.residual=false
  done
fi
# --- narrower and deeper -----------------------------------------------------
if want width; then
  for L in 24 32 48; do
    run "w32_L$L"       --set graph.heads=1 --set graph.num_layers=$L --set graph.hidden_dim=32 --set graph.out_dim=32
  done
  run w16_L64           --set graph.heads=1 --set graph.num_layers=64 --set graph.hidden_dim=16 --set graph.out_dim=16
fi
# --- seeds 7 / 123 for the default and the w64 L5 (with / without residual) configurations ---
if want seeds; then
  for s in 7 123; do
    run "s$s"                  --seed $s
    run "w64_L5_s$s"           --seed $s --set graph.heads=1 --set graph.num_layers=5
    run "w64_L5_nores_s$s"     --seed $s --set graph.heads=1 --set graph.num_layers=5 --set graph.residual=false
  done
fi
# --- join encoding (string id vs. compositional), incl. held-out join pairs --
if want joins; then
  J="mi.movie_id = mi_idx.movie_id,ci.movie_id = mc.movie_id,mi_idx.movie_id = mk.movie_id"
  [ -s "$R/gnto_joinid/summary.json" ] || python3 exp/exp1_qerror/run_gnto.py --preset gnto_joinid --out "$R/gnto_joinid"
  [ -s "$R/gnto_holdout/summary.json" ] || python3 exp/exp1_qerror/run_gnto.py --exclude-joins "$J" --out "$R/gnto_holdout"
  [ -s "$R/gnto_joinid_holdout/summary.json" ] || python3 exp/exp1_qerror/run_gnto.py --preset gnto_joinid --exclude-joins "$J" --out "$R/gnto_joinid_holdout"
fi
# --- input features: what GNTO sees beyond QueryFormer's input ---
# fm = feature-matched: exactly QueryFormer's 1165-d node features (no Plan Rows channel, join as string id).
# gnto_joinid (group joins) = fm + Plan Rows; gnto = + compositional join columns.
if want features; then
  run fm                --preset gnto_joinid --set node.plan_rows=false
  run fm_s7   --seed 7   --preset gnto_joinid --set node.plan_rows=false
  run fm_s123 --seed 123 --preset gnto_joinid --set node.plan_rows=false
  run nostats           --set node.use_hist=false --set node.use_sample=false
  run fm_nostats        --preset gnto_joinid --set node.plan_rows=false --set node.use_hist=false --set node.use_sample=false
fi
# --- backbone: no message passing, GAT (v1) - GCN / tree conv / GATv2 are in group encoder ---
if want backbone; then
  run gat               --set graph.conv=gat
  run nognn             --set graph.conv=none
fi
# --- fewer layers than the default 3 (receptive field 1 and 2 hops) ---
if want shallow; then
  run L1                --set graph.num_layers=1 --set graph.heads=4
  run L2                --set graph.num_layers=2 --set graph.heads=4
fi
# --- QueryFormer seeds (Exp1 baseline; written next to results/exp1_qerror/qf) ---
if want qfseeds; then
  for s in 7 123; do
    d=results/exp1_qerror/qf_s$s
    if [ -s "$d/summary.json" ]; then echo "[skip] qf_s$s"; continue; fi
    if [ -n "${DRY:-}" ]; then echo "[dry ] qf_s$s"; continue; fi
    echo "[run ] qf_s$s  $(date '+%F %T')"
    python3 exp/exp1_qerror/run_qf.py --seed $s --out "$d" > "results/exp1_qerror/qf_s$s.stdout" 2>&1 && echo "[ok  ] qf_s$s  $(date '+%F %T')" || echo "[FAIL] qf_s$s"
    ln -sfn "../exp1_qerror/qf_s$s" "$R/qf_s$s"
  done
fi
echo "ABLATIONS DONE $(date '+%F %T')"
