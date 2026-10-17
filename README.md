# GNTO

Code and experiment results for GNTO, a graph neural network cost model for PostgreSQL query plans, and for the
baselines it is run against (QueryFormer, DACE, Bao, LIMAO, Reqo / BiGG, zero-shot cost estimation).

## Repository layout

| Path | Contents |
|---|---|
| `models/gnto/` | GNTO: node encoder, graph encoder (GATv2 / GAT / GCN / tree convolution / none), readout, head; presets and `--set` overrides in `config.py` |
| `models/qf/` | QueryFormer, ported from the released code |
| `models/dace/` | DACE, ported from the released code, and its node featuriser |
| `adapters/` | plan featurisation shared by the runners (QueryFormer features, PostgreSQL-statistics features, plan representation) and the Bao loader |
| `exp/exp1_qerror/` | Exp1: Q-error on the QueryFormer IMDb plan corpus |
| `exp/exp2_dace/` | Exp2: cross-database estimation on DACE's 20-database workload |
| `exp/exp3_e2e/` | Exp3: plan selection over 49 hint sets on JOB and TPC-H (hint-plan collection, steerers, k-fold training) |
| `exp/exp4_ablation/` | Exp4: architecture ablations, trained with the Exp1 runner |
| `analysis/` | further studies: `dependency_study/`, `attn/`, `baselines/` (Reqo, BiGG, zero-shot), `online/` (online plan selection), `tlr/` |
| `scripts/` | config-driven GNTO training and evaluation on the QueryFormer corpus |
| `tests/` | unit tests of the experiment protocol |
| `data/` | small inputs; large inputs are obtained as described under [Data](#data) |
| `results/` | experiment outputs, see [Results](#results) |

All commands below are run from the repository root.

## Environment

Python 3.13 (results produced with 3.13.11). Install the PyTorch build for your platform first, then

```bash
pip install -r requirements-lock.txt   # exact versions used for the results
# or
pip install -r requirements.txt        # minimum versions
```

The Exp1 / Exp2 / Exp4 runners use a GPU when one is available (`--device cuda|cpu`); the plan-selection runners
and the studies under `analysis/` run on CPU. PostgreSQL is only needed to collect hint plans.

### External code

Clone next to this repository (sibling directories):

| Directory | Repository | Used by |
|---|---|---|
| `../BaoForPostgreSQL` | https://github.com/learnedsystems/BaoForPostgreSQL | Bao (`adapters/bao_adapter.py`) |
| `../LIMAOLifeLongRLDB` | https://github.com/Tsihan/LIMAOLifeLongRLDB, branch `LIMAO_Bao` | LIMAO (`exp/exp3_e2e/limao_worker.py`) |
| `../baselines_src/Reqo-on-PostgreSQL` | https://github.com/BaomingChang/Reqo-on-PostgreSQL | Reqo / BiGG (`analysis/baselines/`) |
| `../baselines_src/zero-shot-cost-estimation` | https://github.com/DataManagementLab/zero-shot-cost-estimation | zero-shot model (`analysis/baselines/run_zeroshot.py`) |

LIMAO loads `kproto_processor.pkl` from an absolute path; point it to the file at the root of the checkout:

```bash
sed -i 's#load_from_disk("[^"]*kproto_processor.pkl")#load_from_disk(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "kproto_processor.pkl"))#' \
    ../LIMAOLifeLongRLDB/bao_server/model.py
```

The zero-shot model runs in its own environment (Python 3.8 and its `requirements.txt`, as in its README); set
`ZS_PYTHON` to that interpreter. It trains on the parsed plans of its data repository
(https://osf.io/ga2xj/?view_only=60916f2446e844edb32e3b4d676cb6ee, `runs/parsed_plans.zip`), unzipped to
`../baselines_src/zero-shot-data/parsed_plans/` and linked per database:

```bash
mkdir -p ../baselines_src/zs_runs/plans
for db in $(python -c "from models.dace import WORKLOADS; print(*WORKLOADS)"); do
  src=$db; [ "$db" = imdb_full ] && src=imdb
  ln -s "../../zero-shot-data/parsed_plans/$src/workload_100k_s1_c8220.json" "../baselines_src/zs_runs/plans/$db.json"
done
```

## Data

### Included

| File | Contents |
|---|---|
| `data/e2e/queries_job112.csv` | 112 IMDb queries (JOB and two CEB queries) with template ids |
| `data/e2e/queries_tpch52.csv` | 52 TPC-H queries (13 templates); `queries_tpch52.sources.json` holds the hashes of the source SQL files |
| `data/e2e/pg_catalog_snapshot.json` | pg_class / pg_stats / settings snapshot of the IMDb database (default `--catalog` of `exp/exp3_e2e/run_kfold.py`) |
| `data/qf/template_split.json` | template-disjoint 80 / 10 / 10 split of the QueryFormer corpus (`--split template`) |

### QueryFormer plan corpus

Exp1, Exp4, `scripts/`, and the Bao / GNTO models of Exp3 that are trained on the corpus use the data released with
QueryFormer (https://github.com/zhaoyue-ntu/QueryFormer):

```bash
git clone https://github.com/zhaoyue-ntu/QueryFormer ../QueryFormer
cp ../QueryFormer/data/imdb/plan_and_cost/train_plan_part*.csv data/
cp ../QueryFormer/data/imdb/{job-light_plan,synthetic_plan,column_min_max_vals}.csv data/
mkdir -p data/qf
cp -r ../QueryFormer/data/imdb/{histogram_string.csv,train.csv,train.bitmaps,workloads} data/qf/
cp ../QueryFormer/checkpoints/{encoding.pt,cost_model.pt} data/qf/
```

### DACE workload

Exp2 and the cross-database runs read `data/dace/workload1/` (`<database>_filted.json` for 20 databases and
`statistics.json`). The files are built from the parsed plans of the zero-shot cost estimation benchmark
(https://osf.io/rb5tn/, `runs.zip` -> `runs/parsed_plans/<database>/workload_100k_s1_c8220.json`; `imdb_full` is
taken from `imdb`) in the format DACE (https://github.com/liang-zibo/DACE) reads: every plan node becomes a
PostgreSQL EXPLAIN node (`op_name` -> `Node Type`, `est_cost` -> `Total Cost`, `est_card` -> `Plan Rows`,
`act_time` -> `Actual Total Time`, `children` -> `Plans`), each plan is wrapped as `[[[{"Plan": ...}]]]`, plans
with a missing field or more than 50 nodes are dropped, and DACE's `setup.filter_plans()` and
`setup.get_statistic()` write the `_filted.json` files and `statistics.json`.

### Hint plans

The plan-selection experiments read one JSON file per query with the plans and measured latencies of the 49 Bao
hint sets. The packs are collected with `exp/exp3_e2e/collect_benchmark.py` (read-only statements, libpq environment
variables for the connection):

```bash
PGDATABASE=imdbload python exp/exp3_e2e/collect_benchmark.py --queries data/e2e/queries_job112.csv --out data/e2e/hint_plans_v2_job112
PGDATABASE=tpch_sf1 python exp/exp3_e2e/collect_benchmark.py --queries data/e2e/queries_tpch52.csv --out data/e2e/hint_plans_v2_tpch52
```

The packs behind the results were collected on PostgreSQL 16.9 (the IMDb database of the Join Order Benchmark,
https://github.com/gregrahn/join-order-benchmark, and TPC-H at scale factor 1) with the default protocol: statement
timeout 32 s, one warm-up and three timed executions per plan in randomized order, seed 42. Each pack also holds
`collection_manifest.json` and the `pg_catalog_snapshot.json` of the database it was collected on.

The runs in `results/exp3_e2e/` use an earlier pack, `data/e2e/hint_plans_job112/`, with one execution per hint set
of the same 112 queries on PostgreSQL 16 (per query: `query_id`, `sql`, `results` = list of `hint_id`,
`total_cost`, `actual_exec_time_ms`, `wall_clock_ms`, `plan`). It was not produced by `collect_benchmark.py`.

Feature caches are written to `data/process/cache/` on first use.

## Running the experiments

Every run directory under `results/` has a `config.json` with the complete arguments of the runner that wrote it,
so any single run can be repeated from it. The shell scripts are idempotent: runs whose `summary.json` exists are
skipped, `DRY=1` lists the missing runs, and `JOBS=<n>` (where supported) sets the number of parallel runs.

### Exp1: Q-error on the QueryFormer corpus

```bash
python exp/exp1_qerror/run_qf.py                                         # -> results/exp1_qerror/qf
python exp/exp1_qerror/run_gnto.py                                       # -> results/exp1_qerror/gnto
python exp/exp1_qerror/run_gnto.py --seed 7 --out results/exp1_qerror/gnto_s7   # same for qf and seed 123
bash exp/exp1_qerror/run_template_split.sh                               # template-disjoint split -> *_tmpl*
python exp/exp1_qerror/analyze.py                                        # -> results/exp1_qerror/tables/
python exp/exp1_qerror/plot.py                                           # -> results/exp1_qerror/figures/
python exp/exp1_qerror/bench_latency.py                                  # inference latency of the trained models
```

### Exp2: cross-database estimation

```bash
bash exp/exp2_dace/run_matrix.sh          # DACE and GNTO, splits paper / rand1 / rand2 x seeds 42 / 7 / 123
bash exp/exp2_dace/run_e100.sh            # 100 epochs, final-epoch model; SPLIT=rand1|rand2 for the other splits
python exp/exp2_dace/analyze.py
python exp/exp2_dace/analyze_e100.py
python exp/exp2_dace/plot.py
```

`paper` is a fixed 10 / 10 split of the 20 databases (listed in `models/dace/resources.py`); `rand1` / `rand2` are
seeded random 10 / 10 splits.

### Exp3: plan selection

```bash
python exp/exp3_e2e/run_baselines.py      # PostgreSQL default plan, PostgreSQL cost, oracle
python exp/exp3_e2e/run_bao.py            # Bao trained on the QueryFormer corpus
python exp/exp3_e2e/run_gnto.py           # GNTO trained in Exp1 (results/exp1_qerror/gnto/best_model.pt)
python exp/exp3_e2e/run_kfold.py --help   # models trained on the hint plans, k folds by query or template
python exp/exp3_e2e/analyze.py            # -> results/exp3_e2e/tables/
python exp/exp3_e2e/plot.py
```

### Main plan-selection, cross-database and capacity runs (`results/main_runs/`)

`job112/` and `tpch52/` are `exp/exp3_e2e/run_kfold.py` on the v2 hint plans, five template folds, seeds 42 / 7 /
123. Directory names are `<features>_<encoder>_rank<0|1>_s<seed>` (GNTO with DACE or PostgreSQL-statistics inputs),
`bao_extended_capped_s<seed>`, `limao_extended_capped_s<seed>`; `references/` holds the PostgreSQL and oracle
choices. Example:

```bash
python exp/exp3_e2e/run_kfold.py --model gnto --hint-plans data/e2e/hint_plans_v2_job112 \
    --features pg --catalog data/e2e/hint_plans_v2_job112/pg_catalog_snapshot.json \
    --set graph.conv=gatv2 --set graph.heads=1 --grouped-batches --rank-weight 1 \
    --group template --folds 5 --split-seed 42 --seed 42 --epochs 100 \
    --plan-representation ordered-binary-v1 --device cpu --threads 2 \
    --out results/main_runs/job112/pg_gatv2_rank1_s42
```

`crossdb/` are the Exp2 runners with `--epochs 100 --select last --statistics train` on the three database splits
and three seeds, e.g.

```bash
python exp/exp2_dace/run_gnto.py --split paper --split-seed 1 --statistics train --epochs 100 --select last \
    --seed 42 --device cpu --threads 4 --out results/main_runs/crossdb/gnto_paper1_s42_trainstats
python exp/exp2_dace/run_dace.py --split rand --split-seed 1 --statistics train --epochs 100 --select last \
    --seed 42 --device cpu --threads 4 --out results/main_runs/crossdb/dace_rand1_s42_trainstats
```

`capacity/` is QueryFormer at reduced width on the template split:

```bash
python exp/exp1_qerror/run_qf.py --emb-size 24 --layers 3 --heads 4 --ffn-dim 64 --pred-hid 64 --split template \
    --seed 42 --device cpu --threads 4 --out results/main_runs/capacity/qf_small_s42
```

`results/main_runs_stats/` holds per-seed and paired tables computed from these runs.

### Exp4: architecture ablations

```bash
bash exp/exp4_ablation/run_ablations.sh             # all groups; or name groups, e.g. `depth res`
python exp/exp4_ablation/analyze.py
python exp/exp4_ablation/plot.py
```

`results/exp4_ablation/{gnto,gnto_s7,gnto_s123,qf,qf_s7,qf_s123}` are links to the Exp1 runs.

### Studies under `analysis/`

| Command | Output |
|---|---|
| `bash analysis/dependency_study/run_all.sh` | `results/analysis/dependency_20261003/` |
| `bash analysis/attn/run_all.sh` | `results/analysis/attn_20261008/` |
| `bash analysis/baselines/run_all.sh`, `run_subset_all.sh`, `zeroshot_more.sh` | `results/analysis/dependency_20261003/`, `results/analysis/zs_subset_20261008/` |
| `bash analysis/online/run_all.sh`, `run_all_tpch.sh`, `run_async_and_none.sh` | `results/analysis/online_20261005/` |
| `bash analysis/tlr/run_all.sh`, `run_selection.sh`, `run_tlr_base.sh` | `results/analysis/tlr_20261008/` |
| `bash analysis/tlr/run_e2e_tlr.sh` (TLR inputs, online and TPC-H plan selection) | `results/analysis/online_20261005/`, `results/analysis/{attn,tlr}_20261008/selection/` |

The header of each script lists its run groups. The Python scripts next to them (`analyze_*.py`, `report.py`, ...)
build the tables from the run directories.

## Results

```
results/<experiment>/<run>/
    config.json          runner arguments and model configuration
    summary.json         metrics of the run
    training_log*.csv    per-epoch training / validation metrics
    predictions*.csv     per-plan predictions (where small)
    picks.csv            chosen hint set per query (plan selection)
    folds.json           query / template folds (plan selection)
results/<experiment>/tables/, figures/   aggregated tables and plots
```

Not included: model weights, raw logs, and the per-plan prediction files of the larger runs; rerun the
corresponding command to recreate them.

## Tests

```bash
python -m pytest tests
```
