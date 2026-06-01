# GNTO - Graph Neural Tree Optimizer

GNTO is a deep learning-based framework for database query optimization and cost prediction. It focuses on node-level encoding and structure-level modeling of query plans, leveraging Graph Neural Networks (GNNs) to capture complex structural features of query plans for accurate execution time or cost prediction.

## Key Highlights

*   **SOTA Hybrid Encoding Architecture**: Combines data-distribution-based statistical features (QueryFormer-style Histograms) and optimizer prior knowledge (Optimizer Cost Estimates), achieving better performance than purely data-driven methods.
*   **DeepSets Theory Alignment**: The node encoder (V4) strictly follows DeepSets theory, using Sum Pooling to aggregate predicate features and preserving the complete information quantity (Total Filtering Mass).
*   **Dynamic Graph Attention Mechanism**: Introduces GATv2 and Global Attention Pooling, addressing the static attention bottleneck of traditional GAT, and dynamically capturing the influence of critical paths and operators in query plans.
*   **Modular Evolution**: Features a complete model evolution history (V1 → V4 → QF+), supporting flexible switching between different encoder combinations for ablation experiments.

## Quick Start

### 1. Environment Setup

```bash
pip install -r requirements.txt
```

### 2. Train the Model

To reproduce the best results from the paper, run the SOTA training script:

```bash
python examples/1216_compGntoWithQF_addPlanrows.py
```

This script will automatically:
1. Load the QueryFormer-format dataset (including histograms and table samples).
2. Inject `Plan Rows` features.
3. Train the GNTO (QF+ / GATv2) model.
4. Output validation set Q-Error metrics (Median, 90th, 95th, etc.).

**Note**: The training script depends on the [QueryFormer_VLDB2022](https://github.com/...) project for data loading. Ensure it is cloned as a sibling directory or adjust the path in the script.

### 3. Run Ablation Experiments

```bash
python examples/0204_run_ablation_gnto.py
```

### 4. Further Documentation

*   **Model evolution details**: See `docs/Model_Evolution.md`.
*   **Experiment-to-script mapping**: See `docs/Experiment_Results_Source.md`.

---

## Project Structure

```
GNTO/
├── config/                  # Training configuration management
├── data/                    # Dataset files
├── docs/                    # Project documentation
│   ├── Model_Evolution.md          # Detailed model module evolution history
│   └── Experiment_Results_Source.md # Experiment results and script mapping
├── examples/                # Example code and experiment scripts (all use centralized adapters)
│   ├── 1216_compGntoWithQF_addPlanrows.py # [Core] SOTA training (QF+ & GATv2, via qf_adapter)
│   ├── 1216_compGntoWithQF.py             # Comparison without PlanRows (via qf_adapter)
│   ├── 1203_train_qf_standard.py          # Baseline: QF reproduction (via qf_adapter)
│   ├── 0120_test_dace_workload1.py        # DACE cross-DB test, sequential split (via dace_adapter)
│   ├── 0121_test_dace_workload1.py        # DACE cross-DB test, random split (via dace_adapter)
│   ├── 0204_run_ablation_gnto.py          # Ablation experiment automation
│   └── ...
├── models/                  # Core model implementations (single source of truth)
│   ├── NodeEncoder.py       # Node encoder (V4, QF, QF_AddPlanrows)
│   ├── TreeEncoder.py       # Tree encoder (GATv2, GlobalAttention)
│   ├── PredictionHead.py    # Prediction head (ResNet-style V2)
│   └── ...
├── adapters/                # Baseline integration adapters
│   ├── qf_adapter.py       # QueryFormer benchmark adapter (static regression)
│   ├── limao_adapter.py    # LIMAO/Bao end-to-end adapter
│   └── dace_adapter.py     # DACE cross-database benchmark adapter
├── archive/                 # Archived code
├── requirements.txt         # Project dependencies
└── README.md                # Project overview
```

## Core Architecture (Current SOTA)

The best-performing model configuration (implemented in `examples/1216_compGntoWithQF_addPlanrows.py`):

1.  **Node Encoder: `NodeEncoder_QF_AddPlanrows` (Hybrid)**
    *   **Base**: Inherits from QueryFormer, using 150-dim histograms and 1000-dim table samples to capture data distribution.
    *   **Enhancement**: Explicitly injects optimizer-estimated `Plan Rows` as an additional feature channel.
    *   **Advantage**: Combines data-driven fine-grained statistics with the optimizer's global cost estimation capability.

2.  **Tree Encoder: `GATv2TreeEncoder_V3`**
    *   **Mechanism**: Uses **GATv2** (Dynamic Graph Attention) instead of standard GAT.
    *   **Structure**: 3-layer GATv2 + LayerNorm + Residual Connections.
    *   **Aggregation**: Supports Global Attention Pooling (GAP), automatically learning node weights for graph-level aggregation.

3.  **Prediction Head: `PredictionHead_V2`**
    *   **Structure**: ResNet-style deep prediction network.
    *   **Features**: Residual connections and LayerNorm for stronger nonlinear fitting and training stability compared to simple MLPs.

## Baseline Integration Architecture

GNTO uses an **adapter pattern** to integrate with different baselines. All GNTO model code lives exclusively in `models/`, and each baseline is connected through a thin adapter in `adapters/`.

| Baseline | Task Paradigm | Data | Adapter | Integration Pattern |
|----------|---|---|---------|-------------------|
| **QueryFormer** | offline plan prediction | QF's IMDB 100k / JOB-light (70) / synthetic (500) | `qf_adapter.py` | GNTO imports QF's FeatureEmbed (1165-dim) and PlanTreeDataset; converts to PyG; replaces QF's transformer tree-encoder with GATv2 |
| **DACE** | offline cross-database | workload1 (10+ DBs; half train / half test) | `dace_adapter.py` | GNTO loads DACE workload JSON; `GNTO_DACE_Model` swaps NodeEncoder to match DACE features (node-type one-hot + scaled cost/rows), reuses GATv2 + head |
| **LIMAO** | online end-to-end steering | live PG + bao_server framework | `limao_adapter.py` | LIMAO's `bao_server` imports GNTO components; predictions logged to `gnto_predictions.csv`. Workflow: `LIMAOLifeLongRLDB/gnto_ex/workflow.sh`. |

**Key design principle**: GNTO's `models/` is the **single source of truth**. Baseline forks (QueryFormer, LIMAO, DACE) should NOT contain copies of GNTO model code. Instead, they import from this repository via adapters.

Each adapter provides:
- **Data conversion**: Transform baseline-specific data format → PyG `Data` objects
- **Model assembly**: Combine appropriate NodeEncoder + TreeEncoder + PredictionHead for the baseline's benchmark
- **Evaluation utilities**: Baseline-compatible metrics (Q-Error, etc.)

## Experiment Scripts

All experiment scripts are in `examples/`. Clone the baseline repositories first:

```shell
git clone https://github.com/AiChaosN/DACE.git
git clone https://github.com/AiChaosN/QueryFormer_VLDB2022.git
git clone https://github.com/AiChaosN/LIMAOLifeLongRLDB.git
```

### Training experiments

| Script | Description | Baseline | Note |
|--------|-------------|----------|------|
| `1216_compGntoWithQF_addPlanrows.py` | **SOTA** GNTO (QF + PlanRows + GATv2) | QueryFormer | Best config |
| `1216_compGntoWithQF.py` | GNTO without PlanRows | QueryFormer | Ablation: PlanRows effect |
| `0519_eval_real_qf.py` | Re-evaluate real QueryFormer (4.48M-param transformer) ckpt from `QueryFormer_VLDB2022/results/full/cost/` | QueryFormer | Pulls in the actual baseline; no re-training |
| `0519_eval_qf_jobsynth.py` | GNTO + real QF on JOB-light / synthetic datasets | QueryFormer | Adds the missing 2/3 datasets |
| `0120_test_dace_workload1.py` | DACE cross-DB, sequential split (DB 0-9 train / 10-19 test), hidden=128, epochs=15 | DACE | |
| `0121_test_dace_workload1.py` | DACE cross-DB, random split, hidden=64, epochs=10 | DACE | |
| `0204_run_ablation_gnto.py` | Full ablation (Hist/Sample/GNN/Head combinations, 6 configs) | Self | Outputs to `results/Ablation_GNTO_*` |

### Visualization / plotting (run after training)

| Script | Description | Depends on |
|--------|-------------|------------|
| `0202_compare_logs_QFvsGNTO.py` | QF vs GNTO training curve comparison | `1216_compGntoWithQF_addPlanrows.py` results |
| `0202_compare_logs_GNTO_GAT1vsGAT2.py` | GAT vs GATv2 comparison | `1216_*` and `1203_*` results |
| `0204_plot_ablation_gnto.py` | Ablation results visualization | `0204_run_ablation_gnto.py` results |

### LIMAO end-to-end

LIMAO experiments run inside `LIMAOLifeLongRLDB/`, not in GNTO. The Bao server loads GNTO via `adapters/limao_adapter.py`. See the [LIMAOLifeLongRLDB README](https://github.com/AiChaosN/LIMAOLifeLongRLDB) for instructions.

### Reproduce all experiments

```shell
# 1. GNTO vs QueryFormer (plan-level cost prediction)
python examples/1216_compGntoWithQF_addPlanrows.py   # train GNTO SOTA on QF 100k IMDB
python examples/0519_eval_real_qf.py                 # eval the real (transformer) QF ckpt on the same val split
python examples/0519_eval_qf_jobsynth.py             # GNTO + real QF on JOB-light + synthetic
python examples/0202_compare_logs_QFvsGNTO.py        # training-curve plot

# 2. GNTO vs DACE (cross-database generalization)
python examples/0120_test_dace_workload1.py          # sequential split
python examples/0121_test_dace_workload1.py          # random split

# 3. GNTO vs LIMAO (end-to-end via Bao server)
# See LIMAOLifeLongRLDB README

# 4. GAT vs GATv2 ablation
python examples/1216_compGntoWithQF_addPlanrows.py
python examples/1216_compGntoWithQF.py
python examples/1203_train_qf_standard.py
python examples/0202_compare_logs_GNTO_GAT1vsGAT2.py  # plot comparison

# 5. Full ablation study (6 configs x 50 epochs)
python examples/0204_run_ablation_gnto.py
python examples/0204_plot_ablation_gnto.py             # plot results
```

## Experimental Findings Summary

*   **GNTO vs QueryFormer**: With `Plan Rows` and GATv2, GNTO significantly outperforms original QueryFormer on Q-Error (95th/99th) for complex queries.
*   **GAT vs GATv2**: The dynamic attention mechanism (GATv2) performs better when handling long-range path dependencies.
*   **Hybrid Encoding**: Hybrid encoding (QF+PlanRows) demonstrates that optimizer estimates, though imperfect, contain important high-level logical information that effectively assists the neural network.
