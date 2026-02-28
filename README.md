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
├── examples/                # Example code and experiment scripts
│   ├── 1216_compGntoWithQF_addPlanrows.py # [Core] Current SOTA model training script (QF+ & GATv2)
│   ├── 1216_compGntoWithQF.py             # Comparison: version without PlanRows
│   ├── 1203_train_qf_standard.py          # Baseline: QueryFormer reproduction
│   ├── 0204_run_ablation_gnto.py          # Ablation experiment automation
│   └── ...
├── models/                  # Core model implementations
│   ├── NodeEncoder.py       # Node encoder (V4, QF, QF_AddPlanrows)
│   ├── TreeEncoder.py       # Tree encoder (GATv2, GlobalAttention)
│   ├── PredictionHead.py    # Prediction head (ResNet-style V2)
│   └── ...
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

## Experimental Findings Summary

*   **GNTO vs QueryFormer**: With `Plan Rows` and GATv2, GNTO significantly outperforms original QueryFormer on Q-Error (95th/99th) for complex queries.
*   **GAT vs GATv2**: The dynamic attention mechanism (GATv2) performs better when handling long-range path dependencies.
*   **Hybrid Encoding**: Hybrid encoding (QF+PlanRows) demonstrates that optimizer estimates, though imperfect, contain important high-level logical information that effectively assists the neural network.
