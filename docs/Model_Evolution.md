# GNTO Model Framework Evolution

This document details the architectural evolution of the GNTO (Graph Neural Tree Optimizer) framework components. The model is modular, consisting of three main parts: **Node Encoder**, **Tree Encoder**, and **Prediction Head**.

## 1. Node Encoder Evolution

The Node Encoder is responsible for converting raw query plan node attributes (operations, predicates, statistics) into a fixed-size vector.

| Version | Class Name | Key Features & Changes |
| :--- | :--- | :--- |
| **V1** | `NodeEncoder_V1` | **Baseline.** Basic embedding lookup for node types. Predicates are processed via mean pooling. |
| **V2** | `NodeEncoder_V2` | **Enhanced MLP.** Added a deeper MLP for numerical features, Batch Normalization, and Dropout for better regularization. |
| **V3** | `NodeEncoder_V3` | **Complex Predicates.** Designed for richer predicate structures, handling specific `op`, `lhs`, `rhs` components with residual connections. |
| **V4** | `NodeEncoder_V4` | **DeepSets Alignment.** Key architectural shift. Changed predicate aggregation from **Mean** to **Sum** to preserve "Total Filtering Mass," aligning with DeepSets theory for set-based features. |
| **QF** | `NodeEncoder_QF` | **QueryFormer Port.** Ported from VLDB'22 QueryFormer. Uses sophisticated **Histograms** (150-dim) and **Table Samples** (1000-dim) to learn data distributions directly. |
| **QF+** | `NodeEncoder_QF_AddPlanrows` | **Hybrid SOTA.** The current best version. Extends `NodeEncoder_QF` by explicitly injecting the optimizer's estimated `plan_rows` as an additional feature channel. This combines data-driven statistics with optimizer domain knowledge. |

## 2. Tree Encoder Evolution

The Tree Encoder aggregates node information across the query plan structure to form a final plan embedding.

| Version | Class Name | Key Features & Changes |
| :--- | :--- | :--- |
| **Mini** | `TreeEncoder_GATMini` | **Lightweight.** A simple 2-layer GAT (Graph Attention Network) with standard concatenation. |
| **V1** | `GATTreeEncoder` | **Standard GAT.** 2-layer GAT with Residual connections and LayerNorm. Configurable pooling (Mean/Max/Sum). |
| **V2** | `GATTreeEncoder_V2` | **Deep GAT.** Increased to **3 layers**. Added **Jumping Knowledge (JK)** aggregation (combining features from all layers) and Edge Dropout for robustness. |
| **V3** | `GATv2TreeEncoder_V3` | **Dynamic Attention.** Switched from standard `GATConv` to **`GATv2Conv`**. GATv2 fixes the static attention problem of GAT, allowing for more expressive dynamic attention weights. |
| **V4** | `GATv2TreeEncoder_V4` | **Global Attention Pooling.** Replaces simple Mean/Max pooling with **Global Attention Pooling (GAP)**, where the model learns a gating function to weigh the importance of each node in the final graph embedding. |

## 3. Prediction Head Evolution

The Prediction Head maps the final plan embedding to the estimated cost/latency.

| Version | Class Name | Key Features & Changes |
| :--- | :--- | :--- |
| **Mini** | `PredictionHead_FNNMini` | **Simple FNN.** A standard Feed-Forward Network (Linear -> ReLU -> Dropout -> Linear). |
| **V2** | `PredictionHead_V2` | **ResNet Style.** Uses **Residual Blocks** (ResBlock) with LayerNorm and GELU activation. This deeper structure allows for better gradient flow and representation learning. |

---

## Current SOTA Configuration

The best-performing configuration (as of Feb 2026) is the **Hybrid GNTO** model:

*   **Node Encoder:** `NodeEncoder_QF_AddPlanrows` (Rich stats + Optimizer estimates)
*   **Tree Encoder:** `GATv2TreeEncoder_V3` (Dynamic GATv2 attention)
*   **Prediction Head:** `PredictionHead_V2` (Deep Residual Network)

This configuration is implemented in `examples/1216_compGntoWithQF_addPlanrows.py`.
