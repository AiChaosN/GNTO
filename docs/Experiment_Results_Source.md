# Experiment Results & Source Mapping

This document maps the experimental results to the specific scripts that generated them. Use this as a reference when writing the "Experiments" section of your paper.

## 1. Main Performance Comparison (GNTO vs. Baselines)

GNTO is compared against three baselines, each evaluated under its own task paradigm:

| Baseline | Paradigm | Data | Script (GNTO-side) | Notes |
| :--- | :--- | :--- | :--- | :--- |
| **QueryFormer** | offline plan-prediction (in-distribution) | QF's IMDB 100k / JOB-light (70 queries) / synthetic (500 queries) | `examples/1216_compGntoWithQF_addPlanrows.py` (train) + `examples/0519_eval_real_qf.py` (re-eval real QF ckpt) | Real QF = 4.48M-param transformer, ckpt at `QueryFormer_VLDB2022/results/full/cost/best_model.pt`. GNTO uses QF's FeatureEmbed + GATv2 tree encoder. |
| **DACE** | offline cross-database | DACE workload1 (10 DBs, half train / half test) | `examples/0120_test_dace_workload1.py`, `examples/0121_test_dace_workload1.py` | GNTO uses `adapters/dace_adapter.GNTO_DACE_Model` (NodeEncoder swapped to match DACE feature dim; same tree encoder + head). |
| **LIMAO** | online end-to-end steering | live PG + bao_server framework | `adapters/limao_adapter.py` (GNTO plugs into LIMAO's `bao_server`); workflow in `LIMAOLifeLongRLDB/gnto_ex/workflow.sh`. Q-Error doesn't apply (different scales); use ranking metrics on `gnto_predictions.csv` plus end-to-end latency. | Predictions logged at `LIMAOLifeLongRLDB/gnto_ex/gnto_predictions.csv`; compared via `compare_models.py`. |

### Result Analysis
*   **Comparison Plot:** Use `examples/0202_compare_logs_QFvsGNTO.py` to generate the Q-Error comparison plots between the best GNTO model and the QueryFormer baseline.
*   **Key Metrics:** Q50 / Q90 / Q99 for plan-prediction; Spearman / Kendall / pairwise-acc for ranking; end-to-end runtime for LIMAO steering.

## 2. Ablation Studies (Architecture Choices)

These scripts justify specific design choices (e.g., why GATv2 is better than GAT).

| Experiment Goal | Script Path | Description |
| :--- | :--- | :--- |
| **GAT vs. GATv2** | `examples/0202_compare_logs_GNTO_GAT1vsGAT2.py` | Compares training logs of models using standard `GATTreeEncoder` vs. `GATv2TreeEncoder`. Proves the benefit of dynamic attention. |
| **Ablation Plotting** | `examples/0204_plot_ablation_gnto.py` | Generates visual charts for the ablation studies. |
| **Ablation Runner** | `examples/0204_run_ablation_gnto.py` | Automated script to run multiple ablation configurations in sequence. |

## 3. Workload Specific Tests

| Experiment Goal | Script Path | Description |
| :--- | :--- | :--- |
| **Workload 1 Test** | `examples/0120_test_dace_workload1.py` | Tests the model on a specific DACE workload subset to verify generalization. |

## Summary of Best Results

To reproduce your best reported results in the paper:
1.  Run `examples/1216_compGntoWithQF_addPlanrows.py`.
2.  Wait for training to complete (approx. 100 epochs).
3.  Check the `training_log.csv` in the output directory.
4.  Use the `0202_compare_logs...` scripts to visualize the improvement over the baseline.
