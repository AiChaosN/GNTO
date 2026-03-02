# Experiment Results & Source Mapping

This document maps the experimental results to the specific scripts that generated them. Use this as a reference when writing the "Experiments" section of your paper.

## 1. Main Performance Comparison (GNTO vs. Baselines)

These scripts generate the core results proving GNTO's effectiveness against QueryFormer.

| Experiment Goal | Script Path | Model Configuration | Output / Log File |
| :--- | :--- | :--- | :--- |
| **Best GNTO Model** | `examples/1216_compGntoWithQF_addPlanrows.py` | **Node:** QF_AddPlanrows<br>**Tree:** GATv2<br>**Head:** V2 | `../results/GNTO_QF_[timestamp]/` |
| **GNTO (No PlanRows)** | `examples/1216_compGntoWithQF.py` | **Node:** QF (Standard)<br>**Tree:** GATv2<br>**Head:** V2 | `../results/GNTO_QF_[timestamp]/` |
| **QueryFormer (Baseline)** | `examples/1203_train_qf_standard.py` | **Node:** QF<br>**Tree:** Transformer (Native)<br>**Head:** MLP | `../results/QF_Standard_[timestamp]/` |

### Result Analysis
*   **Comparison Plot:** Use `examples/0202_compare_logs_QFvsGNTO.py` to generate the Q-Error comparison plots between the best GNTO model and the QueryFormer baseline.
*   **Key Metric:** Look for `Val Q90` and `Val Q95` in the training logs.

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
