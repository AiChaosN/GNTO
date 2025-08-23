"""
Evaluation metrics for query optimization tasks
"""

import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple, Union
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from scipy.stats import spearmanr, kendalltau
import logging


logger = logging.getLogger(__name__)


def compute_regression_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task_name: str = "regression"
) -> Dict[str, float]:
    """
    Compute regression metrics
    
    Args:
        predictions: Model predictions [N, 1] or [N]
        targets: Ground truth targets [N, 1] or [N]
        task_name: Name of the task for logging
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy and flatten
    pred_np = predictions.detach().cpu().numpy().flatten()
    target_np = targets.detach().cpu().numpy().flatten()
    
    # Remove any NaN or inf values
    valid_mask = np.isfinite(pred_np) & np.isfinite(target_np)
    pred_np = pred_np[valid_mask]
    target_np = target_np[valid_mask]
    
    if len(pred_np) == 0:
        logger.warning(f"No valid predictions for {task_name}")
        return {f"{task_name}_mse": float('inf')}
    
    metrics = {}
    
    # Basic regression metrics
    metrics[f"{task_name}_mse"] = mean_squared_error(target_np, pred_np)
    metrics[f"{task_name}_rmse"] = np.sqrt(metrics[f"{task_name}_mse"])
    metrics[f"{task_name}_mae"] = mean_absolute_error(target_np, pred_np)
    
    # R-squared
    try:
        metrics[f"{task_name}_r2"] = r2_score(target_np, pred_np)
    except:
        metrics[f"{task_name}_r2"] = -float('inf')
    
    # Mean Absolute Percentage Error (MAPE)
    non_zero_mask = target_np != 0
    if non_zero_mask.sum() > 0:
        mape = np.mean(np.abs((target_np[non_zero_mask] - pred_np[non_zero_mask]) / target_np[non_zero_mask])) * 100
        metrics[f"{task_name}_mape"] = mape
    
    # Q-error (for cost/cardinality estimation)
    if task_name in ["cost", "cardinality"]:
        # Ensure positive values for Q-error
        pred_pos = np.maximum(pred_np, 1e-8)
        target_pos = np.maximum(target_np, 1e-8)
        
        q_errors = np.maximum(pred_pos / target_pos, target_pos / pred_pos)
        metrics[f"{task_name}_q_error_mean"] = np.mean(q_errors)
        metrics[f"{task_name}_q_error_median"] = np.median(q_errors)
        metrics[f"{task_name}_q_error_95th"] = np.percentile(q_errors, 95)
        
        # Percentage of predictions within 2x, 5x, 10x
        metrics[f"{task_name}_within_2x"] = np.mean(q_errors <= 2.0) * 100
        metrics[f"{task_name}_within_5x"] = np.mean(q_errors <= 5.0) * 100
        metrics[f"{task_name}_within_10x"] = np.mean(q_errors <= 10.0) * 100
    
    return metrics


def compute_ranking_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    k_values: List[int] = [1, 5, 10],
    task_name: str = "ranking"
) -> Dict[str, float]:
    """
    Compute ranking metrics
    
    Args:
        predictions: Model predictions [N] (higher is better)
        targets: Ground truth rankings/scores [N] (higher is better)
        k_values: Values of k for top-k metrics
        task_name: Name of the task for logging
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    pred_np = predictions.detach().cpu().numpy().flatten()
    target_np = targets.detach().cpu().numpy().flatten()
    
    # Remove any NaN or inf values
    valid_mask = np.isfinite(pred_np) & np.isfinite(target_np)
    pred_np = pred_np[valid_mask]
    target_np = target_np[valid_mask]
    
    if len(pred_np) == 0:
        logger.warning(f"No valid predictions for {task_name}")
        return {f"{task_name}_spearman": 0.0}
    
    metrics = {}
    
    # Rank correlation metrics
    try:
        spearman_corr, _ = spearmanr(pred_np, target_np)
        metrics[f"{task_name}_spearman"] = spearman_corr if not np.isnan(spearman_corr) else 0.0
    except:
        metrics[f"{task_name}_spearman"] = 0.0
    
    try:
        kendall_tau, _ = kendalltau(pred_np, target_np)
        metrics[f"{task_name}_kendall"] = kendall_tau if not np.isnan(kendall_tau) else 0.0
    except:
        metrics[f"{task_name}_kendall"] = 0.0
    
    # Top-k accuracy metrics
    n = len(pred_np)
    
    # Get true top-k indices
    true_top_indices = {}
    for k in k_values:
        if k <= n:
            true_top_indices[k] = set(np.argsort(target_np)[-k:])
    
    # Get predicted top-k indices
    pred_top_indices = {}
    for k in k_values:
        if k <= n:
            pred_top_indices[k] = set(np.argsort(pred_np)[-k:])
    
    # Compute top-k accuracy
    for k in k_values:
        if k <= n and k in true_top_indices and k in pred_top_indices:
            intersection = len(true_top_indices[k] & pred_top_indices[k])
            accuracy = intersection / k
            metrics[f"{task_name}_top_{k}_accuracy"] = accuracy
    
    # NDCG (Normalized Discounted Cumulative Gain)
    for k in k_values:
        if k <= n:
            ndcg_k = compute_ndcg(pred_np, target_np, k)
            metrics[f"{task_name}_ndcg_{k}"] = ndcg_k
    
    return metrics


def compute_ndcg(predictions: np.ndarray, targets: np.ndarray, k: int) -> float:
    """
    Compute NDCG@k
    
    Args:
        predictions: Predicted scores
        targets: True relevance scores
        k: Cut-off rank
    
    Returns:
        NDCG@k score
    """
    # Get top-k predicted indices
    top_k_indices = np.argsort(predictions)[-k:][::-1]
    
    # Compute DCG
    dcg = 0.0
    for i, idx in enumerate(top_k_indices):
        relevance = targets[idx]
        dcg += (2**relevance - 1) / np.log2(i + 2)
    
    # Compute IDCG (Ideal DCG)
    ideal_indices = np.argsort(targets)[-k:][::-1]
    idcg = 0.0
    for i, idx in enumerate(ideal_indices):
        relevance = targets[idx]
        idcg += (2**relevance - 1) / np.log2(i + 2)
    
    # Return NDCG
    if idcg == 0:
        return 0.0
    return dcg / idcg


def compute_pairwise_ranking_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    task_name: str = "pairwise_ranking"
) -> Dict[str, float]:
    """
    Compute pairwise ranking metrics
    
    Args:
        predictions: Model predictions for pairs [N_pairs, 2]
        targets: Ground truth preferences [N_pairs] (0 or 1)
        task_name: Name of the task for logging
    
    Returns:
        Dictionary of metrics
    """
    # Convert to numpy
    pred_np = predictions.detach().cpu().numpy()
    target_np = targets.detach().cpu().numpy()
    
    metrics = {}
    
    # Compute pairwise accuracy
    if pred_np.shape[1] == 2:
        # Predictions are [N, 2] - scores for each item in pair
        pred_preferences = (pred_np[:, 0] > pred_np[:, 1]).astype(int)
    else:
        # Predictions are [N, 1] - preference scores
        pred_preferences = (pred_np.flatten() > 0.5).astype(int)
    
    accuracy = np.mean(pred_preferences == target_np)
    metrics[f"{task_name}_accuracy"] = accuracy
    
    return metrics


def compute_uncertainty_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: torch.Tensor,
    task_name: str = "uncertainty"
) -> Dict[str, float]:
    """
    Compute uncertainty estimation metrics
    
    Args:
        predictions: Dictionary containing uncertainty predictions
        targets: Ground truth targets
        task_name: Name of the task for logging
    
    Returns:
        Dictionary of metrics
    """
    metrics = {}
    
    if "aleatoric_mean" in predictions and "aleatoric_var" in predictions:
        mean_pred = predictions["aleatoric_mean"].detach().cpu().numpy().flatten()
        var_pred = predictions["aleatoric_var"].detach().cpu().numpy().flatten()
        target_np = targets.detach().cpu().numpy().flatten()
        
        # Negative log-likelihood
        nll = 0.5 * (np.log(2 * np.pi * var_pred) + (target_np - mean_pred)**2 / var_pred)
        metrics[f"{task_name}_nll"] = np.mean(nll)
        
        # Calibration metrics (simplified)
        residuals = np.abs(target_np - mean_pred)
        predicted_std = np.sqrt(var_pred)
        
        # Fraction of predictions within 1, 2, 3 standard deviations
        for n_std in [1, 2, 3]:
            within_n_std = np.mean(residuals <= n_std * predicted_std)
            expected_fraction = 0.6827 if n_std == 1 else (0.9545 if n_std == 2 else 0.9973)
            metrics[f"{task_name}_within_{n_std}std"] = within_n_std
            metrics[f"{task_name}_calibration_{n_std}std"] = abs(within_n_std - expected_fraction)
    
    if "epistemic_mean" in predictions and "epistemic_var" in predictions:
        epistemic_var = predictions["epistemic_var"].detach().cpu().numpy().flatten()
        metrics[f"{task_name}_epistemic_var_mean"] = np.mean(epistemic_var)
        metrics[f"{task_name}_epistemic_var_std"] = np.std(epistemic_var)
    
    return metrics


def compute_metrics(
    predictions: Dict[str, torch.Tensor],
    targets: Dict[str, torch.Tensor],
    tasks: List[str]
) -> Dict[str, float]:
    """
    Compute metrics for all tasks
    
    Args:
        predictions: Model predictions
        targets: Ground truth targets
        tasks: List of tasks to evaluate
    
    Returns:
        Dictionary of all metrics
    """
    all_metrics = {}
    
    for task in tasks:
        if task not in predictions or task not in targets:
            logger.warning(f"Task {task} not found in predictions or targets")
            continue
        
        pred = predictions[task]
        target = targets[task]
        
        try:
            if task in ["cost", "latency", "memory", "mem", "cardinality"]:
                # Regression tasks
                task_metrics = compute_regression_metrics(pred, target, task)
            elif task in ["ranking", "rank_scores"]:
                # Ranking tasks
                task_metrics = compute_ranking_metrics(pred, target, task_name=task)
            elif task == "uncertainty":
                # Uncertainty estimation
                task_metrics = compute_uncertainty_metrics(predictions, target, task)
            else:
                # Default to regression
                task_metrics = compute_regression_metrics(pred, target, task)
            
            all_metrics.update(task_metrics)
            
        except Exception as e:
            logger.error(f"Error computing metrics for task {task}: {e}")
            all_metrics[f"{task}_error"] = 1.0
    
    return all_metrics


def ranking_metrics(
    predicted_costs: List[float],
    actual_costs: List[float],
    predicted_rankings: Optional[List[int]] = None
) -> Dict[str, float]:
    """
    Compute ranking-specific metrics for plan selection
    
    Args:
        predicted_costs: Predicted execution costs
        actual_costs: Actual execution costs
        predicted_rankings: Predicted plan rankings (optional)
    
    Returns:
        Dictionary of ranking metrics
    """
    predicted_costs = np.array(predicted_costs)
    actual_costs = np.array(actual_costs)
    
    metrics = {}
    
    # Best plan selection accuracy
    pred_best_idx = np.argmin(predicted_costs)
    actual_best_idx = np.argmin(actual_costs)
    
    metrics["best_plan_accuracy"] = float(pred_best_idx == actual_best_idx)
    
    # Cost of selected plan vs optimal
    selected_cost = actual_costs[pred_best_idx]
    optimal_cost = actual_costs[actual_best_idx]
    
    if optimal_cost > 0:
        metrics["cost_ratio"] = selected_cost / optimal_cost
        metrics["relative_error"] = (selected_cost - optimal_cost) / optimal_cost
    else:
        metrics["cost_ratio"] = 1.0
        metrics["relative_error"] = 0.0
    
    # Ranking correlation
    pred_ranking = np.argsort(predicted_costs)
    actual_ranking = np.argsort(actual_costs)
    
    try:
        spearman_corr, _ = spearmanr(pred_ranking, actual_ranking)
        metrics["ranking_spearman"] = spearman_corr if not np.isnan(spearman_corr) else 0.0
    except:
        metrics["ranking_spearman"] = 0.0
    
    # Top-k plan selection (if more than k plans)
    n_plans = len(predicted_costs)
    for k in [1, 3, 5]:
        if k < n_plans:
            pred_top_k = set(np.argsort(predicted_costs)[:k])
            actual_top_k = set(np.argsort(actual_costs)[:k])
            
            intersection = len(pred_top_k & actual_top_k)
            metrics[f"top_{k}_overlap"] = intersection / k
    
    return metrics


def print_metrics(metrics: Dict[str, float], title: str = "Metrics"):
    """
    Pretty print metrics
    
    Args:
        metrics: Dictionary of metrics
        title: Title for the metrics display
    """
    print(f"\n{title}")
    print("=" * len(title))
    
    # Group metrics by task
    task_metrics = {}
    for key, value in metrics.items():
        if "_" in key:
            task = key.split("_")[0]
            if task not in task_metrics:
                task_metrics[task] = {}
            task_metrics[task][key] = value
        else:
            task_metrics.setdefault("general", {})[key] = value
    
    # Print grouped metrics
    for task, task_dict in task_metrics.items():
        print(f"\n{task.upper()}:")
        for metric, value in sorted(task_dict.items()):
            if isinstance(value, float):
                print(f"  {metric}: {value:.4f}")
            else:
                print(f"  {metric}: {value}")


def compute_model_performance_summary(
    all_metrics: List[Dict[str, float]],
    tasks: List[str]
) -> Dict[str, Dict[str, float]]:
    """
    Compute summary statistics across multiple evaluation runs
    
    Args:
        all_metrics: List of metric dictionaries from multiple runs
        tasks: List of tasks
    
    Returns:
        Dictionary with mean, std, min, max for each metric
    """
    if not all_metrics:
        return {}
    
    # Collect all metric keys
    all_keys = set()
    for metrics in all_metrics:
        all_keys.update(metrics.keys())
    
    summary = {}
    
    for key in all_keys:
        values = []
        for metrics in all_metrics:
            if key in metrics and isinstance(metrics[key], (int, float)):
                values.append(metrics[key])
        
        if values:
            summary[key] = {
                "mean": np.mean(values),
                "std": np.std(values),
                "min": np.min(values),
                "max": np.max(values),
                "count": len(values)
            }
    
    return summary
