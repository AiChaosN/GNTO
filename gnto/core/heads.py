"""
预测头模块 - 将GNN输出的向量转换为具体的预测结果
包含成本预测、延迟预测、排序等多种任务的预测头
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple
import math


class RegressionHead(nn.Module):
    """
    回归预测头 - 用于成本、延迟、内存等数值预测任务
    
    功能：
    - 将计划向量转换为具体的数值预测（如执行成本）
    - 支持多层MLP结构
    - 可配置输出激活函数（如确保正数输出）
    """
    
    def __init__(
        self,
        input_dim: int,          # 输入向量维度（来自GNN）
        hidden_dims: List[int] = [128, 64],  # 隐藏层维度列表
        output_dim: int = 1,     # 输出维度（通常为1，表示单个数值）
        dropout: float = 0.1,    # Dropout比例
        activation: str = "relu", # 隐藏层激活函数
        output_activation: Optional[str] = None  # 输出激活函数
    ):
        super().__init__()
        
        # 构建回归预测网络
        layers = []
        prev_dim = input_dim
        
        # 添加隐藏层
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))  # 线性变换
            # 添加激活函数
            if activation == "relu":
                layers.append(nn.ReLU())
            elif activation == "gelu":
                layers.append(nn.GELU())
            layers.append(nn.Dropout(dropout))  # 防止过拟合
            prev_dim = hidden_dim
        
        # 输出层
        layers.append(nn.Linear(prev_dim, output_dim))
        
        # 输出激活函数（确保输出符合要求）
        if output_activation == "sigmoid":
            layers.append(nn.Sigmoid())      # 输出0-1之间
        elif output_activation == "softplus":
            layers.append(nn.Softplus())     # 输出正数（用于成本预测）
        elif output_activation == "exp":
            layers.append(lambda x: torch.exp(x))  # 指数激活
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        前向传播 - 预测数值结果
        
        参数:
            x: [B, input_dim] 计划向量（来自GNN）
            
        返回:
            [B, output_dim] 预测的数值结果（如成本）
        """
        return self.network(x)


class RankingHead(nn.Module):
    """Head for plan ranking tasks"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        ranking_type: str = "pointwise",  # pointwise, pairwise, listwise
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.ranking_type = ranking_type
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        if ranking_type == "pointwise":
            # Output single score per plan
            layers.append(nn.Linear(prev_dim, 1))
        elif ranking_type == "pairwise":
            # Output for pairwise comparison
            layers.append(nn.Linear(prev_dim, 1))
        elif ranking_type == "listwise":
            # Output scores for listwise ranking
            layers.append(nn.Linear(prev_dim, 1))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [B, input_dim] or [B*K, input_dim] plan embeddings
        Returns:
            [B, 1] or [B*K, 1] ranking scores
        """
        return self.network(x)


class CardinalityHead(nn.Module):
    """Head for node-level cardinality estimation"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        dropout: float = 0.1,
        log_scale: bool = True
    ):
        super().__init__()
        
        self.log_scale = log_scale
        
        layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(prev_dim, hidden_dim))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        layers.append(nn.Linear(prev_dim, 1))
        
        if log_scale:
            # Output log cardinality, then exp to get actual cardinality
            layers.append(nn.Softplus())  # Ensures positive output
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [N, input_dim] node embeddings
        Returns:
            [N, 1] cardinality estimates
        """
        output = self.network(x)
        if self.log_scale:
            # Convert log scale to actual cardinality
            output = torch.exp(output)
        return output


class UncertaintyHead(nn.Module):
    """Head for uncertainty estimation"""
    
    def __init__(
        self,
        input_dim: int,
        hidden_dims: List[int] = [128, 64],
        uncertainty_type: str = "aleatoric",  # aleatoric, epistemic, both
        dropout: float = 0.1
    ):
        super().__init__()
        
        self.uncertainty_type = uncertainty_type
        
        # Shared layers
        shared_layers = []
        prev_dim = input_dim
        
        for hidden_dim in hidden_dims[:-1]:
            shared_layers.append(nn.Linear(prev_dim, hidden_dim))
            shared_layers.append(nn.ReLU())
            shared_layers.append(nn.Dropout(dropout))
            prev_dim = hidden_dim
        
        self.shared = nn.Sequential(*shared_layers)
        
        # Uncertainty-specific heads
        final_dim = hidden_dims[-1] if hidden_dims else input_dim
        
        if uncertainty_type in ["aleatoric", "both"]:
            self.aleatoric_mean = nn.Linear(prev_dim, final_dim)
            self.aleatoric_var = nn.Linear(prev_dim, final_dim)
        
        if uncertainty_type in ["epistemic", "both"]:
            self.epistemic_head = nn.Linear(prev_dim, final_dim)
            self.dropout_layers = nn.ModuleList([
                nn.Dropout(dropout) for _ in range(10)  # For MC Dropout
            ])
    
    def forward(self, x: torch.Tensor, mc_samples: int = 10) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: [B, input_dim] input embeddings
            mc_samples: Number of MC samples for epistemic uncertainty
        
        Returns:
            Dictionary with uncertainty estimates
        """
        shared_features = self.shared(x)
        result = {}
        
        if self.uncertainty_type in ["aleatoric", "both"]:
            # Aleatoric uncertainty (data uncertainty)
            mean = self.aleatoric_mean(shared_features)
            log_var = self.aleatoric_var(shared_features)
            var = torch.exp(log_var)
            
            result["aleatoric_mean"] = mean
            result["aleatoric_var"] = var
        
        if self.uncertainty_type in ["epistemic", "both"]:
            # Epistemic uncertainty (model uncertainty) via MC Dropout
            mc_outputs = []
            for i in range(mc_samples):
                dropout_features = shared_features
                for dropout_layer in self.dropout_layers:
                    dropout_features = dropout_layer(dropout_features)
                mc_output = self.epistemic_head(dropout_features)
                mc_outputs.append(mc_output)
            
            mc_outputs = torch.stack(mc_outputs, dim=0)  # [mc_samples, B, dim]
            epistemic_mean = mc_outputs.mean(dim=0)
            epistemic_var = mc_outputs.var(dim=0)
            
            result["epistemic_mean"] = epistemic_mean
            result["epistemic_var"] = epistemic_var
        
        return result


class Heads(nn.Module):
    """
    多任务预测头 - 整合所有预测任务的主要模块
    
    功能：
    - 接收GNN输出的计划向量
    - 同时进行多种预测：成本、延迟、排序等
    - 统一管理所有预测头，支持多任务学习
    """
    
    def __init__(
        self,
        plan_emb_dim: int,       # 计划向量维度（来自GNN）
        node_emb_dim: int,       # 节点向量维度（用于节点级预测）
        tasks: List[str] = ["cost", "latency", "ranking"],  # 要执行的任务列表
        hidden_dims: List[int] = [128, 64],  # 预测头隐藏层维度
        dropout: float = 0.1,    # Dropout比例
        uncertainty: bool = False # 是否启用不确定性估计
    ):
        super().__init__()
        
        self.tasks = tasks
        self.uncertainty = uncertainty
        
        # 任务特定的预测头
        self.task_heads = nn.ModuleDict()
        
        # 成本预测头
        if "cost" in tasks:
            self.task_heads["cost"] = RegressionHead(
                plan_emb_dim, hidden_dims, output_dim=1, 
                dropout=dropout, output_activation="softplus"  # 确保输出正数
            )
        
        # 延迟预测头
        if "latency" in tasks:
            self.task_heads["latency"] = RegressionHead(
                plan_emb_dim, hidden_dims, output_dim=1,
                dropout=dropout, output_activation="softplus"  # 确保输出正数
            )
        
        if "memory" in tasks or "mem" in tasks:
            self.task_heads["memory"] = RegressionHead(
                plan_emb_dim, hidden_dims, output_dim=1,
                dropout=dropout, output_activation="softplus"
            )
        
        if "ranking" in tasks or "rank_scores" in tasks:
            self.task_heads["ranking"] = RankingHead(
                plan_emb_dim, hidden_dims, ranking_type="pointwise",
                dropout=dropout
            )
        
        if "cardinality" in tasks:
            self.task_heads["cardinality"] = CardinalityHead(
                node_emb_dim, hidden_dims, dropout=dropout, log_scale=True
            )
        
        if uncertainty:
            self.task_heads["uncertainty"] = UncertaintyHead(
                plan_emb_dim, hidden_dims, uncertainty_type="both",
                dropout=dropout
            )
    
    def forward(
        self, 
        plan_emb: torch.Tensor,  # [B, plan_emb_dim] 计划向量
        node_emb: torch.Tensor,  # [N, node_emb_dim] 节点向量
        extra_ctx: Optional[Dict[str, torch.Tensor]] = None  # 额外上下文
    ) -> Dict[str, torch.Tensor]:
        """
        多任务预测的前向传播
        
        参数:
            plan_emb: [B, plan_emb_dim] 计划级嵌入向量（来自GNN）
            node_emb: [N, node_emb_dim] 节点级嵌入向量
            extra_ctx: 额外的上下文信息（可选）
        
        返回:
            包含所有任务预测结果的字典
        """
        predictions = {}
        
        # 计划级预测（基于整个计划的向量表示）
        for task in ["cost", "latency", "memory", "ranking"]:
            if task in self.task_heads:
                if task == "memory" and "mem" in self.tasks:
                    # 内存预测
                    predictions["mem"] = self.task_heads["memory"](plan_emb)
                elif task == "ranking" and "rank_scores" in self.tasks:
                    # 排序分数预测
                    predictions["rank_scores"] = self.task_heads["ranking"](plan_emb)
                else:
                    # 其他任务预测（成本、延迟等）
                    predictions[task] = self.task_heads[task](plan_emb)
        
        # 节点级预测（基于每个节点的向量表示）
        if "cardinality" in self.task_heads and node_emb is not None:
            # 基数估计（每个节点的数据量预测）
            predictions["cardinality"] = self.task_heads["cardinality"](node_emb)
        
        # Uncertainty estimation
        if "uncertainty" in self.task_heads:
            uncertainty_pred = self.task_heads["uncertainty"](plan_emb)
            predictions["uncertainty"] = uncertainty_pred
        
        # Handle extra context if provided
        if extra_ctx is not None:
            for key, value in extra_ctx.items():
                if key not in predictions:
                    predictions[key] = value
        
        return predictions


class MultiTaskLoss(nn.Module):
    """Multi-task loss with automatic weighting"""
    
    def __init__(
        self,
        tasks: List[str],
        loss_weights: Optional[Dict[str, float]] = None,
        adaptive_weighting: bool = True
    ):
        super().__init__()
        
        self.tasks = tasks
        self.adaptive_weighting = adaptive_weighting
        
        # Initialize loss weights
        if loss_weights is None:
            loss_weights = {task: 1.0 for task in tasks}
        
        if adaptive_weighting:
            # Learnable loss weights (uncertainty-based)
            self.log_vars = nn.Parameter(torch.zeros(len(tasks)))
        else:
            self.loss_weights = loss_weights
    
    def forward(
        self, 
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Compute multi-task loss
        
        Args:
            predictions: Model predictions
            targets: Ground truth targets
        
        Returns:
            Dictionary with individual and total losses
        """
        losses = {}
        total_loss = 0.0
        
        for i, task in enumerate(self.tasks):
            if task not in predictions or task not in targets:
                continue
            
            pred = predictions[task]
            target = targets[task]
            
            # Compute task-specific loss
            if task in ["cost", "latency", "memory", "mem"]:
                # Regression loss (MSE or MAE)
                loss = F.mse_loss(pred, target)
            elif task in ["ranking", "rank_scores"]:
                # Ranking loss (can be extended with pairwise/listwise losses)
                loss = F.mse_loss(pred, target)
            elif task == "cardinality":
                # Log-scale MSE for cardinality
                log_pred = torch.log(pred + 1e-8)
                log_target = torch.log(target + 1e-8)
                loss = F.mse_loss(log_pred, log_target)
            else:
                loss = F.mse_loss(pred, target)
            
            losses[f"{task}_loss"] = loss
            
            # Apply weighting
            if self.adaptive_weighting:
                # Uncertainty-based weighting
                precision = torch.exp(-self.log_vars[i])
                weighted_loss = precision * loss + self.log_vars[i]
            else:
                weight = self.loss_weights.get(task, 1.0)
                weighted_loss = weight * loss
            
            total_loss += weighted_loss
        
        losses["total_loss"] = total_loss
        return losses
