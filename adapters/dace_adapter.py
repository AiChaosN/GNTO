"""
DACE Adapter - 适配 DACE benchmark 的数据加载和模型组装。

DACE 使用跨数据库 workload 评测，数据格式为 {db_name}_filted.json，
特征使用 node type one-hot + RobustScaler 缩放的 (Total Cost, Plan Rows)。

用法:
    from adapters.dace_adapter import (
        DaceWorkloadDataset, GNTO_DACE_Model,
        load_dace_statistics, q_error_np, WORKLOADS
    )
"""

import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.data import Data, InMemoryDataset
from tqdm import tqdm

# Import GNTO core models
_gnto_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _gnto_root not in sys.path:
    sys.path.insert(0, _gnto_root)

from models.TreeEncoder import GATv2TreeEncoder_V3
from models.PredictionHead import PredictionHead_V2


# ---------------------------------------------------------------------------
# DACE Workload 列表 (20 个数据库)
# ---------------------------------------------------------------------------
WORKLOADS = [
    "accidents", "airline", "baseball", "basketball", "carcinogenesis",
    "consumer", "credit", "employee", "fhnk", "financial",
    "geneea", "genome", "hepatitis", "imdb_full", "movielens",
    "seznam", "ssb", "tournament", "tpc_h", "walmart",
]


# ---------------------------------------------------------------------------
# DACE 资源加载
# ---------------------------------------------------------------------------
def get_dace_path():
    """自动查找 DACE 仓库路径。"""
    candidates = [
        os.path.join(_gnto_root, "..", "DACE"),
        "/home/AiChaosN/Project/Workspace/01_Research/DACE",
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "DACE repo not found. Expected as sibling directory of GNTO "
        "or at /home/AiChaosN/Project/Workspace/01_Research/DACE"
    )


def load_dace_statistics(dace_path=None, workload="workload1"):
    """加载 DACE 的 statistics.json。

    Returns:
        dict: statistics (含 node_types, Total Cost, Plan Rows 的 center/scale)
        str: workload 数据目录路径
    """
    if dace_path is None:
        dace_path = get_dace_path()

    workload_dir = os.path.join(dace_path, "data", workload)
    stats_path = os.path.join(workload_dir, "statistics.json")

    if not os.path.exists(stats_path):
        raise FileNotFoundError(
            f"Statistics not found at {stats_path}. Run DACE setup first."
        )

    with open(stats_path, 'r') as f:
        stats = json.load(f)

    return stats, workload_dir


# ---------------------------------------------------------------------------
# Dataset: DACE JSON plans → PyG Data
# ---------------------------------------------------------------------------
def _scale_value(val, stats, key):
    center = stats[key]["center"]
    scale = stats[key]["scale"]
    if scale == 0:
        return 0.0
    return (val - center) / scale


class DaceWorkloadDataset(InMemoryDataset):
    """加载 DACE workload 的 {db_name}_filted.json 并转为 PyG 数据集。

    Args:
        root: workload 数据目录
        db_names: 要加载的数据库名称列表
        stats: DACE statistics dict
    """

    def __init__(self, root, db_names, stats, transform=None, pre_transform=None):
        self.db_names = db_names
        self.stats = stats
        self.node_type_dict = stats["node_types"]["value_dict"]
        self.num_node_types = len(self.node_type_dict)
        super().__init__(root, transform, pre_transform)
        self.data, self.slices = self._process_data()

    def _process_data(self):
        data_list = []

        for db_name in self.db_names:
            file_path = os.path.join(self.root, f"{db_name}_filted.json")
            if not os.path.exists(file_path):
                print(f"Warning: {file_path} not found. Skipping.")
                continue

            print(f"Processing {db_name}...")
            try:
                with open(file_path, 'r') as f:
                    plans = json.load(f)
            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                continue

            for plan_wrapper in tqdm(plans, desc=db_name, leave=False):
                plan_content = plan_wrapper
                while isinstance(plan_content, list):
                    plan_content = plan_content[0]

                if "Plan" not in plan_content:
                    continue

                root_node = plan_content["Plan"]
                x, edge_index = self._plan_to_graph(root_node)
                exec_time = root_node.get("Actual Total Time", 0.0)
                y = torch.tensor([exec_time], dtype=torch.float)

                data_list.append(Data(x=x, edge_index=edge_index, y=y))

        return self.collate(data_list)

    def _plan_to_graph(self, root):
        nodes = []
        edges = []

        def dfs(node, parent_idx):
            current_idx = len(nodes)

            # Node Type one-hot
            node_type = node.get("Node Type", "Unknown")
            type_idx = self.node_type_dict.get(node_type, 0)
            type_vec = [0] * self.num_node_types
            if type_idx < self.num_node_types:
                type_vec[type_idx] = 1

            # Numerical features (RobustScaler)
            cost_scaled = _scale_value(
                node.get("Total Cost", 0.0), self.stats, "Total Cost")
            rows_scaled = _scale_value(
                node.get("Plan Rows", 0.0), self.stats, "Plan Rows")

            nodes.append(type_vec + [cost_scaled, rows_scaled])

            # Bidirectional edges
            if parent_idx is not None:
                edges.append([parent_idx, current_idx])
                edges.append([current_idx, parent_idx])

            if "Plans" in node:
                for child in node["Plans"]:
                    dfs(child, current_idx)

        dfs(root, None)

        x = torch.tensor(nodes, dtype=torch.float)
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return x, edge_index


# ---------------------------------------------------------------------------
# Model: GNTO for DACE benchmark
# ---------------------------------------------------------------------------
class GNTO_DACE_Model(nn.Module):
    """GNTO model adapted for DACE's feature format (node_type one-hot + 2 scalars)。

    Args:
        input_dim: num_node_types + 2
        hidden_dim: 隐层维度
    """

    def __init__(self, input_dim, hidden_dim=64):
        super().__init__()
        self.node_encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.tree_encoder = GATv2TreeEncoder_V3(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            heads1=4,
            heads2=4,
            drop=0.1,
        )
        self.prediction_head = PredictionHead_V2(
            in_dim=hidden_dim,
            out_dim=1,
            hidden_dims=(hidden_dim, hidden_dim),
        )

    def forward(self, data):
        x = self.node_encoder(data.x)
        g = self.tree_encoder(x, data.edge_index, data.batch)
        out = self.prediction_head(g)
        return torch.sigmoid(out)


# ---------------------------------------------------------------------------
# 评测工具函数
# ---------------------------------------------------------------------------
def q_error_np(preds, targets):
    """计算逐样本 Q-Error。"""
    qerrors = []
    for p, t in zip(preds, targets):
        if p == 0 and t == 0:
            qerrors.append(1.0)
        elif p == 0 or t == 0:
            qerrors.append(float('inf'))
        else:
            qerrors.append(max(p / t, t / p))
    return np.array(qerrors)


def q_error_loss(preds, targets):
    """DACE 风格的 Q-Error loss (用于训练)。"""
    preds = torch.clamp(preds, min=1e-7)
    targets = torch.clamp(targets, min=1e-7)
    q = torch.max(preds / targets, targets / preds)
    return torch.mean(torch.log(q))
