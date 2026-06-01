"""
QueryFormer Adapter - 适配 QueryFormer (VLDB2022) benchmark 的数据转换和模型组装。

将 QueryFormer 的 PlanTreeDataset 数据格式转成 PyG Data，
消除 3 个实验脚本中的重复代码 (1203, 1216, 1216_addPlanrows)。

依赖: QueryFormer_VLDB2022 仓库需要作为 sibling directory 存在。

用法:
    from adapters.qf_adapter import (
        QueryFormerToPyGDataset, GNTO_QF_Model,
        load_qf_resources, unnormalize, calc_q_error, evaluate
    )
"""

import os
import sys
import json

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
from torch_geometric.data import Data
from tqdm import tqdm

# Import GNTO core models
_gnto_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _gnto_root not in sys.path:
    sys.path.insert(0, _gnto_root)

from models.TreeEncoder import GATv2TreeEncoder_V3
from models.PredictionHead import PredictionHead_V2
from models.NodeEncoder import NodeEncoder_QF, NodeEncoder_QF_AddPlanrows


# ---------------------------------------------------------------------------
# QueryFormer 资源加载
# ---------------------------------------------------------------------------
def get_qf_path():
    """自动查找 sibling 目录下的 QueryFormer_VLDB2022。"""
    candidates = [
        os.path.join(_gnto_root, "..", "QueryFormer_VLDB2022"),
        os.path.join(_gnto_root, "..", "..", "QueryFormer_VLDB2022"),
    ]
    for c in candidates:
        p = os.path.abspath(c)
        if os.path.isdir(p):
            return p
    raise FileNotFoundError(
        "QueryFormer_VLDB2022 not found. Expected as sibling directory of GNTO."
    )


def load_qf_resources(qf_path=None):
    """加载 QueryFormer 的 encoding, hist_file, table_sample, Normalizer。

    Returns:
        dict with keys: encoding, hist_file, table_sample, cost_norm, data_path, qf_path
    """
    if qf_path is None:
        qf_path = get_qf_path()

    if qf_path not in sys.path:
        sys.path.insert(0, qf_path)

    from model.database_util import get_hist_file, get_job_table_sample
    from model.util import Normalizer

    data_path = os.path.join(qf_path, 'data/imdb/')
    checkpoint_path = os.path.join(qf_path, 'checkpoints/encoding.pt')

    hist_file = get_hist_file(data_path + 'histogram_string.csv')
    table_sample = get_job_table_sample(data_path + 'train')
    encoding_ckpt = torch.load(checkpoint_path, weights_only=False)
    encoding = encoding_ckpt['encoding']
    cost_norm = Normalizer(-3.61192, 12.290855)

    return {
        "encoding": encoding,
        "hist_file": hist_file,
        "table_sample": table_sample,
        "cost_norm": cost_norm,
        "data_path": data_path,
        "qf_path": qf_path,
    }


# ---------------------------------------------------------------------------
# Dataset: QueryFormer features → PyG Data
# ---------------------------------------------------------------------------
class QueryFormerToPyGDataset(Dataset):
    """把 QueryFormer CSV DataFrame 转成 PyG Data list，支持缓存。

    Args:
        df: pandas DataFrame，含 'json' 和 'id' 列
        encoding: QueryFormer Encoding 对象
        hist_file: histogram 数据
        table_sample: table sample 数据
        cost_norm: Normalizer 对象
        cache_name: 缓存文件名前缀，None 则不缓存
        add_plan_rows: 是否额外拼接 Plan Rows 特征 (1165 → 1166 维)
    """

    def __init__(self, df, encoding, hist_file, table_sample, cost_norm,
                 cache_name=None, add_plan_rows=False):
        self.df = df.reset_index(drop=True)
        self.cost_norm = cost_norm
        self.data_list = []

        # 缓存路径
        cache_file = None
        if cache_name:
            suffix = "_planrows" if add_plan_rows else ""
            cache_dir = os.path.join(_gnto_root, "data", "process", "cache")
            os.makedirs(cache_dir, exist_ok=True)
            cache_file = os.path.join(cache_dir, f"{cache_name}{suffix}_{len(df)}.pt")
            if os.path.exists(cache_file):
                print(f"Loading cached data from {cache_file}...")
                self.data_list = torch.load(cache_file, weights_only=False)
                return

        # 需要 QF 的 PlanTreeDataset
        qf_path = os.path.dirname(sys.modules['model.dataset'].__file__) \
            if 'model.dataset' in sys.modules else None
        from model.dataset import PlanTreeDataset
        from model.util import Normalizer as QFNormalizer

        print(f"Preprocessing {len(df)} samples...")
        dummy_norm = QFNormalizer(1, 100)
        qf_dataset = PlanTreeDataset(
            self.df, None, encoding, hist_file,
            dummy_norm, dummy_norm, 'cost', table_sample
        )

        for idx in tqdm(range(len(self.df)), desc="Processing"):
            json_str = self.df.iloc[idx]['json']
            plan = json.loads(json_str)['Plan']
            query_id = self.df.iloc[idx]['id']

            root = qf_dataset.traversePlan(plan, query_id, encoding)
            node_dict = qf_dataset.node2dict(root)
            qf_dataset.treeNodes.clear()

            x = node_dict['features']
            adj = node_dict['adjacency_list']
            if len(adj) > 0:
                edge_index = adj.t().long()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            # 额外拼接 Plan Rows
            if add_plan_rows:
                rows_list = []
                self._extract_rows(plan, rows_list)
                rows_tensor = torch.tensor(rows_list, dtype=torch.float).view(-1, 1)
                if x.size(0) == rows_tensor.size(0):
                    x = torch.cat([x, rows_tensor], dim=1)

            # 标签
            exec_time = json.loads(json_str)['Execution Time']
            y_val = np.log(float(exec_time) + 0.001)
            y_norm = (y_val - self.cost_norm.mini) / (self.cost_norm.maxi - self.cost_norm.mini)
            y_norm = np.clip(y_norm, 0.001, 1.0)

            data = Data(
                x=x, edge_index=edge_index,
                y=torch.tensor([y_norm], dtype=torch.float),
            )
            data.raw_y = torch.tensor([exec_time], dtype=torch.float)
            self.data_list.append(data)

        if cache_file:
            print(f"Saving cache to {cache_file}...")
            torch.save(self.data_list, cache_file)

    @staticmethod
    def _extract_rows(node, rows_list):
        """DFS 提取每个节点的 Plan Rows (log1p)。"""
        rows_list.append(np.log1p(float(node['Plan Rows'])))
        if 'Plans' in node:
            for child in node['Plans']:
                QueryFormerToPyGDataset._extract_rows(child, rows_list)

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        return self.data_list[idx]


# ---------------------------------------------------------------------------
# Model: GNTO with QF-style Node Encoder
# ---------------------------------------------------------------------------
class GNTO_QF_Model(nn.Module):
    """组合 NodeEncoder_QF (或 AddPlanrows) + GATv2TreeEncoder_V3 + PredictionHead_V2。

    Args:
        encoding: QueryFormer Encoding 对象
        hidden_dim: 隐层维度
        add_plan_rows: 是否使用 NodeEncoder_QF_AddPlanrows
    """

    def __init__(self, encoding, hidden_dim=64, add_plan_rows=False):
        super().__init__()

        num_types = len(encoding.idx2type)
        num_tables = len(encoding.idx2table)
        num_joins = len(encoding.idx2join)
        num_ops = len(encoding.idx2op)
        num_columns = len(encoding.idx2col)

        encoder_cls = NodeEncoder_QF_AddPlanrows if add_plan_rows else NodeEncoder_QF
        self.node_encoder = encoder_cls(
            embed_size=64,
            tables=num_tables,
            types=num_types,
            joins=num_joins,
            columns=num_columns,
            ops=num_ops,
            use_sample=True,
            use_hist=True,
            bin_number=50,
        )

        self.gnn = GATv2TreeEncoder_V3(
            in_dim=64,  # NodeEncoder_QF 输出 embed_size=64
            hidden_dim=hidden_dim,
            out_dim=hidden_dim,
            heads1=4,
            heads2=2,
            drop=0.0,
        )

        self.head = PredictionHead_V2(
            in_dim=hidden_dim,
            out_dim=1,
            hidden_dims=(64, 64),
            dropout=0.0,
        )

    def forward(self, data):
        x = self.node_encoder(data.x)
        x = self.gnn(x, data.edge_index, data.batch)
        return torch.sigmoid(self.head(x))


# ---------------------------------------------------------------------------
# 评测工具函数
# ---------------------------------------------------------------------------
def unnormalize(y_norm, cost_norm):
    """把归一化的预测值还原为真实执行时间。"""
    val = y_norm * (cost_norm.maxi - cost_norm.mini) + cost_norm.mini
    return np.exp(val) - 0.001


def calc_q_error(preds, targets):
    """计算 Q-Error 百分位数 [50, 75, 90, 95, 99]。"""
    from utils.metrics import qerror_percentiles
    q = qerror_percentiles(preds, targets, percentiles=(50, 75, 90, 95, 99))
    return np.array([q["q50"], q["q75"], q["q90"], q["q95"], q["q99"]])


def _run_inference(model, loader, cost_norm, device):
    """Shared loop: returns (avg_loss, preds_raw, targets_raw) on the raw scale."""
    model.eval()
    preds_all, targets_all = [], []
    total_loss = 0

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            out = model(batch).view(-1)
            loss = torch.nn.functional.mse_loss(out, batch.y)
            total_loss += loss.item() * batch.num_graphs

            pred_raw = unnormalize(out.cpu().numpy(), cost_norm)
            target_raw = batch.raw_y.cpu().numpy()
            preds_all.extend(pred_raw)
            targets_all.extend(target_raw)

    avg_loss = total_loss / len(loader.dataset)
    return avg_loss, np.asarray(preds_all), np.asarray(targets_all)


def evaluate(model, loader, cost_norm, device):
    """在验证集上评测，返回 (avg_loss, q50, q75, q90, q95, q99)。"""
    avg_loss, preds, targets = _run_inference(model, loader, cost_norm, device)
    q50, q75, q90, q95, q99 = calc_q_error(preds, targets)
    return avg_loss, q50, q75, q90, q95, q99


def evaluate_full(model, loader, cost_norm, device, group_ids=None):
    """Rich evaluation: Q-Error + Spearman/Kendall/pairwise-acc + (optional) Top-1 regret.

    Returns:
        dict with keys: avg_loss, q50/q75/q90/q95/q99, spearman, kendall,
        pairwise_acc, and (if group_ids given) top1_regret_mean/median/relative.
        Also embeds preds and targets arrays for downstream use.
    """
    from utils.metrics import evaluation_summary
    avg_loss, preds, targets = _run_inference(model, loader, cost_norm, device)
    summary = {"avg_loss": float(avg_loss)}
    summary.update(evaluation_summary(preds, targets, group_ids=group_ids))
    summary["preds"] = preds
    summary["targets"] = targets
    return summary
