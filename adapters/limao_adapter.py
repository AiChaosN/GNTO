"""
LIMAO Adapter - 适配 LIMAO (Bao) end2end benchmark 的接口。

LIMAO 需要:
1. GNTOModel: nn.Module，接收 PyG Batch，输出预测值
2. GNTOFeaturizer: 把 PostgreSQL EXPLAIN JSON → PyG Data（在线学习 vocab）
3. GntoRegression: sklearn 风格的 fit/predict/save/load 接口

用法 (在 LIMAOLifeLongRLDB/bao_server/ 中):
    import sys
    sys.path.insert(0, "/path/to/GNTO")
    from adapters.limao_adapter import GNTOModel, GNTOFeaturizer, GntoRegression
"""

import json
import os
import re
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
import joblib
from sklearn import preprocessing
from sklearn.pipeline import Pipeline

try:
    from torch_geometric.data import Data, Batch
    from torch_geometric.loader import DataLoader
except ImportError:
    raise ImportError("torch_geometric is required for LIMAO adapter. "
                      "Install with: pip install torch-geometric")

# Import GNTO core models
import sys
_gnto_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _gnto_root not in sys.path:
    sys.path.append(_gnto_root)  # append, not insert — avoid shadowing LIMAO's config.py

from models.NodeEncoder import NodeEncoder_V2
from models.TreeEncoder import GATv2TreeEncoder_V3
from models.PredictionHead import PredictionHead_V2


# ---------------------------------------------------------------------------
# GNTOModel: 组合 NodeEncoder + TreeEncoder + PredictionHead
# ---------------------------------------------------------------------------
class GNTOModel(nn.Module):
    def __init__(self, num_node_types, num_cols, num_ops,
                 type_dim=16, hidden_dim=64, out_dim=64, heads=8):
        super().__init__()
        self.node_enc = NodeEncoder_V2(
            num_node_types=num_node_types,
            num_cols=num_cols,
            num_ops=num_ops,
            type_dim=type_dim,
            out_dim=hidden_dim,
            hidden_dim=hidden_dim,
        )
        self.tree_enc = GATv2TreeEncoder_V3(
            in_dim=hidden_dim,
            hidden_dim=hidden_dim,
            out_dim=out_dim,
            heads1=heads,
            heads2=4,
            pooling="mean",
        )
        self.head = PredictionHead_V2(
            in_dim=out_dim,
            out_dim=1,
            hidden_dims=(32, 32),
        )

    def forward(self, data):
        if isinstance(data, list):
            data = Batch.from_data_list(data)

        device = next(self.parameters()).device
        data = data.to(device)
        x, edge_index, batch = data.x, data.edge_index, data.batch

        h = self.node_enc(x)
        g = self.tree_enc(h, edge_index, batch)
        out = self.head(g)
        return out


# ---------------------------------------------------------------------------
# GNTOFeaturizer: PostgreSQL EXPLAIN JSON → PyG Data
# ---------------------------------------------------------------------------
class GNTOFeaturizer:
    """把 PostgreSQL EXPLAIN JSON plan 转成 PyG Data 对象。

    支持在线学习 vocab（fit 阶段 update_vocab=True）。
    """

    def __init__(self, max_preds=3):
        self.max_preds = max_preds
        self.node_type_vocab = {}
        self.col_vocab = {}
        self.op_vocab = {}

        # 常见操作符
        for op in ['=', '!=', '<', '>', '<=', '>=', '~~', '~~*', '!~~', '!~~*']:
            if op not in self.op_vocab:
                self.op_vocab[op] = len(self.op_vocab)

    def num_node_types(self):
        return max(100, len(self.node_type_vocab) + 1)

    def num_cols(self):
        return max(1000, len(self.col_vocab) + 1)

    def num_ops(self):
        return max(50, len(self.op_vocab) + 1)

    # ---- 内部方法 ----

    def _parse_val(self, val):
        try:
            return float(val)
        except (ValueError, TypeError):
            pass
        s = str(val).strip()
        if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
            return s[1:-1]
        return s

    def _parse_predicates(self, node):
        preds = []
        keys = ["Filter", "Index Cond", "Recheck Cond",
                "Join Filter", "Hash Cond", "Merge Cond"]
        for key in keys:
            if key not in node:
                continue
            try:
                cond = str(node[key])
                clean = cond.replace('(', '').replace(')', '')
                parts = re.split(r'\s+AND\s+', clean)
                for p in parts:
                    for op in self.op_vocab:
                        if op in p:
                            split_p = p.split(op)
                            if len(split_p) == 2:
                                lhs = split_p[0].strip()
                                rhs = split_p[1].strip()
                                preds.append((lhs, op, rhs))
                            break
            except Exception:
                continue
        return preds

    def _process_node(self, node, update_vocab=False):
        nt = node.get("Node Type", "Unknown")
        if update_vocab:
            if nt not in self.node_type_vocab:
                self.node_type_vocab[nt] = len(self.node_type_vocab)
            nt_id = self.node_type_vocab[nt]
        else:
            nt_id = self.node_type_vocab.get(nt, 0)

        rows = np.log1p(float(node.get("Plan Rows", 0)))
        width = np.log1p(float(node.get("Plan Width", 0)))

        raw_preds = self._parse_predicates(node)
        pred_vecs = []

        for lhs, op, rhs in raw_preds[:self.max_preds]:
            if update_vocab:
                if lhs not in self.col_vocab:
                    self.col_vocab[lhs] = len(self.col_vocab)
                lhs_id = self.col_vocab[lhs]
                if op not in self.op_vocab:
                    self.op_vocab[op] = len(self.op_vocab)
                op_id = self.op_vocab[op]
            else:
                lhs_id = self.col_vocab.get(lhs, 0)
                op_id = self.op_vocab.get(op, 0)

            rhs_val = self._parse_val(rhs)
            is_join = False
            rhs_feat = 0.0

            if isinstance(rhs_val, str):
                try:
                    rhs_feat = float(rhs_val)
                except (ValueError, TypeError):
                    if update_vocab:
                        if rhs_val not in self.col_vocab:
                            self.col_vocab[rhs_val] = len(self.col_vocab)
                        rhs_feat = float(self.col_vocab[rhs_val])
                    else:
                        rhs_feat = float(self.col_vocab.get(rhs_val, 0))
                    is_join = True
            else:
                rhs_feat = float(rhs_val)

            pred_vecs.append([float(lhs_id), float(op_id), rhs_feat,
                              1.0 if is_join else 0.0])

        # 补齐到 max_preds
        while len(pred_vecs) < self.max_preds:
            pred_vecs.append([0.0, 0.0, 0.0, 0.0])

        pred_flat = [x for p in pred_vecs for x in p]
        return [float(nt_id), rows, width] + pred_flat

    def plan_to_graph(self, plan_json, update_vocab=False):
        node_feats = []
        edges = []
        curr_idx = 0

        def recurse(node, parent_idx):
            nonlocal curr_idx
            my_idx = curr_idx
            curr_idx += 1

            feat = self._process_node(node, update_vocab=update_vocab)
            node_feats.append(feat)

            if parent_idx != -1:
                edges.append([parent_idx, my_idx])
                edges.append([my_idx, parent_idx])

            if "Plans" in node:
                for child in node["Plans"]:
                    recurse(child, my_idx)

        recurse(plan_json, -1)

        x = torch.tensor(node_feats, dtype=torch.float32)
        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)

        return Data(x=x, edge_index=edge_index)

    def fit(self, plans):
        """在线学习 vocab（扫描所有 plan 更新词表）。"""
        for p in plans:
            if isinstance(p, str):
                p = json.loads(p)
            root = p["Plan"] if "Plan" in p else p
            self.plan_to_graph(root, update_vocab=True)

    def transform(self, plans):
        """把 plan list 转成 PyG Data list。"""
        data_list = []
        for p in plans:
            if isinstance(p, str):
                p = json.loads(p)
            root = p["Plan"] if "Plan" in p else p
            data_list.append(self.plan_to_graph(root, update_vocab=False))
        return data_list


# ---------------------------------------------------------------------------
# GntoRegression: sklearn 风格接口，供 LIMAO main.py 调用
# ---------------------------------------------------------------------------
def _inv_log1p(x):
    return np.exp(x) - 1


class GntoRegression:
    """Drop-in replacement for Bao's BaoRegression, using GNTO model."""

    def __init__(self, verbose=False, have_cache_data=False):
        self.__net = None
        self.__verbose = verbose
        self.__pipeline = Pipeline([
            ("log", preprocessing.FunctionTransformer(np.log1p, _inv_log1p, validate=True)),
            ("scale", preprocessing.MinMaxScaler()),
        ])
        self.__tree_transform = GNTOFeaturizer()
        self.__have_cache_data = have_cache_data
        self.__n = 0

    def __log(self, *args):
        if self.__verbose:
            print(*args)

    def num_items_trained_on(self):
        return self.__n

    def load(self, path):
        with open(os.path.join(path, "n"), "rb") as f:
            self.__n = joblib.load(f)
        with open(os.path.join(path, "x_transform"), "rb") as f:
            self.__tree_transform = joblib.load(f)
        self.__net = GNTOModel(
            num_node_types=self.__tree_transform.num_node_types(),
            num_cols=self.__tree_transform.num_cols(),
            num_ops=self.__tree_transform.num_ops(),
        )
        self.__net.load_state_dict(
            torch.load(os.path.join(path, "nn_weights"), weights_only=False))
        self.__net.eval()
        with open(os.path.join(path, "y_transform"), "rb") as f:
            self.__pipeline = joblib.load(f)

    def save(self, path):
        os.makedirs(path, exist_ok=True)
        torch.save(self.__net.state_dict(), os.path.join(path, "nn_weights"))
        with open(os.path.join(path, "y_transform"), "wb") as f:
            joblib.dump(self.__pipeline, f)
        with open(os.path.join(path, "x_transform"), "wb") as f:
            joblib.dump(self.__tree_transform, f)
        with open(os.path.join(path, "n"), "wb") as f:
            joblib.dump(self.__n, f)

    def fit(self, X, y, epochs=100):
        if isinstance(y, list):
            y = np.array(y)

        X = [json.loads(x) if isinstance(x, str) else x for x in X]
        self.__n = len(X)

        y = self.__pipeline.fit_transform(y.reshape(-1, 1)).astype(np.float32)

        self.__tree_transform.fit(X)
        graphs = self.__tree_transform.transform(X)

        data_list = []
        for g, target in zip(graphs, y):
            g.y = torch.as_tensor(target, dtype=torch.float).view(1)
            data_list.append(g)

        if self.__net is None:
            self.__net = GNTOModel(
                num_node_types=self.__tree_transform.num_node_types(),
                num_cols=self.__tree_transform.num_cols(),
                num_ops=self.__tree_transform.num_ops(),
            )

        cuda = torch.cuda.is_available()
        if cuda:
            self.__net = self.__net.cuda()

        optimizer = torch.optim.Adam(self.__net.parameters())
        loss_fn = torch.nn.MSELoss()
        dataset = DataLoader(data_list, batch_size=16, shuffle=True)

        for epoch in range(epochs):
            loss_accum = 0
            num_batches = 0
            self.__net.train()
            for batch in dataset:
                if cuda:
                    batch = batch.cuda()
                pred = self.__net(batch)
                loss = loss_fn(pred.view(-1), batch.y.view(-1))
                loss_accum += loss.item()
                num_batches += 1
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_accum /= max(1, num_batches)
            if epoch % 15 == 0:
                self.__log("Epoch", epoch, "training loss:", loss_accum)

    def predict(self, X, module_assigner=None):
        if not isinstance(X, list):
            X = [X]
        X = [json.loads(x) if isinstance(x, str) else x for x in X]

        graphs = self.__tree_transform.transform(X)
        loader = DataLoader(graphs, batch_size=len(graphs), shuffle=False)

        self.__net.eval()
        preds = []
        with torch.no_grad():
            for batch in loader:
                if torch.cuda.is_available():
                    batch = batch.cuda()
                out = self.__net(batch)
                preds.append(out.cpu().numpy())

        if preds:
            pred_raw = np.concatenate(preds, axis=0)
        else:
            pred_raw = np.array([])

        return self.__pipeline.inverse_transform(pred_raw)
