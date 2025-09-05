from __future__ import annotations
from typing import Dict, Iterable, List, Tuple
import csv, re, numpy as np
from .DataPreprocessor import PlanNode

# -------- 通用工具 --------
def collect_all_nodes(roots: Iterable[PlanNode]) -> List[PlanNode]:
    nodes: List[PlanNode] = []
    def dfs(n: PlanNode):
        nodes.append(n)
        for c in getattr(n, "children", []):
            dfs(c)
    for r in roots:
        dfs(r)
    return nodes

def _safe_float(v) -> float:
    try:
        if v is None: return 0.0
        s = str(v).replace(",", "").strip()
        return float(s)
    except Exception:
        return 0.0


# -------- 列词表 + 列统计（来自 CSV）--------
# CSV 格式: name,min,max,cardinality,num_unique_values
# 例如: t.id,1,2528312,2528312,2528312
def load_column_stats(csv_path: str):
    rows = []
    with open(csv_path, newline='', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append({
                "name": r["name"].strip(),
                "min": _safe_float(r.get("min")),
                "max": _safe_float(r.get("max")),
                "cardinality": _safe_float(r.get("cardinality")),
                "num_unique_values": _safe_float(r.get("num_unique_values")),
            })
    # 词表
    col_vocab: Dict[str, int] = {"<UNK_COL>": 0}
    for r in rows:
        if r["name"] not in col_vocab:
            col_vocab[r["name"]] = len(col_vocab)

    # 为每个数值字段做 log1p 缩放统计
    def compute_mu_std(key: str):
        arr = np.array([np.log1p(max(0.0, rr[key])) for rr in rows], dtype=np.float32)
        if arr.size == 0:
            return (0.0, 1.0)
        return (float(arr.mean()), float(arr.std() + 1e-6))

    scalers = {
        "min": compute_mu_std("min"),
        "max": compute_mu_std("max"),
        "cardinality": compute_mu_std("cardinality"),
        "num_unique_values": compute_mu_std("num_unique_values"),
    }
    # 查表字典
    col_stats_map = {r["name"]: r for r in rows}
    return col_vocab, scalers, col_stats_map

# NodeEncoder阶段处理不同NodeType的不同key

# Join Cond字段处理 格式为 (t.id = mc.movie_id)
def process_join_cond_field(join_cond_field: str) -> List[str]:
    s = join_cond_field.strip("()")
    parts = s.split()
    field, op, value = parts
    return [field, op, value]


# Index Cond字段处理 格式为 (id = 154)
def process_index_cond_field(index_cond_field: str) -> List[str]:
    s = index_cond_field.strip("()")
    parts = s.split()
    field, op, value = parts
    return [field, op, value]