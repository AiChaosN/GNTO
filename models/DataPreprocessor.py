"""Utilities for turning raw query plans into structured objects.

The real project uses database specific JSON plans.  For the purpose of this
skeleton repository we provide a tiny, generic representation so that other
modules can operate on a predictable structure.  The :class:`DataPreprocessor`
turns nested dictionaries that resemble PostgreSQL's ``EXPLAIN`` output into
:class:`PlanNode` trees.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Optional, Tuple
import numpy as np
import pandas as pd
import re
import glob
import json
import math

@dataclass
class PlanNode:
    node_type: str
    children: List["PlanNode"] = field(default_factory=list)
    extra_info: Dict[str, Any] = field(default_factory=dict)
    node_vector: Optional[np.ndarray] = None

    def getExtraInfo(self) -> Dict[str, Any]:
        return self.extra_info

#####################
# 获取json文件数据 #
#####################
def get_plans_dict(path: str):
    # 匹配所有 train_plan_0*.csv
    files = glob.glob(path)
    print("找到的文件:", files)

    # 读入并合并
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

    print("总数据行数:", len(df))
    print("df:\n", df.head())

    #获取json字符串
    plans_json = df['json']
    print("plans_json:\n", plans_json.iloc[0])

    #字符串转json
    plans_dict = []
    ExecutionTimes = []
    idx = 0
    for json_str in plans_json:
        idx += 1
        plan_dict = json.loads(json_str)
        plans_dict.append(plan_dict['Plan'])
        try:
            ExecutionTimes.append(plan_dict['Execution Time'])
        except:
            print(f"idx: {idx} 不存在Execution Time")
            print(plan_dict)
    return plans_dict, ExecutionTimes

class DataPreprocessor:
    """Convert raw JSON style plans into :class:`PlanNode` trees."""

    plan_key: str = "Plans"  # key that contains child nodes in the raw dict

    def preprocess(self, plan: Dict[str, Any]) -> PlanNode:
        """Recursively convert ``plan`` into a :class:`PlanNode` tree.

        Parameters
        ----------
        plan:
            JSON-like dictionary describing a plan node.  Nested plans are
            expected under ``self.plan_key``.
        """

        node = PlanNode(
            node_type=plan.get("Node Type", "Unknown"),
            extra_info={k: v for k, v in plan.items() if k != self.plan_key},
        )

        for child in plan.get(self.plan_key, []):
            node.children.append(self.preprocess(child))

        return node

    # Small convenience wrappers -------------------------------------------------
    def preprocess_all(self, plans: Iterable[Dict[str, Any]]) -> List[PlanNode]:
        """Preprocess a list of raw plans."""

        return [self.preprocess(p) for p in plans]
    
    # 下面是树可视化函数 两个函数 display_tree 和 get_tree_stats ------------------------------------------------
    def display_tree(self, node: PlanNode, show_details: bool = True, max_depth: int = None) -> str:
        lines = []
        self._build_tree_lines(node, lines, "", True, show_details, 0, max_depth)
        return "\n".join(lines)
    
    def print_tree(self, node: PlanNode, show_details: bool = True, max_depth: int = None):
        print(self.display_tree(node, show_details, max_depth))
    
    def get_tree_stats(self, node: PlanNode) -> Dict[str, Any]:
        stats = {
            'total_nodes': 0,
            'max_depth': 0,
            'node_types': {},
            'nodes_by_depth': {},
            'leaf_nodes': 0,
            'internal_nodes': 0
        }
        
        self._collect_tree_stats(node, stats, 0)
        
        return stats
    
    def _build_tree_lines(self, node: PlanNode, lines: List[str], prefix: str, 
                         is_last: bool, show_details: bool, current_depth: int, max_depth: int):
        
        # Check depth limit
        if max_depth is not None and current_depth > max_depth:
            return
        
        # Create the tree structure symbols
        connector = "└── " if is_last else "├── "
        
        # Build the main line
        main_line = f"{prefix}{connector}{node.node_type}"
        
        # Add details if requested
        if show_details and node.extra_info:
            details = []
            
            # Show important cost/performance info
            for key in ['Total Cost', 'Startup Cost', 'Plan Rows', 'Plan Width', 
                       'Actual Total Time', 'Actual Rows']:
                if key in node.extra_info:
                    value = node.extra_info[key]
                    if isinstance(value, float):
                        details.append(f"{key}: {value:.2f}")
                    else:
                        details.append(f"{key}: {value}")
            
            # Show relation/index names
            for key in ['Relation Name', 'Alias', 'Index Name']:
                if key in node.extra_info:
                    details.append(f"{key}: {node.extra_info[key]}")
            
            # Show join type
            if 'Join Type' in node.extra_info:
                details.append(f"Join Type: {node.extra_info['Join Type']}")
            
            if details:
                main_line += f" ({', '.join(details[:])})"
                # main_line += f", node_vector: {node.node_vector}"
                if node.node_vector is not None:
                    main_line += f", node_vector_shape: {node.node_vector.shape}"
                # if len(details) > 3:
                #     main_line += f" [+{len(details)-3} more]"
        
        lines.append(main_line)
        
        # Prepare prefix for children
        if is_last:
            child_prefix = prefix + "    "
        else:
            child_prefix = prefix + "│   "
        
        # Process children
        if node.children:
            for i, child in enumerate(node.children):
                is_last_child = (i == len(node.children) - 1)
                self._build_tree_lines(child, lines, child_prefix, is_last_child, 
                                     show_details, current_depth + 1, max_depth)
    
    def _collect_tree_stats(self, node: PlanNode, stats: Dict[str, Any], depth: int):
        """Recursively collect statistics about the tree."""
        
        # Update total nodes
        stats['total_nodes'] += 1
        
        # Update max depth
        stats['max_depth'] = max(stats['max_depth'], depth)
        
        # Update node types count
        node_type = node.node_type
        stats['node_types'][node_type] = stats['node_types'].get(node_type, 0) + 1
        
        # Update nodes by depth
        stats['nodes_by_depth'][depth] = stats['nodes_by_depth'].get(depth, 0) + 1
        
        # Check if leaf or internal node
        if node.children:
            stats['internal_nodes'] += 1
            # Process children
            for child in node.children:
                self._collect_tree_stats(child, stats, depth + 1)
        else:
            stats['leaf_nodes'] += 1

#####################
# PlanNode 转 Graph #
#####################
def plan_trees_to_graphs(
    plan_roots: List,
    add_self_loops: bool = False,
    undirected: bool = False,
) -> Tuple[List[List[Tuple[int, int]]], List[List[Any]]]:
    graphs_edges: List[List[Tuple[int, int]]] = []
    graphs_nodes: List[List[Any]] = []

    for root in plan_roots:
        edges: List[Tuple[int, int]] = []
        nodes: List[Any] = []

        def dfs(node, parent_idx: int | None):
            idx = len(nodes)
            nodes.append(getattr(node, "extra_info", None))

            if add_self_loops:
                edges.append((idx, idx))

            if parent_idx is not None:
                edges.append((parent_idx, idx))
                if undirected:
                    edges.append((idx, parent_idx))

            # 安全访问 children
            for ch in getattr(node, "children", []) or []:
                dfs(ch, idx)

        dfs(root, None)
        graphs_edges.append(edges)
        graphs_nodes.append(nodes)

    return graphs_edges, graphs_nodes

class DFStatisticsInfo:
    """
    直接对现成的 DataFrame 做统计：
      - 节点类型集合
      - 全局 must/all 键（交集）与 all keys（并集）
      - 每列唯一值计数与样本
      - 按 Node Type 的 must/optional/max
      - 质量问题：Alias 缺失、谓词格式异常
    约定：
      - df 中每行代表一个节点
      - 若无 'Node Type' 列，会自动视为 'Unknown'
    """
    PREDICATE_KEYS = ("Recheck Cond", "Index Cond", "Filter")
    PRED_ALLOWED = re.compile(r"[=><]")

    def __init__(self, 
                 df: pd.DataFrame,
                 sample_threshold: int = 100,
                 sample_k: int = 10,
                 strict_alias_check: bool = True):
        self.df = df.copy()
        if "Node Type" not in self.df.columns:
            self.df["Node Type"] = "Unknown"
        self.df["Node Type"] = self.df["Node Type"].fillna("Unknown").astype("string")

        self.sample_threshold = sample_threshold
        self.sample_k = sample_k
        self.strict_alias_check = strict_alias_check

        # 仅用于统计的列（去掉内部列名规则可自行调整）
        self.data_cols = [c for c in self.df.columns if not c.startswith("_")]

    # ---------- 基础集合 ----------
    def get_node_type_set(self) -> set[str]:
        return set(self.df["Node Type"].dropna().unique())

    def global_must_keys(self) -> set[str]:
        """所有行都非空(notna)的列集合 = 列交集（必有键）"""
        mask = self.df[self.data_cols].notna()
        return set(mask.columns[mask.all(axis=0)])

    def global_all_keys(self) -> set[str]:
        """至少一行非空(notna)的列集合 = 列并集（出现过的键）"""
        mask = self.df[self.data_cols].notna()
        return set(mask.columns[mask.any(axis=0)])

    # ---------- 每列唯一值计数与样本 ----------
    @staticmethod
    def _safe_hashable(x):
        try:
            hash(x)
            return x
        except Exception:
            try:
                return json.dumps(x, ensure_ascii=False, sort_keys=True, default=str)
            except Exception:
                return str(x)

    def per_key_values(self) -> dict[str, dict]:
        """
        返回 {col: {"count": 唯一值数, "values": 样本列表}}
        - 为避免不可哈希对象导致报错，这里统一转为“可哈希字符串”对唯一值做近似统计
        """
        out = {}
        for k in self.data_cols:
            s = self.df[k].dropna().map(self._safe_hashable)
            if s.empty:
                out[k] = {"count": 0, "values": []}
                continue
            uniq = s.unique()
            cnt = len(uniq)
            vals = uniq[:self.sample_k].tolist() if cnt > self.sample_threshold else uniq.tolist()
            out[k] = {"count": cnt, "values": vals}
        return out

    # ---------- 按 Node Type 的键统计 ----------
    def per_nodetype_key_stats(self) -> dict[str, dict[str, set[str]]]:
        """
        对每个 Node Type 给出：
          - must_all:   该类型所有行都非空的列
          - must_only_type: 该类型必有但非全局必有
          - max:        该类型出现过的列（任意一行非空）
          - optional:   该类型出现过但并非所有该类型行都有（max - must_all）
        """
        res = {}
        global_must = self.global_must_keys()
        for nt, g in self.df.groupby("Node Type"):
            gmask = g[self.data_cols].notna()
            must = set(gmask.columns[gmask.all(axis=0)])
            mx   = set(gmask.columns[gmask.any(axis=0)])
            optional = mx - must
            must_only_type = must - global_must
            res[nt] = {
                "must_all": must,
                "must_only_type": must_only_type,
                "max": mx,
                "optional": optional,
            }
        return res

    # ---------- 质量检查 ----------
    def report_issues(self) -> dict:
        issues = {"alias_violations": [], "bad_predicates": []}

        # 1) Relation Name 存在但 Alias 缺失
        if self.strict_alias_check and ("Relation Name" in self.df.columns):
            rn_ok = self.df["Relation Name"].notna()
            alias_col = self.df["Alias"] if "Alias" in self.df.columns else pd.Series(index=self.df.index, dtype="object")
            alias_missing = ~alias_col.notna()
            viol = self.df.loc[rn_ok & alias_missing, ["Node Type", "Relation Name"]]
            if len(viol):
                issues["alias_violations"] = viol.to_dict(orient="records")

        # 2) 谓词必须包含 = / > / <
        for key in self.PREDICATE_KEYS:
            if key in self.df.columns:
                s = self.df[key]
                is_str = s.apply(lambda x: isinstance(x, str))
                non_empty = s.where(is_str, "").str.strip().ne("")
                # 不含 =, >, < 的当作 bad
                bad_mask = non_empty & ~s.where(is_str, "").str.contains(self.PRED_ALLOWED)
                # 取若干样本
                bad_vals = s[bad_mask].head(50).tolist()
                for v in bad_vals:
                    issues["bad_predicates"].append((key, v))

        return issues

    # ---------- 可选的简单打印 ----------
    def pretty_print_report(self):
        print(f"[Node Types] {len(self.get_node_type_set())}: {sorted(self.get_node_type_set())}\n")
        gm = sorted(self.global_must_keys())
        ga = sorted(self.global_all_keys())
        print(f"[Global MUST keys] {len(gm)}: {gm}\n")
        print(f"[Global ALL keys]  {len(ga)}: {ga}\n")

        print("[Per-key unique values] (sample if large)")
        kv = self.per_key_values()
        for k in sorted(kv):
            info = kv[k]
            print(f"  - {k}: {info['count']}  sample: {info['values']}")

        print("\n[Per NodeType key stats]")
        per = self.per_nodetype_key_stats()
        for nt in sorted(per):
            must = sorted(per[nt]["must_all"])
            must_only = sorted(per[nt]["must_only_type"])
            mx   = sorted(per[nt]["max"])
            opt  = sorted(per[nt]["optional"])
            print(f"## {nt}")
            print(f"   must_all({len(must)}): {must}")
            print(f"   must_only_type({len(must_only)}): {must_only}")
            print(f"   max({len(mx)}): {mx}")
            print(f"   optional({len(opt)}): {opt}")



#####################
# 图 转 DataFrame  #
#####################
def graphs_to_df(data: list[list[dict]]) -> pd.DataFrame:
    rows = []
    for pid, plan in enumerate(data):
        for nid, node in enumerate(plan):
            rows.append({"plan_id": pid, "node_idx": nid, **node})
    df = pd.json_normalize(rows, sep='.')

    df = df.sort_values(["plan_id", "node_idx"], kind="stable").reset_index(drop=True)
    return df

#####################
# DataFrame 转 图  #
#####################
def df_to_graphs(df: pd.DataFrame, keep_extra_cols=False) -> list[list[dict]]:
    orig_cols = [c for c in df.columns if c not in {"plan_id", "node_idx"}]
    cols = orig_cols if not keep_extra_cols else [c for c in df.columns if c not in {"plan_id", "node_idx"}]

    out = []
    for pid, g in df.sort_values(["plan_id","node_idx"], kind="stable").groupby("plan_id", sort=True):
        plan_nodes = []
        for _, row in g.sort_values("node_idx", kind="stable").iterrows():
            d = {}
            for k in cols:
                v = row[k]
                if v is None or []:
                    continue
                d[k] = v
            plan_nodes.append(d)
        out.append(plan_nodes)
    return out

#####################
# 多个 Cond 解析    #
#####################
_ID = r'(?:[`"]?[A-Za-z_][\w$]*[`"]?(?:\s*\.\s*[`"]?[A-Za-z_][\w$]*[`"]?)*)'
_OP = r'(=|!=|<>|<=|>=|<|>)'
# 右值尽量宽松，后面再判定其类型
_PATTERN = re.compile(rf'^\s*({_ID})\s*{_OP}\s*(.+?)\s*$')

def _strip_quotes(s: str) -> str:
    s = s.strip()
    if (s.startswith("'") and s.endswith("'")) or (s.startswith('"') and s.endswith('"')):
        return s[1:-1]
    if s.startswith('`') and s.endswith('`'):
        return s[1:-1]
    return s

def _parse_rhs(val: str):
    v = val.strip()
    # 字符串字面量
    if (v[:1] in "'\"`" and v[-1:] in "'\"`") and len(v) >= 2:
        return _strip_quotes(v)
    # 数字（int/float）
    try:
        if re.fullmatch(r'[+-]?\d+', v):
            return int(v)
        if re.fullmatch(r'[+-]?\d*\.\d+', v):
            return float(v)
    except ValueError:
        pass
    # 列标识符（含点号）
    ident = re.sub(r'\s+', '', v)  # 去掉点号两侧多余空格
    return _strip_quotes(ident)

def parse_conditions(expr: str):
    """
    将形如：
      ((kind_id > 1) AND (production_year > 1998) AND (mi.movie_id = ci.movie_id) AND (name != "Tom"))
    解析为：
      [['kind_id','>',1], ['production_year','>',1998], ['mi.movie_id','=', 'ci.movie_id'], ['name','!=','Tom']]
    """
    s = expr.strip()
    # 去一层最外括号（可选）
    if s.startswith("(") and s.endswith(")"):
        s = s[1:-1].strip()

    # 按 AND 拆分（忽略大小写，且不处理字符串中的 AND——一般足够）
    parts = re.split(r'\s+AND\s+', s, flags=re.IGNORECASE)

    out = []
    for p in parts:
        seg = p.strip()
        # 去掉包裹的小括号
        if seg.startswith("(") and seg.endswith(")"):
            seg = seg[1:-1].strip()

        m = _PATTERN.match(seg)
        if not m:
            raise ValueError(f"无法解析条件: {seg}")
        lhs, op, rhs = m.groups()
        lhs = _strip_quotes(lhs).replace(' ', '')   # 统一去掉点号周围空格
        rhs = _parse_rhs(rhs)
        out.append([lhs, op, rhs])
    return out

def safe_cond_parse(expr):
    if pd.isna(expr):  # 处理 NaN
        return []
    try:
        return parse_conditions(str(expr))
    except Exception as e:
        print(f"解析失败: {expr}, 错误: {e}")
        return []


########################################################
# 从 db_info 处理谓词列表
########################################################
# 操作符映射
OP_DICT = {"=":0, ">":1, ">=":2, "<":3, "<=":4}

def _build_maps(db_info: pd.DataFrame):
    df = db_info.copy()
    if "name" not in df.columns and {"table_name","column_name"}.issubset(df.columns):
        df["name"] = df["table_name"].astype(str) + "." + df["column_name"].astype(str)

    # full -> id（稳定：按字典序）
    full_sorted = sorted(df["name"].astype(str).tolist())
    full2id = {f:i for i,f in enumerate(full_sorted)}

    # (alias/bare) 反查辅助：bare -> 所有 full（可能多表同名列）
    col_to_fulls = {}
    for _, r in df.iterrows():
        full = str(r["name"])
        bare = full.split(".")[-1]
        col_to_fulls.setdefault(bare, []).append(full)

    # 元信息
    def _f(x):
        try: return float(x)
        except: return None
    meta = {str(r["name"]): {"min":_f(r["min"]), "max":_f(r["max"])} for _, r in df.iterrows()}

    return full2id, col_to_fulls, meta

def _resolve_col(token: str, default_alias: Optional[str], full2id, col_to_fulls, index_name: Optional[str] = None) -> Optional[int]:
    """
    解析列名优先级：
    1) 已带别名：'t.id' 直接匹配
    2) 裸列名 + 行别名：尝试 f'{Alias}.{col}'
    3) 裸列名且在 db_info 唯一：用该唯一列
    否则返回 None（歧义）
    """
    s = str(token)
    if "." in s and s in full2id:
        return full2id[s]

    bare = s
    if default_alias:
        cand = f"{default_alias}.{bare}"
        if cand in full2id:
            return full2id[cand]
    if index_name:
        index_name_str = index_name[len(bare):].split("_")[1:]
        index_name_head = ""
        # print(f"index_name_str: {index_name_str}")
        for i in range(min(2, len(index_name_str))):
            index_name_head += index_name_str[i][0]
        if len(index_name_str) > 2:
            index_name_head += "_" + index_name_str[-1]
        cand = f"{index_name_head}.{bare}"
        if cand in full2id:
            return full2id[cand]
    fulls = col_to_fulls.get(bare, [])
    if len(fulls) == 1:
        return full2id[fulls[0]]

    return None  # 歧义或不存在

def _to_num(x: Any) -> Optional[float]:
    if isinstance(x, (int, float)): return float(x)
    if isinstance(x, str):
        try: return float(x.strip())
        except: return None
    return None

def _scale01(v: float, vmin: Optional[float], vmax: Optional[float]) -> float:
    if vmin is None or vmax is None or vmax <= vmin or math.isnan(v): return 0.0
    t = (v - vmin) / (vmax - vmin)
    return 0.0 if t < 0 else (1.0 if t > 1 else float(t))

def process_predicate_list(
    predicate_list: List[List[Any]],
    db_info: pd.DataFrame,
    default_alias: Optional[str] = None,
    index_name: Optional[str] = None,
    op_dict: Dict[str,int] = OP_DICT
) -> List[Tuple[int, int, Any, bool]]:
    """
    输入：一行的 predicate_list（如 [['id','=',3], ['t.id','=','mk.movie_id']]）
    输出：[(lhs_col_id, op_id, rhs_value, is_join), ...]
         is_join=True → rhs_value 为列 id；False → rhs_value 为 [0,1] 浮点
    """
    full2id, col_to_fulls, meta = _build_maps(db_info)
    out = []
    for trip in predicate_list or []:
        if not isinstance(trip, (list, tuple)) or len(trip) < 3:
            continue
        lhs, op, rhs = trip[0], str(trip[1]), trip[2]
        if op not in op_dict:
            raise KeyError(f"未知操作符: {op}")

        lhs_id = _resolve_col(lhs, default_alias, full2id, col_to_fulls, index_name)
        if lhs_id is None:
            raise KeyError(f"无法解析 lhs 列名（需要别名）：{lhs}，行别名={default_alias}, 索引名={index_name}")

        # 尝试把 rhs 当数字
        rhs_num = _to_num(rhs)
        if rhs_num is not None:
            # 用 lhs 的 min/max 进行缩放
            # 找出 lhs 的 full 名字来取元信息
            # 反查 full 名称：
            lhs_full = [k for k,v in full2id.items() if v == lhs_id][0]
            vmin = meta[lhs_full]["min"]; vmax = meta[lhs_full]["max"]
            rhs_scaled = _scale01(rhs_num, vmin, vmax)
            out.append((lhs_id, op_dict[op], rhs_scaled, False))
        else:
            # rhs 当作列名解析（可带别名；如裸列且无别名，默认也尝试用同一 default_alias）
            rhs_id = _resolve_col(rhs, default_alias, full2id, col_to_fulls, index_name)
            if rhs_id is None:
                raise KeyError(f"无法解析 rhs 列名：{rhs}（请在谓词中带别名或提供唯一列）")
            out.append((lhs_id, op_dict[op], rhs_id, True))
    return out


#####################
# 测试              #
#####################
if __name__ == "__main__":

    # 多个 Cond 解析示例
    print("多个 Cond 解析示例")
    print(parse_conditions("((kind_id > 1) AND (production_year > 1998) AND (mi.movie_id = cimovie_id))"))