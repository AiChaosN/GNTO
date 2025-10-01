from __future__ import annotations
from typing import Dict, Iterable, List, Tuple, Any, Iterable, Tuple, Set, DefaultDict
import csv, re, numpy as np, json
from .DataPreprocessor import PlanNode
from collections import defaultdict

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

class StatisticsInfo:
    """
    matrix_plans: List[List[dict_or_obj]]
      - Outer list: multiple plans
      - Inner list: nodes of a plan
      - Each node is a dict, or an object exposing getExtraInfo() -> dict
    """
    PREDICATE_KEYS = {"Recheck Cond", "Index Cond", "Filter"}
    PRED_ALLOWED = re.compile(r"[=><]")

    def __init__(self,
                 matrix_plans: List[List[Any]],
                 sample_threshold: int = 100,
                 sample_k: int = 10,
                 strict_alias_check: bool = True):
        self.matrix_plans = matrix_plans
        self.sample_threshold = sample_threshold
        self.sample_k = sample_k
        self.strict_alias_check = strict_alias_check

        # caches
        self._node_type_set: Set[str] = set()
        self._all_keys_union: Set[str] = set()
        self._all_keys_intersection: Set[str] | None = None
        self._key_to_values: DefaultDict[str, Set[Any]] = defaultdict(set)

        # per node-type unions/intersections
        self._type_union: DefaultDict[str, Set[str]] = defaultdict(set)
        self._type_intersection: Dict[str, Set[str]] = {}

        # data quality
        self._alias_violations: List[Dict[str, Any]] = []
        self._bad_predicates: List[Tuple[str, str]] = []

        self._built = False

    # ---------- helpers ----------
    @staticmethod
    def _as_dict(node: Any) -> Dict[str, Any]:
        if isinstance(node, dict):
            return node
        if hasattr(node, "getExtraInfo"):
            return node.getExtraInfo()
        raise TypeError("node must be a dict or expose getExtraInfo() -> dict")

    def _iter_all_nodes(self) -> Iterable[Dict[str, Any]]:
        for plan in self.matrix_plans:
            for node in plan:
                yield self._as_dict(node)

    # ---------- main build ----------
    def build(self):
        for dic in self._iter_all_nodes():
            keys = set(dic.keys())
            node_type = dic.get("Node Type", "Unknown")
            self._node_type_set.add(node_type)

            # global union / intersection
            self._all_keys_union.update(keys)
            if self._all_keys_intersection is None:
                self._all_keys_intersection = set(keys)
            else:
                self._all_keys_intersection &= keys

            # per-type union / intersection
            if node_type not in self._type_intersection:
                self._type_intersection[node_type] = set(keys)
            else:
                self._type_intersection[node_type] &= keys
            self._type_union[node_type].update(keys)

            # key -> unique values (hashable only; else JSON-stringify)
            for k, v in dic.items():
                try:
                    self._key_to_values[k].add(v)
                except TypeError:
                    self._key_to_values[k].add(json.dumps(v, ensure_ascii=False, sort_keys=True))

            # quality: Relation Name exists but Alias missing
            if self.strict_alias_check and dic.get("Relation Name") and not dic.get("Alias"):
                self._alias_violations.append({
                    "Node Type": node_type,
                    "Relation Name": dic.get("Relation Name"),
                })

            # quality: predicates should contain one of =, >, <
            for k in self.PREDICATE_KEYS:
                val = dic.get(k)
                if isinstance(val, str) and val.strip():
                    if self.PRED_ALLOWED.search(val) is None:
                        self._bad_predicates.append((k, val))

        if self._all_keys_intersection is None:
            self._all_keys_intersection = set()

        self._built = True
        return self

    # ---------- public APIs (English names) ----------

    def get_node_type_set(self) -> Set[str]:
        """All observed node types."""
        assert self._built, "call build() first"
        return set(self._node_type_set)

    def global_must_keys(self) -> Set[str]:
        """Keys present in ALL nodes (global intersection)."""
        assert self._built, "call build() first"
        return set(self._all_keys_intersection)

    def global_all_keys(self) -> Set[str]:
        """All keys ever observed (global union)."""
        assert self._built, "call build() first"
        return set(self._all_keys_union)

    def per_key_values(self) -> Dict[str, Dict[str, Any]]:
        """
        Returns: { key: {"count": #unique, "values": list (sample if large)} }
        """
        assert self._built, "call build() first"
        out: Dict[str, Dict[str, Any]] = {}
        for k, s in self._key_to_values.items():
            cnt = len(s)
            if cnt > self.sample_threshold:
                out[k] = {"count": cnt, "values": list(s)[:self.sample_k]}
            else:
                out[k] = {"count": cnt, "values": list(s)}
        return out

    def per_nodetype_key_stats(self, strict: bool = True) -> Dict[str, Dict[str, Set[str]]]:
        """
        For each node type, return:
          - must_all:   keys that are present in ALL nodes of this type
          - must_only_type: keys that are must for THIS type but not global_must
          - max:        keys that ever appeared for this type
          - optional:   keys that are not must within this type (max - must_all)
        """
        assert self._built, "call build() first"
        res: Dict[str, Dict[str, Set[str]]] = {}
        global_must = set(self._all_keys_intersection)

        for nt in self._node_type_set:
            must = set(self._type_intersection.get(nt, set()))
            mx   = set(self._type_union.get(nt, set()))

            if strict and not global_must.issubset(must):
                missing = global_must - must
                raise AssertionError(f"[data anomaly] {nt} misses global must keys: {missing}")

            optional = mx - must
            must_only_type = must - global_must

            res[nt] = {
                "must_all": must,
                "must_only_type": must_only_type,  # === 类型特有必有 ===
                "max": mx,
                "optional": optional,              # 已不含 must_all（因此也不含 global_must）
            }
        return res

    def report_issues(self) -> Dict[str, Any]:
        """Data-quality report: alias violations & bad predicate samples."""
        assert self._built, "call build() first"
        return {
            "alias_violations": list(self._alias_violations),
            "bad_predicates": list(self._bad_predicates),
        }

    def pretty_print_report(self):
        """Optional pretty-printer."""
        assert self._built, "call build() first"
        print(f"[Node Types] {len(self._node_type_set)}: {sorted(self._node_type_set)}\n")
        print(f"[Global MUST keys] {len(self._all_keys_intersection)}: {sorted(self._all_keys_intersection)}\n")
        print(f"[Global ALL keys] {len(self._all_keys_union)}: {sorted(self._all_keys_union)}\n")

        kv = self.per_key_values()
        print("[Per-key unique values] (sample if large)")
        for k in sorted(kv.keys()):
            info = kv[k]
            print(f"  - {k}: {info['count']}  sample: {info['values']}")

        print("\n[Per NodeType key stats]")
        per = self.per_nodetype_key_stats()
        for nt in sorted(per.keys()):
            must = sorted(per[nt]["must_all"])
            must_only = sorted(per[nt]["must_only_type"])
            mx   = sorted(per[nt]["max"])
            opt  = sorted(per[nt]["optional"])
            print(f"## {nt}")
            print(f"   must_all({len(must)}): {must}")
            print(f"   must_only_type({len(must_only)}): {must_only}")
            print(f"   max({len(mx)}): {mx}")
            print(f"   optional({len(opt)}): {opt}")
