"""
Template-disjoint split builder for the QF-100k corpus (Exp1, ``--split template``).

    python exp/exp1_qerror/build_template_split.py      # -> data/qf/template_split.json

Corpus: data/train_plan_part{0..19}.csv (100,000 plans, one per distinct query, ids 0..99999).

TEMPLATE SIGNATURE of a plan = tuple of three frozensets extracted from the plan JSON:
  1. relation names        (every 'Relation Name' in the tree)
  2. join-condition strings (Hash Cond / Join Filter / non-numeric Index Cond,
                             normalized exactly like QueryFormer's formatJoin:
                             alias-qualified, the two sides sorted)
  3. filter COLUMN names   (columns referenced in Filter / numeric Index Cond /
                             Recheck Cond; literal values and operators ignored)

Two queries that differ only in constants therefore share a signature.

SPLIT: whole templates greedily assigned to train/val/test ~ 80/10/10 by query
count. Largest templates go to train first until train reaches 80k; each
remaining template (still in descending size order) goes to whichever of
val/test currently holds fewer queries. No template spans two sets.

Output: data/qf/template_split.json (id lists + stats).
"""
import json
import os
import re
import sys
from collections import defaultdict

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.qf import QF_DATA_DIR, plan_csv_path  # noqa: E402

OUT_PATH = os.path.join(QF_DATA_DIR, "template_split.json")

# column identifier directly left of a comparison operator
COL_RE = re.compile(r"([A-Za-z_][A-Za-z0-9_.]*)\s*(?:=|<>|<=|>=|<|>)")


def plan_signature(plan):
    rels, joins, filtcols = set(), set(), set()

    def walk(node, inherited_alias):
        alias = node.get("Alias", inherited_alias)
        if "Relation Name" in node:
            rels.add(node["Relation Name"])

        # ---- join conditions (mirrors QueryFormer model/database_util.formatJoin)
        join = None
        if "Hash Cond" in node:
            join = node["Hash Cond"]
        elif "Join Filter" in node:
            join = node["Join Filter"]
        elif "Index Cond" in node and len(node["Index Cond"]) >= 2 \
                and not node["Index Cond"][-2].isnumeric():
            join = node["Index Cond"]
        if join is not None:
            two = join[1:-1].split(" = ")
            if len(two) == 2:
                two = [(alias + "." + c) if (alias and "." not in c) else c
                       for c in two]
                joins.add(" = ".join(sorted(two)))
            else:  # unexpected shape -> keep raw string (still literal-free)
                joins.add(join)

        # ---- filter columns (mirrors formatFilter routing; keep columns only)
        filts = []
        if "Filter" in node:
            filts.append(node["Filter"])
        if "Index Cond" in node and len(node["Index Cond"]) >= 2 \
                and node["Index Cond"][-2].isnumeric():
            filts.append(node["Index Cond"])
        if "Recheck Cond" in node:
            filts.append(node["Recheck Cond"])
        for f in filts:
            for col in COL_RE.findall(f):
                filtcols.add((alias + "." + col) if (alias and "." not in col)
                             else col)

        for ch in node.get("Plans", []):
            walk(ch, alias)

    walk(plan, None)
    return (frozenset(rels), frozenset(joins), frozenset(filtcols))


def main():
    print("Loading corpus...")
    dfs = [pd.read_csv(plan_csv_path(i)) for i in range(20)]
    df = pd.concat(dfs, ignore_index=True)
    assert len(df) == 100000, f"expected 100k rows, got {len(df)}"
    assert df["id"].is_unique

    print("Computing template signatures...")
    tmpl = defaultdict(list)          # full signature -> [query ids]
    tmpl_coarse = defaultdict(list)   # relations+joins only -> [query ids]
    for qid, js in zip(df["id"].tolist(), df["json"].tolist()):
        plan = json.loads(js)["Plan"]
        sig = plan_signature(plan)
        tmpl[sig].append(int(qid))
        tmpl_coarse[(sig[0], sig[1])].append(int(qid))

    n_full, n_coarse = len(tmpl), len(tmpl_coarse)
    print(f"Distinct FULL templates (rels+joins+filter-cols): {n_full}")
    print(f"Distinct COARSE templates (rels+joins only):      {n_coarse}")

    # sanity-check granularity: if nearly every query is its own template, coarsen
    coarsened = False
    if n_full > 0.5 * len(df):
        print("FULL signature too fine (most queries singleton) -> coarsening "
              "to relation-set + join-set only.")
        tmpl = tmpl_coarse
        coarsened = True

    sizes = sorted((len(v) for v in tmpl.values()), reverse=True)
    total = sum(sizes)
    top10 = sizes[:10]
    singletons = sum(1 for s in sizes if s == 1)
    stats = {
        "n_queries": total,
        "n_templates": len(tmpl),
        "coarsened_to_rels_plus_joins": coarsened,
        "n_templates_full_signature": n_full,
        "n_templates_coarse_signature": n_coarse,
        "largest_template_size": sizes[0],
        "largest_template_share": sizes[0] / total,
        "top10_template_sizes": top10,
        "median_template_size": sizes[len(sizes) // 2],
        "singleton_templates": singletons,
    }
    print(json.dumps(stats, indent=2))

    # ---- greedy whole-template assignment, deterministic order
    order = sorted(tmpl.items(), key=lambda kv: (-len(kv[1]), min(kv[1])))
    train_ids, val_ids, test_ids = [], [], []
    train_templates = val_templates = test_templates = 0
    TRAIN_TARGET = int(0.8 * total)
    for sig, ids in order:
        if len(train_ids) < TRAIN_TARGET:
            train_ids.extend(ids)
            train_templates += 1
        elif len(val_ids) <= len(test_ids):
            val_ids.extend(ids)
            val_templates += 1
        else:
            test_ids.extend(ids)
            test_templates += 1

    split_stats = {
        "train_queries": len(train_ids), "train_templates": train_templates,
        "val_queries": len(val_ids), "val_templates": val_templates,
        "test_queries": len(test_ids), "test_templates": test_templates,
    }
    print(json.dumps(split_stats, indent=2))
    assert len(train_ids) + len(val_ids) + len(test_ids) == total
    assert not (set(train_ids) & set(val_ids))
    assert not (set(train_ids) & set(test_ids))
    assert not (set(val_ids) & set(test_ids))

    out = {
        "description": "Template-disjoint 3-way split of QF-100k "
                       "(train_plan_part0..19). No template spans two sets.",
        "signature_definition": "frozenset(relation names) + frozenset("
                                "formatJoin-normalized join condition strings) "
                                "+ frozenset(filter column names; literals and "
                                "operators ignored)",
        "assignment": "templates sorted by size desc; to train until 80k, then "
                      "each to the smaller of val/test",
        "stats": stats,
        "split_stats": split_stats,
        "train": sorted(train_ids),
        "val": sorted(val_ids),
        "test": sorted(test_ids),
    }
    out_path = OUT_PATH
    with open(out_path, "w") as f:
        json.dump(out, f)
    print(f"Saved {out_path}")


if __name__ == "__main__":
    main()
