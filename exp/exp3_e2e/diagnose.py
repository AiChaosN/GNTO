"""Per-steerer diagnostics of the selected plans, from ``predictions.csv``:

* how far the chosen plan's predicted runtime is from its measured runtime (and the same for
  the PG-default and oracle plans)
* prediction / actual ratio grouped by the join methods and scan types used in the plan
* the hint sets each steerer picks vs. the oracle's
* worked examples: the queries with the largest regret, top-5 by prediction vs. actual

Usage::

    python exp/exp3_e2e/diagnose.py --steerer gnto          # -> results/exp3_e2e/tables/diagnose_gnto.md
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import Counter

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HINT_PLANS, RESULTS, candidates, load_packs, md  # noqa: E402
from hint_sets import HINT_SETS  # noqa: E402

JOINS = ("Nested Loop", "Hash Join", "Merge Join")
SCANS = ("Seq Scan", "Index Scan", "Index Only Scan", "Bitmap Heap Scan")


def plan_shape(plan):
    c = Counter()
    stack = [plan["Plan"]]
    while stack:
        n = stack.pop()
        c[n["Node Type"]] += 1
        stack.extend(n.get("Plans", []))
    joins = "+".join(j.split()[0] for j in JOINS if c[j]) or "none"
    scans = "+".join({"Seq Scan": "Seq", "Index Scan": "Idx", "Index Only Scan": "IdxOnly", "Bitmap Heap Scan": "Bitmap"}[s] for s in SCANS if c[s]) or "none"
    return joins, scans, c


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steerer", default="gnto")
    ap.add_argument("--results", default=RESULTS)
    ap.add_argument("--hint-plans", default=HINT_PLANS)
    ap.add_argument("--examples", type=int, default=3)
    a = ap.parse_args()
    d = os.path.join(a.results, a.steerer)
    preds = pd.read_csv(os.path.join(d, "predictions.csv"))
    picks = pd.read_csv(os.path.join(d, "picks.csv"))
    packs = {p["query_id"]: p for p in load_packs(a.hint_plans)}
    hint_flags = {h["hint_id"]: h["flags"] for h in HINT_SETS}

    # shapes of every candidate plan
    shape = {}
    for qid, pack in packs.items():
        for h, plan, _ in candidates(pack):
            shape[(qid, h)] = plan_shape(plan)
    preds["joins"] = [shape[(q, h)][0] for q, h in zip(preds.query_id, preds.hint_id)]
    preds["scans"] = [shape[(q, h)][1] for q, h in zip(preds.query_id, preds.hint_id)]
    preds["ratio"] = preds.pred / preds.actual_ms          # < 1: under-estimated

    # 1. pick vs default vs oracle: predicted / actual
    rows = []
    for _, r in picks.iterrows():
        g = preds[preds.query_id == r.query_id].set_index("hint_id")
        for name, h in (("pick", r.pick_hint), ("pg_default", 0), ("oracle", r.oracle_hint)):
            if h in g.index:
                rows.append({"query_id": r.query_id, "plan": name, "pred_ms": g.loc[h, "pred"], "actual_ms": g.loc[h, "actual_ms"]})
    pv = pd.DataFrame(rows)
    pv["ratio"] = pv.pred_ms / pv.actual_ms
    t1 = pv.groupby("plan").agg(n=("ratio", "size"), median_pred_over_actual=("ratio", "median"),
                                 share_underestimated_2x=("ratio", lambda x: float((x < 0.5).mean())),
                                 median_pred_ms=("pred_ms", "median"), median_actual_ms=("actual_ms", "median")).reset_index()

    # 2. prediction / actual by plan shape (all candidates)
    t2 = (preds.groupby("joins").agg(n=("ratio", "size"), median_pred_over_actual=("ratio", "median"),
                                     median_actual_ms=("actual_ms", "median")).reset_index().sort_values("median_pred_over_actual"))
    t2b = (preds.groupby("scans").agg(n=("ratio", "size"), median_pred_over_actual=("ratio", "median"),
                                      median_actual_ms=("actual_ms", "median")).reset_index().sort_values("median_pred_over_actual"))

    # 3. join methods of the chosen plans vs oracle vs PG default
    def joins_of(col):
        return Counter(shape[(q, h)][0] for q, h in zip(picks.query_id, picks[col]))
    t3 = pd.DataFrame([{"which": w, **joins_of(c)} for w, c in (("pick", "pick_hint"), ("oracle", "oracle_hint"))]).fillna(0)
    t3.insert(1, "pg_default", 0); t3 = t3.drop(columns="pg_default")
    pgd = Counter(shape[(q, 0)][0] for q in picks.query_id if (q, 0) in shape)
    t3 = pd.concat([t3, pd.DataFrame([{"which": "pg_default", **pgd}])]).fillna(0)

    # which toggles are off in the picks
    def off_share(col):
        fl = [hint_flags[h] for h in picks[col]]
        return {t: float(np.mean([not f[t] for f in fl])) for t in fl[0]}
    t4 = pd.DataFrame([{"which": "pick", **off_share("pick_hint")}, {"which": "oracle", **off_share("oracle_hint")}])

    # 4. spread of the predictions vs. spread of the measured runtimes per query (max / min)
    rng = preds.groupby("query_id").agg(pred_max_over_min=("pred", lambda x: x.max() / max(x.min(), 1e-6)),
                                        actual_max_over_min=("actual_ms", lambda x: x.max() / max(x.min(), 1e-6)))
    t5 = pd.DataFrame([{"median_pred_range": rng.pred_max_over_min.median(), "median_actual_range": rng.actual_max_over_min.median(),
                        "queries_where_pred_range_lt_2x": int((rng.pred_max_over_min < 2).sum()), "n_queries": len(rng)}])

    # 5. worked examples: the queries that cost the most seconds relative to PG default
    picks["loss_vs_pg_s"] = (picks.pick_ms - picks.pg_default_ms) / 1000
    ranks = []
    for _, r in picks.iterrows():
        g = preds[preds.query_id == r.query_id].sort_values("pred").reset_index(drop=True)
        rank_of = lambda h: int(g.index[g.hint_id == h][0]) + 1 if (g.hint_id == h).any() else None
        ranks.append({"query_id": r.query_id, "n": len(g), "rank_pg_default_by_model": rank_of(0), "rank_oracle_by_model": rank_of(r.oracle_hint),
                      "rank_pick_by_actual": int(g.sort_values("actual_ms").reset_index(drop=True).index[g.sort_values("actual_ms").reset_index(drop=True).hint_id == r.pick_hint][0]) + 1})
    rk = pd.DataFrame(ranks)
    t6 = pd.DataFrame([{"median_rank_of_pg_default_by_model": rk.rank_pg_default_by_model.median(),
                        "median_rank_of_oracle_by_model": rk.rank_oracle_by_model.median(),
                        "median_actual_rank_of_pick": rk.rank_pick_by_actual.median(),
                        "queries_total_loss_vs_pg_s": float(picks.loss_vs_pg_s.sum()),
                        "loss_from_10_worst_queries_s": float(picks.sort_values("loss_vs_pg_s", ascending=False).head(10).loss_vs_pg_s.sum())}])
    ex = []
    for _, r in picks.sort_values("loss_vs_pg_s", ascending=False).head(a.examples).iterrows():
        g = preds[preds.query_id == r.query_id].sort_values("pred").head(5)
        pgrow = preds[(preds.query_id == r.query_id) & (preds.hint_id == 0)]
        ex.append(f"\n**{r.query_id}** (+{r.loss_vs_pg_s:.1f}s vs PG default {r.pg_default_ms/1000:.2f}s; oracle {r.oracle_ms/1000:.2f}s = hint {int(r.oracle_hint)}; "
                  f"PG default predicted {float(pgrow.pred.iloc[0])/1000:.2f}s, ranked #{int(rk[rk.query_id == r.query_id].rank_pg_default_by_model.iloc[0])} of {len(preds[preds.query_id == r.query_id])}) - top 5 by prediction:\n")
        ex.append(md(g[["hint_id", "pred", "actual_ms", "joins", "scans"]].assign(pred=lambda x: x.pred / 1000, actual_ms=lambda x: x.actual_ms / 1000)
                     .rename(columns={"pred": "pred_s", "actual_ms": "actual_s"}), "{:.2f}"))

    out = os.path.join(a.results, "tables", f"diagnose_{a.steerer}.md")
    with open(out, "w") as f:
        f.write(f"# Plan-selection diagnostics: {a.steerer}\n\n## Predicted / actual for the chosen, PG-default and oracle plan\n" + md(t1)
                + "\n## Prediction / actual by join methods in the plan (all candidates)\n" + md(t2)
                + "\n## Prediction / actual by scan types (all candidates)\n" + md(t2b)
                + "\n## Join methods of the chosen plans\n" + md(t3)
                + "\n## Share of picks with each toggle OFF\n" + md(t4)
                + "\n## Spread of predictions vs. spread of measured runtimes within a query (max / min)\n" + md(t5)
                + "\n## Ranks\n" + md(t6)
                + "\n## Queries that cost the most seconds vs PG default\n" + "".join(ex))
    fmt = lambda v: f"{v:.3f}"
    print("== predicted / actual: chosen vs PG default vs oracle plan ==\n" + t1.to_string(index=False, float_format=fmt))
    print("\n== prediction / actual by join methods ==\n" + t2.to_string(index=False, float_format=fmt))
    print("\n== prediction / actual by scan types ==\n" + t2b.to_string(index=False, float_format=fmt))
    print("\n== join methods of chosen plans ==\n" + t3.to_string(index=False))
    print("\n== share of picks with toggle OFF ==\n" + t4.to_string(index=False, float_format=fmt))
    print("\n== spread within a query ==\n" + t5.to_string(index=False, float_format=fmt))
    print("\n== ranks ==\n" + t6.to_string(index=False, float_format=fmt))
    print("".join(ex))
    print(f"-> {out}")


if __name__ == "__main__":
    main()
