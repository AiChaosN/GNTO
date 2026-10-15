"""Error propagation of the optimizer's estimates through the cumulative Total Cost and through the per-operator residual.
Per operator of the 20-database benchmark: cumulative cost c vs cumulative actual time t, and the residual
r = c - sum(children c) vs the operator's own time tau = t - sum(children t). A log-linear calibration
log y = a + b log(1 + x) is fitted on the training databases of the published split; signed log errors on the
test databases. Reports (1) median |error| by subtree size, (2) the correlation of an operator's signed error
with its children's (inheritance), for the cumulative and the residual representation.
    python analysis/tlr/error_propagation.py
"""
import json, os, sys
import numpy as np
D = "data/dace/workload1"
TRAIN = ['basketball', 'geneea', 'genome', 'hepatitis', 'imdb_full', 'movielens', 'seznam', 'ssb', 'tpc_h', 'walmart']
TEST = ['accidents', 'airline', 'baseball', 'carcinogenesis', 'consumer', 'credit', 'employee', 'fhnk', 'financial', 'tournament']

def nodes(db):
    out = []
    for p in json.load(open(f"{D}/{db}_filted.json")):
        def walk(n, parent_idx):
            ch = n.get("Plans", [])
            c, t = float(n.get("Total Cost", 0)), float(n.get("Actual Total Time", 0))
            idx = len(out); out.append(dict(c=c, t=t, parent=parent_idx, kids=[], size=1))
            for k in ch:
                j = walk(k, idx); out[idx]["kids"].append(j); out[idx]["size"] += out[j]["size"]
            out[idx]["r"] = c - sum(out[j]["c"] for j in out[idx]["kids"])
            out[idx]["tau"] = t - sum(out[j]["t"] for j in out[idx]["kids"])
            return idx
        walk(p["Plan"], -1)
    return out

def fit(xs, ys):
    X = np.log1p(np.asarray(xs)); Y = np.log(np.asarray(ys))
    b, a = np.polyfit(X, Y, 1); return a, b
def err(a, b, x, y):
    return np.log(y) - (a + b * np.log1p(x))

tr = [n for db in TRAIN for n in nodes(db)]
te = {db: nodes(db) for db in TEST}
pos = lambda ns, k1, k2: [n for n in ns if n[k1] > 0 and n[k2] > 0]
a_c, b_c = fit([n["c"] for n in pos(tr, "c", "t")], [n["t"] for n in pos(tr, "c", "t")])
a_r, b_r = fit([n["r"] for n in pos(tr, "r", "tau")], [n["tau"] for n in pos(tr, "r", "tau")])
print(f"calibration on training DBs: cumulative log t = {a_c:.2f} + {b_c:.2f} log(1+c); residual log tau = {a_r:.2f} + {b_r:.2f} log(1+r)")
allte = [n for db in TEST for n in te[db]]
kept_r = sum(1 for n in allte if n["r"] > 0 and n["tau"] > 0) / len(allte)
print(f"test operators: {len(allte):,}; residual analysis keeps operators with r>0 and tau>0: {kept_r:.0%}")
ROWS = [dict(stat="n_test_operators", repr="all", n=len(allte), r="", kept_fraction=round(kept_r, 3))]
for n in allte:
    n["e_c"] = err(a_c, b_c, n["c"], n["t"]) if n["c"] > 0 and n["t"] > 0 else np.nan
    n["e_r"] = err(a_r, b_r, n["r"], n["tau"]) if n["r"] > 0 and n["tau"] > 0 else np.nan
print("\n(1) median |log error| (and Q-error median = exp of it) by subtree size, test DBs")
print(f"{'subtree size':14s} {'n':>8s} {'cumulative':>22s} {'residual':>22s}")
for lo, hi in ((1, 1), (2, 3), (4, 7), (8, 15), (16, 10**6)):
    sel = [n for n in allte if lo <= n["size"] <= hi]
    ec = np.array([n["e_c"] for n in sel]); er = np.array([n["e_r"] for n in sel])
    mc, mr = np.nanmedian(np.abs(ec)), np.nanmedian(np.abs(er))
    print(f"{lo:>3d}-{hi if hi < 10**6 else 'max':<9} {len(sel):8,d}   {mc:6.3f} (Q {np.exp(mc):5.2f})      {mr:6.3f} (Q {np.exp(mr):5.2f})")
    ROWS.append(dict(stat="by_size", repr="cumulative", n=len(sel), bucket=f"{lo}-{hi if hi < 10**6 else 'max'}", median_abs_log_err=round(float(mc), 4)))
    ROWS.append(dict(stat="by_size", repr="residual", n=len(sel), bucket=f"{lo}-{hi if hi < 10**6 else 'max'}", median_abs_log_err=round(float(mr), 4)))
print("\n(2) inheritance: correlation of a parent's signed error with the cost-weighted mean of its children's signed error")
for key, lab, w in (("e_c", "cumulative", "c"), ("e_r", "residual", "r")):
    xs, ys = [], []
    for db in TEST:
        ns = te[db]
        for n in ns:
            if not n["kids"] or np.isnan(n[key]): continue
            kids = [ns[j] for j in n["kids"] if not np.isnan(ns[j][key]) and ns[j][w] > 0]
            if not kids: continue
            ws = np.array([k[w] for k in kids]); es = np.array([k[key] for k in kids])
            xs.append(float((ws * es).sum() / ws.sum())); ys.append(n[key])
    xs, ys = np.array(xs), np.array(ys)
    print(f"  {lab:10s}: n={len(xs):,}  Pearson r = {np.corrcoef(xs, ys)[0,1]:.3f}  (parent error explained by children: R^2 = {np.corrcoef(xs, ys)[0,1]**2:.2f})")
    ROWS.append(dict(stat="inherit_all", repr=lab, n=len(xs), r=float(np.corrcoef(xs, ys)[0, 1])))
print("\n(3) the same, root operators only (error of the whole-plan estimate vs its immediate children)")
for key, lab, w in (("e_c", "cumulative", "c"), ("e_r", "residual", "r")):
    xs, ys = [], []
    for db in TEST:
        ns = te[db]
        for n in ns:
            if n["parent"] != -1 or not n["kids"] or np.isnan(n[key]): continue
            kids = [ns[j] for j in n["kids"] if not np.isnan(ns[j][key]) and ns[j][w] > 0]
            if not kids: continue
            ws = np.array([k[w] for k in kids]); es = np.array([k[key] for k in kids])
            xs.append(float((ws * es).sum() / ws.sum())); ys.append(n[key])
    xs, ys = np.array(xs), np.array(ys)
    print(f"  {lab:10s}: n={len(xs):,}  Pearson r = {np.corrcoef(xs, ys)[0,1]:.3f}")
    ROWS.append(dict(stat="inherit_root", repr=lab, n=len(xs), r=float(np.corrcoef(xs, ys)[0, 1])))
os.makedirs("results/analysis/tlr_20261008/tables", exist_ok=True)
import csv
with open("results/analysis/tlr_20261008/tables/error_propagation.csv", "w", newline="") as fh:
    w = csv.DictWriter(fh, fieldnames=["stat", "repr", "n", "r", "bucket", "median_abs_log_err", "kept_fraction"]); w.writeheader()
    for row in ROWS: w.writerow({k: row.get(k, "") for k in w.fieldnames})
print("-> results/analysis/tlr_20261008/tables/error_propagation.csv")
