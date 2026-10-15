"""Exp3 analysis: speed-up of the selected plan vs. cardinality error of the default plan.

Per steerer:

* take the queries with little headroom (best candidate < ``--headroom`` x faster than the
  default plan), where deviating from the default can gain little;
* split them at the median cardinality error of their default plan (largest Q-error between
  ``Plan Rows`` and ``Actual Rows`` over the operators of the plan);
* compare the speed-up of the selected plan in the two halves (Mann-Whitney U) and the ratio
  predicted / measured latency of the selected plan.

Also evaluates the margin rule "deviate from the default plan only if the predicted gain exceeds m".

    python exp/exp3_e2e/analyze_mechanism.py            # -> results/exp3_e2e/tables/mechanism/
"""
from __future__ import annotations

import argparse
import os
import sys

import numpy as np
import pandas as pd
from scipy.stats import mannwhitneyu, spearmanr

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from common import HINT_PLANS, PG_DEFAULT_HINT, RESULTS, SPEED_TOL, candidates, load_packs, md  # noqa: E402

STEERERS = [("gnto_kfold_tpl", "GNTO, DB-independent"), ("gnto_kfold_pg_tpl", "GNTO, optimizer inputs"),
            ("gnto_kfold_pg_ranklog_tpl", "GNTO, optimizer inputs + ranking"), ("bao_kfold_tpl", "Bao"),
            ("limao_kfold_tpl", "LIMAO model")]


def walk(n):
    yield n
    for c in n.get("Plans", []):
        yield from walk(c)


def card_error(plan, per_loop: bool) -> float:
    errs = []
    for n in walk(plan["Plan"]):
        est = float(n.get("Plan Rows", 0))
        act = float(n.get("Actual Rows", 0)) * (1.0 if per_loop else float(n.get("Actual Loops", 1)))
        if est >= 1 and act >= 1:
            errs.append(max(est / act, act / est))
    return float(np.max(errs)) if errs else float("nan")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--headroom", type=float, default=1.3)
    ap.add_argument("--total-rows", action="store_true", help="compare Plan Rows with Actual Rows x Actual Loops instead of per loop")
    ap.add_argument("--out", default=os.path.join(RESULTS, "tables", "mechanism"))
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)

    feat = {}
    for pack in load_packs(HINT_PLANS):
        default = next((p for h, p, _ in candidates(pack) if h == PG_DEFAULT_HINT), None)
        if default is not None:
            feat[pack["query_id"]] = card_error(default, per_loop=not a.total_rows)
    feat = pd.Series(feat, name="card_err")

    cells, margins, corr = [], [], []
    for d, label in STEERERS:
        pp, pr = os.path.join(RESULTS, d, "picks.csv"), os.path.join(RESULTS, d, "predictions.csv")
        if not (os.path.exists(pp) and os.path.exists(pr)):
            continue
        picks = pd.read_csv(pp).set_index("query_id").join(feat)
        preds = pd.read_csv(pr)
        pv = preds.set_index(["query_id", "hint_id"]).pred
        truth = preds.set_index(["query_id", "hint_id"]).actual_ms
        picks["speedup"] = picks.pg_default_ms / picks.pick_ms.clip(lower=1e-9)
        picks["headroom"] = picks.pg_default_ms / picks.oracle_ms.clip(lower=1e-9)
        picks["pred_over_actual"] = [float(pv[(q, h)]) / max(ms, 1e-9) for q, h, ms in zip(picks.index, picks.pick_hint, picks.pick_ms)]
        low = picks[picks.headroom < a.headroom]
        thr = low.card_err.median()
        hi_err, lo_err = low[low.card_err >= thr], low[low.card_err < thr]
        u = mannwhitneyu(np.log(hi_err.speedup), np.log(lo_err.speedup), alternative="two-sided")
        for name, g in (("large cardinality error", hi_err), ("small cardinality error", lo_err),
                        ("headroom >= %.1f (all)" % a.headroom, picks[picks.headroom >= a.headroom])):
            cells.append({"steerer": label, "queries": name, "n": len(g), "geomean_speedup": float(np.exp(np.log(g.speedup).mean())),
                          "slower": int((g.speedup < 1 - SPEED_TOL).sum()), "faster": int((g.speedup > 1 + SPEED_TOL).sum()),
                          "pick_pred_over_actual_median": float(g.pred_over_actual.median()),
                          "mannwhitney_p_large_vs_small": float(u.pvalue) if name != "headroom >= %.1f (all)" % a.headroom else np.nan})
        r = spearmanr(low.card_err, low.speedup)
        corr.append({"steerer": label, "n_low_headroom": len(low), "spearman_card_err_vs_speedup": float(r.statistic), "p": float(r.pvalue)})
        for m in (0.0, 0.1, 0.2, 0.3, 0.5, 1.0):
            sp, taken = [], 0
            for q, row in picks.iterrows():
                ok = (q, PG_DEFAULT_HINT) in pv.index and float(pv[(q, row.pick_hint)]) < float(pv[(q, PG_DEFAULT_HINT)]) / (1 + m)
                ms = float(truth[(q, row.pick_hint)]) if ok else row.pg_default_ms
                taken += bool(ok and row.pick_hint != PG_DEFAULT_HINT)
                sp.append(row.pg_default_ms / max(ms, 1e-9))
            sp = np.array(sp)
            margins.append({"steerer": label, "margin": m, "deviations": taken, "geomean_speedup": float(np.exp(np.log(sp).mean())),
                            "faster_pct": float((sp > 1 + SPEED_TOL).mean() * 100), "slower_pct": float((sp < 1 - SPEED_TOL).mean() * 100),
                            "regressions_2x": int((sp < 0.5).sum())})
    C, M, K = pd.DataFrame(cells), pd.DataFrame(margins), pd.DataFrame(corr)
    C.to_csv(os.path.join(a.out, "controlled_cell.csv"), index=False)
    M.to_csv(os.path.join(a.out, "margin_rule.csv"), index=False)
    K.to_csv(os.path.join(a.out, "correlation.csv"), index=False)
    with open(os.path.join(a.out, "summary.md"), "w") as f:
        f.write(f"# Exp3 - selected-plan speed-up vs. cardinality error of the default plan (template folds)\n\nLow headroom = best candidate < {a.headroom}x "
                f"faster than the default plan. Cardinality error = largest Q-error between Plan Rows and Actual Rows "
                f"({'x loops' if a.total_rows else 'per loop'}) over the operators of the default plan; split at the median.\n\n")
        f.write("## Low-headroom queries by cardinality error\n\n" + md(C, "{:.3f}"))
        f.write("\n## Rank correlation between cardinality error and speed-up (low-headroom queries)\n\n" + md(K, "{:.3f}"))
        f.write("\n## Margin rule: deviate only if the predicted gain exceeds the margin\n\n" + md(M, "{:.3f}"))
    pd.set_option("display.width", 250)
    print(C.round(3).to_string(index=False)); print(); print(K.round(3).to_string(index=False)); print(); print(M.round(3).to_string(index=False))
    print("->", a.out)


if __name__ == "__main__":
    main()
