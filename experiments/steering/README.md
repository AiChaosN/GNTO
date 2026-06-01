# Bao-style Steering Experiment (R1-D5)

This is the experiment infrastructure for the **48 hint-set steering** evaluation
the reviewer asked for in R1-D5. Pipeline overview:

```
   queries.csv  -->  collect_hint_plans.py  -->  results/HintPlans/q_<id>.json
                          (needs live PG)
                                                     |
                                                     v
   trained model     -->     run_steering_eval.py  -->  steering_summary.json
                              (offline scoring)
```

## Status

- [x] `hint_sets.py` — 49 hint sets (6-toggle GUC space, filtered to valid combos).
  Bao paper says 48; we get one extra because we include "all toggles on" as
  hint_id 0 explicitly (Bao counts it but doesn't double-count).
- [x] `collect_hint_plans.py` — connects to PG, runs `EXPLAIN (FORMAT JSON [, ANALYZE])`
  for each query × hint set, caches to disk.
- [x] `run_steering_eval.py` — offline scoring; compares PG default vs. oracle
  vs. learned model. Stubbed `gnto` / `bao` predict_fn entries marked with
  `NotImplementedError` (see TODOs in the file).
- [ ] Wire GNTO predictor (needs fresh GNTO checkpoint — see
  `results/Recompute_0519_*/summary.json::note.GNTO_status`).
- [ ] Wire Bao predictor (load `adapters.bao_adapter.BaoCostPredictor` from
  `results/Bao_0506_1435/bao_model`, call `.predict([plan])`).
- [ ] Actually run end-to-end with `--ground-truth actual` against IMDB.

## Quick smoke test (no PG needed)

```bash
# Generate hint sets
python experiments/steering/hint_sets.py

# Sanity check the eval script with pre-collected mock data:
# (assumes results/HintPlans/q_*.json exists; see collect step below)
python experiments/steering/run_steering_eval.py \
    --hint-plans-dir results/HintPlans \
    --ground-truth cost \
    --model pg_cost \
    --out results/Steering_pg_cost.json
```

## Real run

### 1. Pick test queries

Best candidates: JOB-light (`/home/AiChaosN/Project/Phd/project/GNTO/data/job-light_plan.csv`)
or extract SQL from `data/train_plan_part18.csv` etc.

### 2. Collect hint plans (needs PG 12.5 + IMDB)

```bash
PGHOST=localhost PGUSER=postgres PGDATABASE=imdbload \
    python experiments/steering/collect_hint_plans.py \
        --queries-csv path/to/queries.csv \
        --query-col sql \
        --query-id-col query_id \
        --output-dir results/HintPlans \
        --analyze \
        --timeout-ms 120000
```

Time budget: 49 hint sets × N queries × T per query. With ANALYZE on JOB-light
(~120s timeout, average ~5s real), 100 queries ≈ 1-2 hours. Without ANALYZE
(cost-only) ≈ 5 minutes.

### 3. Score with each model

```bash
# PG cost (sanity)
python experiments/steering/run_steering_eval.py \
    --hint-plans-dir results/HintPlans --ground-truth actual \
    --model pg_cost --out results/Steering_pg.json

# GNTO (after wiring + retrain)
python experiments/steering/run_steering_eval.py \
    --hint-plans-dir results/HintPlans --ground-truth actual \
    --model gnto --out results/Steering_gnto.json

# Bao (after wiring)
python experiments/steering/run_steering_eval.py \
    --hint-plans-dir results/HintPlans --ground-truth actual \
    --model bao --out results/Steering_bao.json
```

### 4. Report

For each steerer the summary contains:
- `total_runtime_ms` — sum of actual runtimes for the model's picks
- `mean_relative_regret` — average `pick_runtime / oracle_runtime` (>= 1)
- `wins_over_pg_default` — # queries where the steerer beat PG's default

These directly answer R1-D5's "is GNTO actually getting closer to the oracle
than Bao within the same 48-hint search space?"

## Why LIMAO isn't an offline `--model limao` entry

Bao, GNTO and `pg_cost` are evaluated in **offline** mode: load a frozen
cost model, score all 48 variants of each query independently, pick the
argmin. That's appropriate for static predictors.

LIMAO is a **lifelong / online** learner — its policy updates as it sees
each query's outcome. Plug-replacing it into the offline harness would
freeze it after some arbitrary point and miss its design point entirely.
The fair LIMAO comparison requires its own driver:

  1. Run LIMAO's bao_server (online) against the same query workload used
     by `collect_hint_plans.py`.
  2. Log the hint LIMAO picks per query + the actual runtime PG reports.
  3. Feed those picks into the same regret/runtime aggregator
     (`summarize_steerer`) so the table column is comparable.

This is bookkeeping, not infrastructure — but it sits outside the
predict-then-rank loop. Left as a follow-up.

## What this experiment *cannot* answer

GNTO + Bao are both capped by the 48-hint search space (R1-D5 itself notes
this). To respond to that critique we add Section X language clarifying that
GNTO is orthogonal to search-space expansion (see review_v2.md item 3).
