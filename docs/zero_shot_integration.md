# Zero-Shot Cost Model Integration Plan

Status: **scaffold only**. Adapter at `adapters/zeroshot_adapter.py` raises
`NotImplementedError` at the two integration points below. This doc lists the
concrete steps to finish the integration so the rebuttal can include Hilprecht
& Binnig (VLDB 2022) as a learned cost-estimator baseline (review item R3-D2).

## Upstream

Repo: `/home/AiChaosN/Project/Workspace/01_Research/zero-shot-cost-estimation/`
Paper: Hilprecht & Binnig, "Zero-Shot Cost Models for Out-of-the-box Learned
Cost Prediction", VLDB 2022. arXiv:2201.00561

Key facts about upstream:

- Uses **DGL** for graph modeling, not PyTorch Geometric. We must add `dgl` to
  the GNTO env (`pip install dgl`).
- Plan format is **not raw PG EXPLAIN JSON**. They parse plans via
  `cross_db_benchmark/parse_plan.py` into their own schema that captures
  per-column statistics from a database stats file.
- Two model variants: `MSCNZeroShotModel` (their MSCN baseline) and
  `MAESTROZeroShotModel` (their proposed graph model). Use the latter.

## Integration steps

### 1. Add dependency
```
pip install dgl  # match torch version, ~CPU/CUDA build
```
Add to `requirements.txt`.

### 2. Decide data-conversion strategy

**Option A (recommended for rebuttal speed):**
Run upstream's `run_benchmark.py setup` pipeline against our PG 12.5 + IMDB
to produce their parsed-plan JSON files. Then point the adapter at those.

- Pros: matches their training pipeline exactly, no reimplementation risk.
- Cons: couples us to their CLI; need to run their setup against IMDB.

**Option B (cleaner long-term):**
Reimplement `convert_pg_plan_to_zeroshot()` inside `zeroshot_adapter.py` to
translate from our `data/train_plan_part*.csv` rows directly. This is ~200-300
LOC mirroring `cross_db_benchmark/parse_plan.py` + a stats-file builder.

### 3. Fill in `ZeroShotPredictor.fit / predict`

Inside `adapters/zeroshot_adapter.py` (search for `NotImplementedError`):

- Replace `convert_pg_plan_to_zeroshot()` per the chosen strategy.
- In `fit()`: build their `PlanDataset` + DGL DataLoader, instantiate
  `MAESTROZeroShotModel`, run a training loop with MSE on log1p-MinMax-scaled
  targets (mirror `bao_adapter.BaoCostPredictor`).
- In `predict()`: single forward pass, inverse-transform.

### 4. Wire into the recompute script

Once the adapter is callable, add a `Zero-shot` block to
`examples/0519_recompute_metrics.py` (mirror the Bao block: load checkpoint
or saved predictions, run `evaluation_summary`).

### 5. Train + evaluate

Run on the same train/val split (parts 0-17 / 18-19) the other baselines
use. Cross-database (R3-D2's specific ask) is the natural follow-up: train on
IMDB, evaluate on TPC-H or STATS without retraining.

## Estimated effort

- Option A: 0.5 day setup + 0.5 day adapter wiring + 1 day train/debug
- Option B: 2-3 days adapter + 1 day train/debug

If the rebuttal deadline is tight, take Option A and note in the paper that
the integration uses upstream's reference pipeline.
