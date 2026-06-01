# 周报 (2026-04-27 ~ 2026-05-01)

## 本周工作

1. **新增统一 baseline 适配层 (`adapters/`)**
   - 实现 `qf_adapter.py`、`dace_adapter.py`、`limao_adapter.py`,把 QueryFormer / DACE / LIMAO 三个 baseline 的数据加载与模型调用收口到统一接口。
   - 解决了之前各 example 脚本各自 import 外部仓库、路径与参数散乱的问题,后续做对比实验只需替换 adapter。

2. **重构 `examples/` 实验脚本**
   - 把消融脚本(`0204_run_ablation_gnto.py` 等)和对比脚本切到新的 adapter 接口,移除冗余的数据预处理代码。
   - 脚本结构更一致,便于复用与新增实验。

3. **更新 README 与文档**
   - 同步 adapter-based 用法说明,修正 DACE / LIMAO 的引用与运行命令。

4. **整理 VLDB 审稿意见**
   - 通读 3 位审稿人的意见,在 `VLDB_review/Review.md` 中按"必须修改 / 应该修改 / 小修"分类汇总,并初步规划了对应的实验与修改方案,作为下一阶段返修工作的依据。

## 下周计划 (05-04 ~ 05-08)

围绕 VLDB 审稿意见中**优先级最高**的几项展开:

1. **补 Bao 作为 steering baseline (回应 R1-D5 / R3-D2)** — *cost-prediction 部分已完成 (2026-05-06)*
   - 基于本周做好的 adapter 接口,新增 `bao_adapter.py`,在同一组 48 hint sets 上跑 GNTO vs Bao 原版 tree-CNN 的对比,证明在固定搜索空间内 GNTO 的 ranking 模型确实更优。
   - **进度**:
     - `adapters/bao_adapter.py` + `examples/0505_train_bao_baseline.py` 已落地。
     - 适配过程修了两处:(a) Bao 的 `model.py` 与 QueryFormer 的 `model/` 包同名,改用 `importlib` 按文件路径加载并注册到 `sys.modules` 的唯一别名;(b) Bao 的 `TreeFeaturizer` 不识别 `BitmapAnd`/`BitmapOr`,在 adapter 里把 `Bitmap Heap Scan` 折叠成 `Bitmap Index Scan` 叶节点(保留 Relation Name + cost/rows)。
     - **首轮 cost-prediction 结果** (parts 0-17 train / 18-19 val,见 `results/Bao_0506_1435/summary.json`):Bao Val Q50=1.181、Q90=3.460、Q99=18.185,在所有百分位上均显著差于 GNTO (Q50=1.054、Q90=2.150、Q99=9.830) 与 QueryFormer (Q50=1.131、Q90=2.425、Q99=11.471)。详见 `results/RESULTS.md`。
   - **待办**:48 hint-sets 下的 steering / latency 对比(R1-D5 真正问的部分)还要单独做,目前只是先把 cost-prediction 这条 baseline 对齐。

2. **新增 ranking / latency 指标 (回应 R3-D1)**
   - 在评估代码里加入 Spearman、Kendall's tau、Top-1 Regret、Pairwise accuracy。
   - 补单 plan 推理延迟(CPU/GPU)、batch=48 的吞吐与峰值显存,扩展 Table 4。

3. **消融结果按 join 数量分层 (回应 R2-W2)**
   - 把现有 ablation 结果按 join 数(1-2 / 3-4 / 5+)重新分组出表,验证 GNTO 在复杂 join 上优势更明显的 claim。

4. **Related Work 中补 JGMP / Reqo 讨论 (回应 R1-D6 / R3-D2)**
   - 起草一段说明 GNTO 与 JGMP (VLDB 2025) 在 graph 粒度(plan DAG vs join graph)与任务(latency vs cardinality)上的差异,避免 novelty 撞车。
