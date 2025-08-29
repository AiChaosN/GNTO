**Node Encoder** 的职责就是：

 **输入**：一个节点（包含算子类型、基数、代价估计、谓词等属性）
 **输出**：该节点的向量表示 $\mathbf{h}_i \in \mathbb{R}^d$

它其实是 **LQO 最关键的环节之一**，因为如果节点向量表达力不足，后面的 Tree/GNN 就算再强，也学不到好的 plan embedding。

---

## Node Encoder 的关键作用

* 把 **异构信息**（类别 + 数值 + 文本/表达式）映射到同一向量空间；
* 保留算子语义和数据统计，使得同类算子在向量空间里相近，不同算子/不同条件区分开；
* 形成下游结构编码的“输入粒子”（像 NLP 里的 token embedding）。

---

## Node Encoder 的常见输入维度

1. **算子类型 (Operator Type)**

   * 离散 → one-hot 或 embedding。
   * 例如：`Seq Scan`, `Hash Join`, `Merge Join`, `Sort` …

2. **数据统计 (Estimated Stats)**

   * 连续 → 标准化数值输入。
   * 常见：`plan_rows`, `plan_width`, `startup_cost`, `total_cost`。

3. **谓词/条件 (Predicate Info)**

   * 文本表达式，可以处理方式有：

     * Bag-of-Words / Token Embedding（比如谓词里的列名/操作符/常数）；
     * 小型 Transformer/BERT（如果论文资源允许）；
     * 简化为“谓词复杂度特征”：谓词数、是否有范围过滤、是否包含子查询。

4. **关系/索引 (Relation / Index Info)**

   * 表名/索引名：离散 embedding；
   * 是否使用索引：布尔特征。

5. **执行上下文 (Execution Context)**

   * 是否并行：布尔；
   * join 的左右输入基数比；
   * pipeline/阻塞算子标志。

6. **历史/真实反馈 (可选)**

   * 如果数据集包含 `actual_rows`, `actual_time`，可以在训练时作为辅助监督。

---

## 编码方式（论文里常见几类）

1. **简单拼接 (Concatenation + MLP)**

   * 把离散 embedding、连续数值、布尔特征拼接 → 全连接层 → $\mathbf{h}_i$。
   * 公式：

     $$
     \mathbf{h}_i = \text{MLP}([e_{\text{type}}, x_{\text{stats}}, e_{\text{rel}}, x_{\text{ctx}}])
     $$

2. **分块编码 (Multi-View Encoding)**

   * 每类特征单独编码（算子 embedding / 统计数值 MLP / 谓词 encoder），最后 concat。
   * 优点：不同模态的信息分开建模。

3. **文本编码 (Predicate Encoder)**

   * 如果强调谓词/表达式，可以额外加一个轻量文本 encoder（如 BiLSTM / Tiny-BERT）。
   * 例如：`filter = "age > 30 AND salary < 10000"` → 词向量 → LSTM → 谓词 embedding。

4. **知识增强 (Hybrid Encoding)**

   * 把传统代价模型输出的 `estimated_cost` 作为额外特征输入 Node Encoder。
   * 这样模型既利用“经典代价模型先验”，又能纠正其偏差。

---

##  论文里写法

你可以在论文里写成公式：

* 输入特征：

  $$
  \mathbf{x}_i = [e(\text{type}_i), \, \text{stats}_i, \, e(\text{rel}_i), \, \text{ctx}_i]
  $$
* 节点编码器：

  $$
  \mathbf{h}_i = f_{\text{node}}(\mathbf{x}_i) = \sigma(W\mathbf{x}_i + b)
  $$

其中 $\mathbf{h}_i \in \mathbb{R}^d$ 是节点向量。

---

## 小结

* Node Encoder 是 **信息整合器**，输入可能非常多（类型、数值、谓词、索引、上下文…）。
* 常见做法是 **多模态 embedding + MLP**。
* 论文里通常会对比不同的 Node Encoder（简单 vs 多模态 vs 加先验）。