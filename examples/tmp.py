# 1.读取数据
import glob
import pandas as pd
import json

# 匹配所有 train_plan_0*.csv
files = glob.glob("../data/train_plan_*.csv")
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
print("plans_dict:\n", plans_dict[0])

# 2.数据格式转换 json -> PlanNode
import sys, os
sys.path.append(os.path.abspath(".."))  # 确保当前目录加入路径

# json -> PlanNode
from models.DataPreprocessor import PlanNode, DataPreprocessor
preprocessor = DataPreprocessor()
plans_tree = preprocessor.preprocess_all(plans_dict)

# 3.数据格式转换 planNode -> graph 格式
# PlanNode -> edges_list, extra_info_list
def tree_to_graph(root):
    edges_list, extra_info_list = [], []

    def dfs(node, parent_idx):
        idx = len(extra_info_list)
        extra_info_list.append(node.extra_info)
        edges_list.append((idx, idx))
        if parent_idx is not None:
            edges_list.append((parent_idx, idx))
        for ch in node.children:
            dfs(ch, idx)

    dfs(root, None)
    return edges_list, extra_info_list

edges_list, matrix_plans = [], []
for i in plans_tree:
    edges_matrix, extra_info_matrix = tree_to_graph(i)
    # if len(edges_matrix) == 0:
    #     print(i)
    #     assert False
    edges_list.append(edges_matrix)
    matrix_plans.append(extra_info_matrix)

type(matrix_plans[0][0])

# 统计信息

from models.Utils import StatisticsInfo

statisticsInfo = StatisticsInfo(matrix_plans, sample_threshold=100, sample_k=10).build()
statisticsInfo.pretty_print_report()

# NodeVectorizer
import re, math
from collections import defaultdict
import numpy as np
import torch
from typing import List

from models.Utils import process_join_cond_field, process_index_cond_field, load_column_stats

# -------- 词表 --------
class Vocab:
    def __init__(self): self.idx = {"<pad>":0, "<unk>":1}
    def add(self, s):
        if s not in self.idx: self.idx[s] = len(self.idx)
    def get(self, s): return self.idx.get(s, 1)
    @property
    def size(self): return len(self.idx)

NodeTypeVocab = ['Bitmap Heap Scan', 'Bitmap Index Scan', 'BitmapAnd', 'Gather', 'Gather Merge', 'Hash', 'Hash Join', 'Index Scan', 'Materialize', 'Merge Join', 'Nested Loop', 'Seq Scan', 'Sort']


# ===== BPE-Enhanced Node Vectorizer =====
import re
from collections import Counter
from typing import Dict, List, Set, Tuple

class BPETokenizer:
    """简化版BPE分词器，用于处理SQL查询计划中的字符串特征"""
    
    def __init__(self, vocab_size: int = 1000):
        self.vocab_size = vocab_size
        self.word_freqs = Counter()
        self.vocab = {}
        self.merges = []
        
    def _get_word_tokens(self, word: str) -> List[str]:
        """将单词拆分为字符级tokens"""
        return list(word) + ['</w>']
    
    def _get_pairs(self, word_tokens: List[str]) -> Set[Tuple[str, str]]:
        """获取相邻字符对"""
        pairs = set()
        prev_char = word_tokens[0]
        for char in word_tokens[1:]:
            pairs.add((prev_char, char))
            prev_char = char
        return pairs
    
    def train(self, texts: List[str]):
        """训练BPE模型"""
        # 1. 预处理文本，收集词频
        for text in texts:
            if text is None:
                continue
            # 清理文本，处理SQL相关字符
            words = re.findall(r'\w+|[^\w\s]', str(text).lower())
            for word in words:
                self.word_freqs[word] += 1
        
        # 2. 初始化词汇表（字符级）
        vocab = set()
        for word in self.word_freqs.keys():
            vocab.update(self._get_word_tokens(word))
        
        # 为每个字符分配ID
        self.vocab = {token: i for i, token in enumerate(sorted(vocab))}
        self.vocab['<UNK>'] = len(self.vocab)
        self.vocab['<PAD>'] = len(self.vocab)
        
        # 3. 创建单词的token表示
        word_tokens = {}
        for word in self.word_freqs.keys():
            word_tokens[word] = self._get_word_tokens(word)
        
        # 4. BPE合并过程
        for _ in range(self.vocab_size - len(self.vocab)):
            # 统计所有相邻字符对的频率
            pairs_freq = Counter()
            for word, freq in self.word_freqs.items():
                pairs = self._get_pairs(word_tokens[word])
                for pair in pairs:
                    pairs_freq[pair] += freq
            
            if not pairs_freq:
                break
                
            # 找到最频繁的字符对
            best_pair = pairs_freq.most_common(1)[0][0]
            self.merges.append(best_pair)
            
            # 合并这个字符对
            new_token = ''.join(best_pair)
            self.vocab[new_token] = len(self.vocab)
            
            # 更新所有单词的token表示
            for word in word_tokens:
                new_tokens = []
                i = 0
                while i < len(word_tokens[word]):
                    if (i < len(word_tokens[word]) - 1 and 
                        word_tokens[word][i] == best_pair[0] and 
                        word_tokens[word][i + 1] == best_pair[1]):
                        new_tokens.append(new_token)
                        i += 2
                    else:
                        new_tokens.append(word_tokens[word][i])
                        i += 1
                word_tokens[word] = new_tokens
    
    def encode(self, text: str, max_length: int = 32) -> List[int]:
        """将文本编码为token ID序列"""
        if text is None:
            return [self.vocab.get('<PAD>', 0)] * max_length
            
        words = re.findall(r'\w+|[^\w\s]', str(text).lower())
        tokens = []
        
        for word in words:
            word_tokens = self._get_word_tokens(word)
            
            # 应用学习到的合并规则
            for merge in self.merges:
                new_tokens = []
                i = 0
                while i < len(word_tokens):
                    if (i < len(word_tokens) - 1 and 
                        word_tokens[i] == merge[0] and 
                        word_tokens[i + 1] == merge[1]):
                        new_tokens.append(''.join(merge))
                        i += 2
                    else:
                        new_tokens.append(word_tokens[i])
                        i += 1
                word_tokens = new_tokens
            
            # 转换为ID
            for token in word_tokens:
                token_id = self.vocab.get(token, self.vocab.get('<UNK>', 0))
                tokens.append(token_id)
        
        # 截断或填充到指定长度
        if len(tokens) > max_length:
            tokens = tokens[:max_length]
        else:
            pad_id = self.vocab.get('<PAD>', 0)
            tokens.extend([pad_id] * (max_length - len(tokens)))
            
        return tokens

class BPENodeVectorizer:
    """使用BPE增强的节点向量化器"""
    
    def __init__(self, node_types: List[str], use_bpe: bool = True, bpe_vocab_size: int = 500):
        self.node_types = node_types
        self.use_bpe = use_bpe
        self.bpe_vocab_size = bpe_vocab_size
        
        # 数值特征维度
        self.node_type_dim = len(node_types)  # one-hot编码
        self.numeric_dim = 2  # Plan Width, Plan Rows
        self.bpe_dim = 32 if use_bpe else 0  # BPE特征维度
        
        self.total_dim = self.node_type_dim + self.numeric_dim + self.bpe_dim
        
        if self.use_bpe:
            self.bpe_tokenizers = {
                'relation_name': BPETokenizer(bpe_vocab_size),
                'index_name': BPETokenizer(bpe_vocab_size),
                'join_condition': BPETokenizer(bpe_vocab_size),
                'filter_condition': BPETokenizer(bpe_vocab_size),
            }
    
    def _extract_text_features(self, matrix_plans: List[List[dict]]) -> Dict[str, List[str]]:
        """提取所有文本特征用于BPE训练"""
        text_features = {
            'relation_name': [],
            'index_name': [],
            'join_condition': [],
            'filter_condition': []
        }
        
        for mp in matrix_plans:
            for node in mp:
                # 关系名
                relation_name = node.get('Relation Name') or node.get('Alias', '')
                text_features['relation_name'].append(str(relation_name))
                
                # 索引名
                index_name = node.get('Index Name', '')
                text_features['index_name'].append(str(index_name))
                
                # Join条件
                join_cond = ''
                if 'Hash Cond' in node:
                    join_cond = str(node['Hash Cond'])
                elif 'Merge Cond' in node:
                    join_cond = str(node['Merge Cond'])
                elif 'Join Filter' in node:
                    join_cond = str(node['Join Filter'])
                text_features['join_condition'].append(join_cond)
                
                # 过滤条件
                filter_cond = ''
                if 'Filter' in node:
                    filter_cond = str(node['Filter'])
                elif 'Index Cond' in node:
                    filter_cond = str(node['Index Cond'])
                text_features['filter_condition'].append(filter_cond)
        
        return text_features
    
    def fit(self, matrix_plans: List[List[dict]]):
        """训练BPE模型"""
        if not self.use_bpe:
            return
            
        print("训练BPE分词器...")
        text_features = self._extract_text_features(matrix_plans)
        
        for feature_name, tokenizer in self.bpe_tokenizers.items():
            texts = text_features[feature_name]
            # 过滤空文本
            texts = [t for t in texts if t and str(t).strip() and str(t) != 'None']
            if texts:
                tokenizer.train(texts)
                print(f"  {feature_name}: 词汇表大小 {len(tokenizer.vocab)}")
    
    def transform(self, matrix_plans: List[List[dict]]) -> List[List[List[float]]]:
        """将计划转换为向量表示"""
        res = []
        
        for mp in matrix_plans:
            plan_matrix = []
            for node in mp:
                node_vector = [0.0] * self.total_dim
                offset = 0
                
                # 1. Node Type one-hot编码 [0:node_type_dim]
                try:
                    node_type_idx = self.node_types.index(node["Node Type"])
                    node_vector[node_type_idx] = 1.0
                except ValueError:
                    print(f"未知节点类型: {node['Node Type']}")
                offset += self.node_type_dim
                
                # 2. 数值特征 [node_type_dim:node_type_dim+numeric_dim]
                plan_width = float(node.get("Plan Width", 0))
                plan_rows = float(node.get("Plan Rows", 0))
                
                # 对数归一化
                node_vector[offset] = np.log1p(plan_width)
                node_vector[offset + 1] = np.log1p(plan_rows)
                offset += self.numeric_dim
                
                # 3. BPE文本特征 [node_type_dim+numeric_dim:total_dim]
                if self.use_bpe:
                    # 关系名特征 (8维)
                    relation_name = node.get('Relation Name') or node.get('Alias', '')
                    relation_tokens = self.bpe_tokenizers['relation_name'].encode(str(relation_name), 8)
                    for i, token_id in enumerate(relation_tokens):
                        if offset + i < len(node_vector):
                            node_vector[offset + i] = float(token_id) / 1000.0  # 归一化
                    offset += 8
                    
                    # 索引名特征 (8维)
                    index_name = node.get('Index Name', '')
                    index_tokens = self.bpe_tokenizers['index_name'].encode(str(index_name), 8)
                    for i, token_id in enumerate(index_tokens):
                        if offset + i < len(node_vector):
                            node_vector[offset + i] = float(token_id) / 1000.0
                    offset += 8
                    
                    # Join条件特征 (8维)
                    join_cond = ''
                    if 'Hash Cond' in node:
                        join_cond = str(node['Hash Cond'])
                    elif 'Merge Cond' in node:
                        join_cond = str(node['Merge Cond'])
                    elif 'Join Filter' in node:
                        join_cond = str(node['Join Filter'])
                    
                    join_tokens = self.bpe_tokenizers['join_condition'].encode(join_cond, 8)
                    for i, token_id in enumerate(join_tokens):
                        if offset + i < len(node_vector):
                            node_vector[offset + i] = float(token_id) / 1000.0
                    offset += 8
                    
                    # 过滤条件特征 (8维)
                    filter_cond = ''
                    if 'Filter' in node:
                        filter_cond = str(node['Filter'])
                    elif 'Index Cond' in node:
                        filter_cond = str(node['Index Cond'])
                    
                    filter_tokens = self.bpe_tokenizers['filter_condition'].encode(filter_cond, 8)
                    for i, token_id in enumerate(filter_tokens):
                        if offset + i < len(node_vector):
                            node_vector[offset + i] = float(token_id) / 1000.0
                
                plan_matrix.append(node_vector)
            res.append(plan_matrix)
        
        return res

# 使用BPE增强的向量化器
print("=== 使用BPE增强的节点向量化 ===")
bpe_vectorizer = BPENodeVectorizer(
    node_types=NodeTypeVocab, 
    use_bpe=True, 
    bpe_vocab_size=500
)

# 训练BPE模型
bpe_vectorizer.fit(matrix_plans)

# 转换数据
res = bpe_vectorizer.transform(matrix_plans)

print(f"BPE向量化完成:")
print(f"  特征维度: {bpe_vectorizer.total_dim}")
print(f"  - 节点类型: {bpe_vectorizer.node_type_dim}")
print(f"  - 数值特征: {bpe_vectorizer.numeric_dim}")
print(f"  - BPE文本特征: {bpe_vectorizer.bpe_dim}")
print(f"  样本形状: {len(res[0])}x{len(res[0][0])}")

# 更新特征维度
F_in = bpe_vectorizer.total_dim

# 原始NodeVectorizer已被BPE版本替代

# 模型搭建

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch

class NodeEncoder(nn.Module):
    """
    输入: data.x 形状 [N, F_in]
    输出: node_embs [N, d_node]
    """
    def __init__(self, in_dim: int, d_node: int):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, d_node),
            nn.ReLU(),
            nn.LayerNorm(d_node),
        )
    def forward(self, x):
        return self.proj(x)

# ---- 组合总模型 ----
class PlanCostModel(nn.Module):
    """
    NodeEncoder → GATTreeEncoder → PredictionHead
    """
    def __init__(self, nodecoder: nn.Module, treeencoder: nn.Module, predict_head: nn.Module):
        super().__init__()
        self.nodecoder = nodecoder
        self.treeencoder = treeencoder
        self.predict_head = predict_head

    def forward(self, data: Data | Batch):
        """
        期望 data 里至少有:
        - x: [N, F_in]
        - edge_index: [2, E]
        - batch: [N]  指示每个节点属于哪张图
        """
        x = self.nodecoder(data.x)                                   # [N, d_node]
        g = self.treeencoder(x, data.edge_index, data.batch)         # [B, d_graph]
        y = self.predict_head(g)                                     # [B, out_dim]
        return y


from models.TreeEncoder import GATTreeEncoder
from models.PredictionHead import PredictionHead
# ---- 使用示例 ----
d_node, d_graph = 32, 64
nodecoder = NodeEncoder(F_in, d_node)
gatTreeEncoder = GATTreeEncoder(
    input_dim=d_node,      # 一定用实际特征维度
    hidden_dim=64,
    output_dim=d_graph,
    num_layers=3,
    num_heads=4,
    dropout=0.1,
    pooling="mean"
)
predict_head = PredictionHead(d_graph, out_dim=1)

model = PlanCostModel(nodecoder, gatTreeEncoder, predict_head)

print(type(ExecutionTimes))
print(type(res))
print(type(edges_list))


print(model)

# 4.构建数据集
import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

def coerce_x_to_tensor(x_plan, in_dim: int):
    """
    x_plan: 很深的 list（最终行向量长度= in_dim）
    变成 [N, in_dim] 的 float32 Tensor
    """
    x = torch.tensor(x_plan, dtype=torch.float32)
    assert x.numel() % in_dim == 0, f"最后一维应为 {in_dim}，拿到形状 {tuple(x.shape)}"
    x = x.view(-1, in_dim)   # 拉平成 [N, in_dim]
    return x

def coerce_edge_index(ei_like):
    """
    ei_like: list/ndarray/tensor, 形状 [2,E] 或 [E,2]
    返回规范 [2,E] 的 long Tensor
    """
    ei = torch.as_tensor(ei_like, dtype=torch.long)
    if ei.ndim != 2:
        raise ValueError(f"edge_index 需要二维，拿到 {tuple(ei.shape)}")
    if ei.shape[0] != 2 and ei.shape[1] == 2:
        ei = ei.t().contiguous()
    elif ei.shape[0] != 2 and ei.shape[1] != 2:
        raise ValueError(f"edge_index 需为 [2,E] 或 [E,2]，拿到 {tuple(ei.shape)}")
    return ei.contiguous()

def build_dataset(res, edges_list, execution_times, in_dim=16, bidirectional=False):
    assert len(res) == len(edges_list) == len(execution_times), "长度必须一致"
    data_list = []
    for i, (x_plan, ei_like, y) in enumerate(zip(res, edges_list, execution_times)):
        x = coerce_x_to_tensor(x_plan, in_dim)      # [N, in_dim]
        edge_index = coerce_edge_index(ei_like)     # [2,E]
        N = x.size(0)

        # 边索引有效性检查
        if edge_index.numel() > 0:
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= N:
                raise ValueError(f"plan[{i}] 的 edge_index 越界：节点数 N={N}，但 edge_index.max={int(edge_index.max())}")

        # 可选：做成双向图（若你的 edges 只有父->子）
        if bidirectional:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        y = torch.tensor([float(y)], dtype=torch.float32)  # 图级回归标签
        data_list.append(Data(x=x, edge_index=edge_index, y=y))
    return data_list

# ===== #
# 构建数据集
dataset = build_dataset(res, edges_list, ExecutionTimes, in_dim=F_in, bidirectional=True)
print(f"数据集大小: {len(dataset)}")
print(f"第一个样本: x.shape={dataset[0].x.shape}, edge_index.shape={dataset[0].edge_index.shape}, y={dataset[0].y}")

# 5. 训练准备
from sklearn.model_selection import train_test_split
import time
import os

# 数据集划分
train_indices, temp_indices = train_test_split(
    range(len(dataset)), test_size=0.3, random_state=42
)
val_indices, test_indices = train_test_split(
    temp_indices, test_size=0.5, random_state=42
)

train_dataset = [dataset[i] for i in train_indices]
val_dataset = [dataset[i] for i in val_indices]
test_dataset = [dataset[i] for i in test_indices]

print(f"训练集: {len(train_dataset)}, 验证集: {len(val_dataset)}, 测试集: {len(test_dataset)}")

# 创建数据加载器
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"训练批次数: {len(train_loader)}, 验证批次数: {len(val_loader)}")

# 6. 训练配置
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"使用设备: {device}")

model = model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
criterion = torch.nn.MSELoss()

# early stopping
class EarlyStopping:
    def __init__(self, patience=15, min_delta=0, restore_best_weights=True):
        self.patience = patience
        self.min_delta = min_delta
        self.restore_best_weights = restore_best_weights
        self.best_loss = None
        self.counter = 0
        self.best_weights = None

    def __call__(self, val_loss, model):
        if self.best_loss is None:
            self.best_loss = val_loss
            self.save_checkpoint(model)
        elif val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
            self.save_checkpoint(model)
        else:
            self.counter += 1

        if self.counter >= self.patience:
            if self.restore_best_weights:
                model.load_state_dict(self.best_weights)
            return True
        return False

    def save_checkpoint(self, model):
        self.best_weights = model.state_dict().copy()

early_stopping = EarlyStopping(patience=15, min_delta=0.001)

# 7. 训练函数
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss = 0
    num_batches = 0
    
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()
        
        # 前向传播
        pred = model(batch)
        loss = criterion(pred, batch.y)
        
        # 反向传播
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        total_loss += loss.item()
        num_batches += 1
    
    return total_loss / num_batches

def validate_epoch(model, loader, criterion, device):
    model.eval()
    total_loss = 0
    num_batches = 0
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y)
            total_loss += loss.item()
            num_batches += 1
    
    return total_loss / num_batches

# 8. 训练循环
def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, 
                early_stopping, device, num_epochs=100):
    
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print("开始训练...")
    print("-" * 60)
    
    for epoch in range(num_epochs):
        start_time = time.time()
        
        # 训练
        train_loss = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # 验证
        val_loss = validate_epoch(model, val_loader, criterion, device)
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 记录损失
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        
        # 计算时间
        epoch_time = time.time() - start_time
        
        # 打印进度
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f"Epoch {epoch+1:3d}/{num_epochs} | "
                  f"Train Loss: {train_loss:.6f} | "
                  f"Val Loss: {val_loss:.6f} | "
                  f"LR: {optimizer.param_groups[0]['lr']:.2e} | "
                  f"Time: {epoch_time:.2f}s")
        
        # 早停检查
        if early_stopping(val_loss, model):
            print(f"\n早停触发在第 {epoch+1} 轮")
            break
        
        # 保存最佳模型
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, '../models/best_model.pth')
    
    print("-" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    return train_losses, val_losses

# 开始训练
train_losses, val_losses = train_model(
    model, train_loader, val_loader, optimizer, scheduler, 
    criterion, early_stopping, device, num_epochs=100
)

# 9. 测试评估
def evaluate_model(model, test_loader, device):
    model.eval()
    predictions = []
    targets = []
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch)
            predictions.extend(pred.cpu().numpy())
            targets.extend(batch.y.cpu().numpy())
    
    predictions = np.array(predictions)
    targets = np.array(targets)
    
    # 计算评估指标
    mse = np.mean((predictions - targets) ** 2)
    rmse = np.sqrt(mse)
    mae = np.mean(np.abs(predictions - targets))
    
    # 计算相对误差
    relative_error = np.mean(np.abs((predictions - targets) / (targets + 1e-8)))
    
    print("\n" + "="*50)
    print("测试集评估结果:")
    print("="*50)
    print(f"MSE:  {mse:.6f}")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"相对误差: {relative_error:.6f}")
    print("="*50)
    
    return predictions, targets, {'mse': mse, 'rmse': rmse, 'mae': mae, 'relative_error': relative_error}

# 加载最佳模型进行测试
try:
    checkpoint = torch.load('../models/best_model.pth', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    print("已加载最佳模型进行测试")
except FileNotFoundError:
    print("未找到保存的模型，使用当前模型进行测试")

predictions, targets, metrics = evaluate_model(model, test_loader, device)

# 10. 可视化训练过程和结果
import matplotlib.pyplot as plt

def plot_training_history(train_losses, val_losses):
    plt.figure(figsize=(12, 4))
    
    # 训练损失曲线
    plt.subplot(1, 2, 1)
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, 'b-', label='训练损失')
    plt.plot(epochs, val_losses, 'r-', label='验证损失')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title('训练和验证损失')
    plt.legend()
    plt.grid(True)
    
    # 预测 vs 真实值
    plt.subplot(1, 2, 2)
    plt.scatter(targets, predictions, alpha=0.5, s=20)
    min_val = min(targets.min(), predictions.min())
    max_val = max(targets.max(), predictions.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2)
    plt.xlabel('真实执行时间')
    plt.ylabel('预测执行时间')
    plt.title('预测 vs 真实值')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('../results/training_results.png', dpi=300, bbox_inches='tight')
    plt.show()

# 创建结果目录
os.makedirs('../results', exist_ok=True)
os.makedirs('../models', exist_ok=True)

# 绘制结果
plot_training_history(train_losses, val_losses)

# 11. 模型保存和加载工具函数
def save_model_with_metadata(model, filepath, metadata=None):
    """保存模型和相关元数据"""
    save_dict = {
        'model_state_dict': model.state_dict(),
        'model_config': {
            'F_in': F_in,
            'd_node': d_node,
            'd_graph': d_graph,
        },
        'training_metadata': metadata or {}
    }
    torch.save(save_dict, filepath)
    print(f"模型已保存到: {filepath}")

def load_model_with_metadata(filepath, model_class, device='cpu'):
    """加载模型和元数据"""
    checkpoint = torch.load(filepath, map_location=device)
    
    # 重建模型
    config = checkpoint['model_config']
    nodecoder = NodeEncoder(config['F_in'], config['d_node'])
    gatTreeEncoder = GATTreeEncoder(
        input_dim=config['d_node'],
        hidden_dim=64,
        output_dim=config['d_graph'],
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        pooling="mean"
    )
    predict_head = PredictionHead(config['d_graph'], out_dim=1)
    model = PlanCostModel(nodecoder, gatTreeEncoder, predict_head)
    
    # 加载权重
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    
    return model, checkpoint.get('training_metadata', {})

# 保存最终模型
final_metadata = {
    'train_size': len(train_dataset),
    'val_size': len(val_dataset),
    'test_size': len(test_dataset),
    'final_metrics': metrics,
    'training_epochs': len(train_losses)
}

save_model_with_metadata(model, '../models/final_model.pth', final_metadata)

print("\n训练完成! 模型已保存，结果已可视化。")
