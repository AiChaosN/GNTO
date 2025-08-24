# 扩展指南

本指南介绍如何扩展GNTO框架，包括自定义组件开发、接口实现和最佳实践。

## 自定义Encoder

### 基础扩展

```python
from models.Encoder import Encoder
import numpy as np

class CustomEncoder(Encoder):
    def __init__(self):
        super().__init__()
        self.feature_extractors = {
            'cost': self._extract_cost,
            'rows': self._extract_rows,
            'selectivity': self._extract_selectivity
        }
    
    def encode(self, node):
        # 获取基础one-hot编码
        base_encoding = super().encode(node)
        
        # 添加数值特征
        numerical_features = self._extract_numerical_features(node)
        
        # 合并特征
        return np.concatenate([base_encoding, numerical_features])
    
    def _extract_numerical_features(self, node):
        """提取数值特征"""
        features = []
        
        # 提取成本特征
        cost = node.extra_info.get('Cost', 0.0)
        features.append(np.log1p(cost))  # 对数变换
        
        # 提取行数特征
        rows = node.extra_info.get('Rows', 1.0)
        features.append(np.log1p(rows))
        
        # 提取选择性特征
        if 'Filter' in node.extra_info:
            features.append(1.0)  # 有过滤条件
        else:
            features.append(0.0)  # 无过滤条件
        
        return np.array(features)
```

### 图神经网络Encoder

```python
class GNNEncoder(Encoder):
    def __init__(self, embedding_dim=64, num_layers=3):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_layers = num_layers
        
    def encode(self, node):
        # 构建图结构
        graph = self._build_graph(node)
        
        # 初始化节点特征
        node_features = self._init_node_features(graph)
        
        # 多层图卷积
        for layer in range(self.num_layers):
            node_features = self._graph_conv_layer(graph, node_features, layer)
        
        # 图级别表示
        return self._graph_pooling(node_features)
    
    def _build_graph(self, root_node):
        """构建图结构"""
        nodes = []
        edges = []
        
        def traverse(node, parent_id=None):
            node_id = len(nodes)
            nodes.append(node)
            
            if parent_id is not None:
                edges.append((parent_id, node_id))
            
            for child in node.children:
                traverse(child, node_id)
        
        traverse(root_node)
        return {'nodes': nodes, 'edges': edges}
```

## 自定义TreeModel

### 注意力机制聚合

```python
from models.TreeModel import TreeModel
import numpy as np

class AttentionTreeModel(TreeModel):
    def __init__(self, attention_dim=32):
        super().__init__()
        self.attention_dim = attention_dim
        # 在实际实现中，这些应该是可训练的参数
        self.W_q = np.random.randn(attention_dim, attention_dim)
        self.W_k = np.random.randn(attention_dim, attention_dim)
        self.W_v = np.random.randn(attention_dim, attention_dim)
    
    def forward(self, vectors):
        if not vectors:
            return np.zeros(0)
        
        # Padding到相同维度
        stacked = self._pad_and_stack(vectors)
        
        # 多头注意力聚合
        attended = self._multi_head_attention(stacked)
        
        # 全局池化
        return np.mean(attended, axis=0)
    
    def _multi_head_attention(self, X):
        """简化的多头注意力机制"""
        # Q, K, V变换
        Q = X @ self.W_q
        K = X @ self.W_k  
        V = X @ self.W_v
        
        # 注意力权重
        attention_weights = self._softmax(Q @ K.T / np.sqrt(self.attention_dim))
        
        # 加权聚合
        return attention_weights @ V
    
    def _softmax(self, x):
        """Softmax激活函数"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
```

### 层次化聚合

```python
class HierarchicalTreeModel(TreeModel):
    def __init__(self, levels=3):
        super().__init__()
        self.levels = levels
    
    def forward(self, vectors):
        """层次化聚合策略"""
        if not vectors:
            return np.zeros(0)
        
        current_vectors = list(vectors)
        
        # 多层聚合
        for level in range(self.levels):
            current_vectors = self._aggregate_level(current_vectors)
            if len(current_vectors) == 1:
                break
        
        return current_vectors[0] if current_vectors else np.zeros(0)
    
    def _aggregate_level(self, vectors):
        """单层聚合"""
        if len(vectors) <= 1:
            return vectors
        
        aggregated = []
        for i in range(0, len(vectors), 2):
            if i + 1 < len(vectors):
                # 成对聚合
                pair_agg = (vectors[i] + vectors[i + 1]) / 2
                aggregated.append(pair_agg)
            else:
                # 奇数个向量，直接添加
                aggregated.append(vectors[i])
        
        return aggregated
```

## 自定义Predictioner

### 深度神经网络预测器

```python
from models.Predictioner import Predictioner
import numpy as np

class DNNPredictioner(Predictioner):
    def __init__(self, hidden_dims=[64, 32, 16]):
        super().__init__()
        self.hidden_dims = hidden_dims
        self.weights = []
        self.biases = []
        
    def _init_network(self, input_dim):
        """初始化网络参数"""
        dims = [input_dim] + self.hidden_dims + [1]
        
        for i in range(len(dims) - 1):
            # Xavier初始化
            w = np.random.randn(dims[i], dims[i+1]) * np.sqrt(2.0 / dims[i])
            b = np.zeros(dims[i+1])
            self.weights.append(w)
            self.biases.append(b)
    
    def predict(self, features):
        if not self.weights:
            self._init_network(len(features))
        
        x = features
        
        # 前向传播
        for i, (w, b) in enumerate(zip(self.weights, self.biases)):
            x = x @ w + b
            
            # 除最后一层外都使用ReLU激活
            if i < len(self.weights) - 1:
                x = np.maximum(0, x)  # ReLU
        
        return float(x[0])  # 返回标量
    
    def _relu(self, x):
        return np.maximum(0, x)
```

### 多任务预测器

```python
class MultiTaskPredictioner(Predictioner):
    def __init__(self, task_weights=None):
        super().__init__()
        self.tasks = ['cost', 'latency', 'memory', 'io']
        self.task_weights = task_weights or {task: None for task in self.tasks}
        
    def predict_all_tasks(self, features):
        """预测所有任务"""
        predictions = {}
        
        for task in self.tasks:
            if self.task_weights[task] is None:
                # 使用默认权重
                self.task_weights[task] = np.ones_like(features)
            
            # 任务特定预测
            pred = np.dot(self.task_weights[task][:len(features)], features)
            predictions[task] = float(pred)
        
        return predictions
    
    def predict(self, features):
        """主任务预测（兼容性）"""
        all_preds = self.predict_all_tasks(features)
        return all_preds['cost']  # 默认返回成本预测
```

## 完整的自定义Pipeline

### 端到端自定义示例

```python
from models.Gnto import GNTO
from models.DataPreprocessor import DataPreprocessor

class CustomGNTO(GNTO):
    def __init__(self, config=None):
        # 使用自定义组件
        custom_encoder = CustomEncoder()
        custom_tree_model = AttentionTreeModel()
        custom_predictor = DNNPredictioner()
        
        super().__init__(
            encoder=custom_encoder,
            tree_model=custom_tree_model,
            predictioner=custom_predictor
        )
        
        self.config = config or {}
    
    def run_with_confidence(self, plan):
        """带置信度的预测"""
        # 多次预测取平均（模拟不确定性）
        predictions = []
        for _ in range(self.config.get('num_samples', 10)):
            pred = self.run(plan)
            predictions.append(pred)
        
        mean_pred = np.mean(predictions)
        std_pred = np.std(predictions)
        
        return {
            'prediction': mean_pred,
            'confidence': 1.0 / (1.0 + std_pred),  # 简单的置信度计算
            'std': std_pred
        }
```

## 最佳实践

### 1. 接口一致性

确保自定义组件遵循相同的接口约定：

```python
class BaseComponent:
    def __init__(self, **kwargs):
        """统一的初始化接口"""
        pass
    
    def process(self, input_data):
        """统一的处理接口"""
        raise NotImplementedError
```

### 2. 错误处理

```python
class RobustEncoder(Encoder):
    def encode(self, node):
        try:
            return super().encode(node)
        except AttributeError as e:
            # 处理缺失属性
            print(f"Warning: Missing attribute {e}, using default encoding")
            return np.zeros(1)
        except Exception as e:
            # 通用错误处理
            print(f"Error in encoding: {e}")
            raise
```

### 3. 配置管理

```python
class ConfigurableComponent:
    def __init__(self, config_file=None, **kwargs):
        self.config = self._load_config(config_file, kwargs)
    
    def _load_config(self, config_file, overrides):
        """加载和合并配置"""
        default_config = self._get_default_config()
        
        if config_file:
            # 从文件加载配置
            file_config = self._load_from_file(config_file)
            default_config.update(file_config)
        
        # 应用覆盖参数
        default_config.update(overrides)
        return default_config
```

### 4. 测试和验证

```python
def test_custom_component():
    """自定义组件测试"""
    # 准备测试数据
    test_plan = {
        "Node Type": "Test",
        "Cost": 100.0,
        "Plans": []
    }
    
    # 测试组件
    component = CustomEncoder()
    result = component.encode(test_plan)
    
    # 验证结果
    assert isinstance(result, np.ndarray)
    assert len(result) > 0
    assert not np.any(np.isnan(result))
    
    print("✓ Custom component test passed")
```

## 部署和集成

### 1. 模块化部署

```python
# components/custom_encoder.py
class CustomEncoder(Encoder):
    pass

# main.py
from components.custom_encoder import CustomEncoder
from models.Gnto import GNTO

def create_custom_pipeline():
    return GNTO(encoder=CustomEncoder())
```

### 2. 配置驱动

```yaml
# config.yaml
encoder:
  type: "CustomEncoder"
  params:
    embedding_dim: 64
    
tree_model:
  type: "AttentionTreeModel"
  params:
    attention_dim: 32
```

### 3. 版本兼容性

```python
class VersionedComponent:
    VERSION = "1.0.0"
    
    def __init__(self):
        self._check_compatibility()
    
    def _check_compatibility(self):
        """检查版本兼容性"""
        # 实现版本检查逻辑
        pass
```

通过遵循这些扩展指南，你可以创建强大且可维护的自定义GNTO组件。
