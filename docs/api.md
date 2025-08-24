# API文档

## GNTO类
主要的Pipeline类，整合所有组件。

```python
class GNTO:
    def __init__(self, 
                 preprocessor=None, 
                 encoder=None, 
                 tree_model=None, 
                 predictioner=None):
        """
        初始化GNTO pipeline
        
        参数:
            preprocessor: DataPreprocessor实例，默认为DataPreprocessor()
            encoder: Encoder实例，默认为Encoder()
            tree_model: TreeModel实例，默认为TreeModel()
            predictioner: Predictioner实例，默认为Predictioner()
        """
    
    def run(self, plan: dict) -> float:
        """
        执行完整的预测pipeline
        
        参数:
            plan: 包含查询计划的字典，必须包含"Node Type"字段
        
        返回:
            float: 性能预测值
        
        示例:
            >>> gnto = GNTO()
            >>> plan = {"Node Type": "Seq Scan", "Cost": 100.0}
            >>> prediction = gnto.run(plan)
        """
```

## DataPreprocessor类

将原始查询计划转换为结构化的PlanNode树。

```python
class DataPreprocessor:
    plan_key: str = "Plans"  # 子计划的键名
    
    def preprocess(self, plan: dict) -> PlanNode:
        """
        将原始计划转换为PlanNode树
        
        参数:
            plan: 包含查询计划的字典
        
        返回:
            PlanNode: 结构化的计划节点树
        """
    
    def preprocess_all(self, plans: list) -> list:
        """
        批量处理多个计划
        
        参数:
            plans: 计划字典的列表
        
        返回:
            list: PlanNode对象的列表
        """
```

### PlanNode类

```python
@dataclass
class PlanNode:
    node_type: str                           # 节点类型
    children: List["PlanNode"] = []          # 子节点列表
    extra_info: Dict[str, Any] = {}          # 额外信息字典
```

## Encoder类

将PlanNode树编码为数值向量。

```python
class Encoder:
    def __init__(self):
        """初始化编码器，创建空的节点索引"""
        self.node_index: Dict[str, int] = {}
    
    def encode(self, node: PlanNode) -> np.ndarray:
        """
        编码单个PlanNode为向量
        
        参数:
            node: 要编码的PlanNode对象
        
        返回:
            np.ndarray: 编码后的向量
        
        说明:
            使用one-hot编码表示节点类型，递归聚合子节点向量
        """
    
    def encode_all(self, nodes: list) -> list:
        """
        批量编码多个节点
        
        参数:
            nodes: PlanNode对象的列表
        
        返回:
            list: 编码向量的列表
        """
```

## TreeModel类

将多个编码向量聚合为单一向量表示。

```python
class TreeModel:
    def __init__(self, reduction="mean"):
        """
        初始化树模型
        
        参数:
            reduction: 聚合方法，支持"mean"或"sum"
        
        异常:
            ValueError: 当reduction不是"mean"或"sum"时抛出
        """
    
    def forward(self, vectors: list) -> np.ndarray:
        """
        将向量列表聚合为单一向量
        
        参数:
            vectors: numpy数组的列表
        
        返回:
            np.ndarray: 聚合后的向量
        
        说明:
            自动处理不同维度的向量，通过padding对齐到最大维度
        """
```

## Predictioner类

线性预测头，将特征向量转换为标量预测值。

```python
class Predictioner:
    def __init__(self, weights=None):
        """
        初始化预测器
        
        参数:
            weights: 预测权重，None时使用全1权重
        """
    
    def predict(self, features: np.ndarray) -> float:
        """
        基于特征向量进行预测
        
        参数:
            features: 特征向量
        
        返回:
            float: 预测值
        
        说明:
            如果权重维度小于特征维度，会自动padding；
            如果权重维度大于特征维度，会自动截断
        """
```

## Utils模块

```python
def set_seed(seed: int) -> None:
    """
    设置随机种子
    
    参数:
        seed: 随机种子值
    
    说明:
        同时设置random和numpy的随机种子，确保结果可重现
    """

def flatten(nested: Iterable[Iterable]) -> List:
    """
    扁平化嵌套列表
    
    参数:
        nested: 嵌套的可迭代对象
    
    返回:
        List: 扁平化后的列表
    
    示例:
        >>> flatten([[1, 2], [3, 4]])
        [1, 2, 3, 4]
    """
```

## 异常处理

### 常见异常

- **ImportError**: 模块导入失败，检查相对导入路径
- **ValueError**: TreeModel的reduction参数错误
- **AttributeError**: PlanNode缺少必要属性
- **TypeError**: 输入数据类型错误

### 调试建议

1. 确保输入的计划字典包含"Node Type"字段
2. 检查嵌套计划是否使用"Plans"作为键名
3. 验证所有numpy数组的维度兼容性
4. 使用Demo.ipynb中的示例作为参考
