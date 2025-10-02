"""
NodeEncoder工厂类
提供便捷的接口来创建和配置不同类型的NodeEncoder
"""

from typing import Dict, List, Optional, Any, Union
import torch
import torch.nn as nn

from .NodeEncoder import (
    NodeEncoder_Mini,
    NodeEncoder_Enhanced,
    NodeEncoder_Vectorized,
    NodeEncoder_Mixed,
    default_emb_dim
)

class NodeEncoderFactory:
    """NodeEncoder工厂类，提供统一的接口创建不同类型的编码器"""
    
    # PostgreSQL查询计划中常见的节点类型
    DEFAULT_NODE_TYPES = [
        'Bitmap Heap Scan', 'Bitmap Index Scan', 'BitmapAnd', 'Gather', 
        'Gather Merge', 'Hash', 'Hash Join', 'Index Scan', 'Materialize', 
        'Merge Join', 'Nested Loop', 'Seq Scan', 'Sort'
    ]
    
    @classmethod
    def create_mini_encoder(
        cls,
        in_dim: int,
        d_node: int
    ) -> NodeEncoder_Mini:
        """
        创建简单的Mini编码器
        
        Args:
            in_dim: 输入特征维度
            d_node: 输出节点嵌入维度
        
        Returns:
            NodeEncoder_Mini实例
        """
        return NodeEncoder_Mini(in_dim=in_dim, d_node=d_node)
    
    @classmethod
    def create_enhanced_encoder(
        cls,
        in_dim: int,
        d_node: int,
        num_node_types: int = 13,
        use_attention: bool = True,
        use_residual: bool = True,
        dropout: float = 0.1,
        hidden_dim: Optional[int] = None
    ) -> NodeEncoder_Enhanced:
        """
        创建增强版编码器
        
        Args:
            in_dim: 输入特征维度
            d_node: 输出节点嵌入维度
            num_node_types: 节点类型数量
            use_attention: 是否使用注意力机制
            use_residual: 是否使用残差连接
            dropout: Dropout概率
            hidden_dim: 隐藏层维度
        
        Returns:
            NodeEncoder_Enhanced实例
        """
        return NodeEncoder_Enhanced(
            in_dim=in_dim,
            d_node=d_node,
            num_node_types=num_node_types,
            use_attention=use_attention,
            use_residual=use_residual,
            dropout=dropout,
            hidden_dim=hidden_dim
        )
    
    @classmethod
    def create_vectorized_encoder(
        cls,
        d_node: int,
        node_types: Optional[List[str]] = None,
        plan_rows_max: float = 2e8,
        use_parallel_feature: bool = True,
        use_cost_features: bool = True,
        dropout: float = 0.1
    ) -> NodeEncoder_Vectorized:
        """
        创建向量化编码器（基于手工特征工程）
        
        Args:
            d_node: 输出节点嵌入维度
            node_types: 节点类型列表
            plan_rows_max: 计划行数的最大值（用于归一化）
            use_parallel_feature: 是否使用并行特征
            use_cost_features: 是否使用成本特征
            dropout: Dropout概率
        
        Returns:
            NodeEncoder_Vectorized实例
        """
        if node_types is None:
            node_types = cls.DEFAULT_NODE_TYPES
        
        return NodeEncoder_Vectorized(
            node_types=node_types,
            d_node=d_node,
            plan_rows_max=plan_rows_max,
            use_parallel_feature=use_parallel_feature,
            use_cost_features=use_cost_features,
            dropout=dropout
        )
    
    @classmethod
    def create_mixed_encoder(
        cls,
        num_in_dim: int,
        cat_cardinalities: List[int],
        d_node: int,
        emb_dims: Optional[List[int]] = None,
        hidden_dim: Optional[int] = None,
        dropout: float = 0.1,
        use_batch_norm: bool = True
    ) -> NodeEncoder_Mixed:
        """
        创建混合编码器（数值+类别特征）
        
        Args:
            num_in_dim: 数值特征维度
            cat_cardinalities: 每个类别特征的基数
            d_node: 输出节点嵌入维度
            emb_dims: 每个类别特征的embedding维度
            hidden_dim: 隐藏层维度
            dropout: Dropout概率
            use_batch_norm: 是否使用BatchNorm
        
        Returns:
            NodeEncoder_Mixed实例
        """
        return NodeEncoder_Mixed(
            num_in_dim=num_in_dim,
            cat_cardinalities=cat_cardinalities,
            d_node=d_node,
            emb_dims=emb_dims,
            hidden_dim=hidden_dim,
            dropout=dropout,
            use_batch_norm=use_batch_norm
        )
    
    @classmethod
    def create_encoder_from_config(
        cls,
        config: Dict[str, Any]
    ) -> Union[NodeEncoder_Mini, NodeEncoder_Enhanced, NodeEncoder_Vectorized, NodeEncoder_Mixed]:
        """
        从配置字典创建编码器
        
        Args:
            config: 包含编码器类型和参数的配置字典
                   必须包含 'type' 字段指定编码器类型
        
        Returns:
            相应类型的NodeEncoder实例
        
        Example:
            config = {
                'type': 'vectorized',
                'd_node': 32,
                'use_cost_features': True,
                'dropout': 0.1
            }
        """
        encoder_type = config.get('type', 'mini').lower()
        
        if encoder_type == 'mini':
            return cls.create_mini_encoder(
                in_dim=config['in_dim'],
                d_node=config['d_node']
            )
        
        elif encoder_type == 'enhanced':
            return cls.create_enhanced_encoder(
                in_dim=config['in_dim'],
                d_node=config['d_node'],
                num_node_types=config.get('num_node_types', 13),
                use_attention=config.get('use_attention', True),
                use_residual=config.get('use_residual', True),
                dropout=config.get('dropout', 0.1),
                hidden_dim=config.get('hidden_dim')
            )
        
        elif encoder_type == 'vectorized':
            return cls.create_vectorized_encoder(
                d_node=config['d_node'],
                node_types=config.get('node_types'),
                plan_rows_max=config.get('plan_rows_max', 2e8),
                use_parallel_feature=config.get('use_parallel_feature', True),
                use_cost_features=config.get('use_cost_features', True),
                dropout=config.get('dropout', 0.1)
            )
        
        elif encoder_type == 'mixed':
            return cls.create_mixed_encoder(
                num_in_dim=config['num_in_dim'],
                cat_cardinalities=config['cat_cardinalities'],
                d_node=config['d_node'],
                emb_dims=config.get('emb_dims'),
                hidden_dim=config.get('hidden_dim'),
                dropout=config.get('dropout', 0.1),
                use_batch_norm=config.get('use_batch_norm', True)
            )
        
        else:
            raise ValueError(f"不支持的编码器类型: {encoder_type}")
    
    @classmethod
    def get_recommended_config(
        cls,
        use_case: str = "query_plan",
        performance_level: str = "balanced"
    ) -> Dict[str, Any]:
        """
        根据使用场景和性能要求推荐配置
        
        Args:
            use_case: 使用场景 ("query_plan", "general", "embedding")
            performance_level: 性能级别 ("fast", "balanced", "accurate")
        
        Returns:
            推荐的配置字典
        """
        configs = {
            "query_plan": {
                "fast": {
                    "type": "mini",
                    "in_dim": 16,
                    "d_node": 32
                },
                "balanced": {
                    "type": "vectorized",
                    "d_node": 32,
                    "use_cost_features": True,
                    "use_parallel_feature": True,
                    "dropout": 0.1
                },
                "accurate": {
                    "type": "enhanced",
                    "in_dim": 16,
                    "d_node": 64,
                    "use_attention": True,
                    "use_residual": True,
                    "dropout": 0.1,
                    "hidden_dim": 128
                }
            },
            "general": {
                "fast": {
                    "type": "mini",
                    "in_dim": 32,
                    "d_node": 32
                },
                "balanced": {
                    "type": "enhanced",
                    "in_dim": 32,
                    "d_node": 64,
                    "use_attention": False,
                    "use_residual": True,
                    "dropout": 0.1
                },
                "accurate": {
                    "type": "enhanced",
                    "in_dim": 32,
                    "d_node": 128,
                    "use_attention": True,
                    "use_residual": True,
                    "dropout": 0.1,
                    "hidden_dim": 256
                }
            }
        }
        
        return configs.get(use_case, configs["general"]).get(performance_level, configs["general"]["balanced"])

class NodeEncoderWrapper:
    """NodeEncoder包装类，提供统一的接口处理不同类型的输入"""
    
    def __init__(self, encoder: nn.Module, encoder_type: str):
        self.encoder = encoder
        self.encoder_type = encoder_type.lower()
    
    def encode(self, data, **kwargs):
        """
        统一的编码接口
        
        Args:
            data: 输入数据，格式根据编码器类型而定
            **kwargs: 额外的参数
        
        Returns:
            编码后的特征张量
        """
        if self.encoder_type == 'vectorized':
            # 对于vectorized编码器，可以直接处理节点字典列表
            return self.encoder(data)
        
        elif self.encoder_type == 'mixed':
            # 对于mixed编码器，需要分别传入数值和类别特征
            x_num = kwargs.get('x_num')
            x_cat = kwargs.get('x_cat')
            return self.encoder(x_num, x_cat)
        
        elif self.encoder_type in ['mini', 'enhanced']:
            # 对于mini和enhanced编码器，传入特征张量
            node_type_ids = kwargs.get('node_type_ids')
            if self.encoder_type == 'enhanced' and node_type_ids is not None:
                return self.encoder(data, node_type_ids)
            else:
                return self.encoder(data)
        
        else:
            raise ValueError(f"不支持的编码器类型: {self.encoder_type}")
    
    def get_output_dim(self) -> int:
        """获取输出维度"""
        if hasattr(self.encoder, 'd_node'):
            return self.encoder.d_node
        elif hasattr(self.encoder, 'out_dim'):
            return self.encoder.out_dim
        else:
            # 通过一个dummy输入来推断输出维度
            with torch.no_grad():
                if self.encoder_type == 'vectorized':
                    dummy_node = [{
                        "Node Type": "Seq Scan",
                        "Parallel Aware": False,
                        "Plan Rows": 1000,
                        "Startup Cost": 0.0,
                        "Total Cost": 100.0
                    }]
                    output = self.encoder(dummy_node)
                    return output.shape[-1]
                else:
                    # 其他类型需要具体的输入维度信息
                    raise ValueError("无法推断输出维度，请手动指定")

# 便捷函数
def create_query_plan_encoder(
    d_node: int = 32,
    performance_level: str = "balanced",
    **kwargs
) -> NodeEncoderWrapper:
    """
    为查询计划任务创建推荐的编码器
    
    Args:
        d_node: 输出节点嵌入维度
        performance_level: 性能级别
        **kwargs: 额外参数
    
    Returns:
        NodeEncoderWrapper实例
    """
    config = NodeEncoderFactory.get_recommended_config("query_plan", performance_level)
    config.update(kwargs)
    config['d_node'] = d_node
    
    encoder = NodeEncoderFactory.create_encoder_from_config(config)
    return NodeEncoderWrapper(encoder, config['type'])
