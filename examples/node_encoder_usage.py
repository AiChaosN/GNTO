#!/usr/bin/env python3
"""
NodeEncoder使用示例
展示不同类型的NodeEncoder的使用方法
"""

import sys
import os
sys.path.append(os.path.abspath(".."))

import torch
import torch.nn as nn
from typing import List, Dict
import numpy as np

# 导入自定义模块
from models.NodeEncoder import (
    NodeEncoder_Mini,
    NodeEncoder_Enhanced,
    NodeEncoder_Vectorized,
    NodeEncoder_Mixed,
    default_emb_dim
)

# PostgreSQL查询计划中常见的节点类型
NODE_TYPES = [
    'Bitmap Heap Scan', 'Bitmap Index Scan', 'BitmapAnd', 'Gather', 
    'Gather Merge', 'Hash', 'Hash Join', 'Index Scan', 'Materialize', 
    'Merge Join', 'Nested Loop', 'Seq Scan', 'Sort'
]

def demo_mini_encoder():
    """演示基础的NodeEncoder_Mini使用"""
    print("=== NodeEncoder_Mini 示例 ===")
    
    # 创建编码器
    encoder = NodeEncoder_Mini(in_dim=16, d_node=32)
    
    # 模拟输入数据（批次大小为5，特征维度为16）
    x = torch.randn(5, 16)
    
    # 编码
    encoded = encoder(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {encoded.shape}")
    print(f"编码器参数数量: {sum(p.numel() for p in encoder.parameters())}")
    print()

def demo_enhanced_encoder():
    """演示增强版NodeEncoder_Enhanced使用"""
    print("=== NodeEncoder_Enhanced 示例 ===")
    
    # 创建增强编码器
    encoder = NodeEncoder_Enhanced(
        in_dim=16,
        d_node=64,
        num_node_types=13,
        use_attention=True,
        use_residual=True,
        dropout=0.1
    )
    
    # 模拟输入数据
    x = torch.randn(5, 16)
    node_type_ids = torch.randint(0, 13, (5,))  # 节点类型ID
    
    # 编码（带节点类型信息）
    encoded = encoder(x, node_type_ids)
    print(f"输入形状: {x.shape}")
    print(f"节点类型ID: {node_type_ids}")
    print(f"输出形状: {encoded.shape}")
    print(f"编码器参数数量: {sum(p.numel() for p in encoder.parameters())}")
    print()

def demo_vectorized_encoder():
    """演示基于向量化的NodeEncoder_Vectorized使用"""
    print("=== NodeEncoder_Vectorized 示例 ===")
    
    # 创建向量化编码器
    encoder = NodeEncoder_Vectorized(
        node_types=NODE_TYPES,
        d_node=32,
        plan_rows_max=2e8,
        use_parallel_feature=True,
        use_cost_features=True,
        dropout=0.1
    )
    
    # 模拟查询计划节点数据
    sample_nodes = [
        {
            "Node Type": "Hash Join",
            "Parallel Aware": False,
            "Plan Rows": 1000000,
            "Startup Cost": 1000.0,
            "Total Cost": 5000.0
        },
        {
            "Node Type": "Seq Scan",
            "Parallel Aware": True,
            "Plan Rows": 500000,
            "Startup Cost": 0.0,
            "Total Cost": 2000.0
        },
        {
            "Node Type": "Index Scan",
            "Parallel Aware": False,
            "Plan Rows": 100,
            "Startup Cost": 0.5,
            "Total Cost": 10.0
        }
    ]
    
    # 方法1: 直接传入节点字典列表
    encoded_from_dict = encoder(sample_nodes)
    print(f"从字典编码 - 输出形状: {encoded_from_dict.shape}")
    
    # 方法2: 先手动向量化，再编码
    vectorized = encoder.vectorize_nodes(sample_nodes)
    encoded_from_vector = encoder(vectorized)
    print(f"从向量编码 - 输出形状: {encoded_from_vector.shape}")
    print(f"向量化特征维度: {vectorized.shape[1]}")
    print(f"编码器参数数量: {sum(p.numel() for p in encoder.parameters())}")
    print()

def demo_mixed_encoder():
    """演示混合编码器NodeEncoder_Mixed使用"""
    print("=== NodeEncoder_Mixed 示例 ===")
    
    # 假设我们有3个数值特征和2个类别特征
    # 类别特征的基数分别为10和5
    encoder = NodeEncoder_Mixed(
        num_in_dim=3,
        cat_cardinalities=[10, 5],
        d_node=32,
        dropout=0.1
    )
    
    # 模拟数据
    batch_size = 4
    x_num = torch.randn(batch_size, 3)  # 数值特征
    x_cat = torch.randint(0, 10, (batch_size, 2))  # 类别特征ID
    x_cat[:, 1] = torch.randint(0, 5, (batch_size,))  # 第二个类别特征基数为5
    
    # 编码
    encoded = encoder(x_num, x_cat)
    print(f"数值特征形状: {x_num.shape}")
    print(f"类别特征形状: {x_cat.shape}")
    print(f"类别特征值: {x_cat}")
    print(f"输出形状: {encoded.shape}")
    print(f"编码器参数数量: {sum(p.numel() for p in encoder.parameters())}")
    print()

def demo_comparison():
    """比较不同编码器的性能"""
    print("=== 编码器性能比较 ===")
    
    # 统一的输入维度和输出维度
    in_dim, d_node = 16, 32
    batch_size = 100
    x = torch.randn(batch_size, in_dim)
    
    encoders = {
        "Mini": NodeEncoder_Mini(in_dim, d_node),
        "Enhanced": NodeEncoder_Enhanced(in_dim, d_node, use_attention=False),
        "Enhanced+Attention": NodeEncoder_Enhanced(in_dim, d_node, use_attention=True),
    }
    
    import time
    
    for name, encoder in encoders.items():
        # 参数数量
        num_params = sum(p.numel() for p in encoder.parameters())
        
        # 推理时间测试
        encoder.eval()
        with torch.no_grad():
            start_time = time.time()
            for _ in range(100):
                if "Enhanced" in name:
                    output = encoder(x)
                else:
                    output = encoder(x)
            end_time = time.time()
        
        avg_time = (end_time - start_time) / 100 * 1000  # 毫秒
        
        print(f"{name:20} | 参数数量: {num_params:6d} | 推理时间: {avg_time:.2f}ms")
    
    print()

def create_production_example():
    """创建一个生产环境使用的完整示例"""
    print("=== 生产环境使用示例 ===")
    
    # 根据你的项目需求选择合适的编码器
    # 这里选择NodeEncoder_Vectorized，因为它最接近你的example实现
    
    encoder = NodeEncoder_Vectorized(
        node_types=NODE_TYPES,
        d_node=32,
        plan_rows_max=2e8,
        use_parallel_feature=True,
        use_cost_features=True,
        dropout=0.1
    )
    
    # 模拟从你的数据预处理管道得到的数据
    def simulate_plan_data(num_plans=5, max_nodes_per_plan=8):
        """模拟查询计划数据"""
        plans = []
        for i in range(num_plans):
            num_nodes = np.random.randint(2, max_nodes_per_plan + 1)
            plan_nodes = []
            
            for j in range(num_nodes):
                node = {
                    "Node Type": np.random.choice(NODE_TYPES),
                    "Parallel Aware": np.random.choice([True, False]),
                    "Plan Rows": np.random.randint(1, 1000000),
                    "Startup Cost": np.random.uniform(0, 1000),
                    "Total Cost": np.random.uniform(100, 10000)
                }
                plan_nodes.append(node)
            
            plans.append(plan_nodes)
        
        return plans
    
    # 生成模拟数据
    plans = simulate_plan_data(num_plans=3, max_nodes_per_plan=5)
    
    print("模拟的查询计划:")
    for i, plan in enumerate(plans):
        print(f"  计划 {i+1}: {len(plan)} 个节点")
        for j, node in enumerate(plan):
            print(f"    节点 {j+1}: {node['Node Type']}, Rows: {node['Plan Rows']}")
    
    # 编码每个计划的节点
    print("\n编码结果:")
    for i, plan_nodes in enumerate(plans):
        encoded_nodes = encoder(plan_nodes)
        print(f"  计划 {i+1} 编码形状: {encoded_nodes.shape}")
    
    print(f"\n编码器总参数数量: {sum(p.numel() for p in encoder.parameters())}")

if __name__ == "__main__":
    print("NodeEncoder 使用示例")
    print("=" * 50)
    
    # 运行各种示例
    demo_mini_encoder()
    demo_enhanced_encoder()
    demo_vectorized_encoder()
    demo_mixed_encoder()
    demo_comparison()
    create_production_example()
    
    print("所有示例运行完成！")
