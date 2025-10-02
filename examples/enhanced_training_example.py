#!/usr/bin/env python3
"""
使用增强版NodeEncoder的完整训练示例
展示如何将新的NodeEncoder集成到现有的训练流程中
"""

import sys
import os
sys.path.append(os.path.abspath(".."))

import torch
import torch.nn as nn
from torch_geometric.data import Data, Batch
from torch_geometric.loader import DataLoader
import numpy as np
import pandas as pd
from typing import List, Dict
from datetime import datetime
from sklearn.model_selection import train_test_split
import time

# 导入项目模块
from models.DataPreprocessor import get_plans_dict, DataPreprocessor, plan_trees_to_graphs, graphs_to_df, df_to_graphs
from models.NodeEncoderFactory import NodeEncoderFactory, create_query_plan_encoder
from models.TreeEncoder import GATTreeEncoder
from models.PredictionHead import PredictionHead
from models.TrainAndEval import (
    build_dataset, EarlyStopping, train_epoch, validate_epoch, evaluate_model
)
from models.NodeVectorizerAll import NodeVectorizerAll, create_enhanced_vectorizer_all

class EnhancedPlanCostModel(nn.Module):
    """
    使用增强版NodeEncoder的查询成本预测模型
    """
    def __init__(self, node_encoder: nn.Module, tree_encoder: nn.Module, prediction_head: nn.Module):
        super().__init__()
        self.node_encoder = node_encoder
        self.tree_encoder = tree_encoder
        self.prediction_head = prediction_head
    
    def forward(self, data: Data | Batch):
        """
        前向传播
        
        Args:
            data: PyTorch Geometric的Data或Batch对象
                 包含 x (节点特征), edge_index (边索引), batch (批次索引)
        
        Returns:
            预测的执行时间
        """
        # 节点编码
        x = self.node_encoder(data.x)  # [N, d_node]
        
        # 图级编码
        g = self.tree_encoder(x, data.edge_index, data.batch)  # [B, d_graph]
        
        # 预测
        y = self.prediction_head(g)  # [B, 1]
        
        return y

def NodeVectorizer(matrix_plans: List[List[dict]], node_type_mapping: Dict[str, int]) -> List[List[List]]:
    """
    使用与demo.py相同的节点向量化方法
    """
    node_type_list = ['Bitmap Heap Scan', 'Bitmap Index Scan', 'BitmapAnd', 'Gather', 'Gather Merge', 'Hash', 'Hash Join', 'Index Scan', 'Materialize', 'Merge Join', 'Nested Loop', 'Seq Scan', 'Sort']
    parallel_list = [True, False]
    plan_rows_max = 2*10**8
    
    res = []
    for mp in matrix_plans:
        plan_matrix = []
        for node in mp:
            node_vector = [0] * (len(node_type_list) + 2 + 1)
            offset = 0
            # 1. node_type
            node_vector[node_type_mapping[node["Node Type"]] + offset] = 1
            offset += len(node_type_list)
            # 2. parallel
            node_vector[parallel_list.index(node["Parallel Aware"]) + offset] = 1
            
            offset += len(parallel_list)
            # 3. rows
            node_vector[offset] = node["Plan Rows"] / plan_rows_max
            plan_matrix.append(node_vector)
        res.append(plan_matrix)
    return res

def demonstrate_different_encoders():
    """演示不同NodeEncoder的效果"""
    print("=== 不同NodeEncoder效果对比 ===")
    
    # 节点类型列表
    node_types = [
        'Bitmap Heap Scan', 'Bitmap Index Scan', 'BitmapAnd', 'Gather', 
        'Gather Merge', 'Hash', 'Hash Join', 'Index Scan', 'Materialize', 
        'Merge Join', 'Nested Loop', 'Seq Scan', 'Sort'
    ]
    
    # 创建不同类型的编码器
    encoders = {
        "Mini_16": NodeEncoderFactory.create_mini_encoder(in_dim=16, d_node=32),
        "Mini_58": NodeEncoderFactory.create_mini_encoder(in_dim=58, d_node=32),
        "Enhanced_16": NodeEncoderFactory.create_enhanced_encoder(
            in_dim=16, d_node=32, use_attention=True
        ),
        "Enhanced_58": NodeEncoderFactory.create_enhanced_encoder(
            in_dim=58, d_node=32, use_attention=True
        ),
        "Vectorized": NodeEncoderFactory.create_vectorized_encoder(
            d_node=32, node_types=node_types, use_cost_features=True
        )
    }
    
    # 模拟数据
    batch_size = 10
    x_16 = torch.randn(batch_size, 16)  # 原始16维特征
    x_58 = torch.randn(batch_size, 58)  # 完整58维特征
    
    # 模拟节点数据（用于Vectorized encoder）
    sample_nodes = []
    for _ in range(batch_size):
        node = {
            "Node Type": np.random.choice(node_types),
            "Parallel Aware": np.random.choice([True, False]),
            "Plan Rows": np.random.randint(1, 1000000),
            "Startup Cost": np.random.uniform(0, 1000),
            "Total Cost": np.random.uniform(100, 10000)
        }
        sample_nodes.append(node)
    
    print(f"原始特征维度: {x_16.shape}")
    print(f"完整特征维度: {x_58.shape}")
    
    for name, encoder in encoders.items():
        encoder.eval()
        with torch.no_grad():
            if name == "Vectorized":
                # Vectorized encoder使用节点字典
                output = encoder(sample_nodes)
            elif "16" in name:
                # 使用16维特征
                output = encoder(x_16)
            else:
                # 使用58维特征
                output = encoder(x_58)
            
            num_params = sum(p.numel() for p in encoder.parameters())
            print(f"{name:15} | 输出形状: {output.shape} | 参数数量: {num_params:6d}")
    
    print()

def demonstrate_vectorizer_comparison():
    """演示不同向量化方法的对比"""
    print("=== 向量化方法对比 ===")
    
    # 创建示例数据
    sample_data = {
        'plan_id': [0, 0, 1, 1, 2],
        'node_idx': [0, 1, 0, 1, 0],
        'Node Type': ['Hash Join', 'Seq Scan', 'Index Scan', 'Sort', 'Nested Loop'],
        'Parallel Aware': [False, True, False, False, True],
        'Relation Name': ['table1', 'table2', '', 'table3', 'table4'],
        'Alias': ['t1', 't2', '', 't3', 't4'],
        'Plan Rows': [1000, 5000, 100, 2000, 500],
        'Startup Cost': [10.5, 0.0, 0.5, 15.2, 5.0],
        'Total Cost': [100.5, 50.0, 5.5, 25.2, 20.0],
        'Actual Rows': [950, 4800, 98, 1950, 480],
        'Join Type': ['Inner', '', '', '', 'Left'],
        'Filter': ['', 'id > 100', '', '', 'status = 1'],
        'Workers Planned': [2, 0, 0, 0, 4],
        'Hash Buckets': [1024, 0, 0, 0, 2048],
        'Peak Memory Usage': [512, 0, 0, 0, 1024]
    }
    
    df = pd.DataFrame(sample_data)
    print(f"示例数据形状: {df.shape}")
    print(f"包含列: {list(df.columns)}")
    
    # 方法1: 原始NodeVectorizer (16维)
    print("\n1. 原始NodeVectorizer (16维):")
    node_type_mapping = {'Hash Join': 0, 'Seq Scan': 1, 'Index Scan': 2, 'Sort': 3, 'Nested Loop': 4}
    
    # 转换为matrix_plans格式
    matrix_plans = []
    for plan_id, plan_group in df.groupby('plan_id'):
        plan_nodes = []
        for _, row in plan_group.iterrows():
            node_dict = row.to_dict()
            plan_nodes.append(node_dict)
        matrix_plans.append(plan_nodes)
    
    # 原始向量化
    original_vectors = NodeVectorizer(matrix_plans, node_type_mapping)
    print(f"  - 特征维度: {len(original_vectors[0][0])}")
    print(f"  - 计划数: {len(original_vectors)}")
    print(f"  - 第一个节点向量: {original_vectors[0][0]}")
    
    # 方法2: NodeVectorizerAll (58维)
    print("\n2. NodeVectorizerAll (58维):")
    vectorizer = NodeVectorizerAll()
    result = vectorizer.vectorize_dataframe(df)
    
    print(f"  - 特征维度: {result['feature_dim']}")
    print(f"  - 计划数: {len(result['vectors'])}")
    print(f"  - 第一个节点向量: {result['vectors'][0][0][:]}")
    print(f"  - 特征名称示例: {result['feature_names'][:10]}")
    
    # 统计信息
    stats = vectorizer.get_statistics(df)
    print(f"  - 平均每计划节点数: {stats['avg_nodes_per_plan']:.1f}")
    print(f"  - 节点类型分布: {stats['node_type_distribution']}")
    
    print()

def train_model(model, train_loader, val_loader, optimizer, scheduler, criterion, 
                early_stopping, device, weight_path, num_epochs=100):
    """
    训练模型的主函数
    """
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
            date = datetime.now().strftime("%m%d")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': train_loss,
                'val_loss': val_loss,
            }, f'../results/weight_{date}.pth')
    
    print("-" * 60)
    print("训练完成!")
    print(f"最佳验证损失: {best_val_loss:.6f}")
    
    return train_losses, val_losses

def main():
    """主函数：完整的训练流程示例"""
    print("增强版NodeEncoder训练示例")
    print("=" * 50)
    
    # 演示不同编码器和向量化方法
    demonstrate_different_encoders()
    demonstrate_vectorizer_comparison()
    
    # 1. 数据加载和预处理
    print("1. 数据处理...")
    json_path = "../data/train_plan_*.csv"
    plans_dict, execution_times = get_plans_dict(json_path)
    print("plans_dict:\n", plans_dict[0:5])
    print("execution_times:\n", execution_times[0:5])

    preprocessor = DataPreprocessor()
    plans_tree = preprocessor.preprocess_all(plans_dict)

    edges_list, matrix_plans = plan_trees_to_graphs(plans_tree, add_self_loops=True, undirected=False)
    print(matrix_plans[0][0])
    print(matrix_plans[0][1])
    print(edges_list[0])
    print(edges_list[99])

    plans_df = graphs_to_df(matrix_plans)
    plans_df.to_csv("../data/process/01_plans_df.csv", index=False)

    # 将node_type转换为id
    node_type = plans_df["Node Type"].unique()
    print(node_type, len(node_type))
    node_type_mapping = {k : i for i, k in enumerate(node_type)}
    plans_df["NodeType_id"] = plans_df["Node Type"].map(node_type_mapping)
    print(plans_df[["Node Type", "NodeType_id"]].head())

    new_matrix_plans = df_to_graphs(plans_df)

    # 2. 节点向量化
    print("2. 节点向量化...")
    
    # 选择向量化方式
    use_full_vectorizer = True  # 设置为True使用完整向量化器，False使用原始方式
    
    if use_full_vectorizer:
        print("使用NodeVectorizerAll进行完整向量化...")
        # 使用完整向量化器
        vectorizer_result, vectorizer = create_enhanced_vectorizer_all(plans_df)
        res = vectorizer_result['vectors']
        F_num = vectorizer_result['feature_dim']
        print(f"完整向量化 - 特征维度: {F_num}")
        print(f"第一个计划的节点数: {len(res[0])}")
        print(f"第一个节点的向量维度: {len(res[0][0])}")
        print(f"第一个节点向量: {res[0][0][:]}")
        
        # 显示特征分布
        feature_names = vectorizer_result['feature_names']
        print(f"特征名称示例: {feature_names[:15]}...")
        
    else:
        print("使用原始NodeVectorizer...")
        # 使用原始向量化方式
        res = NodeVectorizer(new_matrix_plans, node_type_mapping)
        F_num = 16
        print("NodeType[13] : parallel[2] : rows[1]")
        print(len(res[0]))
        print(len(res[0][0]))
        print(res[0][0])
    
    # 3. 创建增强版编码器
    print("3. 创建模型...")
    d_node, d_graph = 32, 64
    print(f"输入特征维度: {F_num}, 节点嵌入维度: {d_node}, 图嵌入维度: {d_graph}")
    
    # 使用增强版NodeEncoder
    node_encoder = NodeEncoderFactory.create_enhanced_encoder(
        in_dim=F_num,
        d_node=d_node,
        use_attention=True,
        use_residual=True,
        dropout=0.1
    )
    
    tree_encoder = GATTreeEncoder(
        input_dim=d_node,
        hidden_dim=64,
        output_dim=d_graph,
        num_layers=3,
        num_heads=4,
        dropout=0.1,
        pooling="mean"
    )
    
    prediction_head = PredictionHead(d_graph, out_dim=1)
    model = EnhancedPlanCostModel(node_encoder, tree_encoder, prediction_head)
    
    # 4. 构建数据集
    print("4. 构建数据集...")
    dataset = build_dataset(res, edges_list, execution_times, in_dim=F_num, bidirectional=True)
    print(f"数据集大小: {len(dataset)}")
    for i in range(20):
        print(f"样本: x.shape={dataset[i].x.shape}, edge_index.shape={dataset[i].edge_index.shape}, y={dataset[i].y}")

    # 5. 训练准备
    print("5. 数据集划分...")
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
    print("6. 训练配置...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")

    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=10, factor=0.5)
    criterion = torch.nn.MSELoss()
    early_stopping = EarlyStopping(patience=15, min_delta=0.001)

    # 7. 开始训练
    print("7. 开始训练...")
    date = datetime.now().strftime("%m%d")
    weight_path = f'../results/{date}.pth'
    train_losses, val_losses = train_model(
        model, train_loader, val_loader, optimizer, scheduler, 
        criterion, early_stopping, device, weight_path, num_epochs=100
    )

    # 8. 测试评估
    print("8. 测试评估...")
    try:
        checkpoint = torch.load(f'../results/{date}.pth', map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print("已加载最佳模型进行测试")
    except FileNotFoundError:
        print("未找到保存的模型，使用当前模型进行测试")

    predictions, targets, metrics = evaluate_model(model, test_loader, device)
    
    print("训练完成!")

if __name__ == "__main__":
    main()
