#!/usr/bin/env python3
"""
GNTO Demo - Complete Architecture Demonstration

This demo showcases the complete GNTO pipeline following the architecture described in README.md:

1. 预处理 (DataPreprocessor): 原始JSON → TreeNode结构
2. 节点级编码 (Node Encoder): TreeNode → 带有node_vector的TreeNode  
3. 结构级编码 (Tree Encoder): 带向量的树 → plan embedding
4. 预测头 (Prediction Head): plan embedding → 最终预测

The demo supports both traditional and GNN modes with automatic fallback.
"""

import sys
import os
import numpy as np
from typing import Dict, Any, List

# Add the project root to the path to import models
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

# Import GNTO components
from models import (
    DataPreprocessor, PlanNode,
    NodeEncoder, create_simple_node_encoder, create_rich_node_encoder,
    TreeEncoder, create_tree_encoder, is_gnn_available,
    PredictionHead,
    GNTO, create_traditional_gnto, create_gnn_gnto, create_auto_gnto,
    print_package_info
)


def create_sample_query_plans() -> List[Dict[str, Any]]:
    """Create sample PostgreSQL-style query plans for demonstration."""
    
    # Simple sequential scan plan
    simple_plan = {
        "Node Type": "Seq Scan",
        "Relation Name": "users",
        "Alias": "u",
        "Startup Cost": 0.00,
        "Total Cost": 15122.68,
        "Plan Rows": 383592,
        "Plan Width": 45,
        "Filter": "(age > 25)"
    }
    
    # Complex join plan with nested structure
    complex_plan = {
        "Node Type": "Gather",
        "Startup Cost": 23540.58,
        "Total Cost": 154548.95,
        "Plan Rows": 567655,
        "Plan Width": 89,
        "Plans": [
            {
                "Node Type": "Hash Join",
                "Join Type": "Inner",
                "Startup Cost": 22540.58,
                "Total Cost": 96783.45,
                "Plan Rows": 236523,
                "Plan Width": 89,
                "Join Filter": "(u.id = o.user_id)",
                "Plans": [
                    {
                        "Node Type": "Seq Scan",
                        "Relation Name": "users",
                        "Alias": "u",
                        "Startup Cost": 0.00,
                        "Total Cost": 49166.46,
                        "Plan Rows": 649574,
                        "Plan Width": 45,
                        "Filter": "(age > 18)"
                    },
                    {
                        "Node Type": "Hash",
                        "Startup Cost": 15122.68,
                        "Total Cost": 15122.68,
                        "Plan Rows": 383592,
                        "Plan Width": 44,
                        "Plans": [
                            {
                                "Node Type": "Seq Scan",
                                "Relation Name": "orders",
                                "Alias": "o",
                                "Startup Cost": 0.00,
                                "Total Cost": 15122.68,
                                "Plan Rows": 383592,
                                "Plan Width": 44,
                                "Filter": "(created_at > '2024-01-01')"
                            }
                        ]
                    }
                ]
            }
        ]
    }
    
    # Index scan plan
    index_plan = {
        "Node Type": "Index Scan",
        "Relation Name": "products",
        "Index Name": "idx_products_category",
        "Startup Cost": 0.43,
        "Total Cost": 8.45,
        "Plan Rows": 1,
        "Plan Width": 32,
        "Index Cond": "(category_id = 5)",
        "Filter": "(price > 100.00)"
    }
    
    return [simple_plan, complex_plan, index_plan]


def demonstrate_step_by_step():
    """Demonstrate each step of the GNTO architecture separately."""
    
    print("=" * 80)
    print("GNTO 架构演示 - 逐步处理流程")
    print("=" * 80)
    
    # Get sample plans
    sample_plans = create_sample_query_plans()
    
    for i, plan in enumerate(sample_plans, 1):
        print(f"\n示例计划 {i}: {plan['Node Type']}")
        print("-" * 50)
        
        #####################################
        # Step 1: 预处理 (DataPreprocessor)
        #####################################
        print("\n步骤 1: 预处理 (DataPreprocessor)")
        print("输入: 原始JSON风格的计划")
        print("输出: TreeNode结构")
        
        preprocessor = DataPreprocessor()
        tree_node = preprocessor.preprocess(plan)
        
        print(f"转换完成:")
        print(f"   - 根节点类型: {tree_node.node_type}")
        print(f"   - 子节点数量: {len(tree_node.children)}")
        print(f"   - 额外信息字段: {list(tree_node.extra_info.keys())}")
        
        # Display tree structure
        print(f"\n树结构可视化:")
        preprocessor.print_tree(tree_node, show_details=True, max_depth=3)
        
        # Tree statistics
        stats = preprocessor.get_tree_stats(tree_node)
        print(f"\n树统计信息:")
        print(f"   - 总节点数: {stats['total_nodes']}")
        print(f"   - 最大深度: {stats['max_depth']}")
        print(f"   - 节点类型分布: {stats['node_types']}")
        
        #####################################
        # Step 2: 节点级编码 (Node Encoder) #
        #####################################
        print(f"\n步骤 2: 节点级编码 (Node Encoder)")
        print("输入: TreeNode (每个Node的vector为空)")
        print("输出: TreeNode (每个Node有自己的node_vector)")
        
        # Try both simple and rich encoders
        simple_encoder = create_simple_node_encoder()
        rich_encoder = create_rich_node_encoder(feature_dim=64)
        
        # Collect all nodes for encoding
        all_nodes = []
        def collect_nodes(node):
            all_nodes.append(node)
            for child in node.children:
                collect_nodes(child)
        
        collect_nodes(tree_node)
        
        print(f"\n简单编码器 (仅节点类型):")
        simple_vectors = []
        for node in all_nodes:
            vector = simple_encoder.encode_node(node)  # 自动存储到node.node_vector
            simple_vectors.append(vector)
            print(f"   - {node.node_type}: 向量维度 {len(vector)} (自动存储到node.node_vector)")
        
        print(f"\n丰富编码器 (节点类型 + 数值 + 类别特征):")
        rich_vectors = []
        for node in all_nodes:
            vector = rich_encoder.encode_node(node)  # 自动存储到node.node_vector
            rich_vectors.append(vector)
            features = rich_encoder.extract_node_features(node)
            print(f"   - {node.node_type}: 向量维度 {len(vector)} (自动存储到node.node_vector)")
            print(f"     * 节点类型维度: {len(features.node_type_vector)}")
            print(f"     * 数值特征维度: {len(features.numerical_features)}")
            print(f"     * 类别特征维度: {len(features.categorical_features)}")
        
        # 验证向量已存储
        print(f"\n向量存储验证:")
        for node in all_nodes:
            if node.node_vector is not None:
                print(f"   - {node.node_type}: node_vector 维度 {len(node.node_vector)}")
            else:
                print(f"   - {node.node_type}: node_vector 为空")
        
        #####################################
        # Step 3: 结构级编码 (Tree Encoder)
        #####################################
        print(f"\n步骤 3: 结构级编码 (Tree Encoder)")
        print("输入: 带有节点向量的树/DAG")
        print("输出: 全局plan embedding")
        
        # Traditional tree encoder
        tree_encoder = create_tree_encoder(use_gnn=False, reduction="mean")
        
        print(f"\n传统树编码器 (统计聚合):")
        plan_embedding_simple = tree_encoder.forward(simple_vectors)
        plan_embedding_rich = tree_encoder.forward(rich_vectors)
        
        print(f"   - 简单特征 plan embedding: 维度 {len(plan_embedding_simple)}")
        print(f"   - 丰富特征 plan embedding: 维度 {len(plan_embedding_rich)}")
        
        # Try GNN encoder if available
        if is_gnn_available():
            print(f"\nGNN树编码器:")
            try:
                gnn_tree_encoder = create_tree_encoder(
                    use_gnn=True, 
                    model_type='gcn',
                    input_dim=64,
                    hidden_dim=128,
                    output_dim=64
                )
                # Note: GNN encoder needs the original tree structure
                gnn_embedding = gnn_tree_encoder.forward([tree_node])
                print(f"   - GNN plan embedding: 维度 {len(gnn_embedding)}")
            except Exception as e:
                print(f"   - GNN编码失败: {e}")
        else:
            print(f"\n警告: GNN不可用 (需要PyTorch和PyTorch Geometric)")
        
        #####################################
        # Step 4: 预测头 (Prediction Head)
        #####################################
        print(f"\n步骤 4: 预测头 (Prediction Head)")
        print("输入: plan embedding")
        print("输出: 最终预测结果")
        
        prediction_head = PredictionHead()
        
        prediction_simple = prediction_head.predict(plan_embedding_simple)
        prediction_rich = prediction_head.predict(plan_embedding_rich)
        
        print(f"   - 简单特征预测: {prediction_simple:.4f}")
        print(f"   - 丰富特征预测: {prediction_rich:.4f}")
        
        print("\n" + "="*50)


def demonstrate_end_to_end():
    """Demonstrate the complete end-to-end GNTO pipeline."""
    
    print("\n" + "=" * 80)
    print("完整端到端GNTO流水线演示")
    print("=" * 80)
    
    sample_plans = create_sample_query_plans()
    
    # Traditional GNTO
    print("\n传统GNTO流水线:")
    traditional_gnto = create_traditional_gnto()
    
    for i, plan in enumerate(sample_plans, 1):
        try:
            prediction = traditional_gnto.run(plan)
            print(f"   计划 {i} ({plan['Node Type']}): 预测值 = {prediction:.4f}")
        except Exception as e:
            print(f"   计划 {i} 处理失败: {e}")
    
    # GNN GNTO (if available)
    if is_gnn_available():
        print("\nGNN增强GNTO流水线:")
        try:
            gnn_gnto = create_gnn_gnto()
            
            for i, plan in enumerate(sample_plans, 1):
                try:
                    prediction = gnn_gnto.run(plan)
                    print(f"   计划 {i} ({plan['Node Type']}): GNN预测值 = {prediction:.4f}")
                except Exception as e:
                    print(f"   计划 {i} GNN处理失败: {e}")
        except Exception as e:
            print(f"   GNN GNTO初始化失败: {e}")
    
    # Auto GNTO (automatic selection)
    print(f"\n自动选择GNTO流水线:")
    auto_gnto = create_auto_gnto()
    
    for i, plan in enumerate(sample_plans, 1):
        try:
            prediction = auto_gnto.run(plan)
            mode = "GNN" if is_gnn_available() else "传统"
            print(f"   计划 {i} ({plan['Node Type']}): {mode}预测值 = {prediction:.4f}")
        except Exception as e:
            print(f"   计划 {i} 自动处理失败: {e}")


def demonstrate_batch_processing():
    """Demonstrate batch processing capabilities."""
    
    print("\n" + "=" * 80)
    print("批量处理演示")
    print("=" * 80)
    
    sample_plans = create_sample_query_plans()
    
    # Create multiple instances of each plan for batch processing
    batch_plans = []
    for plan in sample_plans:
        for j in range(3):  # 3 copies of each plan
            batch_plans.append(plan.copy())
    
    print(f"批量处理 {len(batch_plans)} 个查询计划...")
    
    # Process with auto GNTO
    auto_gnto = create_auto_gnto()
    
    try:
        batch_predictions = auto_gnto.run_batch(batch_plans)
        
        print(f"\n批量处理完成:")
        print(f"   - 处理计划数: {len(batch_plans)}")
        print(f"   - 预测结果数: {len(batch_predictions)}")
        print(f"   - 平均预测值: {np.mean(batch_predictions):.4f}")
        print(f"   - 预测值范围: [{min(batch_predictions):.4f}, {max(batch_predictions):.4f}]")
        
        # Show predictions grouped by plan type
        plan_types = [plan['Node Type'] for plan in sample_plans]
        for i, plan_type in enumerate(plan_types):
            start_idx = i * 3
            end_idx = start_idx + 3
            type_predictions = batch_predictions[start_idx:end_idx]
            avg_pred = np.mean(type_predictions)
            print(f"   - {plan_type}: 平均 {avg_pred:.4f} (范围: {min(type_predictions):.4f}-{max(type_predictions):.4f})")
    
    except Exception as e:
        print(f"批量处理失败: {e}")


def main():
    """Main demonstration function."""
    
    print("GNTO (Graph Neural Tree Optimizer) 演示")
    print("基于README.md架构的完整实现")
    
    # Print package information
    print_package_info()
    
    # Step-by-step demonstration
    demonstrate_step_by_step()
    
    # End-to-end demonstration
    demonstrate_end_to_end()
    
    # Batch processing demonstration
    demonstrate_batch_processing()
    
    print("\n" + "=" * 80)
    print("演示完成!")
    print("=" * 80)
    
    print(f"\n使用说明:")
    print(f"   1. 确保安装了requirements.txt中的依赖")
    print(f"   2. GNN功能需要PyTorch和PyTorch Geometric")
    print(f"   3. 没有GNN依赖时会自动回退到传统方法")
    print(f"   4. 可以通过修改示例数据来测试不同的查询计划")


if __name__ == "__main__":
    main()
