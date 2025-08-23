"""
三个核心模块演示：NodeEncoder -> StructureEncoder -> Heads
展示如何将执行计划从原始特征转换为最终的成本预测
"""

import torch
import torch.nn as nn
from typing import Dict, List

# 导入我们的三个核心模块
from gnto.core.encoders import NodeEncoder, StructureEncoder
from gnto.core.heads import RegressionHead


class SimpleLQODemo(nn.Module):
    """
    简化的LQO演示模型
    清晰展示三个核心模块的数据流向
    """
    
    def __init__(self):
        super().__init__()
        
        # 第一个模块：节点编码器
        # 功能：将每个节点的原始特征（如行数、算子类型）编码为向量
        self.node_encoder = NodeEncoder(
            input_dim=50,      # 假设输入特征有50维
            hidden_dims=[128, 64],  # 隐藏层维度
            output_dim=32      # 输出32维的节点向量
        )
        
        # 第二个模块：结构编码器（GNN）
        # 功能：将所有节点向量和它们之间的连接关系编码为一个计划向量
        self.structure_encoder = StructureEncoder(
            node_dim=32,       # 接收来自NodeEncoder的32维向量
            hidden_dim=64,     # GNN隐藏层维度
            num_layers=2,      # GNN层数
            gnn_type="gcn",    # 使用图卷积网络
            pooling="mean"     # 使用平均池化生成计划向量
        )
        
        # 第三个模块：预测头
        # 功能：将计划向量转换为具体的成本预测
        self.cost_predictor = RegressionHead(
            input_dim=64,      # 接收来自StructureEncoder的64维向量
            hidden_dims=[32, 16],  # 预测头隐藏层
            output_dim=1,      # 输出1个数值：执行成本
            output_activation="softplus"  # 确保成本为正数
        )
    
    def forward(self, plan_data: Dict) -> torch.Tensor:
        """
        完整的前向传播过程
        
        参数:
            plan_data: 包含以下内容的字典
                - node_features: [N, 50] 每个节点的特征
                - edge_index: [2, E] 节点之间的连接关系
                - edge_types: [E] 边的类型
                - batch_idx: [N] 节点所属的批次（用于批处理）
        
        返回:
            [B, 1] 每个计划的预测成本
        """
        
        print("=== 数据流向演示 ===")
        
        # 第一步：节点编码
        print(f"1. 输入节点特征: {plan_data['node_features'].shape}")
        node_embeddings = self.node_encoder(plan_data['node_features'])
        print(f"   -> 节点向量: {node_embeddings.shape}")
        
        # 第二步：结构编码（GNN）
        print(f"2. GNN处理结构信息...")
        structure_output = self.structure_encoder(
            nodes=node_embeddings,
            edges=plan_data['edge_index'],
            edge_types=plan_data['edge_types'],
            batch_idx=plan_data['batch_idx']
        )
        plan_embedding = structure_output['plan_emb']
        print(f"   -> 计划向量: {plan_embedding.shape}")
        
        # 第三步：成本预测
        print(f"3. 成本预测...")
        cost_prediction = self.cost_predictor(plan_embedding)
        print(f"   -> 预测成本: {cost_prediction.shape}")
        
        return cost_prediction


def create_sample_data():
    """创建示例数据"""
    # 模拟一个有3个节点的执行计划
    node_features = torch.randn(3, 50)  # 3个节点，每个节点50维特征
    
    # 边连接：0->1, 1->2 (形成一个链式结构)
    edge_index = torch.tensor([[0, 1], [1, 2]]).t()  # [2, 2]
    edge_types = torch.tensor([0, 0])  # 边类型
    
    # 批次索引（所有节点都属于第一个计划）
    batch_idx = torch.tensor([0, 0, 0])
    
    return {
        'node_features': node_features,
        'edge_index': edge_index, 
        'edge_types': edge_types,
        'batch_idx': batch_idx
    }


def main():
    """主演示函数"""
    print("LQO三个核心模块演示")
    print("=" * 50)
    
    # 创建模型
    model = SimpleLQODemo()
    model.eval()  # 设置为评估模式
    
    # 创建示例数据
    sample_data = create_sample_data()
    
    print("输入数据:")
    print(f"- 节点数量: {sample_data['node_features'].shape[0]}")
    print(f"- 节点特征维度: {sample_data['node_features'].shape[1]}")
    print(f"- 边数量: {sample_data['edge_index'].shape[1]}")
    print()
    
    # 运行预测
    with torch.no_grad():
        predicted_cost = model(sample_data)
    
    print()
    print("=" * 50)
    print(f"最终预测成本: {predicted_cost.item():.4f}")
    print()
    
    print("三个模块的作用总结:")
    print("1. NodeEncoder: 节点特征 -> 节点向量")
    print("2. StructureEncoder: 节点向量 + 图结构 -> 计划向量") 
    print("3. RegressionHead: 计划向量 -> 成本预测")


if __name__ == "__main__":
    main()
