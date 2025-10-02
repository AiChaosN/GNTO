import torch
import numpy as np
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

#  数据集构建

def coerce_edge_index(ei_like):
    """
    将 Edge的Matrix变成[2,E]的long Tensor
    """
    ei = torch.as_tensor(ei_like, dtype=torch.long)
    if ei.ndim != 2: # 维度判断
        raise ValueError(f"edge_index 需要二维，拿到 {tuple(ei.shape)}")
    if ei.shape[0] != 2 and ei.shape[1] == 2: # 形状判断
        ei = ei.t().contiguous()
    elif ei.shape[0] != 2 and ei.shape[1] != 2:
        raise ValueError(f"edge_index 需为 [2,E] 或 [E,2]，拿到 {tuple(ei.shape)}")
    return ei.contiguous()

def build_dataset(res, edges_list, execution_times, in_dim=16, bidirectional=False):
    assert len(res) == len(edges_list) == len(execution_times), "长度必须一致"
    data_list = []
    for i, (x_plan, ei_like, y) in enumerate(zip(res, edges_list, execution_times)):

        # x = x_plan
        # x = torch.tensor(x_plan, dtype=torch.float32) # [N, in_dim]
        # x_shape = x.shape
        # assert x_shape[1] == in_dim, f"维度不一致{x_shape[1]}"
        # edge_index = coerce_edge_index(ei_like)     # [2,E]

        # N = x.size(0)

        # # 边索引有效性检查
        # if edge_index.numel() > 0:
        #     if int(edge_index.min()) < 0 or int(edge_index.max()) >= N:
        #         raise ValueError(f"plan[{i}] 的 edge_index 越界：节点数 N={N}，但 edge_index.max={int(edge_index.max())}")
        # # 可选：做成双向图（若你的 edges 只有父->子）
        # if bidirectional:
        #     edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        # y = torch.tensor([float(y)], dtype=torch.float32)  # 图级回归标签
        # data_list.append(Data(x=x, edge_index=edge_index, y=y))
        data_list.append(Data(x=x_plan, edge_index=ei_like, y=y))
    return data_list


# 早停机制
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


# 训练&评估函数
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
    q_errors_all = []  # 收集整份验证集的 q-error
    eps = 1e-8
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch)
            loss = criterion(pred, batch.y)
            total_loss += loss.item()
            num_batches += 1

            
            # 防 0 防负；Q-Error 定义是正数比例
            p = torch.clamp(pred, min=eps)
            t = torch.clamp(batch.y,    min=eps)
            q_error = torch.maximum(p / t, t / p)              # [B]
            q_errors_all.append(q_error.cpu().numpy())

    if q_errors_all:
        q_all = np.concatenate(q_errors_all, axis=0)
        Q50 = float(np.quantile(q_all, 0.5))
        Q95 = float(np.quantile(q_all, 0.95))
    else:
        Q50 = float("nan")
        Q95 = float("nan")

    avg_loss = total_loss / max(1, num_batches)
    print(f"val_loss: {avg_loss:.6f} | Q50: {Q50:.6f} | Q95: {Q95:.6f}")

    return total_loss / num_batches, Q50, Q95

def evaluate_model(model, test_loader, device):
    model.eval()
    preds_all, targs_all = [], []

    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            pred = model(batch).view(-1).float()   # [B]，先拍平
            y    = batch.y.view(-1).float()        # [B]
            preds_all.append(pred.cpu())
            targs_all.append(y.cpu())

    preds = torch.cat(preds_all)   # [N]
    targs = torch.cat(targs_all)   # [N]

    # MSE（Torch实现）
    mse = torch.mean((preds - targs) ** 2).item()

    # Q-Error（Torch实现）
    eps = 1e-8
    p = torch.clamp(preds, min=eps)
    t = torch.clamp(targs, min=eps)
    q = torch.maximum(p / t, t / p)             # [N]
    Q50 = torch.quantile(q, 0.5).item()
    Q95 = torch.quantile(q, 0.95).item()

    # 如果你需要返回 numpy
    predictions = preds.numpy()
    targets = targs.numpy()

    print("\n" + "="*50)
    print("测试集评估结果:")
    print("="*50)
    print(f"MSE:  {mse:.6f}")
    print(f"Q50: {Q50:.6f}, Q95: {Q95:.6f}")
    print("="*50)

    return predictions, targets, {'mse': mse, 'Q50': Q50, 'Q95': Q95}

