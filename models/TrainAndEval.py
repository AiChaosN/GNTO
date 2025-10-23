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

def build_dataset(plans, edges_list, execution_times, in_dim=15, bidirectional=False, max_len=3):
    assert len(plans) == len(edges_list) == len(execution_times), "长度必须一致"
    data_list = []
    for i, (x_plan, edge_index, y) in enumerate(zip(plans, edges_list, execution_times)):

        rows = []
        for node in x_plan:
            # 固定长度：3 个标量 + max_len*4 个谓词槽
            predicate_list = [0.0] * (max_len * 4)
            preds = node["predicate_list_processed"]

            # 截断或填充到 max_len
            for j, (c, op, v, is_id) in enumerate(preds[:max_len]):
                base = j * 4
                predicate_list[base + 0] = float(c)
                predicate_list[base + 1] = float(op)
                # v 既可能是 id 也可能是数值，这里先原样放 float（id 也转 float）
                predicate_list[base + 2] = float(v)
                predicate_list[base + 3] = 1.0 if bool(is_id) else 0.0  # bool → float

            new_node = [
                float(node["node_type_id"]),
                float(node["plan_rows"]),
                float(node["plan_width"]),
            ] + predicate_list

            assert len(new_node) == in_dim, f"节点特征维度不一致：got {len(new_node)}, expect {in_dim}"
            rows.append(new_node)

        x = torch.tensor(rows, dtype=torch.float32)  # [N, in_dim]
        N = x.size(0)

        # edge_index: 期望 [2, E]
        edge_index = torch.as_tensor(edge_index, dtype=torch.long)
        if edge_index.numel() == 0:
            edge_index = edge_index.view(2, 0)
        elif edge_index.dim() == 2 and edge_index.shape[1] == 2 and edge_index.shape[0] != 2:
            edge_index = edge_index.t().contiguous()

        assert edge_index.shape[0] == 2, f"edge_index 需为 [2, E]，现在 {tuple(edge_index.shape)}"

        # 边越界检查
        if edge_index.numel() > 0:
            if int(edge_index.min()) < 0 or int(edge_index.max()) >= N:
                raise ValueError(
                    f"plan[{i}] 的 edge_index 越界：节点数 N={N}，但 edge_index.max={int(edge_index.max())}"
                )

        # 可选：双向
        if bidirectional and edge_index.numel() > 0:
            edge_index = torch.cat([edge_index, edge_index.flip(0)], dim=1)

        y = torch.tensor([float(y)], dtype=torch.float32)  # [1]

        data_list.append(Data(x=x, edge_index=edge_index, y=y))

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

# qerror
def print_qerror(preds_unnorm, labels_unnorm, prints=True):
    qerror = []
    for i in range(len(preds_unnorm)):
        if preds_unnorm[i] > float(labels_unnorm[i]):
            qerror.append(preds_unnorm[i] / float(labels_unnorm[i]))
        else:
            qerror.append(float(labels_unnorm[i]) / float(preds_unnorm[i]))

    e_50, e_90 = np.median(qerror), np.percentile(qerror,90)    
    e_mean = np.mean(qerror)

    if prints:
        print("Median: {}".format(e_50))
        print("Mean: {}".format(e_mean))

    res = {
        'q_median' : e_50,
        'q_90' : e_90,
        'q_mean' : e_mean,
    }

    return res

# 训练
def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, num_batches = 0.0, 0
    for batch in loader:
        batch = batch.to(device)
        optimizer.zero_grad()

        pred = model(batch)                # [B, 1] 或 [B]
        pred = pred.view(-1)               # -> [B]
        target = batch.y.view(-1).to(pred.dtype)  # -> [B]

        loss = criterion(pred, target)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    return total_loss / max(1, num_batches)

def validate_epoch(model, loader, criterion, device, scaler=None):
    model.eval()
    preds_all, labels_all = [], []
    total_loss, num_batches = 0, 0
    eps = 1e-8

    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            pred = model(batch).view(-1)
            label = batch.y.view(-1).to(pred.dtype)

            if scaler is not None:
                # 如果训练时对 y 做了 log / 标准化，这里反归一化
                pred_unnorm = scaler.inverse_transform(pred.cpu().numpy().reshape(-1, 1)).flatten()
                label_unnorm = scaler.inverse_transform(label.cpu().numpy().reshape(-1, 1)).flatten()
            else:
                pred_unnorm = pred.cpu().numpy()
                label_unnorm = label.cpu().numpy()

            preds_all.append(pred_unnorm)
            labels_all.append(label_unnorm)

            loss = criterion(pred, label)
            total_loss += loss.item()
            num_batches += 1

    preds_unnorm = np.concatenate(preds_all, axis=0)
    labels_unnorm = np.concatenate(labels_all, axis=0)

    # ✅ 打印 Q-Error
    qres = print_qerror(preds_unnorm, labels_unnorm, prints=True)
    avg_loss = total_loss / max(1, num_batches)
    print(f"val_loss: {avg_loss:.6f} | MedianQ: {qres['q_median']:.3f} | Q90: {qres['q_90']:.3f} | MeanQ: {qres['q_mean']:.3f}")

    return avg_loss, qres['q_median'], qres['q_90']

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

    qres = print_qerror(preds, targs)

    print("\n" + "="*50)
    print("测试集评估结果:")
    print("="*50)
    print(f"MSE:  {mse:.6f}")
    print(f"Q50: {qres['q_median']:.3f}, Q90: {qres['q_90']:.3f}, QMean: {qres['q_mean']:.3f}")
    print("="*50)

    return preds.numpy(), targs.numpy(), {'mse': mse, 'q_median': qres['q_median'], 'q_90': qres['q_90'], 'q_mean': qres['q_mean']}

