import torch
import numpy as np

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
    
    Q50_list = []
    Q95_list = []
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

    return total_loss / num_batches

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

