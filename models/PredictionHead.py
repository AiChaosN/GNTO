# models/PredictionHead.py
import torch
import torch.nn as nn

# FNN 模型 64 -> 128 -> 64 -> 1
class PredictionHead_FNNMini(nn.Module):
    def __init__(self, in_dim=64, hidden_dims=(128, 64), out_dim=1, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, plan_emb: torch.Tensor) -> torch.Tensor:
        return self.mlp(plan_emb)

# FNN 模型 64 -> 128 -> 64 -> 1
class PredictionHead(nn.Module):
    def __init__(self, in_dim=64, hidden_dims=(128, 64), out_dim=1, dropout=0.1):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden_dims:
            layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, out_dim)]
        self.mlp = nn.Sequential(*layers)

    def forward(self, plan_emb: torch.Tensor) -> torch.Tensor:
        if plan_emb.dim() == 1:
            plan_emb = plan_emb.unsqueeze(0)
        out = self.mlp(plan_emb)       # [B, 1]
        return out.squeeze(-1)         # [B] 或标量

