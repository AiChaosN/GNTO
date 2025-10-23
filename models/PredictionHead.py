# models/PredictionHead.py
import torch
import torch.nn as nn

# FNN 模型 64 -> 128 -> 64 -> 1
class PredictionHead_FNNMini(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dims=(64, 64), dropout=0.1):
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
