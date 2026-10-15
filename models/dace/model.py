"""DACE model (Liang et al., "DACE: A Database-Agnostic Cost Estimator", ICDE 2024),
verbatim from ``DACE/model.py`` minus the Lightning wrappers.

One single-head Transformer-encoder layer over the plan's node sequence
(``d_model = node_length`` = 20 one-hot node types + 2 scaled numerics), a tree
reachability attention mask, and a LoRA-linear MLP head that predicts a
normalised runtime for *every* node (sub-plan supervision); the root's
prediction is the plan estimate.  ``loralib.Linear`` freezes the base weight
and trains only the low-rank factors + bias, so the model has 23.4K parameters
of which 12.3K are trainable (as printed by the upstream script).
"""
from __future__ import annotations

import loralib as lora
import torch
import torch.nn as nn


class DACELora(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, mlp_activation="ReLU",
                 transformer_activation="gelu", mlp_dropout=0.3, transformer_dropout=0.2):
        super().__init__()
        self.tranformer_encoder = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=input_dim, dim_feedforward=hidden_dim, nhead=1,
                                       batch_first=True, activation=transformer_activation,
                                       dropout=transformer_dropout),
            num_layers=1,
            enable_nested_tensor=False,   # upstream default triggers a warning (nhead=1); no behaviour change
        )
        self.node_length = input_dim
        if mlp_activation == "ReLU":
            self.mlp_activation = nn.ReLU()
        elif mlp_activation == "GELU":
            self.mlp_activation = nn.GELU()
        elif mlp_activation == "LeakyReLU":
            self.mlp_activation = nn.LeakyReLU()
        self.mlp_hidden_dims = [128, 64, 1]
        self.mlp = nn.Sequential(
            lora.Linear(self.node_length, self.mlp_hidden_dims[0], r=16),
            nn.Dropout(mlp_dropout),
            self.mlp_activation,
            lora.Linear(self.mlp_hidden_dims[0], self.mlp_hidden_dims[1], r=8),
            nn.Dropout(mlp_dropout),
            self.mlp_activation,
            lora.Linear(self.mlp_hidden_dims[1], output_dim, r=4),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x, attn_mask=None):
        """``x`` [B, pad_length * node_length] -> [B, pad_length] normalised runtimes."""
        x = x.view(x.shape[0], -1, self.node_length)
        out = self.tranformer_encoder(x, mask=attn_mask)
        out = self.mlp(out)
        return self.sigmoid(out).squeeze(dim=2)


def q_error(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return torch.max(pred / target, target / pred)


def dace_loss(est_run_times: torch.Tensor, run_times: torch.Tensor, loss_mask: torch.Tensor) -> torch.Tensor:
    """Upstream ``PL_DACE.DACE_loss``: per plan, sum over nodes of
    ``0.5^height * log(max(q-error, 1))`` (padding nodes have mask 0); returns [B]."""
    loss = torch.max(est_run_times / run_times, run_times / est_run_times)
    loss = loss * loss_mask
    loss = torch.log(torch.where(loss > 1, loss, torch.ones_like(loss)))
    return torch.sum(loss, dim=1)


DACE_DEFAULT_HPARAMS = dict(hidden_dim=128, mlp_activation="ReLU", transformer_activation="gelu",
                            mlp_dropout=0.3, transformer_dropout=0.2)

__all__ = ["DACELora", "q_error", "dace_loss", "DACE_DEFAULT_HPARAMS"]
