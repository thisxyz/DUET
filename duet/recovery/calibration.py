from __future__ import annotations

import torch

import torch.nn as nn

class EdgeCalibrator(nn.Module):

    def __init__(self, feat_dim: int, topk: int=32):

        super().__init__()

        self.topk = topk

        self.scorer = nn.Linear(4 * feat_dim, 1)

    def forward(self, x_star: torch.Tensor, a_bar: torch.Tensor) -> torch.Tensor:

        n, d = x_star.size()

        device = x_star.device

        k = min(self.topk, n - 1)

        vals, idx = torch.topk(a_bar, k=k, dim=1)

        rows = torch.arange(n, device=device).unsqueeze(1).expand(-1, k).reshape(-1)

        cols = idx.reshape(-1)

        xi = x_star[rows]

        xj = x_star[cols]

        feats = torch.cat([xi, xj, (xi - xj).abs(), xi * xj], dim=-1)

        s = torch.sigmoid(self.scorer(feats)).squeeze(-1)

        a_star = torch.zeros(n, n, device=device, dtype=x_star.dtype)

        a_star[rows, cols] = a_bar[rows, cols] * s

        a_star = 0.5 * (a_star + a_star.t())

        a_star.fill_diagonal_(0.0)

        return a_star
