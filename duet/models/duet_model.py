from __future__ import annotations

import torch

import torch.nn as nn

import torch.nn.functional as F

from .gcn import DenseGCN

class DUETModel(nn.Module):

    def __init__(self, in_dim: int, nclass: int, n_views: int=5, hidden: int=64, dropout: float=0.35, mix_scale: float=0.5):

        super().__init__()

        self.attn = nn.Parameter(torch.zeros(n_views))

        self.gcn = DenseGCN(in_dim, nclass, hidden, dropout)

        self.alpha = nn.Parameter(torch.tensor(1.0))

        self.mix_scale = mix_scale

    def fuse(self, views, a_base: torch.Tensor, soft: torch.Tensor):

        w = torch.softmax(self.attn, dim=0)

        x = sum((w[i] * views[i] for i in range(len(views))))

        xn = F.normalize(x, dim=1)

        sim = (xn @ xn.t()).clamp(0, 1)

        mix = torch.sigmoid(self.alpha) * self.mix_scale

        a = a_base * (1.0 - mix + mix * sim) + soft * (mix * 0.6 * sim)

        a = 0.5 * (a + a.t())

        a = a - torch.diag(torch.diag(a))

        return (x, a)

    def forward(self, views, a_base: torch.Tensor, soft: torch.Tensor) -> torch.Tensor:

        x, a = self.fuse(views, a_base, soft)

        return self.gcn(x, a)
