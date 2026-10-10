from __future__ import annotations

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch_geometric.nn import DenseGATConv, DenseGCNConv, DenseSAGEConv, GCNConv

class DenseGCN(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int=16, dropout: float=0.5):

        super().__init__()

        self.conv1 = DenseGCNConv(in_dim, hidden_dim)

        self.conv2 = DenseGCNConv(hidden_dim, out_dim)

        self.dropout = dropout

        self.act = nn.SELU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:

        squeeze = False

        if x.dim() == 2:

            x = x.unsqueeze(0)

            adj = adj.unsqueeze(0)

            squeeze = True

        x = self.conv1(x, adj)

        x = self.act(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, adj)

        if squeeze:

            x = x.squeeze(0)

        return x

class DenseGAT(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int=16, dropout: float=0.5, heads: int=4):

        super().__init__()

        self.conv1 = DenseGATConv(in_dim, hidden_dim, heads=heads, concat=False)

        self.conv2 = DenseGATConv(hidden_dim, out_dim, heads=1, concat=False)

        self.dropout = dropout

        self.act = nn.SELU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:

        squeeze = False

        if x.dim() == 2:

            x = x.unsqueeze(0)

            adj = adj.unsqueeze(0)

            squeeze = True

        x = self.conv1(x, adj)

        x = self.act(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, adj)

        if squeeze:

            x = x.squeeze(0)

        return x

class DenseSAGE(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int=16, dropout: float=0.5):

        super().__init__()

        self.conv1 = DenseSAGEConv(in_dim, hidden_dim)

        self.conv2 = DenseSAGEConv(hidden_dim, out_dim)

        self.dropout = dropout

        self.act = nn.SELU()

    def forward(self, x: torch.Tensor, adj: torch.Tensor) -> torch.Tensor:

        squeeze = False

        if x.dim() == 2:

            x = x.unsqueeze(0)

            adj = adj.unsqueeze(0)

            squeeze = True

        x = self.conv1(x, adj)

        x = self.act(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, adj)

        if squeeze:

            x = x.squeeze(0)

        return x

class SparseGCN(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden_dim: int=16, dropout: float=0.5):

        super().__init__()

        self.conv1 = GCNConv(in_dim, hidden_dim)

        self.conv2 = GCNConv(hidden_dim, out_dim)

        self.dropout = dropout

        self.act = nn.SELU()

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_weight=None) -> torch.Tensor:

        x = self.conv1(x, edge_index, edge_weight)

        x = self.act(x)

        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.conv2(x, edge_index, edge_weight)

        return x

_MODEL_MAP = {'gcn': DenseGCN, 'gat': DenseGAT, 'sage': DenseSAGE, 'graphsage': DenseSAGE}

def make_dense_model(model_name: str, in_dim: int, out_dim: int, hidden_dim: int=32, dropout: float=0.5) -> nn.Module:

    key = model_name.lower()

    if key not in _MODEL_MAP:

        raise ValueError(f'Unknown model {model_name}; choose from {list(_MODEL_MAP)}')

    return _MODEL_MAP[key](in_dim, out_dim, hidden_dim, dropout)

def sparsify_dense_adj(adj: torch.Tensor, topk: int=32) -> torch.Tensor:

    n = adj.size(0)

    if topk <= 0 or topk >= n - 1:

        return adj

    a = adj.clone()

    a.fill_diagonal_(-1000000000.0)

    vals, idx = torch.topk(a, k=min(topk, n - 1), dim=1)

    out = torch.zeros_like(adj)

    rows = torch.arange(n, device=adj.device).unsqueeze(1).expand_as(idx)

    out[rows, idx] = vals.clamp(min=0)

    out = torch.maximum(out, out.t())

    out.fill_diagonal_(0.0)

    return out
