from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

import torch.nn as nn

import torch.nn.functional as F

from torch_geometric.nn import DenseGCNConv

from duet.train import accuracy, train_dense_gcn

class LinkGuardGCN(nn.Module):

    def __init__(self, in_dim: int, out_dim: int, hidden: int=32, dropout: float=0.5):

        super().__init__()

        self.conv1 = DenseGCNConv(in_dim, hidden)

        self.conv2 = DenseGCNConv(hidden, out_dim)

        self.dropout = dropout

        self.act = nn.SELU()

        self.alpha = nn.Parameter(torch.tensor(0.7))

    def _fuse(self, a_est: torch.Tensor, h: torch.Tensor) -> torch.Tensor:

        hn = F.normalize(h, dim=1)

        sim = (hn @ hn.t()).clamp(0.0, 1.0)

        deg = a_est.sum(dim=1, keepdim=True).clamp(min=1.0)

        alpha = torch.sigmoid(self.alpha)

        sim_term = (sim / deg).clamp(0.0, 1.0)

        a = alpha * a_est + (1.0 - alpha) * a_est * sim_term

        a = 0.5 * (a + a.t())

        a = a - torch.diag(torch.diag(a))

        return a.clamp(0.0, 1.0)

    def forward(self, x: torch.Tensor, a_est: torch.Tensor) -> torch.Tensor:

        squeeze = False

        if x.dim() == 2:

            x = x.unsqueeze(0)

            a_in = a_est.unsqueeze(0)

            squeeze = True

        else:

            a_in = a_est

        h = self.conv1(x, a_in)

        h = self.act(h)

        h = F.dropout(h, p=self.dropout, training=self.training)

        h2 = h.squeeze(0) if squeeze else h

        a_fuse = self._fuse(a_est if a_est.dim() == 2 else a_est.squeeze(0), h2)

        a_fuse_b = a_fuse.unsqueeze(0) if squeeze else a_fuse

        out = self.conv2(h, a_fuse_b)

        if squeeze:

            out = out.squeeze(0)

        return out

def run_linkguard(x: torch.Tensor, a_soft: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor, num_classes: int, hidden: int=32, dropout: float=0.5, lr: float=0.01, weight_decay: float=0.0005, epochs: int=400, patience: int=60, model_factory: Optional[Callable]=None) -> Dict:

    device = x.device

    if model_factory is None:

        model = LinkGuardGCN(x.size(1), num_classes, hidden, dropout).to(device)

        opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

        best_val, best_test, bad = (-1.0, 0.0, 0)

        for _ in range(epochs):

            model.train()

            opt.zero_grad()

            logits = model(x, a_soft)

            F.cross_entropy(logits[train_mask], y[train_mask]).backward()

            opt.step()

            model.eval()

            with torch.no_grad():

                logits = model(x, a_soft)

                val_acc = accuracy(logits, y, val_mask)

                test_acc = accuracy(logits, y, test_mask)

            if val_acc > best_val:

                best_val, best_test, bad = (val_acc, test_acc, 0)

            else:

                bad += 1

                if bad >= patience:

                    break

        return {'test_acc': best_test, 'val_acc': best_val}

    with torch.no_grad():

        hn = F.normalize(x, dim=1)

        sim = (hn @ hn.t()).clamp(0.0, 1.0)

        deg = a_soft.sum(dim=1, keepdim=True).clamp(min=1.0)

        a_fuse = (0.7 * a_soft + 0.3 * a_soft * (sim / deg)).clamp(0.0, 1.0)

        a_fuse = 0.5 * (a_fuse + a_fuse.t())

        a_fuse.fill_diagonal_(0.0)

    model = model_factory(x.size(1), num_classes, hidden, dropout).to(device)

    test_acc, val_acc, _ = train_dense_gcn(model, x, a_fuse, y, train_mask, val_mask, test_mask, lr=lr, weight_decay=weight_decay, epochs=epochs, patience=patience)

    return {'test_acc': test_acc, 'val_acc': val_acc}
