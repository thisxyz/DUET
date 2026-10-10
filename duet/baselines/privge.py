from __future__ import annotations

from typing import Callable, Dict, Optional

import torch

from duet.models import DenseGCN

from duet.train import train_dense_gcn

def _ppr_transition(adj: torch.Tensor, r: float=0.5) -> torch.Tensor:

    a = adj.clone()

    a.fill_diagonal_(0.0)

    deg = a.sum(dim=1).clamp(min=1e-08)

    left = deg.pow(r - 1.0)

    right = deg.pow(-r)

    return left.unsqueeze(1) * a * right.unsqueeze(0)

def personalized_pagerank(x: torch.Tensor, adj: torch.Tensor, alpha: float=0.4, r: float=0.5, steps: int=5) -> torch.Tensor:

    t = _ppr_transition(adj, r=r)

    h = x

    for _ in range(steps):

        h = alpha * x + (1.0 - alpha) * (t @ h)

    return h

def run_privge(x_tilde: torch.Tensor, a_rr: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor, num_classes: int, a_prop: Optional[torch.Tensor]=None, ppr_alpha: float=0.4, ppr_r: float=0.5, ppr_steps: int=5, hidden: int=32, dropout: float=0.5, lr: float=0.01, weight_decay: float=0.0005, epochs: int=400, patience: int=60, model_factory: Optional[Callable]=None) -> Dict:

    factory = model_factory or DenseGCN

    adj = a_rr if a_prop is None else a_prop

    with torch.no_grad():

        z = personalized_pagerank(x_tilde, adj, alpha=ppr_alpha, r=ppr_r, steps=ppr_steps)

    model = factory(z.size(1), num_classes, hidden, dropout).to(z.device)

    test_acc, val_acc, _ = train_dense_gcn(model, z, adj, y, train_mask, val_mask, test_mask, lr=lr, weight_decay=weight_decay, epochs=epochs, patience=patience)

    return {'test_acc': test_acc, 'val_acc': val_acc}
