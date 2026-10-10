from __future__ import annotations

from typing import Dict

import torch

from typing import Callable, Optional

from duet.models import DenseGCN

from duet.train import train_dense_gcn

def _normalize_no_self(a: torch.Tensor) -> torch.Tensor:

    a = a.clone()

    a.fill_diagonal_(0.0)

    deg = a.sum(dim=1).clamp(min=1e-08)

    d_inv = deg.pow(-0.5)

    return d_inv.unsqueeze(1) * a * d_inv.unsqueeze(0)

def hoa(x: torch.Tensor, adj: torch.Tensor, k: int) -> torch.Tensor:

    if k <= 0:

        return x

    a_hat = _normalize_no_self(adj)

    cur = x

    total = torch.zeros_like(x)

    for _ in range(k):

        cur = a_hat @ cur

        total = total + cur

    return total / float(k)

def nfr(x: torch.Tensor, mu: float) -> torch.Tensor:

    return torch.sign(x) * torch.clamp(x.abs() - mu, min=0.0)

def upgnet_recover(x_tilde: torch.Tensor, adj: torch.Tensor, k: int=8, tau: float=0.1, architecture: str='nh') -> torch.Tensor:

    flat = x_tilde.detach().abs().reshape(-1)

    if flat.numel() > 2000000:

        idx = torch.randint(0, flat.numel(), (2000000,), device=flat.device)

        flat = flat[idx]

    b = float(flat.quantile(0.99).item())

    b = max(b, 0.001)

    d_bar = float(adj.sum().item() / max(adj.size(0), 1))

    d_bar = max(d_bar, 1.0)

    if architecture.lower() == 'hn':

        h = hoa(x_tilde, adj, k=k)

        mu = tau * b / (d_bar * max(k, 1))

        return nfr(h, mu)

    mu = tau * b

    x_e = nfr(x_tilde, mu)

    return hoa(x_e, adj, k=k)

def run_upgnet(x_tilde: torch.Tensor, a_rr: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor, num_classes: int, k: int=8, tau: float=0.1, architecture: str='auto', hidden: int=32, dropout: float=0.5, lr: float=0.01, weight_decay: float=0.0005, epochs: int=400, patience: int=60, model_factory: Optional[Callable]=None) -> Dict:

    factory = model_factory or DenseGCN

    arches = ['nh', 'hn'] if architecture == 'auto' else [architecture]

    best = {'test_acc': 0.0, 'val_acc': -1.0}

    for arch in arches:

        x_rec = upgnet_recover(x_tilde, a_rr, k=k, tau=tau, architecture=arch)

        model = factory(x_tilde.size(1), num_classes, hidden, dropout).to(x_tilde.device)

        test_acc, val_acc, _ = train_dense_gcn(model, x_rec, a_rr, y, train_mask, val_mask, test_mask, lr=lr, weight_decay=weight_decay, epochs=epochs, patience=patience)

        if val_acc > best['val_acc']:

            best = {'test_acc': test_acc, 'val_acc': val_acc}

    return best
