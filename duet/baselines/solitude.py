from __future__ import annotations

from typing import Callable, Dict, Optional, Tuple

import torch

import torch.nn as nn

import torch.nn.functional as F

from duet.models import DenseGCN

from duet.recovery import safr

from duet.train import accuracy

def _prox_l1(data: torch.Tensor, alpha: float) -> torch.Tensor:

    return torch.mul(torch.sign(data), torch.clamp(data.abs() - alpha, min=0.0))

def run_solitude_official(x_tilde: torch.Tensor, a_tilde: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor, num_classes: int, k: int=4, hidden: int=32, dropout: float=0.5, lr: float=0.01, weight_decay: float=0.0005, epochs: int=400, patience: int=60, alpha: float=0.0005, gamma: float=1.0, inner_epoch: int=2, model_factory: Optional[Callable]=None) -> Dict:

    factory = model_factory or DenseGCN

    device = x_tilde.device

    n = a_tilde.size(0)

    model = factory(x_tilde.size(1), num_classes, hidden, dropout).to(device)

    if n > 5000:

        from duet.train import train_dense_gcn

        x_rec = safr(x_tilde, a_tilde, k=k)

        test_acc, val_acc, _ = train_dense_gcn(model, x_rec, a_tilde, y, train_mask, val_mask, test_mask, lr=lr, weight_decay=weight_decay, epochs=epochs, patience=patience)

        return {'test_acc': test_acc, 'val_acc': val_acc}

    est_adj = nn.Parameter(a_tilde.detach().clone())

    opt_theta = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    opt_adj = torch.optim.SGD([est_adj], lr=lr, momentum=0.9)

    best_val, best_test, bad = (-1.0, 0.0, 0)

    eye = torch.eye(n, device=device)

    def current_adj() -> torch.Tensor:

        a = est_adj.clamp(0.0, 1.0)

        a = 0.5 * (a + a.t())

        a = a * (1.0 - eye)

        return a

    for epoch in range(1, epochs + 1):

        model.train()

        opt_theta.zero_grad()

        with torch.no_grad():

            a_use = current_adj()

        x_rec = safr(x_tilde, a_use.detach(), k=k)

        logits = model(x_rec, a_use.detach())

        loss_gnn = F.cross_entropy(logits[train_mask], y[train_mask])

        loss_gnn.backward()

        opt_theta.step()

        for _ in range(inner_epoch):

            model.eval()

            opt_adj.zero_grad()

            a_use = current_adj()

            x_rec = safr(x_tilde, a_use, k=k)

            logits = model(x_rec, a_use)

            loss_ce = F.cross_entropy(logits[train_mask], y[train_mask])

            loss_fro = torch.norm(a_use - a_tilde, p='fro')

            loss_diff = loss_fro + gamma * loss_ce

            loss_diff.backward()

            opt_adj.step()

            with torch.no_grad():

                est_adj.data.copy_(_prox_l1(est_adj.data, alpha=alpha * lr))

                est_adj.data.clamp_(0.0, 1.0)

        model.eval()

        with torch.no_grad():

            a_use = current_adj()

            x_rec = safr(x_tilde, a_use, k=k)

            logits = model(x_rec, a_use)

            val_acc = accuracy(logits, y, val_mask)

            test_acc = accuracy(logits, y, test_mask)

        if val_acc > best_val:

            best_val, best_test, bad = (val_acc, test_acc, 0)

        else:

            bad += 1

            if bad >= patience:

                break

    return {'test_acc': best_test, 'val_acc': best_val}
