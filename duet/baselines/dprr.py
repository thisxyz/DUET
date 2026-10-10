from __future__ import annotations

import math

from typing import Callable, Dict, List, Optional, Tuple

import torch

import torch.nn.functional as F

from duet.mechanisms.links import randomized_response, rr_probs

from duet.models import DenseGCN

from duet.train import train_dense_gcn

def _unbiased_degree(a_rr: torch.Tensor, eps: float) -> torch.Tensor:

    n = a_rr.size(0)

    p, q = rr_probs(eps)

    d_rr = a_rr.sum(dim=1, keepdim=True)

    return ((d_rr - (n - 1) * q) / max(p - q, 1e-06)).clamp(min=0.0)

def _feature_sim(x: torch.Tensor) -> torch.Tensor:

    xn = F.normalize(x, p=2, dim=1)

    return (xn @ xn.t() + 1.0) * 0.5

def dprr_graphs(adj: torch.Tensor, x_tilde: torch.Tensor, eps_a: float) -> List[Tuple[str, torch.Tensor]]:

    n = adj.size(0)

    device = adj.device

    eps_a = float(max(eps_a, 1e-06))

    a_rr = randomized_response(adj, eps_a)

    k = _unbiased_degree(a_rr, eps_a)

    p, q_fp = rr_probs(eps_a)

    denom = k * (2.0 * p - 1.0) + (n - 1) * (1.0 - p)

    q_keep = (k / denom.clamp(min=1e-12)).clamp(0.0, 1.0)

    sim = _feature_sim(x_tilde)

    cands: List[Tuple[str, torch.Tensor]] = [('rr', a_rr)]

    qq = (q_keep @ q_keep.t()).sqrt()

    a_soft = a_rr * qq

    a_soft = 0.5 * (a_soft + a_soft.t())

    a_soft.fill_diagonal_(0.0)

    cands.append(('soft', a_soft))

    a_top = torch.zeros_like(a_rr)

    rr_ones = a_rr > 0.5

    scores = sim.masked_fill(~rr_ones, -1000000000.0)

    scores.fill_diagonal_(-1000000000.0)

    k_int = k.view(-1).round().clamp(min=1, max=n - 1).long()

    k_max = int(k_int.max().item())

    if k_max > 0:

        vals, idx = torch.topk(scores, k=k_max, dim=1)

        rows = torch.arange(n, device=device).unsqueeze(1).expand_as(idx)

        keep_mask = torch.arange(k_max, device=device).unsqueeze(0) < k_int.unsqueeze(1)

        keep_mask = keep_mask & (vals > -100000000.0)

        a_top[rows[keep_mask], idx[keep_mask]] = 1.0

    a_top = torch.maximum(a_top, a_top.t())

    a_top.fill_diagonal_(0.0)

    cands.append(('topk_sim', a_top))

    a_blend = torch.clamp(0.6 * a_rr + 0.4 * a_top, 0.0, 1.0)

    a_blend.fill_diagonal_(0.0)

    cands.append(('blend', a_blend))

    return cands

def run_dprr(x_tilde: torch.Tensor, adj_true: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor, num_classes: int, eps_a: float, hidden: int=32, dropout: float=0.5, lr: float=0.01, weight_decay: float=0.0005, epochs: int=400, patience: int=60, model_factory: Optional[Callable]=None, lite: bool=False) -> Dict:

    factory = model_factory or DenseGCN

    cands = dprr_graphs(adj_true, x_tilde, eps_a)

    if lite:

        cands = [c for c in cands if c[0] in ('rr', 'topk_sim', 'blend')]

    scored = []

    for name, a_hat in cands:

        model = factory(x_tilde.size(1), num_classes, hidden, dropout).to(x_tilde.device)

        test_acc, val_acc, _ = train_dense_gcn(model, x_tilde, a_hat, y, train_mask, val_mask, test_mask, lr=lr, weight_decay=weight_decay, epochs=epochs, patience=patience)

        scored.append((name, val_acc, test_acc))

    rr = next((s for s in scored if s[0] == 'rr'))

    best_name, best_val, best_test = rr

    margin = 0.005

    for name, val_acc, test_acc in scored:

        if name == 'rr':

            continue

        if val_acc > best_val + margin:

            best_name, best_val, best_test = (name, val_acc, test_acc)

    return {'test_acc': best_test, 'val_acc': best_val}
