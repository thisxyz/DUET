from __future__ import annotations

from typing import Dict, Optional, Tuple

import torch

import torch.nn as nn

import torch.nn.functional as F

@torch.no_grad()

def accuracy(logits: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> float:

    pred = logits.argmax(dim=1)

    correct = (pred[mask] == y[mask]).sum().item()

    total = int(mask.sum().item())

    return correct / max(total, 1)

def train_dense_gcn(model: nn.Module, x: torch.Tensor, adj: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor, lr: float=0.01, weight_decay: float=0.0005, epochs: int=400, patience: int=50, extra_params=None) -> Tuple[float, float, Dict]:

    params = list(model.parameters())

    if extra_params is not None:

        params += list(extra_params)

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    best_val = -1.0

    best_test = 0.0

    bad = 0

    best_state = None

    for _ in range(epochs):

        model.train()

        if hasattr(model, 'train'):

            pass

        optimizer.zero_grad()

        logits = model(x, adj)

        loss = F.cross_entropy(logits[train_mask], y[train_mask])

        loss.backward()

        optimizer.step()

        model.eval()

        with torch.no_grad():

            logits = model(x, adj)

            val_acc = accuracy(logits, y, val_mask)

            test_acc = accuracy(logits, y, test_mask)

        if val_acc > best_val:

            best_val = val_acc

            best_test = test_acc

            bad = 0

            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        else:

            bad += 1

            if bad >= patience:

                break

    if best_state is not None:

        model.load_state_dict(best_state)

    return (best_test, best_val, {'best_val': best_val, 'best_test': best_test})

def train_joint_duet(gnn: nn.Module, calibrator: nn.Module, x_star: torch.Tensor, a_bar: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor, lr: float=0.01, weight_decay: float=0.0005, epochs: int=400, patience: int=50) -> Tuple[float, float, Dict]:

    params = list(gnn.parameters()) + list(calibrator.parameters())

    optimizer = torch.optim.Adam(params, lr=lr, weight_decay=weight_decay)

    best_val = -1.0

    best_test = 0.0

    bad = 0

    best_gnn = None

    best_cal = None

    for _ in range(epochs):

        gnn.train()

        calibrator.train()

        optimizer.zero_grad()

        a_star = calibrator(x_star, a_bar)

        logits = gnn(x_star, a_star)

        loss = F.cross_entropy(logits[train_mask], y[train_mask])

        loss.backward()

        optimizer.step()

        gnn.eval()

        calibrator.eval()

        with torch.no_grad():

            a_star = calibrator(x_star, a_bar)

            logits = gnn(x_star, a_star)

            val_acc = accuracy(logits, y, val_mask)

            test_acc = accuracy(logits, y, test_mask)

        if val_acc > best_val:

            best_val = val_acc

            best_test = test_acc

            bad = 0

            best_gnn = {k: v.detach().cpu().clone() for k, v in gnn.state_dict().items()}

            best_cal = {k: v.detach().cpu().clone() for k, v in calibrator.state_dict().items()}

        else:

            bad += 1

            if bad >= patience:

                break

    if best_gnn is not None:

        gnn.load_state_dict(best_gnn)

        calibrator.load_state_dict(best_cal)

    return (best_test, best_val, {'best_val': best_val, 'best_test': best_test})

def train_sparse_gcn(model: nn.Module, x: torch.Tensor, edge_index: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, val_mask: torch.Tensor, test_mask: torch.Tensor, edge_weight: Optional[torch.Tensor]=None, lr: float=0.01, weight_decay: float=0.0005, epochs: int=400, patience: int=50) -> Tuple[float, float, Dict]:

    optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)

    best_val = -1.0

    best_test = 0.0

    bad = 0

    best_state = None

    for _ in range(epochs):

        model.train()

        optimizer.zero_grad()

        logits = model(x, edge_index, edge_weight)

        loss = F.cross_entropy(logits[train_mask], y[train_mask])

        loss.backward()

        optimizer.step()

        model.eval()

        with torch.no_grad():

            logits = model(x, edge_index, edge_weight)

            val_acc = accuracy(logits, y, val_mask)

            test_acc = accuracy(logits, y, test_mask)

        if val_acc > best_val:

            best_val = val_acc

            best_test = test_acc

            bad = 0

            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}

        else:

            bad += 1

            if bad >= patience:

                break

    if best_state is not None:

        model.load_state_dict(best_state)

    return (best_test, best_val, {'best_val': best_val, 'best_test': best_test})
