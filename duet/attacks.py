from __future__ import annotations

from typing import Dict, Tuple

import numpy as np

import torch

import torch.nn as nn

import torch.nn.functional as F

from duet.data import edge_index_to_dense

from duet.mechanisms import perturb_features, randomized_response, rr_probs, standardize_features

from duet.recovery import bayesian_soft_adjacency, hybrid_adjacency, safr, soft_threshold

from duet.recovery.calibration import EdgeCalibrator

from duet.methods import _solitude_clean_adj, _sym_zero_diag

SETTINGS = ['NonPriv', 'Feature-Only', 'Link-Only', 'Vanilla', 'Duet']

def roc_auc(y_true: np.ndarray, y_score: np.ndarray) -> float:

    y_true = np.asarray(y_true).astype(np.int64).ravel()

    y_score = np.asarray(y_score).astype(np.float64).ravel()

    n_pos = int((y_true == 1).sum())

    n_neg = int((y_true == 0).sum())

    if n_pos == 0 or n_neg == 0:

        return 0.5

    order = np.argsort(y_score)

    ranks = np.empty_like(order, dtype=np.float64)

    ranks[order] = np.arange(1, len(y_score) + 1, dtype=np.float64)

    sorted_scores = y_score[order]

    i = 0

    while i < len(sorted_scores):

        j = i

        while j + 1 < len(sorted_scores) and sorted_scores[j + 1] == sorted_scores[i]:

            j += 1

        if j > i:

            avg = 0.5 * (ranks[order[i]] + ranks[order[j]])

            ranks[order[i:j + 1]] = avg

        i = j + 1

    sum_pos_ranks = ranks[y_true == 1].sum()

    auc = (sum_pos_ranks - n_pos * (n_pos + 1) / 2.0) / (n_pos * n_neg)

    return float(np.clip(auc, 0.0, 1.0))

def feature_inference_auc(x_true: torch.Tensor, x_obs: torch.Tensor) -> float:

    xt = (x_true.detach().cpu().numpy() > 0.5).astype(np.int64)

    xs = x_obs.detach().cpu().numpy().astype(np.float64)

    if xs.min() < -1e-06:

        xs = np.abs(xs)

    aucs = []

    for d in range(xt.shape[1]):

        col = xt[:, d]

        if col.min() == col.max():

            continue

        aucs.append(roc_auc(col, xs[:, d]))

    if not aucs:

        return 0.5

    return float(np.mean(aucs))

def link_inference_auc(a_true: torch.Tensor, a_obs: torch.Tensor, max_neg: int=200000, seed: int=0) -> float:

    a_t = a_true.detach()

    a_o = a_obs.detach()

    n = a_t.size(0)

    iu, ju = torch.triu_indices(n, n, offset=1, device=a_t.device)

    y = a_t[iu, ju]

    s = a_o[iu, ju]

    pos = torch.nonzero(y > 0.5, as_tuple=False).view(-1)

    neg = torch.nonzero(y <= 0.5, as_tuple=False).view(-1)

    if pos.numel() == 0 or neg.numel() == 0:

        return 0.5

    g = torch.Generator(device='cpu')

    g.manual_seed(seed)

    n_neg = min(int(neg.numel()), max(int(pos.numel()) * 5, 1), max_neg)

    perm = torch.randperm(neg.numel(), generator=g)[:n_neg]

    neg_s = neg[perm.to(neg.device)]

    idx = torch.cat([pos, neg_s], dim=0)

    return roc_auc(y[idx].cpu().numpy(), s[idx].cpu().numpy())

def _duet_recovered_views(x_tilde: torch.Tensor, a_tilde: torch.Tensor, soft: torch.Tensor, y: torch.Tensor, train_mask: torch.Tensor, num_classes: int, k: int=8, lambda_f: float=0.05, hidden: int=32, lr: float=0.01) -> Tuple[torch.Tensor, torch.Tensor]:

    a_hat = _solitude_clean_adj(x_tilde, a_tilde, y, train_mask, num_classes, k=min(k, 4), hidden=max(hidden, 48), lr=lr, steps=120)

    a_mix = _sym_zero_diag(0.6 * a_hat + 0.4 * soft)

    x_bar = soft_threshold(x_tilde, lambda_f)

    x_rec = safr(x_bar, a_mix, k=k)

    cal = EdgeCalibrator(x_tilde.size(1), topk=56).to(x_tilde.device)

    opt = torch.optim.Adam(cal.parameters(), lr=lr)

    for _ in range(40):

        opt.zero_grad()

        a_cal = cal(x_rec.detach(), a_mix)

        loss = F.mse_loss(a_cal, soft) + 0.1 * F.mse_loss(a_cal, a_mix)

        loss.backward()

        opt.step()

    with torch.no_grad():

        a_rec = cal(x_rec, a_mix)

        a_rec = _sym_zero_diag(a_rec)

    return (x_rec.detach(), a_rec.detach())

def build_released_views(data, setting: str, *, mechanism: str='mbm', eps_x: float=1.0, eps_a: float=8.0, seed: int=0, device: str='cuda', k: int=8, lambda_f: float=0.05) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)

    dev = torch.device(device if torch.cuda.is_available() else 'cpu')

    data = data.to(dev)

    a_true = edge_index_to_dense(data.edge_index, data.num_nodes, device=dev)

    x_raw = data.x

    x_std = standardize_features(x_raw)

    setting = setting.strip()

    if setting == 'NonPriv':

        return (x_raw, a_true, x_std, a_true)

    x_tilde = standardize_features(perturb_features(x_raw, mechanism, eps_x, input_range=(0.0, 1.0)))

    a_tilde = randomized_response(a_true, eps_a)

    p_a, q_a = rr_probs(eps_a)

    soft = bayesian_soft_adjacency(a_tilde, p_a, q_a)

    if setting == 'Feature-Only':

        return (x_raw, a_true, x_tilde, a_true)

    if setting == 'Link-Only':

        return (x_raw, a_true, x_std, a_tilde)

    if setting == 'Vanilla':

        return (x_raw, a_true, x_tilde, a_tilde)

    if setting == 'Duet':

        x_rec, a_rec = _duet_recovered_views(x_tilde, a_tilde, soft, data.y, data.train_mask, data.num_classes, k=k, lambda_f=lambda_f)

        return (x_raw, a_true, x_rec, a_rec)

    raise ValueError(f'Unknown setting: {setting}')

def evaluate_setting(data, setting: str, *, mechanism: str='mbm', eps_x: float=1.0, eps_a: float=8.0, seed: int=0, device: str='cuda') -> Dict[str, float]:

    x_true, a_true, x_obs, a_obs = build_released_views(data, setting, mechanism=mechanism, eps_x=eps_x, eps_a=eps_a, seed=seed, device=device)

    feat_auc = feature_inference_auc(x_true, x_obs)

    link_auc = link_inference_auc(a_true, a_obs, seed=seed)

    return {'feature_auc': feat_auc, 'link_auc': link_auc, 'setting': setting}
