from __future__ import annotations

import torch

def _normalize_soft_adj(a_bar: torch.Tensor) -> torch.Tensor:

    a = a_bar + torch.eye(a_bar.size(0), device=a_bar.device, dtype=a_bar.dtype)

    deg = a.sum(dim=1).clamp(min=1e-08)

    d_inv_sqrt = deg.pow(-0.5)

    return d_inv_sqrt.unsqueeze(1) * a * d_inv_sqrt.unsqueeze(0)

def safr(x_bar: torch.Tensor, a_bar: torch.Tensor, k: int) -> torch.Tensor:

    if k <= 0:

        return x_bar

    a_hat = _normalize_soft_adj(a_bar)

    agg = x_bar

    total = x_bar.clone()

    for _ in range(k):

        agg = a_hat @ agg

        total = total + agg

    return total / float(k + 1)

def khop_standard(x_bar: torch.Tensor, a_bar: torch.Tensor, k: int) -> torch.Tensor:

    if k <= 0:

        return x_bar

    hard = (a_bar >= 0.5).to(dtype=x_bar.dtype)

    hard = 0.5 * (hard + hard.t())

    hard.fill_diagonal_(0.0)

    a_hat = _normalize_soft_adj(hard)

    agg = x_bar

    total = x_bar.clone()

    for _ in range(k):

        agg = a_hat @ agg

        total = total + agg

    return total / float(k + 1)
