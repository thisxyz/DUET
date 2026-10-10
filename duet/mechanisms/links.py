from __future__ import annotations

import math

import torch

def rr_probs(eps: float):

    p = math.exp(eps) / (math.exp(eps) + 1.0)

    q = 1.0 / (math.exp(eps) + 1.0)

    return (p, q)

def randomized_response(adj: torch.Tensor, eps: float) -> torch.Tensor:

    n = adj.size(0)

    device = adj.device

    flip = 1.0 / (math.exp(eps) + 1.0)

    triu = torch.triu(torch.ones(n, n, device=device, dtype=torch.bool), diagonal=1)

    u = torch.triu(adj, diagonal=1)

    flip_mask = (torch.rand_like(u) < flip) & triu

    u2 = u.clone()

    u2[flip_mask] = 1.0 - u2[flip_mask]

    out = u2 + u2.t()

    out.fill_diagonal_(0.0)

    return out
