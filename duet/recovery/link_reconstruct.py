from __future__ import annotations

import torch

def unbiased_degrees(a_tilde: torch.Tensor, p_a: float, q_a: float) -> torch.Tensor:

    n = a_tilde.size(0)

    d_tilde = a_tilde.sum(dim=1)

    d_hat = (d_tilde - (n - 1) * q_a) / max(p_a - q_a, 1e-08)

    return d_hat.clamp(1.0, float(n - 2))

def _blink_prior(d_hat: torch.Tensor, iters: int=100) -> torch.Tensor:

    n = d_hat.numel()

    device = d_hat.device

    ones_1xn = torch.ones(1, n, device=device)

    ones_nx1 = torch.ones(n, 1, device=device)

    beta = torch.zeros(n, 1, device=device)

    d = d_hat.view(n, 1)

    for _ in range(iters):

        r = 1.0 / (torch.exp(beta).matmul(ones_1xn) + ones_nx1.matmul(torch.exp(-beta).reshape(1, n)))

        denom = (r.sum(1) - r.diagonal()).clamp(min=1e-08).view(n, 1)

        beta = torch.log(d) - torch.log(denom)

    s = ones_nx1.matmul(beta.transpose(0, 1)) + beta.matmul(ones_1xn)

    prior = torch.sigmoid(s)

    prior.fill_diagonal_(0.0)

    return prior

def bayesian_soft_adjacency(a_tilde: torch.Tensor, p_a: float, q_a: float) -> torch.Tensor:

    d_hat = unbiased_degrees(a_tilde, p_a, q_a)

    prior = _blink_prior(d_hat)

    ell1 = torch.where(a_tilde > 0.5, torch.full_like(a_tilde, p_a), torch.full_like(a_tilde, 1.0 - p_a))

    ell0 = torch.where(a_tilde > 0.5, torch.full_like(a_tilde, q_a), torch.full_like(a_tilde, 1.0 - q_a))

    soft = ell1 * prior / (ell1 * prior + ell0 * (1.0 - prior) + 1e-12)

    soft = 0.5 * (soft + soft.t())

    soft.fill_diagonal_(0.0)

    return soft.clamp(0.0, 1.0)

def hybrid_adjacency(soft: torch.Tensor) -> torch.Tensor:

    k = max(int(soft.sum().item()), 1)

    k = min(k, soft.numel())

    thr = torch.topk(soft.flatten(), k=k).values[-1]

    hyb = soft * (soft >= thr).float()

    hyb = 0.5 * (hyb + hyb.t())

    hyb.fill_diagonal_(0.0)

    return hyb
