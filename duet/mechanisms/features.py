from __future__ import annotations

import math

import torch

def _sample_m(eps: float, d: int, m='best') -> int:

    if m == 'best':

        base = int(max(1, math.floor(max(eps, 1e-06) / 2.18)))

        return int(max(1, min(d, max(base, 16))))

    if m == 'max':

        return d

    return int(m)

class MultiBit:

    def __init__(self, eps: float, input_range=(0.0, 1.0), m='best'):

        self.eps = eps

        self.alpha, self.beta = input_range

        self.m = m

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        n, d = x.size()

        m = _sample_m(self.eps, d, self.m)

        big_s = torch.rand_like(x).topk(m, dim=1).indices

        s = torch.zeros_like(x, dtype=torch.bool).scatter(1, big_s, True)

        em = math.exp(max(self.eps, 1e-06) / m)

        p = (x - self.alpha) / (self.beta - self.alpha)

        p = (p * (em - 1) + 1) / (em + 1)

        t = torch.bernoulli(p.clamp(0, 1))

        x_star = s * (2 * t - 1)

        x_prime = d * (self.beta - self.alpha) / (2 * m)

        x_prime = x_prime * (em + 1) * x_star / (em - 1)

        x_prime = x_prime + (self.alpha + self.beta) / 2

        return x_prime

class PiecewiseMechanism:

    def __init__(self, eps: float, input_range=(0.0, 1.0), m='best'):

        self.eps = eps

        self.alpha, self.beta = input_range

        self.m = m

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        n, d = x.size()

        m = _sample_m(self.eps, d, self.m)

        device = x.device

        t = 2 * (x - self.alpha) / (self.beta - self.alpha) - 1

        t = t.clamp(-1.0, 1.0)

        big_s = torch.rand_like(t).topk(m, dim=1).indices

        mask = torch.zeros_like(t, dtype=torch.bool).scatter(1, big_s, True)

        eps_m = max(self.eps / m, 1e-06)

        c = (math.exp(eps_m / 2) + 1) / (math.exp(eps_m / 2) - 1)

        t_sel = t[mask]

        l = (c + 1) / 2 * t_sel - (c - 1) / 2

        r = l + c - 1

        p_center = math.exp(eps_m / 2) / (math.exp(eps_m / 2) + 1)

        center = torch.rand(t_sel.shape, device=device) < p_center

        u = torch.rand(t_sel.shape, device=device)

        out_sel = torch.empty_like(t_sel)

        out_sel[center] = l[center] + u[center] * (r[center] - l[center]).clamp(min=1e-08)

        if (~center).any():

            left_len = (l[~center] + c).clamp(min=0)

            right_len = (c - r[~center]).clamp(min=0)

            total = (left_len + right_len).clamp(min=1e-08)

            choose_left = u[~center] * total < left_len

            side = torch.empty_like(t_sel[~center])

            if choose_left.any():

                ul = torch.rand(int(choose_left.sum().item()), device=device)

                side[choose_left] = -c + ul * left_len[choose_left].clamp(min=1e-08)

            if (~choose_left).any():

                ur = torch.rand(int((~choose_left).sum().item()), device=device)

                side[~choose_left] = r[~center][~choose_left] + ur * right_len[~choose_left].clamp(min=1e-08)

            out_sel[~center] = side

        center_val = (self.alpha + self.beta) / 2

        t_out = torch.zeros_like(t)

        t_out[mask] = out_sel

        t_clipped = t_out.clamp(-1.0, 1.0)

        contrib = (t_clipped + 1) / 2 * (self.beta - self.alpha) + self.alpha

        x_hat = torch.full_like(x, center_val)

        x_hat = torch.where(mask, center_val + (contrib - center_val) * (d / float(m)), x_hat)

        return x_hat

class SquareWaveMechanism:

    def __init__(self, eps: float, input_range=(0.0, 1.0), m='best'):

        self.eps = eps

        self.alpha, self.beta = input_range

        self.m = m

    def __call__(self, x: torch.Tensor) -> torch.Tensor:

        n, d = x.size()

        m = _sample_m(self.eps, d, self.m)

        device = x.device

        t = 2 * (x - self.alpha) / (self.beta - self.alpha) - 1

        t = t.clamp(-1.0, 1.0)

        big_s = torch.rand_like(t).topk(m, dim=1).indices

        mask = torch.zeros_like(t, dtype=torch.bool).scatter(1, big_s, True)

        eps_m = max(self.eps / m, 1e-06)

        ee = math.exp(eps_m)

        denom = ee * (ee - 1.0 - eps_m)

        b = (eps_m * ee - ee + 1.0) / denom if abs(denom) >= 1e-12 else 1.0 + 2.0 / eps_m

        b = max(float(b), 1.0)

        p = ee / (ee + 1.0)

        t_sel = t[mask]

        u = torch.rand(t_sel.shape, device=device)

        out_sel = torch.empty_like(t_sel)

        left = (t_sel - 1).clamp(min=-b)

        right = (t_sel + 1).clamp(max=b)

        center = u < p

        if center.any():

            uc = torch.rand(int(center.sum().item()), device=device)

            out_sel[center] = left[center] + uc * (right[center] - left[center]).clamp(min=1e-08)

        if (~center).any():

            l_len = (left[~center] + b).clamp(min=0)

            r_len = (b - right[~center]).clamp(min=0)

            total = (l_len + r_len).clamp(min=1e-08)

            us = torch.rand(int((~center).sum().item()), device=device)

            choose_left = us * total < l_len

            side = torch.empty(int((~center).sum().item()), device=device)

            if choose_left.any():

                ul = torch.rand(int(choose_left.sum().item()), device=device)

                side[choose_left] = -b + ul * l_len[choose_left].clamp(min=1e-08)

            if (~choose_left).any():

                ur = torch.rand(int((~choose_left).sum().item()), device=device)

                side[~choose_left] = right[~center][~choose_left] + ur * r_len[~choose_left].clamp(min=1e-08)

            out_sel[~center] = side

        center_val = (self.alpha + self.beta) / 2

        t_out = torch.zeros_like(t)

        t_out[mask] = out_sel.clamp(-b, b)

        t_clipped = t_out.clamp(-1.0, 1.0)

        contrib = (t_clipped + 1) / 2 * (self.beta - self.alpha) + self.alpha

        x_hat = torch.full_like(x, center_val)

        x_hat = torch.where(mask, center_val + (contrib - center_val) * (d / float(m)), x_hat)

        return x_hat

_MECHS = {'mbm': MultiBit, 'pm': PiecewiseMechanism, 'swm': SquareWaveMechanism}

def perturb_features(x: torch.Tensor, mechanism: str, eps: float, input_range=(0.0, 1.0), m='best') -> torch.Tensor:

    mech = _MECHS[mechanism.lower()](eps=eps, input_range=input_range, m=m)

    return mech(x)

def standardize_features(x: torch.Tensor) -> torch.Tensor:

    return (x - x.mean(dim=0, keepdim=True)) / (x.std(dim=0, keepdim=True) + 1e-06)
