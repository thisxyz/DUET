from __future__ import annotations

import torch

def soft_threshold(x_tilde: torch.Tensor, lambda_f: float) -> torch.Tensor:

    return torch.sign(x_tilde) * torch.clamp(x_tilde.abs() - lambda_f, min=0.0)
