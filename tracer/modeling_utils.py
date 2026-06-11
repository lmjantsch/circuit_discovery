import torch

from dataclasses import dataclass
from typing import Callable

def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_inverse_rope(grad, rot_embeds):
    """
    Backpropagates gradients through the RoPE operation by applying 
    the rotation in the opposite direction.
    """
    cos, sin = rot_embeds
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (grad * cos) + (rotate_half(grad) * (-sin))

def calculate_theta(x: torch.Tensor, x_alt: torch.Tensor) -> torch.Tensor:
    cos_theta = torch.cosine_similarity(x.float(), x_alt, dim=-1)
    cos_theta = torch.clamp(cos_theta, min=-1.0, max=1.0)
    
    theta = torch.acos(cos_theta)
    return theta.unsqueeze(-1)