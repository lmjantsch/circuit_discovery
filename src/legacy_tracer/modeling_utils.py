import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Callable

def head_wise_backwards(
    linear: nn.Linear,
    grad: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """
    Computes the per-head pre-linear gradient for a Q/K/V projection.

    Each head owns a (head_dim, model_dim) slice of the weight matrix.
    We apply those slices independently rather than projecting the full
    concatenated gradient and then splitting.

    Args:
        linear: projection layer with weight (n_heads * head_dim, model_dim)
        grad:   upstream gradient  # (B, S, n_heads, head_dim)
        head_dim: dimension of each head

    Returns:
        pre-linear gradient  # (B, S, n_heads, model_dim)
    """
    # (n_heads * head_dim, model_dim) → (n_heads, head_dim, model_dim)
    n_heads = linear.weight.shape[0] // head_dim
    W = linear.weight.view(n_heads, head_dim, -1)  # (n_heads, head_dim, model_dim)
    return torch.einsum("bshd,hdm->bshm", grad, W)  # (B, S, n_heads, model_dim)


def head_wise_forward(
    linear: nn.Linear,
    x: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    """
    Computes each head's independent contribution through an output projection.

    Splits the weight column-wise (one (head_dim, model_dim) block per head)
    and applies each block independently, without summing across heads.

    Args:
        linear: output projection with weight (model_dim, n_heads * head_dim)
        x:      per-head activations  # (B, S, n_heads, head_dim)
        head_dim: dimension of each head

    Returns:
        per-head output contributions  # (B, S, n_heads, model_dim)
    """
    # (model_dim, n_heads * head_dim).T → (n_heads * head_dim, model_dim)
    # → (n_heads, head_dim, model_dim)
    n_heads = linear.weight.shape[1] // head_dim
    W = linear.weight.T.view(n_heads, head_dim, -1)  # (n_heads, head_dim, model_dim)
    return torch.einsum("bshd,hdm->bshm", x, W)  # (B, S, n_heads, model_dim)


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