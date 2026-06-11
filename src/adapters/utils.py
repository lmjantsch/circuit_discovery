import torch
import torch.nn as nn

from dataclasses import dataclass
from typing import Callable

def head_wise_backwards(
    linear: nn.Linear,
    grad: torch.Tensor,
    head_dim: int,
    is_grouped: bool = False,
) -> torch.Tensor:
    """Expects grad tensor of shape (B, S, head_groups, n_kn_heads, head_dim) -> returns (B, S, n_heads, model_dim)"""
    if is_grouped:
        B, S, head_groups, n_kv_heads, head_dim = grad.shape
        W = linear.weight.view(n_kv_heads, head_dim, -1)              # (n_kv_heads, head_dim, model_dim)
        out = torch.einsum("bsgkd,kdm->bsgkm", grad, W)               # (B, S, head_groups, n_kv_heads, model_dim)
        return out.reshape(B, S, head_groups * n_kv_heads, -1)         # (B, S, n_heads, model_dim)
    else:
        B, S, n_heads, head_dim = grad.shape
        W = linear.weight.view(n_heads, head_dim, -1)                  # (n_heads, head_dim, model_dim)
        return torch.einsum("bshd,hdm->bshm", grad, W)                 # (B, S, n_heads, model_dim)


def head_wise_forward(
    linear: nn.Linear,
    x: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    n_heads = linear.weight.shape[1] // head_dim
    W = linear.weight.view(-1, n_heads, head_dim) 
    return torch.einsum("bshd,mhd->hbsm", x, W)


def head_wise_forward_qkv(
    linear: nn.Linear,
    x: torch.Tensor,
    head_dim: int,
) -> torch.Tensor:
    B, S, n_heads, _  = x.shape
    n_kv_heads = linear.weight.shape[0] // head_dim
    group_size = n_heads // n_kv_heads
    W = linear.weight.view(n_kv_heads, head_dim, -1)            # (n_kv_heads, head_dim, model_dim)
    x_g = x.view(B, S, n_kv_heads, group_size, -1)             # (B, S, n_kv_heads, group_size, model_dim)
    out = torch.einsum("bskgm,kdm->bskgd", x_g, W)             # (B, S, n_kv_heads, group_size, head_dim)
    out = out.reshape(B, S, n_heads, head_dim)                  # (B, S, n_heads, head_dim)
    if linear.bias is not None:
        bias = linear.bias.view(n_kv_heads, head_dim).repeat_interleave(group_size, dim=0)
        out = out + bias
    return out


def rotate_half(x):
    """Rotates half the hidden dims of the input."""
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)

def apply_rope(hidden_state, rot_embeds, unsqueeze_dim):
    """Applies Rotary Position Embedding to the query and key tensors.

    Args:
        q (`torch.Tensor`): The query tensor.
        k (`torch.Tensor`): The key tensor.
        cos (`torch.Tensor`): The cosine part of the rotary embedding.
        sin (`torch.Tensor`): The sine part of the rotary embedding.
        unsqueeze_dim (`int`, *optional*, defaults to 1):
            The 'unsqueeze_dim' argument specifies the dimension along which to unsqueeze cos[position_ids] and
            sin[position_ids] so that they can be properly broadcasted to the dimensions of q and k. For example, note
            that cos[position_ids] and sin[position_ids] have the shape [batch_size, seq_len, head_dim]. Then, if q and
            k have the shape [batch_size, heads, seq_len, head_dim], then setting unsqueeze_dim=1 makes
            cos[position_ids] and sin[position_ids] broadcastable to the shapes of q and k. Similarly, if q and k have
            the shape [batch_size, seq_len, heads, head_dim], then set unsqueeze_dim=2.
    Returns:
        `tuple(torch.Tensor)` comprising of the query and key tensors rotated using the Rotary Position Embedding.
    """
    cos, sin = rot_embeds
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (hidden_state * cos) + (rotate_half(hidden_state) * sin)

def apply_inverse_rope(grad, rot_embeds, unsqueeze_dim):
    """
    Backpropagates gradients through the RoPE operation by applying 
    the rotation in the opposite direction.
    """
    cos, sin = rot_embeds
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    return (grad * cos) + (rotate_half(grad) * (-sin))