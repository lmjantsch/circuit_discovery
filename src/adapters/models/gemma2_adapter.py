from typing import Tuple

import torch
import torch.nn as nn

from .llama2_adapter import Llama2ModelAdapter, Llama2LayerAdapter

from ..utils import head_wise_forward


class Gemma2LayerAdapter(Llama2LayerAdapter):

    def head_wise_attn_out_hook(self, detached: bool = True) -> torch.Tensor:
        attn_out = self.attention_interface.output[0]
        attn_out = head_wise_forward(self.o_proj, attn_out, self.parent.head_dim)  # (n_heads, B, S, d_model)
        # post_attention_layernorm operates on the total attn output; apply the same sigma to every head
        attn_out = attn_out.float()
        total = attn_out.sum(0)                                                    # (B, S, d_model)
        sigma = torch.rsqrt(total.pow(2).mean(-1, keepdim=True) + self.post_ln1.eps)  # (B, S, 1)
        w = 1.0 + self.post_ln1.weight.float()                                    # (d_model,)
        attn_out = (attn_out * sigma.unsqueeze(0) * w).to(self.dtype)             # (n_heads, B, S, d_model)
        if detached:
            return attn_out.detach()
        return attn_out

    def mlp_out_hook(self, detached: bool = True) -> torch.Tensor:
        out = self.mlp.output
        out = self.parent._compute_norm_forward(out, self.post_ln2)
        if detached:
            return out.detach()
        return out


class Gemma2ModelAdapter(Llama2ModelAdapter):

    @property
    def wrapped_layers(self):
        return [Gemma2LayerAdapter(l, self, i) for i, l in enumerate(self.layers)]

    def norm_scale(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.ln.eps).to(self.dtype)

    def _compute_norm_forward(
        self, hidden_state: torch.Tensor, norm: nn.Module, post_scaling: bool = False
    ) -> torch.Tensor:
        if post_scaling:
            # Gemma2 uses (1 + weight) with weight initialised to 0
            return (hidden_state.float() * (1.0 + norm.weight.float())).type_as(hidden_state)
        return norm(hidden_state)

    def _compute_norm_backward(
        self, grad: torch.Tensor, norm: nn.Module, forward_state: torch.Tensor, pre_scaling: bool = False
    ) -> torch.Tensor:
        if pre_scaling:
            return grad * (1.0 + norm.weight)
        x = forward_state.float()                                             # (B, S, d_model)
        w = 1.0 + norm.weight.float()                                        # (d_model,)
        sigma = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.eps)     # (B, S, 1)
        dy_w = grad.float() * w                                               # (B, S, [n_heads,] d_model)
        if self.frozen_norm:
            if grad.ndim == 4:
                return (sigma.unsqueeze(-2) * dy_w).to(grad.dtype)
            return (sigma * dy_w).to(grad.dtype)
        x_norm = x * sigma                                                    # (B, S, d_model)
        if grad.ndim == 4:
            sigma  = sigma.unsqueeze(-2)                                      # (B, S, 1, 1)
            x_norm = x_norm.unsqueeze(-2)                                     # (B, S, 1, d_model)
        dx = sigma * (dy_w - x_norm * (dy_w * x_norm).mean(-1, keepdim=True))
        return dx.to(grad.dtype)
