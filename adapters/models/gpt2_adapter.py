from typing import Tuple

import torch
import torch.nn as nn

from adapters.adapter import ModelAdapter, LayerAdapter

from adapters.utils import head_wise_backwards, head_wise_forward_qkv


class GPT2LayerAdapter(LayerAdapter):

    def cache_attn_for_grad(self, cache: dict):
        cache[f"{self.layer_id}.q_out_4grad"] = self.q_proj.output
        cache[f"{self.layer_id}.k_out_4grad"] = self.k_proj.output
        cache[f"{self.layer_id}.v_out_4grad"] = self.v_proj.output

    def head_wise_value_grad_hook(self, cache: dict, pre_scaling: bool = False) -> torch.Tensor:
        grad_v = cache[f"{self.layer_id}.v_out_4grad"].grad         # (B, S, n_heads * head_dim)
        B, S, _ = grad_v.shape
        grad_v = grad_v.view(B, S, self.n_heads, self.head_dim)     # (B, S, n_heads, head_dim)
        grad_v = head_wise_backwards(self.v_proj, grad_v, self.head_dim)  # (B, S, n_heads, model_dim)
        fwd = cache[f"{self.layer_id - 1}.out"].float()
        grad_v = self.parent._compute_norm_backward(grad_v, self.pre_ln1, fwd, pre_scaling)
        return grad_v.permute(2, 0, 1, 3)                           # (n_heads, B, S, model_dim)

    def head_wise_key_grad_hook(self, cache: dict, pre_scaling: bool = False) -> torch.Tensor:
        grad_k = cache[f"{self.layer_id}.k_out_4grad"].grad         # (B, S, n_heads * head_dim)
        B, S, _ = grad_k.shape
        grad_k = grad_k.view(B, S, self.n_heads, self.head_dim)     # (B, S, n_heads, head_dim)
        grad_k = head_wise_backwards(self.k_proj, grad_k, self.head_dim)  # (B, S, n_heads, model_dim)
        fwd = cache[f"{self.layer_id - 1}.out"].float()
        grad_k = self.parent._compute_norm_backward(grad_k, self.pre_ln1, fwd, pre_scaling)
        return grad_k.permute(2, 0, 1, 3)                           # (n_heads, B, S, model_dim)

    def head_wise_query_grad_hook(self, cache: dict, pre_scaling: bool = False) -> torch.Tensor:
        grad_q = cache[f"{self.layer_id}.q_out_4grad"].grad         # (B, S, n_heads * head_dim)
        B, S, _ = grad_q.shape
        grad_q = grad_q.view(B, S, self.n_heads, self.head_dim)     # (B, S, n_heads, head_dim)
        grad_q = head_wise_backwards(self.q_proj, grad_q, self.head_dim)  # (B, S, n_heads, model_dim)
        fwd = cache[f"{self.layer_id - 1}.out"].float()
        grad_q = self.parent._compute_norm_backward(grad_q, self.pre_ln1, fwd, pre_scaling)
        return grad_q.permute(2, 0, 1, 3)                           # (n_heads, B, S, model_dim)

    def head_wise_query_patch_hook(self, hidden_state: torch.Tensor, cache: dict, post_scaling: bool = False):
        # hidden_state: (n_heads, B, S, model_dim)
        hs = hidden_state.permute(1, 2, 0, 3)                       # (B, S, n_heads, model_dim)
        hs = self.parent._compute_norm_forward(hs, self.pre_ln1, post_scaling=post_scaling)
        out = head_wise_forward_qkv(self.q_proj, hs, self.head_dim)  # (B, S, n_heads, head_dim)
        self.q_proj.output = out.reshape(*out.shape[:2], -1)

    def head_wise_key_patch_hook(self, hidden_state: torch.Tensor, cache: dict, post_scaling: bool = False):
        # hidden_state: (n_heads, B, S, model_dim)
        hs = hidden_state.permute(1, 2, 0, 3)                       # (B, S, n_heads, model_dim)
        hs = self.parent._compute_norm_forward(hs, self.pre_ln1, post_scaling=post_scaling)
        out = head_wise_forward_qkv(self.k_proj, hs, self.head_dim)  # (B, S, n_heads, head_dim)
        self.k_proj.output = out.reshape(*out.shape[:2], -1)

    def head_wise_value_patch_hook(self, hidden_state: torch.Tensor, cache: dict, post_scaling: bool = False):
        # hidden_state: (n_heads, B, S, model_dim)
        hs = hidden_state.permute(1, 2, 0, 3)                       # (B, S, n_heads, model_dim)
        hs = self.parent._compute_norm_forward(hs, self.pre_ln1, post_scaling=post_scaling)
        out = head_wise_forward_qkv(self.v_proj, hs, self.head_dim)  # (B, S, n_heads, head_dim)
        self.v_proj.output = out.reshape(*out.shape[:2], -1)


class GPT2ModelAdapter(ModelAdapter):

    @property
    def wrapped_layers(self):
        return [GPT2LayerAdapter(l, self, i) for i, l in enumerate(self.layers)]

    def norm_scale(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(x.float().var(-1, keepdim=True, unbiased=False) + self.ln.eps).to(self.dtype)

    def _compute_norm_forward(
        self, hidden_state: torch.Tensor, norm: nn.Module, post_scaling: bool = False
    ) -> torch.Tensor:
        if post_scaling:
            return norm.weight * hidden_state + norm.bias
        return norm(hidden_state)

    def _compute_norm_backward(
        self, grad: torch.Tensor, norm: nn.Module, forward_state: torch.Tensor, pre_scaling: bool = False
    ) -> torch.Tensor:
        if pre_scaling:
            return grad * norm.weight
        # analytic backward through nn.LayerNorm:
        # dx = sigma * (dy*w - mean(dy*w) - x_norm * mean(dy*w*x_norm))
        x = forward_state.float()                                                               # (B, S, d_model)
        w = norm.weight.float()                                                                 # (d_model,)
        mean = x.mean(-1, keepdim=True)                                                         # (B, S, 1)
        sigma = torch.rsqrt(x.var(-1, keepdim=True, unbiased=False) + norm.eps)                # (B, S, 1)
        dy_w = grad.float() * w                                                                 # (B, S, [n_heads,] d_model)
        if self.frozen_norm:
            if grad.ndim == 4:
                return (sigma.unsqueeze(-2) * dy_w).to(grad.dtype)
            return (sigma * dy_w).to(grad.dtype)
        x_norm = (x - mean) * sigma                                                             # (B, S, d_model)
        if grad.ndim == 4:
            sigma  = sigma.unsqueeze(-2)                                                        # (B, S, 1, 1)
            x_norm = x_norm.unsqueeze(-2)                                                       # (B, S, 1, d_model)
        dx = sigma * (dy_w - dy_w.mean(-1, keepdim=True) - x_norm * (dy_w * x_norm).mean(-1, keepdim=True))
        return dx.to(grad.dtype)
