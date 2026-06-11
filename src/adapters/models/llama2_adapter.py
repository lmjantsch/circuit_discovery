from typing import Tuple

import torch
import torch.nn as nn

from adapters.adapter import ModelAdapter, LayerAdapter
from adapters.utils import head_wise_backwards, head_wise_forward_qkv, apply_inverse_rope, apply_rope


class Llama2LayerAdapter(LayerAdapter):

    def cache_attn_for_grad(self, cache: dict):
        cache[f"{self.layer_id}.q_out_4grad"] = self.q_proj.output
        cache[f"{self.layer_id}.k_rk_4grad"] = self.attention_interface.source.repeat_kv_0.output
        cache[f"{self.layer_id}.v_rk_4grad"] = self.attention_interface.source.repeat_kv_1.output

    def head_wise_value_grad_hook(self, cache: dict, pre_scaling: bool = False) -> torch.Tensor:
        n_kv_heads = self.k_proj.weight.shape[0] // self.head_dim
        group_size = self.n_heads // n_kv_heads

        grad_v = cache[f"{self.layer_id}.v_rk_4grad"].grad          # (B, n_heads, S, head_dim)
        B, _, S, _ = grad_v.shape
        grad_v = grad_v.view(B, n_kv_heads, group_size, S, self.head_dim).permute(0, 3, 2, 1, 4)
        grad_v = head_wise_backwards(self.v_proj, grad_v, self.head_dim, is_grouped=True)  # (B, S, group_size, n_kv_heads, model_dim) before reshape
        # Reorder from group-major (g*n_kv+k) to kv-head-major (k*group_size+g) to match source cache ordering
        model_dim = grad_v.shape[-1]
        grad_v = grad_v.view(B, S, group_size, n_kv_heads, model_dim).permute(0, 1, 3, 2, 4).reshape(B, S, -1, model_dim)

        fwd = cache[f"{self.layer_id -1}.out"].float()                  # (B, S, model_dim)
        grad_v = self.parent._compute_norm_backward(grad_v, self.pre_ln1, fwd, pre_scaling)
        return grad_v.permute(2, 0, 1, 3)                            # (n_heads, B, S, model_dim)

    def head_wise_key_grad_hook(self, cache: dict, pre_scaling: bool = False) -> torch.Tensor:
        n_kv_heads = self.k_proj.weight.shape[0] // self.head_dim
        group_size = self.n_heads // n_kv_heads

        # Reverse rotary emb because the 4grad hook is post-rotation
        grad_k = cache[f"{self.layer_id}.k_rk_4grad"].grad          # (B, n_heads, S, head_dim)
        B, _, S, _ = grad_k.shape
        grad_k = apply_inverse_rope(grad_k, cache['rotary_emb'], unsqueeze_dim=1)  # (B, n_heads, S, head_dim)
        grad_k = grad_k.view(B, n_kv_heads, group_size, S, self.head_dim).permute(0, 3, 2, 1, 4)
        grad_k = head_wise_backwards(self.k_proj, grad_k, self.head_dim, is_grouped=True)  # (B, S, group_size, n_kv_heads, model_dim) before reshape
        # Reorder from group-major (g*n_kv+k) to kv-head-major (k*group_size+g) to match source cache ordering
        model_dim = grad_k.shape[-1]
        grad_k = grad_k.view(B, S, group_size, n_kv_heads, model_dim).permute(0, 1, 3, 2, 4).reshape(B, S, -1, model_dim)

        fwd = cache[f"{self.layer_id -1}.out"].float()                  # (B, S, model_dim)
        grad_k = self.parent._compute_norm_backward(grad_k, self.pre_ln1, fwd, pre_scaling)
        return grad_k.permute(2, 0, 1, 3)                            # (n_heads, B, S, model_dim)

    def head_wise_query_grad_hook(self, cache: dict, pre_scaling: bool = False) -> torch.Tensor:
        grad_q = cache[f"{self.layer_id}.q_out_4grad"].grad          # (B, S, n_heads * head_dim)
        B, S, _ = grad_q.shape
        grad_q = grad_q.view(B, S, self.n_heads, self.head_dim)      # (B, S, n_heads, head_dim)
        grad_q = head_wise_backwards(self.q_proj, grad_q, self.head_dim)  # (B, S, n_heads, model_dim)

        fwd = cache[f"{self.layer_id -1}.out"].float()                  # (B, S, model_dim)
        grad_q = self.parent._compute_norm_backward(grad_q, self.pre_ln1, fwd, pre_scaling)
        return grad_q.permute(2, 0, 1, 3)                            # (n_heads, B, S, model_dim)

    def head_wise_query_patch_hook(self, hidden_state: torch.Tensor, cache: dict, post_scaling: bool = False):
        # hidden_state: (n_heads, B, S, model_dim)
        hs = hidden_state.permute(1, 2, 0, 3)                       # (B, S, n_heads, model_dim)
        hs = self.parent._compute_norm_forward(hs, self.pre_ln1, post_scaling=post_scaling)
        out = head_wise_forward_qkv(self.q_proj, hs, self.head_dim)  # (B, S, n_heads, head_dim)
        self.q_proj.output = out.reshape(*out.shape[:2], -1)         # (B, S, n_heads * head_dim)

    def head_wise_key_patch_hook(self, hidden_state: torch.Tensor, cache: dict, post_scaling: bool = False):
        # hidden_state: (n_heads, B, S, model_dim)
        hs = hidden_state.permute(1, 2, 0, 3)                       # (B, S, n_heads, model_dim)
        hs = self.parent._compute_norm_forward(hs, self.pre_ln1, post_scaling=post_scaling)
        out = head_wise_forward_qkv(self.k_proj, hs, self.head_dim).transpose(1, 2)  # (B, n_heads, S, head_dim)
        out = apply_rope(out, cache['rotary_emb'], unsqueeze_dim=1)
        self.attention_interface.source.repeat_kv_0 = out

    def head_wise_value_patch_hook(self, hidden_state: torch.Tensor, cache: dict, post_scaling: bool = False):
        # hidden_state: (n_heads, B, S, model_dim)
        hs = hidden_state.permute(1, 2, 0, 3)                       # (B, S, n_heads, model_dim)
        hs = self.parent._compute_norm_forward(hs, self.pre_ln1, post_scaling=post_scaling)
        out = head_wise_forward_qkv(self.v_proj, hs, self.head_dim).transpose(1, 2)  # (B, n_heads, S, head_dim)
        self.attention_interface.source.repeat_kv_1 = out


class Llama2ModelAdapter(ModelAdapter):

    @property
    def wrapped_layers(self):
        return [Llama2LayerAdapter(l, self, i) for i, l in enumerate(self.layers)]

    def norm_scale(self, x: torch.Tensor) -> torch.Tensor:
        return torch.rsqrt(x.float().pow(2).mean(-1, keepdim=True) + self.ln.variance_epsilon).to(self.dtype)

    def _compute_norm_forward(
        self, hidden_state: torch.Tensor, norm: nn.Module, post_scaling: bool = False
    ) -> torch.Tensor:
        if post_scaling:
            return norm.weight * hidden_state.to(norm.weight.dtype)
        return norm(hidden_state)

    def _compute_norm_backward(
        self, grad: torch.Tensor, norm: nn.Module, forward_state: torch.Tensor, pre_scaling: bool = False
    ) -> torch.Tensor:
        """
        RMSNorm backward. Accepts grad of shape (B, S, d_model) or (B, S, n_heads, d_model);
        sigma and x_norm are unsqueezed to broadcast over the head dim when present.
        """
        if pre_scaling:
            return grad * norm.weight
        
        x = forward_state.float()                                                        # (B, S, d_model)
        w = norm.weight.float()                                                          # (d_model,)
        sigma = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + norm.variance_epsilon)   # (B, S, 1)
        dy_w = grad.float() * w                                                          # (B, S, [n_heads,] d_model)
        if self.frozen_norm:
            if grad.ndim == 4:
                return (sigma.unsqueeze(-2) * dy_w).to(grad.dtype)
            return (sigma * dy_w).to(grad.dtype)
        x_norm = x * sigma                                                               # (B, S, d_model)
        if grad.ndim == 4:
            sigma  = sigma.unsqueeze(-2)                                                 # (B, S, 1, 1)
            x_norm = x_norm.unsqueeze(-2)                                                # (B, S, 1, d_model)
        dx = sigma * (dy_w - x_norm * (dy_w * x_norm).mean(-1, keepdim=True))
        return dx.to(grad.dtype)
