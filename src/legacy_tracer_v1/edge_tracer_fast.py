"""
Fast vectorized EdgeTracer — optimized for Llama3 and larger models.

Key differences from edge_tracer.py:
1. cache_device='cuda' — no CPU/GPU round-trips for typed updates
2. Typed updates stacked into a single (n_layers, 3, B, S, H, D) tensor
3. All edge scores computed in 3 einsum calls (no per-source Python loops, no .item())
4. Targets ~100-300x speedup on Llama3.1-8B vs edge_tracer.py

MLP is still treated as a single node (matching MIB graph structure).
Output is identical to EdgeTracer.trace(): (n_forward, n_backward) tensor.
"""

from typing import Dict, Any, Tuple
import torch

from tracer.backend import ModelBackend
from tracer.utils import (
    decompose_attention_to_head,
    rotate_half,
)


class FastEdgeTracer:
    VALID_RULES = ('standard', 'dtd', 'counterfactual', 'averaged', 'secant', 'constant')

    def __init__(self, backend: ModelBackend, tokenizer, softmax_rule: str = 'standard'):
        """
        softmax_rule: one of 'standard', 'dtd', 'counterfactual', 'averaged', 'secant', 'constant'
          - 'standard':       A[q,k] * (t_sm[q,k] - Σ_k' A[q,k']*t_sm[q,k'])
          - 'dtd':            t_sm[q,k] - A[q,k] * Σ_k' t_sm[q,k']   (Deep Taylor / Rule 3)
          - 'counterfactual': (A_clean[q,k] - A_corrupt[q,k]) * t_sm[q,k]
          - 'averaged':       A_mid[q,k] * (t_sm_mid[q,k] - Σ_k' A_mid[q,k']*t_sm_mid[q,k'])
          - 'secant':         Secant Jacobian from uniform baseline — factor(x)*(t - mean(t))
          - 'constant':       detach A from grad graph — Q/K delta_ij = 0, V only
        """
        assert softmax_rule in self.VALID_RULES, f"softmax_rule must be one of {self.VALID_RULES}"
        self.backend = backend
        self.tokenizer = tokenizer
        self.config = backend.config
        self.n_layers = self.config.num_hidden_layers
        self.n_heads = self.config.num_attention_heads
        self.d_model = self.config.hidden_size
        self.d_head = self.config.head_dim
        self.dev = backend.compute_device
        self.dtype = backend.dtype
        self.softmax_rule = softmax_rule

    @torch.no_grad()
    def trace(
        self,
        clean_batch: Dict[str, Any],
        corrupt_batch: Dict[str, Any],
        target_ids: torch.Tensor,
    ) -> torch.Tensor:
        n_forward = 1 + self.n_layers * (self.n_heads + 1)
        n_backward = self.n_layers * (3 * self.n_heads + 1) + 1

        clean_cache = self.backend.run_forward_and_cache(clean_batch)
        corrupt_cache = self.backend.run_forward_and_cache(corrupt_batch)

        B, S = clean_batch['input_ids'].shape
        H = self.n_heads
        D = self.d_model

        g = self._init_gradients(clean_batch, clean_cache)
        if 'input_lengths' in clean_batch:
            ans_pos = (clean_batch['input_lengths'].to(self.dev) - 1).long()
        else:
            ans_pos = torch.full((B,), S - 1, device=self.dev, dtype=torch.long)
        b_idx = torch.arange(B, device=self.dev)
        g_logits = g[b_idx, ans_pos].clone()

        # Step 1: Compute source_diffs FIRST (before backward pass)
        source_diffs = torch.zeros(
            n_forward, B, S, D,
            device=self.dev, dtype=self.dtype,
        )
        source_diffs[0] = (clean_cache['emb'] - corrupt_cache['emb']).to(self.dev)
        self._fill_component_diffs(
            source_diffs, clean_batch, corrupt_batch, clean_cache, corrupt_cache
        )

        BSD = B * S * D
        source_2d = source_diffs.reshape(n_forward, BSD)  # (F, BSD)

        # Score accumulators (fp32)
        attn_scores = torch.zeros(
            n_forward, self.n_layers, 3, H,
            device=self.dev, dtype=torch.float32,
        )
        mlp_scores = torch.zeros(
            n_forward, self.n_layers,
            device=self.dev, dtype=torch.float32,
        )

        # Step 2: Streaming backward pass — compute per-layer updates,
        # immediately consume them into score matrix, then discard.
        # This keeps peak memory at ~1 layer worth of typed updates (~500 MB)
        # instead of pre-allocating (L, 3, B, S, H, D) = 16+ GB for Llama3/ARC.
        for l in range(self.n_layers - 1, -1, -1):
            # MLP update — consume immediately
            total_mlp, info_mlp, ctrl_mlp = self.backend.get_mlp_update_dual(l, g, clean_cache)
            mlp_upd = info_mlp + ctrl_mlp  # (B, S, D)
            mlp_flat = mlp_upd.reshape(BSD)  # (BSD,)
            mlp_scores[:, l] = (torch.matmul(source_2d, mlp_flat).float() / B)
            g = g + total_mlp
            del total_mlp, info_mlp, ctrl_mlp, mlp_upd, mlp_flat

            # Attention per-head updates — consume immediately per type
            v_ph, q_ph, k_ph = self._get_attn_update_per_head(l, g, clean_cache, corrupt_cache)
            # q, k, v are stored in that order in the backward index
            for t, upd in enumerate((q_ph, k_ph, v_ph)):
                # upd: (B, S, H, D) → (H, B*S*D)
                upd_2d = upd.permute(2, 0, 1, 3).contiguous().reshape(H, BSD)
                attn_scores[:, l, t, :] = (torch.matmul(source_2d, upd_2d.t()).float() / B)
                del upd_2d

            total_attn = v_ph.sum(dim=2) + q_ph.sum(dim=2) + k_ph.sum(dim=2)
            g = g + total_attn
            del v_ph, q_ph, k_ph, total_attn

        del source_2d

        # Gather source diffs at each sample's answer position (right-padded).
        source_ans = source_diffs[:, b_idx, ans_pos]  # (F, B, D)
        logit_scores = torch.einsum(
            'fbd, bd -> f',
            source_ans, g_logits,
        ).float() / B

        scores = torch.zeros(n_forward, n_backward, device=self.dev, dtype=torch.float32)

        for l in range(self.n_layers):
            base = l * (3 * self.n_heads + 1)
            scores[:, base:base + H] = attn_scores[:, l, 0, :]
            scores[:, base + H:base + 2 * H] = attn_scores[:, l, 1, :]
            scores[:, base + 2 * H:base + 3 * H] = attn_scores[:, l, 2, :]
            scores[:, base + 3 * H] = mlp_scores[:, l]

        scores[:, -1] = logit_scores

        return scores.cpu()

    def _init_gradients(self, batch: Dict[str, Any], cache: Dict[str, Any]) -> torch.Tensor:
        """Initialize backward gradient at the per-sample answer position.

        With right-padding (matching MIB EAP), the answer token sits at
        `input_length - 1` rather than always at `-1`.

        `batch['targets']` shape:
          - (B, 2)  → logit-diff target (EAP standard): W_U[y+] − W_U[y−]
          - (B,)    → single-token target (legacy): W_U[y+]
        """
        B, S = batch['input_ids'].shape
        target_ids = batch['targets']

        if 'input_lengths' in batch:
            ans_pos = (batch['input_lengths'].to(self.dev) - 1).long()
        else:
            ans_pos = torch.full((B,), S - 1, device=self.dev, dtype=torch.long)
        b_idx = torch.arange(B, device=self.dev)

        grad = torch.zeros(B, S, self.d_model, device=self.dev, dtype=self.dtype)
        device = self.backend.model.lm_head.device
        lm_head = self.backend.model.lm_head.weight.data

        # Gather final-LN state at each sample's answer position.
        full_scale = cache['final_ln'].to(device)  # (B, S, 1)
        final_scale = full_scale[b_idx.to(device), ans_pos.to(device)]  # (B, 1)
        norm_W = self.backend.get_final_norm_weight().to(device)

        # eap_compat: apply Gemma-2 final_logit_softcapping derivative.
        # Forward: logit = softcap · tanh(W_U · y / softcap),  y = γ · x / σ (post-LN).
        # ∂L/∂z[i] = (1 − tanh²(z[i]/softcap)) · (δ(i=y+) − δ(i=y−)),  z = W_U · y
        softcap = self.backend.final_logit_softcap if self.backend.eap_compat else None

        if target_ids.dim() == 2 and target_ids.size(1) == 2:
            y_plus = target_ids[:, 0].to(device)
            y_minus = target_ids[:, 1].to(device)
            if softcap is not None:
                full_x = cache['final_ln.x'].to(device)
                x_ans = full_x[b_idx.to(device), ans_pos.to(device)]  # (B, D)
                y_ans = (x_ans.float() * final_scale.float() * norm_W.float())  # post-LN output
                z_plus = (y_ans * lm_head[y_plus].float()).sum(-1)   # pre-softcap logit (B,)
                z_minus = (y_ans * lm_head[y_minus].float()).sum(-1)
                f_plus = 1.0 - torch.tanh(z_plus / softcap) ** 2     # (B,)
                f_minus = 1.0 - torch.tanh(z_minus / softcap) ** 2
                lm_head_W = (
                    lm_head[y_plus].float() * f_plus.unsqueeze(-1)
                    - lm_head[y_minus].float() * f_minus.unsqueeze(-1)
                ).to(lm_head.dtype)
            else:
                lm_head_W = lm_head[y_plus] - lm_head[y_minus]
        else:
            t_ids = target_ids.to(device)
            if softcap is not None:
                full_x = cache['final_ln.x'].to(device)
                x_ans = full_x[b_idx.to(device), ans_pos.to(device)]
                y_ans = (x_ans.float() * final_scale.float() * norm_W.float())
                z_t = (y_ans * lm_head[t_ids].float()).sum(-1)
                f_t = 1.0 - torch.tanh(z_t / softcap) ** 2
                lm_head_W = (lm_head[t_ids].float() * f_t.unsqueeze(-1)).to(lm_head.dtype)
            else:
                lm_head_W = lm_head[t_ids]

        if self.backend.eap_compat and self.backend.use_full_ln_vjp:
            full_x = cache['final_ln.x'].to(device)  # (B, S, D)
            x_final = full_x[b_idx.to(device), ans_pos.to(device)]  # (B, D)
            grad_final = self.backend._rmsnorm_vjp_full(lm_head_W, x_final, norm_W, final_scale)
            grad[b_idx, ans_pos] = grad_final.to(self.dev)
        else:
            grad[b_idx, ans_pos] = (lm_head_W * norm_W * final_scale).to(self.dev)
        return grad

    def _get_attn_update_per_head(
        self, layer_idx: int, grad: torch.Tensor, cache: Dict, corrupt_cache: Dict = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size, seq_len = grad.shape[:2]
        layer = self.backend.model.model.layers[layer_idx]
        dev = self.dev

        norm_w = self.backend._norm_weight_fn(layer.input_layernorm.weight.data).to(dev)
        q_w = layer.self_attn.q_proj.weight.data.view(
            self.config.num_attention_heads, self.d_head, self.d_model).to(dev)
        k_w = layer.self_attn.k_proj.weight.data.view(
            self.config.num_key_value_heads, self.d_head, self.d_model).to(dev)
        v_w = layer.self_attn.v_proj.weight.data.view(
            self.config.num_key_value_heads, self.d_head, self.d_model).to(dev)
        num_head_groups = self.config.num_attention_heads // self.config.num_key_value_heads
        if num_head_groups > 1:
            v_w = v_w.repeat_interleave(num_head_groups, 0)
            k_w = k_w.repeat_interleave(num_head_groups, 0)
        o_w = layer.self_attn.o_proj.weight.data.to(dev)

        sin = cache["rotary.sin"].to(dev)
        cos = cache["rotary.cos"].to(dev)
        v_proj = cache[f"attn.{layer_idx}.v"].to(dev)
        rot_q_proj = cache[f"attn.{layer_idx}.rot_q"].to(dev)
        rot_k_proj = cache[f"attn.{layer_idx}.rot_k"].to(dev)
        attn_weight = cache[f"attn.{layer_idx}.weights"].to(dev)
        z = cache[f"attn.{layer_idx}.z"].to(dev)
        ln_scalar = cache[f"ln.attn.{layer_idx}"].to(dev)
        eap_compat = self.backend.eap_compat
        use_full_ln = eap_compat and self.backend.use_full_ln_vjp
        if use_full_ln:
            x_pre_attn = cache[f"ln.attn.{layer_idx}.x"].to(dev)  # (B, S, D)

        # Optional post-attn norm chain rule (Gemma-2 only)
        if (post_key := f'ln.attn_post.{layer_idx}') in cache:
            post_w = self.backend._norm_weight_fn(
                self.backend.attn_post_norm(layer).weight.data).to(dev)
            post_scalar = cache[post_key].to(dev)
            if use_full_ln:
                post_x = cache[f'{post_key}.x'].to(dev)
                grad = self.backend._rmsnorm_vjp_full(grad, post_x, post_w, post_scalar)
            else:
                grad = grad * post_scalar * post_w

        grad_mid = torch.matmul(grad, o_w).view(
            batch_size, seq_len, self.n_heads, self.d_head).transpose(1, 2)

        # V path: compute post-LN grad per head, then apply LN VJP.
        #   v_post_head[b,h,k,d_h] = Σ_q attn_weight[b,h,q,k] · grad_mid[b,h,q,d_h]
        grad_t_per_head = torch.matmul(
            attn_weight.transpose(-2, -1), grad_mid
        )  # (B, H, K, d_h)
        v_post_per_head = torch.einsum('bhkd, hdo -> bkho', grad_t_per_head, v_w)  # (B, K, H, D)
        if use_full_ln:
            v_per_head = self.backend._rmsnorm_vjp_full(
                v_post_per_head,
                x_pre_attn.unsqueeze(2),      # (B, S, 1, D)
                norm_w,                        # (D,)
                ln_scalar.unsqueeze(2),       # (B, S, 1, 1)
            )
        else:
            v_per_head = self.backend._rmsnorm_vjp_diag(
                v_post_per_head, norm_w, ln_scalar.unsqueeze(2)
            )
        v_per_head = self.backend.scaling['v'] * v_per_head

        term_1 = torch.matmul(grad_mid, v_proj.transpose(-1, -2))
        if self.softmax_rule == 'counterfactual':
            # Counterfactual ΔA: (A_clean - A_corrupt) * t_sm[q,k]
            attn_weight_corrupt = corrupt_cache[f"attn.{layer_idx}.weights"].to(dev)
            delta_ij = (attn_weight - attn_weight_corrupt) * term_1
        elif self.softmax_rule == 'averaged':
            # Trapezoidal IG: A_mid = (A_clean + A_corrupt) / 2 as operating point
            attn_weight_corrupt = corrupt_cache[f"attn.{layer_idx}.weights"].to(dev)
            attn_mid = (attn_weight + attn_weight_corrupt) * 0.5
            z_mid = torch.matmul(attn_mid, v_proj)  # recompute z with A_mid
            term_2_mid = (grad_mid * z_mid).sum(dim=-1, keepdim=True)
            delta_ij = attn_mid * (term_1 - term_2_mid)
        elif self.softmax_rule == 'dtd':
            # Rule 3 (DTD): t_sm[q,k] - A[q,k] * Σ_k' t_sm[q,k']
            term_1_sum = term_1.sum(dim=-1, keepdim=True)
            delta_ij = term_1 - attn_weight * term_1_sum
        elif self.softmax_rule == 'secant':
            # Secant Jacobian: factor(logit) * (term_1 - mean_k(term_1))
            # Logits recomputed from cached rot_q/rot_k (GQA already expanded in cache).
            # Causal mask applied via attn_weight: masked positions (A≈0) zeroed out,
            # matching standard rule's natural A*(...) zeroing.
            logits = torch.matmul(rot_q_proj, rot_k_proj.transpose(-1, -2)) * (self.d_head ** -0.5)
            delta_ij = self._secant_vjp(logits.float(), term_1.float()).to(term_1.dtype)
            delta_ij = delta_ij * (attn_weight > 1e-6)
        elif self.softmax_rule == 'constant':
            # Constant: treat A as detached constant — no gradient through Q/K routing
            delta_ij = torch.zeros_like(term_1)
        else:
            # Standard: A[q,k] * (t_sm[q,k] - Σ_k' A[q,k']*t_sm[q,k'])
            term_2 = (grad_mid * z).sum(dim=-1, keepdim=True)
            delta_ij = attn_weight * (term_1 - term_2)

        # eap_compat: Gemma-2 attention softcap derivative.
        # Forward: A = softmax(softcap · tanh(Q·Kᵀ/√d / softcap) + mask).
        # Chain delta through the tanh_softcap:
        #   ∂L/∂s_raw = ∂L/∂s_softcap · (1 − tanh²(s_raw/softcap))
        attn_softcap = self.backend.attn_logit_softcap if eap_compat else None
        if attn_softcap is not None:
            s_raw = torch.matmul(rot_q_proj.float(), rot_k_proj.transpose(-1, -2).float()) * (self.d_head ** -0.5)
            softcap_factor = 1.0 - torch.tanh(s_raw / attn_softcap) ** 2
            delta_ij = (delta_ij.float() * softcap_factor).to(delta_ij.dtype)

        # Q path: post-LN grad then VJP.
        weighted_k_sum = torch.matmul(delta_ij, rot_k_proj) * (self.d_head ** -0.5)
        grad_q_rot = (weighted_k_sum * cos) - (rotate_half(weighted_k_sum) * sin)
        q_post_per_head = torch.einsum('bhqd, hdo -> bqho', grad_q_rot, q_w)  # (B, Q, H, D)
        if use_full_ln:
            q_per_head = self.backend._rmsnorm_vjp_full(
                q_post_per_head,
                x_pre_attn.unsqueeze(2),
                norm_w,
                ln_scalar.unsqueeze(2),
            )
        else:
            q_per_head = self.backend._rmsnorm_vjp_diag(
                q_post_per_head, norm_w, ln_scalar.unsqueeze(2)
            )
        q_per_head = self.backend.scaling['q'] * q_per_head

        # K path: post-LN grad then VJP.
        weighted_q_sum = torch.matmul(delta_ij.transpose(-2, -1), rot_q_proj) * (self.d_head ** -0.5)
        grad_k_rot = (weighted_q_sum * cos) - (rotate_half(weighted_q_sum) * sin)
        k_post_per_head = torch.einsum('bhkd, hdo -> bkho', grad_k_rot, k_w)  # (B, K, H, D)
        if use_full_ln:
            k_per_head = self.backend._rmsnorm_vjp_full(
                k_post_per_head,
                x_pre_attn.unsqueeze(2),
                norm_w,
                ln_scalar.unsqueeze(2),
            )
        else:
            k_per_head = self.backend._rmsnorm_vjp_diag(
                k_post_per_head, norm_w, ln_scalar.unsqueeze(2)
            )
        k_per_head = self.backend.scaling['k'] * k_per_head

        return v_per_head, q_per_head, k_per_head

    @staticmethod
    def _secant_vjp(logits: torch.Tensor, grad: torch.Tensor) -> torch.Tensor:
        """Secant Jacobian VJP: factor(x) * (grad - mean_k(grad)).

        factor(x) = (exp(x)-1) / (x * (N-1+exp(x)))
        Stable across domains; Taylor near 0 prevents 0/0 NaN.
        """
        N = logits.shape[-1]
        eps = 1e-4

        taylor = 1.0 / N + ((N - 2.0) / (2.0 * N * N)) * logits

        exp_neg = torch.exp(-logits)
        factor_pos = (1.0 - exp_neg) / (logits * ((N - 1) * exp_neg + 1.0))

        exp_x = torch.exp(logits)
        factor_neg = (exp_x - 1.0) / (logits * (N - 1 + exp_x))

        factor = torch.where(logits > 0, factor_pos, factor_neg)
        factor = torch.where(logits.abs() < eps, taylor, factor)

        G_mean = grad.sum(dim=-1, keepdim=True) / N
        return factor * (grad - G_mean)

    def _fill_component_diffs(
        self,
        source_diffs: torch.Tensor,
        clean_batch: Dict,
        corrupt_batch: Dict,
        clean_cache: Dict,
        corrupt_cache: Dict,
    ):
        """
        Compute per-head attention output and MLP output from cached activations
        directly — no second forward pass and no (B, Q, K, H, D) intermediate tensor.

        Attention per-head output:
            z = attn_weight @ v_proj  (already in cache as attn.{l}.z)
            per_head_out = z @ o_proj_W_per_head  (einsum, no K expansion)

        MLP output (from SiLU-gated GLU):
            mlp_out = (gate_act * up_proj) @ down_W
        """
        for l_A in range(self.n_layers):
            layer = self.backend.model.model.layers[l_A]
            o_w = layer.self_attn.o_proj.weight.data.to(self.dev)  # (D, H*d_h)
            # Reshape to (H, d_h, D): for each head, which input dims map to output
            o_w_per_head = o_w.t().view(self.n_heads, self.d_head, self.d_model)
            down_w = layer.mlp.down_proj.weight.data.to(self.dev)  # (D, d_ff)

            # Clean per-head attn output
            z_clean = clean_cache[f'attn.{l_A}.z'].to(self.dev)  # (B, H, Q, d_h)
            head_out_clean = torch.einsum(
                'bhqd, hdD -> bqhD', z_clean, o_w_per_head
            )  # (B, S, H, D)

            # Corrupt per-head attn output
            z_corrupt = corrupt_cache[f'attn.{l_A}.z'].to(self.dev)
            head_out_corrupt = torch.einsum(
                'bhqd, hdD -> bqhD', z_corrupt, o_w_per_head
            )
            head_diff = head_out_clean - head_out_corrupt  # (B, S, H, D)

            for h_A in range(self.n_heads):
                fwd_idx = 1 + l_A * (self.n_heads + 1) + h_A
                source_diffs[fwd_idx] = head_diff[:, :, h_A, :]

            del z_clean, z_corrupt, head_out_clean, head_out_corrupt, head_diff

            # MLP output: (gate_act * up_proj) @ down_W
            gate_act_clean = clean_cache[f'mlp.{l_A}.gate_act'].to(self.dev)
            up_proj_clean = clean_cache[f'mlp.{l_A}.up_proj'].to(self.dev)
            mlp_clean = (gate_act_clean * up_proj_clean) @ down_w.t()  # (B, S, D)

            gate_act_corrupt = corrupt_cache[f'mlp.{l_A}.gate_act'].to(self.dev)
            up_proj_corrupt = corrupt_cache[f'mlp.{l_A}.up_proj'].to(self.dev)
            mlp_corrupt = (gate_act_corrupt * up_proj_corrupt) @ down_w.t()

            mlp_diff = mlp_clean - mlp_corrupt
            fwd_idx = 1 + l_A * (self.n_heads + 1) + self.n_heads
            source_diffs[fwd_idx] = mlp_diff

            del gate_act_clean, up_proj_clean, gate_act_corrupt, up_proj_corrupt
            del mlp_clean, mlp_corrupt, mlp_diff
