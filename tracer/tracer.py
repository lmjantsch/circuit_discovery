from contextlib import contextmanager
import math

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from tracer.model_adapters import ModelAdapter

VARIANCE_TYPES = ('within', 'in_between', 'none')
NORM_MATCHING_TYPES = (None, 'source', 'target')
COS_THRESHOLD_TYPES = ('none', 'hard', 'linear', 'tanh')


class EdgeCircuitTracer:

    def __init__(self, adapter: ModelAdapter, tokenizer: AutoTokenizer, cache_device: torch.device | None = None,
                 variance_type: str = 'in_between', norm_matching: str | None = None,
                 q_weight: float = 1.0, k_weight: float = 1.0, v_weight: float = 1.0,
                 gate_weight: float = 1.0, up_weight: float = 1.0, scale_loc: str = 'post',
                 cos_threshold: str = 'none'):
        assert variance_type in VARIANCE_TYPES, f"variance_type must be one of {VARIANCE_TYPES}"
        assert norm_matching in NORM_MATCHING_TYPES, f"norm_matching must be one of {NORM_MATCHING_TYPES}"
        assert cos_threshold in COS_THRESHOLD_TYPES, f"cos_threshold must be one of {COS_THRESHOLD_TYPES}"
        self.adapter = adapter
        self.model = adapter.model
        self.tokenizer = tokenizer

        self.cache_device = cache_device
        if not cache_device:
            self.cache_device = adapter.device

        self.variance_type = variance_type
        self.norm_matching = norm_matching

        self.q_weight = q_weight
        self.k_weight = k_weight
        self.v_weight = v_weight
        self.gate_weight = gate_weight
        self.up_weight = up_weight
        self.scale_loc = scale_loc
        self.cos_threshold = cos_threshold

        self.circuit_scores = None
        self.circuit_scores_m2 = None
        self.num_processed_samples = 0

        self.cache = {}
        self.source_cache = None   # shape: [source_dims, B, S, d]
        self.baseline_cache = {}
        self.baseline_source_cache = None
        self.disable_source_caching = False

        self.clean_embeds = None
        self.corrupt_embeds = None

        self.curr_batch_size = None
        self.curr_seq_len = None
        self.curr_attention_mask = None  # [B, S], 1 for real tokens
        self.curr_id_mask = None         # [B, S, 1], True where clean_id == corrupt_id

    def _init_circuit_tensor(self) -> torch.Tensor:
        return torch.zeros(self.adapter.source_dims, self.adapter.grad_dims, dtype=self.adapter.dtype, device=self.cache_device)

    def _init_source_cache(self) -> torch.Tensor:
        return torch.zeros(
            self.adapter.source_dims, self.curr_batch_size, self.curr_seq_len, self.adapter.model_dim,
            device=self.adapter.device, dtype=self.adapter.dtype,
        )

    def build_circuit(self, dataloader: DataLoader, use_counterfactual: bool = False, integration_steps: int = 1):
        self.circuit_scores = self._init_circuit_tensor()
        if self.variance_type != 'none':
            self.circuit_scores_m2 = self._init_circuit_tensor()
        self.num_processed_samples = 0

        try:
            output = self._build_circuit_inner(
                dataloader=dataloader,
                use_counterfactual=use_counterfactual,
                integration_steps=integration_steps)
        finally:
            self.circuit_scores = None
            self.circuit_scores_m2 = None
            self.num_processed_samples = None

        return output

    def _build_circuit_inner(self, dataloader: DataLoader, use_counterfactual: bool = False, integration_steps: int = 1):

        for batch in tqdm(dataloader):
            clean_prompts, corrupt_prompts, clean_targets, corrupt_targets = batch

            clean_inputs = self.tokenizer(clean_prompts, padding=True, return_tensors='pt')
            corrupt_inputs = self.tokenizer(corrupt_prompts, padding=True, return_tensors='pt')

            self.curr_batch_size, self.curr_seq_len = clean_inputs['input_ids'].shape
            self.curr_attention_mask = clean_inputs['attention_mask']
            self.curr_id_mask = (clean_inputs['input_ids'] == corrupt_inputs['input_ids']).unsqueeze(-1).to(self.adapter.device)  # [B, S, 1]

            target_idx = torch.full((self.curr_batch_size,), self.curr_seq_len - 1)
            if self.tokenizer.padding_side == 'right':
                target_idx = clean_inputs['attention_mask'].sum(dim=-1) - 1

            with self.manage_batch_context(), self.model.session():

                if use_counterfactual == True or integration_steps > 1:
                    with self.model.trace(**corrupt_inputs):
                        self._forward_pass_and_cache()
                    self.corrupt_embeds = self.cache['emb'].detach().clone()
                    self.baseline_cache = {f'{i}.{k}': self.cache[f'{i}.{k}'] for i in range(self.adapter.n_layers) for k in ['mid', 'out']}
                    self.baseline_cache[f'-1.out'] = self.cache['-1.out']
                    self.baseline_source_cache = self.source_cache
                    self.source_cache = self._init_source_cache()

                with self.model.trace(**clean_inputs):
                    self._forward_pass_and_cache()
                    self.clean_embeds = self.cache['emb'].detach().clone()
                    metric = self._get_metric(clean_targets, corrupt_targets, target_idx, use_counterfactual)

                    if use_counterfactual == True or integration_steps > 1:
                        if self.norm_matching == 'source':
                            source_scale = self.get_norm_matching()
                            self.source_cache -= self.baseline_source_cache * source_scale
                            self.baseline_source_cache = None
                        elif self.norm_matching == 'target':
                            pass  # baseline_source_cache kept; scaling applied per-target in _update_scores
                        else:
                            self.source_cache -= self.baseline_source_cache
                            self.baseline_source_cache = None

                    self.num_processed_samples += self.curr_batch_size
                    self._backward_pass_and_scoring(metric)

                if integration_steps > 1:
                    self.disable_source_caching = True
                    for alpha in torch.linspace(0, 1, integration_steps + 1)[1:-1]:  # k / integration_steps for k in range(integration_steps)
                        with self.model.trace(**clean_inputs):
                            integrated_embeds = (1 - alpha) * self.clean_embeds + alpha * self.corrupt_embeds
                            self._forward_pass_and_cache(integrated_embeds)
                            metric = self._get_metric(clean_targets, corrupt_targets, target_idx, use_counterfactual)
                            self.num_processed_samples += self.curr_batch_size
                            self._backward_pass_and_scoring(metric)

        if self.variance_type != 'none':
            stat = self.circuit_scores_m2 / self.num_processed_samples
            return (self.circuit_scores, stat)

        return (self.circuit_scores, None)

    @contextmanager
    def manage_batch_context(self):
        self.cache = {i: {} for i in range(-1, self.adapter.n_layers)}
        self.source_cache = self._init_source_cache()
        try:
            yield
        except Exception as e:
            print(e)
        finally:
            self.empty_cache()
            self.disable_source_caching = False

    def empty_cache(self):
        self.cache = {}
        self.baseline_cache = {}
        self.source_cache = None
        self.baseline_source_cache = None

    def _forward_pass_and_cache(self, integrated_embeds: torch.Tensor | None = None):
        B, S, d = self.curr_batch_size, self.curr_seq_len, self.adapter.model_dim  # (B, S, d)

        if integrated_embeds is not None:
            self.adapter.emb.output = integrated_embeds

        self.cache['emb'] = self.adapter.emb_out().detach()

        if self.adapter.uses_rotary_emb:
            self.cache['rotary_emb'] = self.adapter.rotary_emb_out()

        first_layer = self.adapter.layers[0]
        self.cache['-1.out'] = self.adapter.residual_in(first_layer).detach()
        self._update_source_cache(
            self.adapter.residual_in(first_layer).detach().reshape(1, B, S, d), type='emb'
        )

        for layer_id, layer in enumerate(self.adapter.layers):

            if self.adapter.norm_approx in ('dynamic_thr', 'dynamic_msk') and self.baseline_cache:
                self.adapter.ln_1_baseline_hook(layer, self.baseline_cache[f'{layer_id - 1}.out'], self.curr_id_mask)

            self.adapter.build_attn_source(layer)  # call to build source
            self.cache[f'{layer_id}.query_out'] = self.adapter.query_out(layer)
            self.cache[f'{layer_id}.key_out'] = self.adapter.key_out(layer)
            self.cache[f'{layer_id}.value_out'] = self.adapter.value_out(layer)

            self._update_source_cache(
                self.adapter.per_head_attn_out(layer).detach().reshape(self.adapter.n_heads, B, S, d),
                type='attn', layer_id=layer_id)

            self.cache[f'{layer_id}.mid'] = self.adapter.residual_mid(layer).detach()

            if self.adapter.norm_approx in ('dynamic_thr', 'dynamic_msk') and self.baseline_cache:
                self.adapter.ln_2_baseline_hook(layer, self.baseline_cache[f'{layer_id}.mid'], self.curr_id_mask)

            self.cache[f'{layer_id}.gate_out'] = self.adapter.gate_out(layer)
            self.cache[f'{layer_id}.up_out'] = self.adapter.up_out(layer)
            self._update_source_cache(
                self.adapter.mlp_out(layer).detach().reshape(1, B, S, d), type='mlp', layer_id=layer_id)

            self.cache[f'{layer_id}.out'] = self.adapter.residual_out(layer)

        self.cache['logits'] = self.adapter.logits()

    def _update_source_cache(self, new_tensor: torch.Tensor, type: str, layer_id: int = None):
        if self.disable_source_caching == False:
            source_slice = self.adapter.get_src_slice(type=type, layer_id=layer_id)
            self.source_cache[source_slice] = new_tensor.detach()

    def _get_metric(self, clean_targets, corrupt_targets, target_idx, use_counterfactual):
        clean_logits = self.cache['logits'][range(len(clean_targets)), target_idx, clean_targets]
        corrupt_logits = self.cache['logits'][range(len(corrupt_targets)), target_idx, corrupt_targets]

        if use_counterfactual:
            return (clean_logits - corrupt_logits).sum()

        return clean_logits.sum()

    def _backward_pass_and_scoring(self, metric: torch.Tensor):
        B, S, d = self.curr_batch_size, self.curr_seq_len, self.adapter.model_dim  # (B, S, d)

        with metric.backward():
            logit_grad = self.cache[f'{self.adapter.n_layers - 1}.out'].grad.detach().reshape(1, B, S, d)
            self._update_scores(logit_grad, type='lm_head')
            del logit_grad

            for layer_id in range(self.adapter.n_layers - 1, -1, -1):
                layer = self.adapter.layers[layer_id]
                layer_input = self.cache[f'{layer_id - 1}.out'].detach()

                up_grad = self._scale_and_detach_grad(self.cache[f'{layer_id}.up_out'], self.up_weight, scale_loc=self.scale_loc)
                gate_grad = self._scale_and_detach_grad(self.cache[f'{layer_id}.gate_out'], self.gate_weight, scale_loc=self.scale_loc)

                mlp_in_grad_pre_norm = self.adapter.mlp_in_grad(
                    gate_grad=gate_grad, up_grad=up_grad,
                    x_pre_norm=self.cache[f'{layer_id}.mid'],
                    x_baseline=self.baseline_cache.get(f'{layer_id}.mid', self.cache[f'{layer_id}.mid']),
                    mask=self.curr_id_mask,
                    layer=layer,
                ).reshape(1, B, S, d)
                self._update_scores(mlp_in_grad_pre_norm, type='mlp', layer_id=layer_id)
                del mlp_in_grad_pre_norm

                v_grad = self._scale_and_detach_grad(self.cache[f'{layer_id}.value_out'], self.v_weight, scale_loc=self.scale_loc)

                v_proj_in_grad_pre_norm = self.adapter.value_in_grad(
                    grad=v_grad, x_pre_norm=layer_input,
                    x_baseline=self.baseline_cache.get(f'{layer_id - 1}.out', layer_input),
                    mask=self.curr_id_mask,
                    layer=layer, cache=self.cache,
                ).reshape(self.adapter.n_heads, B, S, d)
                self._update_scores(v_proj_in_grad_pre_norm, type='attn_v', layer_id=layer_id)
                del v_proj_in_grad_pre_norm

                k_grad = self._scale_and_detach_grad(self.cache[f'{layer_id}.key_out'], self.k_weight, scale_loc=self.scale_loc)

                k_proj_in_grad_pre_norm = self.adapter.key_in_grad(
                    grad=k_grad, x_pre_norm=layer_input,
                    x_baseline=self.baseline_cache.get(f'{layer_id - 1}.out', layer_input),
                    mask=self.curr_id_mask,
                    layer=layer, cache=self.cache,
                ).reshape(self.adapter.n_heads, B, S, d)
                self._update_scores(k_proj_in_grad_pre_norm, type='attn_k', layer_id=layer_id)
                del k_proj_in_grad_pre_norm

                q_grad = self._scale_and_detach_grad(self.cache[f'{layer_id}.query_out'], self.q_weight, scale_loc=self.scale_loc)

                q_proj_in_grad_pre_norm = self.adapter.query_in_grad(
                    grad=q_grad, x_pre_norm=layer_input,
                    x_baseline=self.baseline_cache.get(f'{layer_id - 1}.out', layer_input),
                    mask=self.curr_id_mask,
                    layer=layer, cache=self.cache,
                ).reshape(self.adapter.n_heads, B, S, d)
                self._update_scores(q_proj_in_grad_pre_norm, type='attn_q', layer_id=layer_id)
                del q_proj_in_grad_pre_norm

    def _get_target_norm_ratio(self, type: str, layer_id: int | None) -> torch.Tensor:
        """Norm ratio (clean/baseline) of the residual that target `type` reads from. Returns (B, S, 1)."""
        if type == 'lm_head':
            key = f'{self.adapter.n_layers - 1}.out'
        elif type == 'mlp':
            key = f'{layer_id}.mid'
        else:  # attn_q, attn_k, attn_v
            key = f'{layer_id - 1}.out' if layer_id > 0 else '-1.out'
        clean = self.cache[key]
        baseline = self.baseline_cache[key]
        return (clean.detach().norm(dim=-1, keepdim=True)
                / baseline.detach().norm(dim=-1, keepdim=True).clamp(min=1e-8))  # (B, S, 1)

    def _apply_cos_threshold(
        self,
        scores: torch.Tensor,   # (B, n_src, n_tgt)
        sources: torch.Tensor,  # (n_src, B, S, d)
        grad: torch.Tensor,     # (n_tgt, B, S, d)
    ) -> torch.Tensor:
        threshold = math.sqrt(2.0 / (math.pi * self.adapter.model_dim))
        norms_u = sources.detach().norm(dim=-1)  # (n_src, B, S)
        norms_g = grad.detach().norm(dim=-1)     # (n_tgt, B, S)
        denom = torch.einsum('ibs,jbs->bij', norms_u, norms_g).to(self.cache_device)  # (B, n_src, n_tgt)
        cos_sim = scores / denom.clamp(min=1e-8)  # (B, n_src, n_tgt)
        abs_cos = cos_sim.abs()

        if self.cos_threshold == 'hard':
            weight = (abs_cos > threshold).to(scores.dtype)
        elif self.cos_threshold == 'linear':
            weight = (abs_cos - threshold).clamp(min=0) / abs_cos.clamp(min=1e-8)
        else:  # 'tanh'
            ratio = ((abs_cos - threshold) / threshold).clamp(min=0)
            weight = torch.where(abs_cos > threshold, torch.tanh(ratio), torch.zeros_like(ratio))

        return scores * weight

    def _update_scores(self, grad: torch.Tensor, type: str, layer_id: int = None):
        # grad: [n_tgt, B, S, d]
        src2d_slice = self.adapter.get_src_slice(type, layer_id)
        src_slice = slice(None, src2d_slice.start)
        tgt_slice = self.adapter.get_tgt_slice(type, layer_id)

        sources = self.source_cache[src_slice]  # (n_src, B, S, d)
        if self.norm_matching == 'target' and self.baseline_source_cache is not None:
            target_ratio = self._get_target_norm_ratio(type, layer_id)  # (B, S, 1)
            sources = sources - self.baseline_source_cache[src_slice] * target_ratio

        # per_sample_scores[b, i, j] = dot(sources[i,b,:,:], grad[j,b,:,:]) over S*d  # [B, n_src, n_tgt]
        per_sample_scores = torch.einsum('ibsd,jbsd->bij', sources, grad).to(self.cache_device).detach()
        if self.cos_threshold != 'none':
            per_sample_scores = self._apply_cos_threshold(per_sample_scores, sources, grad)

        n_real = self.curr_attention_mask.sum(dim=-1).float().to(self.cache_device)  # [B]
        per_token_scores = per_sample_scores / n_real[:, None, None]                 # [B, n_src, n_tgt], μ_b

        # --- Running mean update (per-token normalised) ---
        batch_mean = per_token_scores.mean(dim=0)
        batch_mean_delta = batch_mean - self.circuit_scores[src_slice, tgt_slice]
        batch_weight = self.curr_batch_size / self.num_processed_samples
        self.circuit_scores[src_slice, tgt_slice] += batch_weight * batch_mean_delta

        if self.variance_type == 'none':
            return

        # --- Variance/stat update ---
        if self.variance_type == 'within':
            # [B, S, n_src, n_tgt]: per-position attribution scores
            per_pos = torch.einsum('ibsd,jbsd->bsij', sources, grad).to(self.cache_device).detach()
            mask = self.curr_attention_mask.to(self.cache_device).float()            # [B, S]
            masked_mean = (per_pos * mask[:, :, None, None]).sum(dim=1) / n_real[:, None, None]  # [B, n_src, n_tgt]
            sq_dev = ((per_pos - masked_mean.unsqueeze(1)) * mask[:, :, None, None]) ** 2        # [B, S, n_src, n_tgt]
            within_var = sq_dev.sum(dim=1) / n_real[:, None, None]                               # [B, n_src, n_tgt]
            # Accumulate sum; final division by num_processed_samples gives E_b[Var_s[x_{b,s}]]
            self.circuit_scores_m2[src_slice, tgt_slice] += within_var.sum(dim=0)

        else:  # 'in_between': Var_b[μ_b], Welford running mean reuses circuit_scores
            # Chan's algorithm for batched M2 update: M2_total = M2_prev + M2_batch + delta² * n_prev*n_new / n_total
            batch_m2 = ((per_token_scores - batch_mean.unsqueeze(0)) ** 2).sum(dim=0)
            batch_m2_weight = (self.num_processed_samples - self.curr_batch_size) * self.curr_batch_size / self.num_processed_samples
            self.circuit_scores_m2[src_slice, tgt_slice] += batch_m2 + (batch_mean_delta ** 2) * batch_m2_weight

    def get_norm_matching(self) -> torch.Tensor:
        """Compute per-source norm ratio (clean / baseline) for counterfactual attribution.

        For each source slot, computes norm(r_clean) / norm(r_baseline) where r is the
        residual state *after* that source's contribution has been added:
          - embedding  → ratio at '-1.out'
          - attn heads of layer L → ratio at 'L.mid'
          - MLP of layer L       → ratio at 'L.out'

        Returns shape: (source_dims, B, S, 1)
        """
        B, S = self.curr_batch_size, self.curr_seq_len

        def _ratio(clean: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
            # (B, S, 1)
            return (clean.detach().norm(dim=-1, keepdim=True)
                    / baseline.detach().norm(dim=-1, keepdim=True).clamp(min=1e-8))

        parts = []

        # Embedding slot (index 0)
        parts.append(_ratio(self.cache['-1.out'], self.baseline_cache['-1.out']).unsqueeze(0))  # (1, B, S, 1)

        for layer_id in range(self.adapter.n_layers):
            # Attention heads: residual at mid (after attn, before MLP)
            mid_ratio = _ratio(self.cache[f'{layer_id}.mid'], self.baseline_cache[f'{layer_id}.mid'])
            parts.append(mid_ratio.unsqueeze(0).expand(self.adapter.n_heads, B, S, 1))  # (n_heads, B, S, 1)

            # MLP: residual at out (after MLP)
            out_ratio = _ratio(self.cache[f'{layer_id}.out'], self.baseline_cache[f'{layer_id}.out'])
            parts.append(out_ratio.unsqueeze(0))  # (1, B, S, 1)

        return torch.cat(parts, dim=0)  # (source_dims, B, S, 1)

    def _scale_and_detach_grad(self, grad_tensor: torch.Tensor, weight: float, scale_loc: str = 'post'):
        if grad_tensor == None:  # handle GPT2 missing gate tensor
            return None

        if scale_loc == 'pre':
            grad_tensor.grad = weight * grad_tensor.grad

        detached_grad = grad_tensor.grad.detach()

        if scale_loc == 'post':
            grad_tensor.grad = weight * grad_tensor.grad

        return detached_grad
