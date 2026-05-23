from contextlib import contextmanager
import math

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from adapters import ModelAdapter

VARIANCE_TYPES = ('within', 'in_between', 'none')
NORM_MATCHING_TYPES = (None, 'source', 'target')


class EdgeCircuitTracer:

    def __init__(self, adapter: ModelAdapter, tokenizer: AutoTokenizer, cache_device: torch.device | None = None,
                  use_counterfactual: bool = False, integration_steps: int = 1, variance_type: str = 'in_between', norm_matching: str | None = None,):
        assert variance_type in VARIANCE_TYPES, f"variance_type must be one of {VARIANCE_TYPES}"
        assert norm_matching in NORM_MATCHING_TYPES, f"norm_matching must be one of {NORM_MATCHING_TYPES}"
        self.adapter = adapter
        self.model = adapter.model
        self.tokenizer = tokenizer

        self.cache_device = cache_device
        if not cache_device:
            self.cache_device = adapter.device

        self.integration_steps = integration_steps
        self.use_counterfactual = use_counterfactual or integration_steps > 1
        self.variance_type = variance_type
        self.norm_matching = norm_matching

        self.circuit_scores = None
        self.circuit_scores_m2 = None
        self.num_processed_samples = 0

        self._empty_cache()

    def _empty_cache(self):
        self._c_base_cache = {}
        self._c_base_source = None

        self._c_cache = {}
        self._c_source = None

        self._c_act_cache = ({}, None, {})

        self._c_inputs = None
        self._c_base_inputs = None
        self._c_batch_size, self._c_seq_length = None, None
        self._c_identity_mask = None
        self._c_target_idx = None
        self._c_base_target_idx = None

    def _init_circuit_tensor(self) -> torch.Tensor:
        return torch.zeros(self.adapter.source_dims, self.adapter.grad_dims, dtype=self.adapter.dtype, device=self.cache_device)

    def _init_source_cache(self) -> torch.Tensor:
        return torch.zeros(
            self.adapter.source_dims, self._c_batch_size, self._c_seq_length, self.adapter.model_dim,
            device=self.adapter.device, dtype=self.adapter.dtype,
        )

    @contextmanager
    def circuit_context(self):
        self.circuit_scores = self._init_circuit_tensor()
        if self.variance_type != 'none':
            self.circuit_scores_m2 = self._init_circuit_tensor()
        self.num_processed_samples = 0

        try:
            yield
        except Exception as e:
            raise(e)
        finally:
            self.circuit_scores = None
            self.circuit_scores_m2 = None
            self.num_processed_samples = None

    @contextmanager
    def batch_context(self, batch):
        prompts, base_prompts, target, base_target = batch

        self._c_inputs = self.tokenizer(prompts, padding=True, return_tensors='pt')
        self._c_base_inputs = self.tokenizer(base_prompts, padding=True, return_tensors='pt')

        B, S = self._c_inputs['input_ids'].shape
        self._c_batch_size, self._c_seq_length = B, S
        self._c_identity_mask = ((self._c_inputs['input_ids'] == self._c_base_inputs['input_ids']) & \
            self._c_inputs['attention_mask'].bool()).unsqueeze(-1).to(self.adapter.device) # [B, S, 1]
        self._c_source = self._init_source_cache()
        self._c_base_source = self._init_source_cache()

        last_token_idx = torch.full((B,), S - 1)
        if self.tokenizer.padding_side == 'right':
            last_token_idx = self._c_inputs['attention_mask'].sum(dim=-1) - 1
        self._c_target_idx = (range(B), last_token_idx, target)
        self._c_base_target_idx = (range(B), last_token_idx, base_target)

        try:
            yield
        finally:
            self._empty_cache()

    @contextmanager
    def trace_context(self, is_baseline:bool = False, is_integration:bool = False):
        if is_baseline:
            inputs = self._c_base_inputs
            self._c_act_cache = (self._c_base_cache, self._c_base_source, {})
        elif is_integration:
            inputs = self._c_inputs
            self._c_act_cache = ({}, None, {})
        else:
            inputs = self._c_inputs
            self._c_act_cache = (self._c_cache, self._c_source, {})

        try:
            yield inputs
        finally:
            self._c_act_cache = ({}, None, {})


    def build_circuit(self, dataloader: DataLoader):
        with self.circuit_context():
            for batch in tqdm(dataloader):
                self._process_batch(batch)

            if self.variance_type != 'none':
                stat = self.circuit_scores_m2 / self.num_processed_samples
                return (self.circuit_scores, stat)

            return (self.circuit_scores, None)

    def _process_batch(self, batch):
        with self.batch_context(batch), self.model.session():

            if self.use_counterfactual:
                with self.trace_context(is_baseline=True), self.model.trace(**self._c_base_inputs):
                    self._forward_pass_and_cache()

            with self.trace_context(), self.model.trace(**self._c_inputs):
                self._forward_pass_and_cache()
                metric = self._get_metric()

                if self.use_counterfactual:
                    if self.norm_matching == 'source':
                        source_scale = self.get_norm_matching()
                        self._c_source -= self._c_base_source * source_scale
                        self._c_base_source = None
                    elif self.norm_matching == 'target':
                        pass  # _c_base_source kept; scaling applied per-target in _update_scores
                    else:
                        self._c_source -= self._c_base_source
                        self._c_base_source = None

                self.num_processed_samples += self._c_batch_size
                self._backward_pass_and_scoring(metric)

            if self.integration_steps > 1:
                for alpha in torch.linspace(0, 1, self.integration_steps + 1)[1:-1]:  # k / integration_steps, k=1 already computed
                    with self.trace_context(is_integration=True), self.model.trace(**self._c_inputs):
                        integrated_embeds = (1 - alpha) * self._c_cache['emb'] + alpha * self._c_base_cache['emb']
                        self._forward_pass_and_cache(integrated_embeds)
                        metric = self._get_metric()

                        self.num_processed_samples += self._c_batch_size
                        self._backward_pass_and_scoring(metric)

    def _update_source_cache(self, new_tensor: torch.Tensor, type: str, layer_id: int = None):
        if self._c_act_cache[1] is not None:
            source_slice = self.adapter.get_src_slice(type=type, layer_id=layer_id)
            self._c_act_cache[1][source_slice] = new_tensor.detach()

    def _get_metric(self):
        cache = self._c_act_cache[0]
        target_logits = cache['logits'][self._c_target_idx]
        base_logits = cache['logits'][self._c_base_target_idx]

        if self.use_counterfactual:
            return (target_logits - base_logits).sum()
        return target_logits.sum()

    def _forward_pass_and_cache(self, integrated_embeds: torch.Tensor | None = None):
        B, S, d = self._c_batch_size, self._c_seq_length, self.adapter.model_dim  # (B, S, d)
        cache, _, grad_cache = self._c_act_cache

        if integrated_embeds is not None:
            self.adapter.embedding_patch_hook(integrated_embeds)

        cache['emb'] = self.adapter.embed.output.detach()

        if self.adapter.uses_rotary_emb:
            cos, sin = self.adapter.rotary_emb.output
            cache['rotary_emb'] = (cos.detach(), sin.detach())

        cache['-1.out'] = self.adapter.embed_out_hook(detached=False)
        self._update_source_cache(
            self.adapter.embed_out_hook().reshape(1, B, S, d), type='emb')

        for layer_id, layer in enumerate(self.adapter.wrapped_layers):
            layer.attention_interface # needs to be included to access attn.source
            layer.cache_attn_for_grad(grad_cache)
            self._update_source_cache(
                layer.head_wise_attn_out_hook(), type='attn', layer_id=layer_id)
            cache[f'{layer_id}.mid'] = layer.residual_mid_hook()

            layer.cache_mlp_for_grad(grad_cache)
            self._update_source_cache(
                layer.mlp_out_hook().reshape(1, B, S, d), type='mlp', layer_id=layer_id)
            cache[f'{layer_id}.out'] = layer.residual_out_hook()

        cache['post'] = cache[f'{self.adapter.n_layers - 1}.out']
        self.adapter.cache_logits_for_grad(grad_cache)
        cache['logits'] = self.model.output['logits']

    def _backward_pass_and_scoring(self, metric: torch.Tensor):
        B, S, d = self._c_batch_size, self._c_seq_length, self.adapter.model_dim  # (B, S, d)
        cache, _, grad_cache = self._c_act_cache
        full_cache = grad_cache | cache

        with metric.backward():
            self._update_scores(
                self.adapter.logit_grad_hook(full_cache).reshape(1, B, S, d), type='lm_head')

            for layer in self.adapter.wrapped_layers[::-1]:
                self._update_scores(
                    layer.mlp_grad_hook(full_cache).reshape(1, B, S, d), type='mlp', layer_id=layer.layer_id)

                self._update_scores(
                    layer.head_wise_value_grad_hook(full_cache), type='attn_v', layer_id=layer.layer_id)

                self._update_scores(
                    layer.head_wise_key_grad_hook(full_cache), type='attn_k', layer_id=layer.layer_id)

                self._update_scores(
                    layer.head_wise_query_grad_hook(full_cache), type='attn_q', layer_id=layer.layer_id)


    def _get_target_norm_ratio(self, type: str, layer_id: int | None) -> torch.Tensor:
        """Norm ratio (clean/baseline) of the residual that target `type` reads from. Returns (B, S, 1)."""
        if type == 'lm_head':
            key = f'{self.adapter.n_layers - 1}.out'
        elif type == 'mlp':
            key = f'{layer_id}.mid'
        else:  # attn_q, attn_k, attn_v
            key = f'{layer_id - 1}.out' if layer_id > 0 else '-1.out'
        clean = self._c_cache[key]
        baseline = self._c_base_cache[key]
        return (clean.detach().norm(dim=-1, keepdim=True)
                / baseline.detach().norm(dim=-1, keepdim=True).clamp(min=1e-8))  # (B, S, 1)

    def _update_scores(self, grad: torch.Tensor, type: str, layer_id: int = None):
        # grad: [n_tgt, B, S, d]
        src2d_slice = self.adapter.get_src_slice(type, layer_id)
        src_slice = slice(None, src2d_slice.start)
        tgt_slice = self.adapter.get_tgt_slice(type, layer_id)

        sources = self._c_source[src_slice]  # (n_src, B, S, d)
        if self.norm_matching == 'target' and self._c_base_source is not None:
            target_ratio = self._get_target_norm_ratio(type, layer_id)  # (B, S, 1)
            sources = sources - self._c_base_source[src_slice] * target_ratio

        # per_sample_scores[b, i, j] = dot(sources[i,b,:,:], grad[j,b,:,:]) over S*d  # [B, n_src, n_tgt]
        per_sample_scores = torch.einsum('ibsd,jbsd->bij', sources, grad).to(self.cache_device).detach()
        n_real = self._c_inputs['attention_mask'].sum(dim=-1).float().to(self.cache_device)  # [B]
        per_token_scores = per_sample_scores / n_real[:, None, None]                 # [B, n_src, n_tgt], μ_b

        # --- Running mean update (per-token normalised) ---
        batch_mean = per_token_scores.mean(dim=0)
        batch_mean_delta = batch_mean - self.circuit_scores[src_slice, tgt_slice]
        batch_weight = self._c_batch_size / self.num_processed_samples
        self.circuit_scores[src_slice, tgt_slice] += batch_weight * batch_mean_delta

        if self.variance_type == 'none':
            return

        # --- Variance/stat update ---
        if self.variance_type == 'within':
            # [B, S, n_src, n_tgt]: per-position attribution scores
            per_pos = torch.einsum('ibsd,jbsd->bsij', sources, grad).to(self.cache_device).detach()
            mask = self._c_inputs['attention_mask'].to(self.cache_device).float()            # [B, S]
            masked_mean = (per_pos * mask[:, :, None, None]).sum(dim=1) / n_real[:, None, None]  # [B, n_src, n_tgt]
            sq_dev = ((per_pos - masked_mean.unsqueeze(1)) * mask[:, :, None, None]) ** 2        # [B, S, n_src, n_tgt]
            within_var = sq_dev.sum(dim=1) / n_real[:, None, None]                               # [B, n_src, n_tgt]
            # Accumulate sum; final division by num_processed_samples gives E_b[Var_s[x_{b,s}]]
            self.circuit_scores_m2[src_slice, tgt_slice] += within_var.sum(dim=0)

        else:  # 'in_between': Var_b[μ_b], Welford running mean reuses circuit_scores
            # Chan's algorithm for batched M2 update: M2_total = M2_prev + M2_batch + delta² * n_prev*n_new / n_total
            batch_m2 = ((per_token_scores - batch_mean.unsqueeze(0)) ** 2).sum(dim=0)
            batch_m2_weight = (self.num_processed_samples - self._c_batch_size) * self._c_batch_size / self.num_processed_samples
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
        B, S = self._c_batch_size, self._c_seq_length

        def _ratio(clean: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
            # (B, S, 1)
            return (clean.detach().norm(dim=-1, keepdim=True)
                    / baseline.detach().norm(dim=-1, keepdim=True).clamp(min=1e-8))

        parts = []

        # Embedding slot (index 0)
        parts.append(_ratio(self._c_cache['-1.out'], self._c_base_cache['-1.out']).unsqueeze(0))  # (1, B, S, 1)

        for layer_id in range(self.adapter.n_layers):
            # Attention heads: residual at mid (after attn, before MLP)
            mid_ratio = _ratio(self._c_cache[f'{layer_id}.mid'], self._c_base_cache[f'{layer_id}.mid'])
            parts.append(mid_ratio.unsqueeze(0).expand(self.adapter.n_heads, B, S, 1))  # (n_heads, B, S, 1)

            # MLP: residual at out (after MLP)
            out_ratio = _ratio(self._c_cache[f'{layer_id}.out'], self._c_base_cache[f'{layer_id}.out'])
            parts.append(out_ratio.unsqueeze(0))  # (1, B, S, 1)

        return torch.cat(parts, dim=0)  # (source_dims, B, S, 1)
    

class NodeCircuitTracer:

    def _init_circuit_tensor(self) -> torch.Tensor:
        return torch.zeros(self.adapter.source_dims, 1, dtype=self.adapter.dtype, device=self.cache_device)

    def _forward_pass_and_cache(self, integrated_embeds: torch.Tensor | None = None):
        B, S, d = self._c_batch_size, self._c_seq_length, self.adapter.model_dim  # (B, S, d)
        cache, _, grad_cache = self._c_act_cache

        if integrated_embeds is not None:
            self.adapter.embedding_patch_hook(integrated_embeds)

        cache['emb'] = self.adapter.embed.output.detach()

        if self.adapter.uses_rotary_emb:
            cos, sin = self.adapter.rotary_emb.output
            cache['rotary_emb'] = (cos.detach(), sin.detach())

        cache['-1.out'] = self.adapter.embed_out_hook(detached=False)
        self._update_source_cache(
            self.adapter.embed_out_hook().reshape(1, B, S, d), type='emb')

        for layer_id, layer in enumerate(self.adapter.wrapped_layers):
            layer.attention_interface # needs to be included to access attn.source
            self._update_source_cache(
                layer.head_wise_attn_out_hook(), type='attn', layer_id=layer_id)
            cache[f'{layer_id}.mid'] = layer.residual_mid_hook()

            self._update_source_cache(
                layer.mlp_out_hook().reshape(1, B, S, d), type='mlp', layer_id=layer_id)
            cache[f'{layer_id}.out'] = layer.residual_out_hook()

        cache['logits'] = self.model.output['logits']

    def _backward_pass_and_scoring(self, metric: torch.Tensor):
        B, S, d = self._c_batch_size, self._c_seq_length, self.adapter.model_dim  # (B, S, d)
        cache, _, grad_cache = self._c_act_cache
        full_cache = grad_cache | cache

        with metric.backward():
            self._update_scores(
                self.adapter.logit_grad_hook(full_cache).reshape(1, B, S, d), type='lm_head')

            for layer in self.adapter.wrapped_layers[::-1]:
                self._update_scores(
                    layer.mlp_grad_hook(full_cache).reshape(1, B, S, d), type='mlp', layer_id=layer.layer_id)

                self._update_scores(
                    layer.head_wise_value_grad_hook(full_cache), type='attn_v', layer_id=layer.layer_id)

                self._update_scores(
                    layer.head_wise_key_grad_hook(full_cache), type='attn_k', layer_id=layer.layer_id)

                self._update_scores(
                    layer.head_wise_query_grad_hook(full_cache), type='attn_q', layer_id=layer.layer_id)


    def _get_target_norm_ratio(self, type: str, layer_id: int | None) -> torch.Tensor:
        """Norm ratio (clean/baseline) of the residual that target `type` reads from. Returns (B, S, 1)."""
        if type == 'lm_head':
            key = f'{self.adapter.n_layers - 1}.out'
        elif type == 'mlp':
            key = f'{layer_id}.mid'
        else:  # attn_q, attn_k, attn_v
            key = f'{layer_id - 1}.out' if layer_id > 0 else '-1.out'
        clean = self._c_cache[key]
        baseline = self._c_base_cache[key]
        return (clean.detach().norm(dim=-1, keepdim=True)
                / baseline.detach().norm(dim=-1, keepdim=True).clamp(min=1e-8))  # (B, S, 1)

    def _update_scores(self, grad: torch.Tensor, type: str, layer_id: int = None):
        # grad: [n_tgt, B, S, d]
        src2d_slice = self.adapter.get_src_slice(type, layer_id)
        src_slice = slice(None, src2d_slice.start)
        tgt_slice = self.adapter.get_tgt_slice(type, layer_id)

        sources = self._c_source[src_slice]  # (n_src, B, S, d)
        if self.norm_matching == 'target' and self._c_base_source is not None:
            target_ratio = self._get_target_norm_ratio(type, layer_id)  # (B, S, 1)
            sources = sources - self._c_base_source[src_slice] * target_ratio

        # per_sample_scores[b, i, j] = dot(sources[i,b,:,:], grad[j,b,:,:]) over S*d  # [B, n_src, n_tgt]
        per_sample_scores = torch.einsum('ibsd,jbsd->bij', sources, grad).to(self.cache_device).detach()
        n_real = self._c_inputs['attention_mask'].sum(dim=-1).float().to(self.cache_device)  # [B]
        per_token_scores = per_sample_scores / n_real[:, None, None]                 # [B, n_src, n_tgt], μ_b

        # --- Running mean update (per-token normalised) ---
        batch_mean = per_token_scores.mean(dim=0)
        batch_mean_delta = batch_mean - self.circuit_scores[src_slice, tgt_slice]
        batch_weight = self._c_batch_size / self.num_processed_samples
        self.circuit_scores[src_slice, tgt_slice] += batch_weight * batch_mean_delta

        if self.variance_type == 'none':
            return

        # --- Variance/stat update ---
        if self.variance_type == 'within':
            # [B, S, n_src, n_tgt]: per-position attribution scores
            per_pos = torch.einsum('ibsd,jbsd->bsij', sources, grad).to(self.cache_device).detach()
            mask = self._c_inputs['attention_mask'].to(self.cache_device).float()            # [B, S]
            masked_mean = (per_pos * mask[:, :, None, None]).sum(dim=1) / n_real[:, None, None]  # [B, n_src, n_tgt]
            sq_dev = ((per_pos - masked_mean.unsqueeze(1)) * mask[:, :, None, None]) ** 2        # [B, S, n_src, n_tgt]
            within_var = sq_dev.sum(dim=1) / n_real[:, None, None]                               # [B, n_src, n_tgt]
            # Accumulate sum; final division by num_processed_samples gives E_b[Var_s[x_{b,s}]]
            self.circuit_scores_m2[src_slice, tgt_slice] += within_var.sum(dim=0)

        else:  # 'in_between': Var_b[μ_b], Welford running mean reuses circuit_scores
            # Chan's algorithm for batched M2 update: M2_total = M2_prev + M2_batch + delta² * n_prev*n_new / n_total
            batch_m2 = ((per_token_scores - batch_mean.unsqueeze(0)) ** 2).sum(dim=0)
            batch_m2_weight = (self.num_processed_samples - self._c_batch_size) * self._c_batch_size / self.num_processed_samples
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
        B, S = self._c_batch_size, self._c_seq_length

        def _ratio(clean: torch.Tensor, baseline: torch.Tensor) -> torch.Tensor:
            # (B, S, 1)
            return (clean.detach().norm(dim=-1, keepdim=True)
                    / baseline.detach().norm(dim=-1, keepdim=True).clamp(min=1e-8))

        parts = []

        # Embedding slot (index 0)
        parts.append(_ratio(self._c_cache['-1.out'], self._c_base_cache['-1.out']).unsqueeze(0))  # (1, B, S, 1)

        for layer_id in range(self.adapter.n_layers):
            # Attention heads: residual at mid (after attn, before MLP)
            mid_ratio = _ratio(self._c_cache[f'{layer_id}.mid'], self._c_base_cache[f'{layer_id}.mid'])
            parts.append(mid_ratio.unsqueeze(0).expand(self.adapter.n_heads, B, S, 1))  # (n_heads, B, S, 1)

            # MLP: residual at out (after MLP)
            out_ratio = _ratio(self._c_cache[f'{layer_id}.out'], self._c_base_cache[f'{layer_id}.out'])
            parts.append(out_ratio.unsqueeze(0))  # (1, B, S, 1)

        return torch.cat(parts, dim=0)  # (source_dims, B, S, 1)
