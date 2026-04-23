from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader

from transformers import AutoTokenizer
from nnsight import NNsight

from backend import compute_headwise_input_gradients, compute_rmsnorm_input_gradients
from modeling_utils import per_head_attn_out

class CircuitTracer:

    def __init__(self, model: NNsight, tokenizer: AutoTokenizer, batch_size: int, cache_device: torch.device | None = None):
        self.model = model
        self.compute_device = model.device
        self.dtype = model.dtype

        self.tokenizer = tokenizer

        self.batch_size = batch_size

        self.cache_device = cache_device
        if not cache_device:
            self.cache_device = model.device

        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.head_dim = model.config.head_dim
        self.model_dim = model.config.hidden_size

        self.source_dims = (1 + self.n_layers * (self.n_heads + 1)) # emb + layer * (heads_out + mlp_out)
        self.grad_dims = (self.n_layers * (3 * self.n_heads + 1) + 1) # layer * (heads_in + mlp_in) + lm_head_in
        self.circuit_scores = torch.zeros(self.source_dims, self.grad_dims, device=self.cache_device, dtype=model.dtype)

        self.cache = {}
        self.source_2d_cache = None

        self.curr_batch_size = None
        self.curr_seq_len = None
    

    def __call__(self, batch: dict):
        clean_prompt, _, clean_targets, corrupt_targes = batch
        inputs = self.tokenizer(clean_prompt, padding=True, return_tensors='pt')

        with self.trace(**inputs):
            self._forward_pass_and_cache()

            metric = self._get_metric(clean_targets, corrupt_targes)

            self._backward_pass_and_scoring(metric)

    @property
    def BSD(self) -> int | None:
        """Agggregate batch, seq_len and model_dim dimensions"""
        if not self.curr_batch_size or self.curr_seq_len:
            raise Warning('BSD can only be provided during tracing')
        return self.curr_batch_size * self.curr_seq_len * self.model_dim
    

    def _trace(self, **kwargs):
        self.curr_batch_size, self.curr_seq_len = kwargs['input_ids'].shape
        self.cache = {i:{} for i in range(-1, self.n_layers)}
        self.source_2d_cache = torch.zeros(self.source_dims, self.BSD, device=self.compute_device, dtype=self.dtype)
        try:
            with self.model.trace(**kwargs) as tracer:
                yield tracer
        finally:
            self.cache = {}
            self.source_2d_cache = None
    

    def _forward_pass_and_cache(self):
        self.cache[-1]['out'] = self.model.model.embed_tokens.output
        self._update_source_2d_cache(self.model.model.embed_tokens.output.reshape(1, self.BSD), type='emb')

        rot_cos, rot_sin = self.model.model.rotary_emb.output
        self.cache['rot_embeds'] = (rot_cos.detach(), rot_sin.detach())

        for layer_id, layer in enumerate(self.model.model.layers):
            layer.self_attn.source # call to build source
            self.cache[layer_id]['q_proj_out'] = layer.self_attn.q_proj.output
            self.cache[layer_id]['k_proj_out'] = layer.self_attn.source.attention_interface_0.source.repeat_kv_0.output
            self.cache[layer_id]['v_proj_out'] = layer.self_attn.source.attention_interface_0.source.repeat_kv_1.output

            attn_out = per_head_attn_out(layer.self_attn.source.attention_interface_0.output[0].detach(),
                layer.self_attn.o_proj.weight.data)
            self._update_source_2d_cache(attn_out.reshape(self.n_heads, self.BSD), type='attn', layer_id=layer_id)

            self.cache[layer_id]['mid'] = layer.post_attention_layernorm.input

            self.cache[layer_id]['mlp_in'] = layer.mlp.input
            self._update_source_2d_cache(layer.mlp.output.reshape(1, self.BSD), type='mlp', layer=layer_id)

            self.cache[layer_id]['out'] = layer.output
        
        self.cache['logits'] = self.model.lm_head.output

    def _update_source_2d_cache(self, new_tensor: torch.Tensor, type: str, layer_id: int = None):
        source_slice = self._get_source_2d_cache_slice(type=type, layer_id = layer_id)
        self.source_2d_cache[source_slice] = new_tensor.detach()

    def _get_source_2d_cache_slice(self, type: str, layer_id: int | None) -> slice:
        if type == 'emb':
            return slice(0, 1)
        if type == 'attn':
            start = 1 + layer_id * (self.n_heads + 1)
            return slice(start, start + self.n_heads)
        if type == 'mlp':
            start = 1 + layer_id * (self.n_heads + 1) + self.n_heads
            return slice(start, start + 1)
        if type == 'lm_head':
            start = 1 + self.n_layers * (self.n_heads + 1) + self.n_heads
            return slice(start, None)
        raise NotImplementedError
    

    def _get_metric(self, clean_targets, corrupt_targes):
        clean_logits = self.cache['logits'][range(len(clean_targets)), -1, clean_targets]
        corrupt_logits = self.cache['logits'][range(len(corrupt_targes)), -1, corrupt_targes]
        
        metric = (clean_logits - corrupt_logits).sum()
        return metric
    
    def _backward_pass_and_scoring(self, metric: torch.Tensor):
        with metric.backward():
            self._update_scores(self.cache[self.n_layers - 1]['out'].grad.detach().reshape(1, self.BSD), type='lm_head')

            for layer_id, layer in enumerate(self.model.model.layers[::-1]):
                layer_input = self.cache[layer_id - 1]['out'].detach()

                mlp_in_grad_pre_norm = compute_rmsnorm_input_gradients(
                    grad=self.cache[layer_id]['mlp_in'].grad.detach(),
                    x_pre_norm=self.cache[layer_id]['mid'].detach(),
                    rmsnorm_weight=layer.post_attention_layernorm.weight.data
                ).reshape(1, self.BSD)
                self._update_scores(mlp_in_grad_pre_norm, type='mlp', layer=layer_id)
                del mlp_in_grad_pre_norm

                v_proj_in_grad_pre_norm = compute_headwise_input_gradients(
                    grad_heads=self.cache[layer_id]['v_proj_out'].grad.detach(),
                    W_linear=layer.self_attn.v_proj.weight.data,
                    x_pre_norm=layer_input,
                    rmsnorm_weight=layer.input_layernorm.weight.data,
                ).reshape(self.n_heads, self.BSD)
                self._update_scores(v_proj_in_grad_pre_norm, type='attn_v', layer=layer_id)
                del v_proj_in_grad_pre_norm

                k_proj_in_grad_pre_norm = compute_headwise_input_gradients(
                    grad_heads=self.cache[layer_id]['k_proj_out'].grad.detach(),
                    W_linear=layer.self_attn.k_proj.weight.data,
                    rot_embeds=self.cache['rot_embeds'],
                    x_pre_norm=layer_input,
                    rmsnorm_weight=layer.input_layernorm.weight.data,
                ).reshape(self.n_heads, self.BSD)
                self._update_scores(k_proj_in_grad_pre_norm, type='attn_k', layer=layer_id)
                del k_proj_in_grad_pre_norm

                q_proj_in_grad_pre_norm = compute_headwise_input_gradients( # does not require inverse rot_embeds as cached at linear out
                    grad_heads=self.cache[layer_id]['q_proj_out'].grad.detach()\
                        .reshape(self.curr_batch_size, self.curr_seq_len, self.n_heads, self.head_dim).transpose(1, 2),
                    W_linear=layer.self_attn.q_proj.weight.data,
                    x_pre_norm=layer_input,
                    rmsnorm_weight=layer.input_layernorm.weight.data,
                ).reshape(self.n_heads, self.BSD)
                self._update_scores(q_proj_in_grad_pre_norm, type='attn_q', layer=layer_id)
                del q_proj_in_grad_pre_norm

    def _update_scores(self, grad: torch.Tensor, type: str, layer_id: int = None):
        src2d_slice = self._get_source_2d_cache_slice(type, layer_id)
        src_slice = slice(None, src2d_slice.start)

        tgt_slice = self._get_score_tgt_slice(type, layer_id)
        
        weight = self.curr_batch_size / self.batch_size**2
        new_scores = weight * torch.matmul(self.source_2d_cache[src_slice], grad.T)
        self.circuit_scores[src_slice, tgt_slice] += new_scores.to(self.cache_device).detach()

    def _get_score_tgt_slice(self, type: str, layer_id: int | None) -> slice:
        if type == 'lm_head':
            return slice(-1,)
        elif type == 'mlp':
            start = layer_id * (3 * self.n_heads + 1) + 3 * self.n_heads
            return slice(start, start + 1)
        elif type == 'attn_v':
            start = layer_id * (3 * self.n_heads + 1) + 2 * self.n_heads
            return slice(start, start + self.n_heads)
        elif type == 'attn_k':
            start = layer_id * (3 * self.n_heads + 1) + self.n_heads
            return slice(start, start + self.n_heads)
        elif type == 'attn_q':
            start = layer_id * (3 * self.n_heads + 1)
            return slice(start, start + self.n_heads)
        else:
            raise NotImplementedError(f"Type '{type}' is not implemented.")