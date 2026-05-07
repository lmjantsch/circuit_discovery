from contextlib import contextmanager
from abc import ABC

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from tracer.model_adapters import ModelAdapter

class EdgeCircuitTracer:

    def __init__(self, adapter: ModelAdapter, tokenizer: AutoTokenizer, cache_device: torch.device | None = None, return_variance: bool = True, 
                 q_weight: float = 1.0, k_weight: float = 1.0, v_weight: float = 1.0, gate_weight: float = 1.0, up_weight: float = 1.0, scale_loc: str = 'post'):
        self.adapter = adapter
        self.model = adapter.model
        self.tokenizer = tokenizer

        self.cache_device = cache_device
        if not cache_device:
            self.cache_device = adapter.device

        self.return_variance = return_variance

        self.q_weight = q_weight
        self.k_weight = k_weight
        self.v_weight = v_weight
        self.gate_weight = gate_weight
        self.up_weight = up_weight
        self.scale_loc = scale_loc
        
        self.circuit_scores = None
        self.circuit_scores_m2 = None
        self.num_processed_samples = 0

        self.cache = {}
        self.source_2d_cache = None
        self.baseline_2d_cache = None
        self.disable_source_caching = False

        self.clean_embeds = None
        self.corrupt_embeds = None

        self.curr_batch_size = None
        self.curr_seq_len = None


    @property
    def BSD(self):
        return self.curr_batch_size * self.curr_seq_len * self.adapter.model_dim
    
    def _init_circuit_tensor(self) -> torch.Tensor:
        return torch.zeros(self.adapter.source_dims, self.adapter.grad_dims, dtype=self.adapter.dtype, device=self.cache_device)
    
    def _init_source_cache(self) -> torch.Tensor:
        return torch.zeros(self.adapter.source_dims, self.BSD, device=self.adapter.device, dtype=self.adapter.dtype)
    
    def build_circuit(self, dataloader: DataLoader, use_counterfactual: bool = False, integration_steps: int = 1):
        self.circuit_scores = self._init_circuit_tensor()
        if self.return_variance:
            self.circuit_scores_m2 = self._init_circuit_tensor()
        self.num_processed_samples = 0
        
        try: 
            output = self._build_circuit_inner(
                dataloader=dataloader, 
                use_counterfactual=use_counterfactual,
                integration_steps= integration_steps)
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
            target_idx = torch.full((self.curr_batch_size,), self.curr_seq_len - 1)
            if self.tokenizer.padding_side == 'right':
                target_idx = clean_inputs['attention_mask'].sum(dim=-1) - 1

            with self.manage_batch_context(), self.model.session():

                if use_counterfactual == True or integration_steps > 1:
                    with self.model.trace(**corrupt_inputs):
                        self._forward_pass_and_cache()
                    self.corrupt_embeds = self.cache['emb'].detach().clone()
                    self.baseline_2d_cache = self.source_2d_cache
                    self.source_2d_cache = self._init_source_cache()
                    

                with self.model.trace(**clean_inputs):
                    self._forward_pass_and_cache()
                    self.clean_embeds = self.cache['emb'].detach().clone()
                    metric = self._get_metric(clean_targets, corrupt_targets, target_idx, use_counterfactual)

                    if use_counterfactual == True or integration_steps > 1:
                        self.source_2d_cache -= self.baseline_2d_cache
                        self.baseline_2d_cache = None

                    self.num_processed_samples += self.curr_batch_size
                    self._backward_pass_and_scoring(metric)

                if integration_steps > 1:
                    self.disable_source_caching = True
                    for alpha in torch.linspace(0, 1, integration_steps + 1)[1:-1]: # k / integration_steps for k in range(integration_steps)
                        with self.model.trace(**clean_inputs):
                            integrated_embeds = (1 - alpha) * self.clean_embeds + alpha * self.corrupt_embeds
                            self._forward_pass_and_cache(integrated_embeds)
                            metric = self._get_metric(clean_targets, corrupt_targets, target_idx, use_counterfactual)
                            self.num_processed_samples += self.curr_batch_size
                            self._backward_pass_and_scoring(metric)
        
        if self.return_variance:
            variance = self.circuit_scores_m2 / self.num_processed_samples
            return (self.circuit_scores, variance)
        
        return (self.circuit_scores, None)
    
    @contextmanager
    def manage_batch_context(self):
        self.cache = {i:{} for i in range(-1, self.adapter.n_layers)}
        self.source_2d_cache = self._init_source_cache()
        try:
            yield
        except Exception as e:
            print(e)
        finally:
            self.empty_cache()
            self.disable_source_caching = False
 
    def empty_cache(self):
        self.cache = {}
        self.source_2d_cache = None
        self.baseline_2d_cache = None

    def _forward_pass_and_cache(self, integrated_embeds: torch.Tensor | None = None):

        if integrated_embeds is not None:
            self.adapter.emb.output = integrated_embeds

        self.cache['emb'] = self.adapter.emb_out().detach()

        if self.adapter.uses_rotary_emb:
            self.cache['rotary_emb'] = self.adapter.rotary_emb_out()

        first_layer = self.adapter.layers[0]
        self.cache[-1]['out'] = self.adapter.residual_in(first_layer).detach()
        self._update_source_2d_cache(
            self.adapter.residual_in(first_layer).detach().reshape(1, self.BSD), type='emb'
        )

        for layer_id, layer in enumerate(self.adapter.layers):

            self.adapter.build_attn_source(layer) # call to build source
            self.cache[layer_id]['query_out'] = self.adapter.query_out(layer)
            self.cache[layer_id]['key_out'] = self.adapter.key_out(layer)
            self.cache[layer_id]['value_out'] = self.adapter.value_out(layer)

            self._update_source_2d_cache(
                self.adapter.per_head_attn_out(layer).detach().reshape(self.adapter.n_heads, self.BSD), 
                type='attn', layer_id=layer_id)

            self.cache[layer_id]['mid'] = self.adapter.residual_mid(layer).detach()

            self.cache[layer_id]['gate_out'] = self.adapter.gate_out(layer)
            self.cache[layer_id]['up_out'] = self.adapter.up_out(layer)
            self._update_source_2d_cache(
                self.adapter.mlp_out(layer).detach().reshape(1, self.BSD), type='mlp', layer_id=layer_id)

            self.cache[layer_id]['out'] = self.adapter.residual_out(layer)
        
        self.cache['logits'] = self.adapter.logits()

    def _update_source_2d_cache(self, new_tensor: torch.Tensor, type: str, layer_id: int = None):
        if self.disable_source_caching == False:
            source_slice = self.adapter.get_src_slice(type=type, layer_id = layer_id)
            self.source_2d_cache[source_slice] = new_tensor.detach()
    
    def _get_metric(self, clean_targets, corrupt_targets, target_idx, use_counterfactual):
        clean_logits = self.cache['logits'][range(len(clean_targets)), target_idx, clean_targets]
        corrupt_logits = self.cache['logits'][range(len(corrupt_targets)), target_idx, corrupt_targets]
        
        if use_counterfactual:
            return (clean_logits - corrupt_logits).sum()
        
        return clean_logits.sum()
    
    def _backward_pass_and_scoring(self, metric: torch.Tensor):
        with metric.backward():
            logit_grad = self.cache[self.adapter.n_layers - 1]['out'].grad.detach().reshape(1, self.BSD)
            self._update_scores(logit_grad, type='lm_head')
            del logit_grad

            for layer_id in range(self.adapter.n_layers - 1, -1, -1):
                layer = self.adapter.layers[layer_id]
                layer_input = self.cache[layer_id - 1]['out'].detach()

                up_grad = self._scale_and_detach_grad(self.cache[layer_id]['up_out'], self.up_weight, scale_loc=self.scale_loc)
                gate_grad = self._scale_and_detach_grad(self.cache[layer_id]['gate_out'], self.gate_weight, scale_loc=self.scale_loc)

                mlp_in_grad_pre_norm = self.adapter.mlp_in_grad(
                    gate_grad=gate_grad, up_grad=up_grad,
                    x_pre_norm=self.cache[layer_id]['mid'],
                    layer=layer
                ).reshape(1, self.BSD)
                self._update_scores(mlp_in_grad_pre_norm, type='mlp', layer_id=layer_id)
                del mlp_in_grad_pre_norm
                

                v_grad = self._scale_and_detach_grad(self.cache[layer_id]['value_out'], self.v_weight, scale_loc=self.scale_loc)

                v_proj_in_grad_pre_norm = self.adapter.value_in_grad(
                    grad=v_grad, x_pre_norm=layer_input, layer=layer, cache=self.cache
                ).reshape(self.adapter.n_heads, self.BSD)
                self._update_scores(v_proj_in_grad_pre_norm, type='attn_v', layer_id=layer_id)
                del v_proj_in_grad_pre_norm

                
                k_grad = self._scale_and_detach_grad(self.cache[layer_id]['key_out'], self.k_weight, scale_loc=self.scale_loc)
                
                k_proj_in_grad_pre_norm = self.adapter.key_in_grad(
                    grad=k_grad, x_pre_norm=layer_input, layer=layer, cache=self.cache
                ).reshape(self.adapter.n_heads, self.BSD)
                self._update_scores(k_proj_in_grad_pre_norm, type='attn_k', layer_id=layer_id)
                del k_proj_in_grad_pre_norm


                q_grad = self._scale_and_detach_grad(self.cache[layer_id]['query_out'], self.q_weight, scale_loc=self.scale_loc)

                q_proj_in_grad_pre_norm = self.adapter.query_in_grad(
                    grad=q_grad, x_pre_norm=layer_input, layer=layer, cache=self.cache
                ).reshape(self.adapter.n_heads, self.BSD)
                self._update_scores(q_proj_in_grad_pre_norm, type='attn_q', layer_id=layer_id)
                del q_proj_in_grad_pre_norm
    
    def _update_scores(self, grad: torch.Tensor, type: str, layer_id: int = None):
        src2d_slice = self.adapter.get_src_slice(type, layer_id)
        src_slice = slice(None, src2d_slice.start)
        tgt_slice = self.adapter.get_tgt_slice(type, layer_id)
        
        batch_scores = torch.matmul(self.source_2d_cache[src_slice], grad.T).to(self.cache_device).detach()
        batch_scores_mean = (batch_scores / self.curr_batch_size)

        batch_scores_mean_delta = (batch_scores_mean - self.circuit_scores[src_slice, tgt_slice])
        batch_scores_weight = self.curr_batch_size / self.num_processed_samples

        self.circuit_scores[src_slice, tgt_slice] += batch_scores_weight * batch_scores_mean_delta
        
        if self.return_variance:
            SD = self.curr_seq_len * self.adapter.model_dim
            sources = self.source_2d_cache[src_slice].reshape(-1, self.curr_batch_size, SD)  # [n_src, B, S*d]
            grad_3d = grad.reshape(-1, self.curr_batch_size, SD)                             # [n_tgt, B, S*d]

            # per_sample_scores[b, i, j] = dot(sources[i,b,:], grad[j,b,:]) over S*d  # [B, n_src, n_tgt]
            per_sample_scores = torch.einsum('ibk,jbk->bij', sources, grad_3d).to(self.cache_device).detach()
            batch_m2 = ((per_sample_scores - batch_scores_mean.unsqueeze(0)) ** 2).sum(dim=0)

            # Chan's algorithm for batched M2 update: M2_total = M2_prev + M2_batch + delta² * n_prev*n_new / n_total
            batch_m2_weight = (self.num_processed_samples - self.curr_batch_size) * self.curr_batch_size / self.num_processed_samples
            self.circuit_scores_m2[src_slice, tgt_slice] += batch_m2 + (batch_scores_mean_delta ** 2) * batch_m2_weight

    def _scale_and_detach_grad(self, grad_tensor: torch.Tensor, weight: float, scale_loc: str = 'post'):
        if grad_tensor == None: # handle GPT2 missing gate tensor
            return None

        if scale_loc == 'pre':
            grad_tensor.grad = weight * grad_tensor.grad

        detached_grad = grad_tensor.grad.detach()

        if scale_loc == 'post':
            grad_tensor.grad = weight * grad_tensor.grad
            
        return detached_grad
        
# # ioi, mcqa / qwen, gemma
# # ioi gpt
# # eap+FrozenNorm+Scaling
# 0.0 0.05 0.1 0.2 0.5 1.0 1.5 3.0
# 0.0 0.05 0.1 0.2 0.5 1.0 1.5 3.0
# 0.0 0.05 0.1 0.2 0.5 1.0 1.5 3.0
# 0.0 0.05 0.1 0.2 0.5 1.0 1.5 3.0
# 0.0 0.05 0.1 0.2 0.5 1.0 1.5 3.0
