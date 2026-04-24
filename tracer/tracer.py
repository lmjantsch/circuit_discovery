from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

from tracer.model_adapters import ModelAdapter

class EdgeCircuitTracer:

    def __init__(self, adapter: ModelAdapter, tokenizer: AutoTokenizer, cache_device: torch.device | None = None):
        self.adapter = adapter
        self.model = adapter.model
        self.tokenizer = tokenizer

        self.cache_device = cache_device
        if not cache_device:
            self.cache_device = adapter.device
        
        self.circuit_scores = None
        self.num_processed_samples = 0

        self.cache = {}
        self.source_2d_cache = None
        self.baseline_2d_cache = None

        self.curr_batch_size = None
        self.curr_seq_len = None

    @property
    def BSD(self):
        return self.curr_batch_size * self.curr_seq_len * self.adapter.model_dim
    
    def _init_source_cache(self) -> torch.Tensor:
        return torch.zeros(self.adapter.source_dims, self.BSD, device=self.adapter.device, dtype=self.adapter.dtype)
    
    def build_circuit(self, dataloader: DataLoader, use_counterfactual: bool = False):
        self.circuit_scores = torch.zeros(self.adapter.source_dims, self.adapter.grad_dims, dtype=self.adapter.dtype, device=self.cache_device)
        self.num_processed_samples = 0

        try: 
            for batch in tqdm(dataloader):
                clean_prompts, corrupt_prompts, clean_targets, corrupt_targets = batch

                clean_inputs = self.tokenizer(clean_prompts, padding=True, return_tensors='pt')
                corrupt_inputs = self.tokenizer(corrupt_prompts, padding=True, return_tensors='pt')

                self.curr_batch_size, self.curr_seq_len = clean_inputs['input_ids'].shape
                self.num_processed_samples += self.curr_batch_size

                with self.manage_batch_context(), self.model.session():

                    if use_counterfactual == True:
                        with self.model.trace(**corrupt_inputs):
                            self._forward_pass_and_cache()
                        self.baseline_2d_cache = self.source_2d_cache
                        self.source_2d_cache = self._init_source_cache()
                        

                    with self.model.trace(**clean_inputs):
                        self._forward_pass_and_cache()
                        metric = self._get_metric(clean_targets, corrupt_targets)

                        if use_counterfactual == True:
                            self.source_2d_cache -= self.baseline_2d_cache
                            self.baseline_2d_cache = None

                        self._backward_pass_and_scoring(metric)


        finally:
            circuit_scores = self.circuit_scores
            self.circuit_scores = None
            self.num_processed_samples = None

        return circuit_scores
    
    @contextmanager
    def manage_batch_context(self):
        self.cache = {i:{} for i in range(-1, self.adapter.n_layers)}
        self.source_2d_cache = self._init_source_cache()
        try:
            yield
        except Exception as e:
            print(e)
        finally:
            self.cache = {}
            self.source_2d_cache = None
            self.baseline_2d_cache = None # also clean up any allocated baseline cache
 

    def _forward_pass_and_cache(self):

        if self.adapter.uses_rotary_emb:
            rot_cos, rot_sin = self.adapter.rotary_emb.output
            self.cache['rotary_emb'] = (rot_cos.detach(), rot_sin.detach())


        # use first layer input to aggregate gpt2 embeddings
        first_layer = self.adapter.layers[0]
        self.cache[-1]['out'] = self.adapter.ln_1(first_layer).input
        self._update_source_2d_cache(
            self.adapter.ln_1(first_layer).input.reshape(1, self.BSD), type='emb'
        )


        for layer_id, layer in enumerate(self.adapter.layers):

            self.adapter.build_attn_source(layer) # call to build source
            self.cache[layer_id]['q_proj_out'] = self.adapter.q_proj_out(layer).output
            self.cache[layer_id]['k_proj_out'] = self.adapter.k_proj_out(layer).output
            self.cache[layer_id]['v_proj_out'] = self.adapter.v_proj_out(layer).output

            attn_out = self.adapter.compute_per_head_attn_out(
                self.adapter.attn_interface_out(layer).output[0].detach(), self.adapter.o_proj_weights(layer)
            ).reshape(self.adapter.n_heads, self.BSD)
            self._update_source_2d_cache(attn_out, type='attn', layer_id=layer_id)

            self.cache[layer_id]['mid'] = self.adapter.ln_2(layer).input

            self.cache[layer_id]['mlp_in'] = self.adapter.mlp(layer).input
            self._update_source_2d_cache(self.adapter.mlp(layer).output.reshape(1, self.BSD), type='mlp', layer_id=layer_id)

            self.cache[layer_id]['out'] = layer.output
        
        self.cache['logits'] = self.adapter.lm_head.output

    def _update_source_2d_cache(self, new_tensor: torch.Tensor, type: str, layer_id: int = None):
        source_slice = self.adapter.get_src_slice(type=type, layer_id = layer_id)
        self.source_2d_cache[source_slice] += new_tensor.detach()

    
    def _get_metric(self, clean_targets, corrupt_targets):
        clean_logits = self.cache['logits'][range(len(clean_targets)), -1, clean_targets]
        corrupt_logits = self.cache['logits'][range(len(corrupt_targets)), -1, corrupt_targets]
        
        metric = (clean_logits - corrupt_logits).sum()
        return metric
    
    def _backward_pass_and_scoring(self, metric: torch.Tensor):
        with metric.backward():
            self._update_scores(self.cache[self.adapter.n_layers - 1]['out'].grad.detach().reshape(1, self.BSD), type='lm_head')

            for layer_id in range(self.adapter.n_layers - 1, -1, -1):
                layer = self.adapter.layers[layer_id]
                layer_input = self.cache[layer_id - 1]['out'].detach()

                mlp_in_grad_pre_norm = self.adapter.compute_mlp_input_gradient(
                    grad=self.cache[layer_id]['mlp_in'].grad.detach(),
                    x_pre_norm=self.cache[layer_id]['mid'].detach(),
                    layer=layer
                ).reshape(1, self.BSD)
                self._update_scores(mlp_in_grad_pre_norm, type='mlp', layer_id=layer_id)
                del mlp_in_grad_pre_norm

                v_proj_in_grad_pre_norm = self.adapter.compute_headwise_value_input_gradient(
                    grad=self.cache[layer_id]['v_proj_out'].grad.detach(),
                    x_pre_norm=layer_input, layer=layer, cache=self.cache
                ).reshape(self.adapter.n_heads, self.BSD)
                self._update_scores(v_proj_in_grad_pre_norm, type='attn_v', layer_id=layer_id)
                del v_proj_in_grad_pre_norm

                k_proj_in_grad_pre_norm = self.adapter.compute_headwise_key_input_gradient(
                    grad=self.cache[layer_id]['k_proj_out'].grad.detach(),
                    x_pre_norm=layer_input, layer=layer, cache=self.cache
                ).reshape(self.adapter.n_heads, self.BSD)
                self._update_scores(k_proj_in_grad_pre_norm, type='attn_k', layer_id=layer_id)
                del k_proj_in_grad_pre_norm

                q_proj_in_grad_pre_norm = self.adapter.compute_headwise_query_input_gradient(
                      grad=self.cache[layer_id]['q_proj_out'].grad.detach(),
                    x_pre_norm=layer_input, layer=layer, cache=self.cache
                ).reshape(self.adapter.n_heads, self.BSD)
                self._update_scores(q_proj_in_grad_pre_norm, type='attn_q', layer_id=layer_id)
                del q_proj_in_grad_pre_norm
    
    def _update_scores(self, grad: torch.Tensor, type: str, layer_id: int = None):
        src2d_slice = self.adapter.get_src_slice(type, layer_id)
        src_slice = slice(None, src2d_slice.start)
        tgt_slice = self.adapter.get_tgt_slice(type, layer_id)
        
        new_score = torch.matmul(self.source_2d_cache[src_slice], grad.T) 
        new_score_mean = new_score / self.curr_batch_size

        new_score_weight = self.curr_batch_size / self.num_processed_samples
        new_score_delta = (new_score_mean.to(self.cache_device).detach() - self.circuit_scores[src_slice, tgt_slice])
        self.circuit_scores[src_slice, tgt_slice] += new_score_weight * new_score_delta
