import os
import sys

os.environ['CUDA_VISIBLE_DEVICES']='0'
proj_path = '/home/dacslab/lasse_jantsch/circuit_discovery'
if proj_path not in sys.path:
    sys.path.insert(0, proj_path)

from contextlib import contextmanager

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm


from modular_transformer import patch_model
from modular_transformer.models import GPT2_ARC, LLAMA2_ARC, GEMMA2_ARC
from src.adapters import GPT2ModelAdapter, Gemma2ModelAdapter, Llama2ModelAdapter, ModelAdapter


MIB_MODEL_TO_HF_ID: dict[str, str] = {
    "gpt2": "gpt2",
    "qwen2.5": "Qwen/Qwen2.5-0.5B",
    "llama3": "meta-llama/Llama-3.1-8B",
    "gemma2": "google/gemma-2-2b",
}

MIB_MODEL_TO_ADAPTER_CLS: dict[str, type | None] = {
    "gpt2": GPT2ModelAdapter,
    "qwen2.5": Llama2ModelAdapter,
    "llama3": Llama2ModelAdapter,
    "gemma2": Gemma2ModelAdapter,
}

MIB_MODEL_TO_ARC: dict[str, type | None] = {
    "gpt2": GPT2_ARC,
    "qwen2.5": LLAMA2_ARC,
    "llama3": LLAMA2_ARC,
    "gemma2": GEMMA2_ARC,
}

PERCENTAGES = (.001, .002, .005, .01, .02, .05, .1, .2, .5, 1)


class EdgeCircuitPatcher:

    def __init__(self, adapter: ModelAdapter, tokenizer: AutoTokenizer, cache_device: torch.device | None = None, norm_matching: bool = False):
        self.adapter = adapter
        self.model = adapter.model
        self.tokenizer = tokenizer

        self.cache_device = cache_device
        if not cache_device:
            self.cache_device = adapter.device

        self.norm_matching = norm_matching

        self.faithfulness_agg = None

        self._empty_cache()

    def _empty_cache(self):
        self._c_base_cache = {}
        self._c_base_source = None
        self._c_clean_cache = {}

        self._c_cache = {}
        self._c_source = None

        self._c_inputs = None
        self._c_base_inputs = None
        self._c_batch_size, self._c_seq_length = None, None
        self._c_target_idx = None
        self._c_base_target_idx = None

        self._c_clean_logit_diff = None
        self._c_base_logit_diff = None

    def _init_source_cache(self) -> torch.Tensor:
        return torch.zeros(
            self.adapter.source_dims, self._c_batch_size, self._c_seq_length, self.adapter.model_dim,
            device=self.adapter.device, dtype=self.adapter.dtype,
        )

    @contextmanager
    def circuit_context(self, circuit_scores: torch.Tensor):
        self.faithfulness_agg = [[] for _ in PERCENTAGES]
        self.valid_edge_mask = self.get_valid_edge_mask()
        self.n_edges = self.valid_edge_mask.sum()
        self.forward_to_backward = self.get_forward_to_backward()

        self.circuit_scores = circuit_scores
        self.circuit_scores[~self.valid_edge_mask] = torch.finfo(self.adapter.dtype).min
        self.sorted_score_idx = torch.argsort(self.circuit_scores.view(-1), descending=True)

        try:
            yield
        except Exception as e:
            raise(e)
        finally:
            self.faithfulness_agg = None
            self.valid_edge_mask = None
            self.n_edges = None
            self.forward_to_backward = None
            self.circuit_scores = None
            self.sorted_score_idx = None

    @contextmanager
    def batch_context(self, batch):
        prompts, base_prompts, target, base_target = batch

        self._c_inputs = self.tokenizer(prompts, padding=True, return_tensors='pt')
        self._c_base_inputs = self.tokenizer(base_prompts, padding=True, return_tensors='pt')

        B, S = self._c_inputs['input_ids'].shape
        self._c_batch_size, self._c_seq_length = B, S
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


    def patch_circuit(self, dataloader: DataLoader, circuit_scores):
        with self.circuit_context(circuit_scores):
            for batch in tqdm(dataloader):
                self._process_batch(batch)

            faithfulness = [sum(s) / len(s) for s in self.faithfulness_agg]
            weighted_edge_counts = [int(self.n_edges * p) for p in PERCENTAGES]

            return (faithfulness, PERCENTAGES, weighted_edge_counts)

    def _process_batch(self, batch):
        with self.batch_context(batch), self.model.session():
            
            with self.model.trace(**self._c_base_inputs):
                self._cache_base()
            
            with self.model.trace(**self._c_inputs):
                self._cache_clean()

            for p_id, percentage in enumerate(PERCENTAGES[:-1]): # exlude 1.0
                in_graph = self._get_in_graph(percentage)
                in_graph = in_graph
                with self.model.trace(**self._c_inputs):
                    patched_diff = self._cache_patched(in_graph)
                
                batch_score = (patched_diff - self._c_base_logit_diff) / (self._c_clean_logit_diff - self._c_base_logit_diff)
                self.faithfulness_agg[p_id].extend(batch_score.tolist())

            # set 1.0 for all 100% circuits
            self.faithfulness_agg[-1].extend([1 for _ in range(self._c_batch_size)])

    def _update_source_cache(self, new_tensor: torch.Tensor, source_cache: torch.Tensor, type: str, layer_id: int = None):
        source_slice = self.adapter.get_src_slice(type=type, layer_id=layer_id)
        source_cache[source_slice] = new_tensor.detach()

    def _get_logit_diff(self, circuit_logits: torch.Tensor):
        base_logits = circuit_logits[self._c_base_target_idx]
        logits = circuit_logits[self._c_target_idx]
        return logits - base_logits

    def _cache_base(self):
        B, S, d = self._c_batch_size, self._c_seq_length, self.adapter.model_dim

        self._update_source_cache(
            self.adapter.embed_out_hook(detached=False).reshape(1, B, S, d), self._c_base_source, type='emb'
        )
        for layer_id, layer in enumerate(self.adapter.wrapped_layers):
            self._c_base_cache[f"{layer_id}.in"] = layer.residual_in_hook()
            self._update_source_cache(
                layer.head_wise_attn_out_hook(), self._c_base_source, type='attn', layer_id=layer_id)
            
            self._c_base_cache[f"{layer_id}.mid"] = layer.residual_mid_hook()
            self._update_source_cache(
                layer.mlp_out_hook().reshape(1, B, S, d), self._c_base_source, type='mlp', layer_id=layer_id)
            
            self._c_base_cache[f"{layer_id}.out"] = layer.residual_out_hook()
        
        self._c_base_logit_diff = self._get_logit_diff(self.model.output['logits'])

    def _cache_clean(self):
        for layer_id, layer in enumerate(self.adapter.wrapped_layers):
            self._c_clean_cache[f"{layer_id}.in"] = layer.residual_in_hook()
            self._c_clean_cache[f"{layer_id}.mid"] = layer.residual_mid_hook()
            self._c_clean_cache[f"{layer_id}.out"] = layer.residual_out_hook()
        self._c_clean_logit_diff = self._get_logit_diff(self.model.output['logits'])

    def _cache_patched(self, in_graph: torch.Tensor):
        B, S, d = self._c_batch_size, self._c_seq_length, self.adapter.model_dim
        patch_cache = {}

        if self.adapter.uses_rotary_emb:
            cos, sin = self.adapter.rotary_emb.output
            patch_cache['rotary_emb'] = (cos.detach(), sin.detach())

        self._update_source_cache(
            self.adapter.embed_out_hook(detached=False).reshape(1, B, S, d), self._c_source, type='emb'
        )
        for layer_id, layer in enumerate(self.adapter.wrapped_layers):
            self._c_cache[f"{layer_id}.in"] = layer.residual_in_hook()
            in_state = layer.residual_in_hook().unsqueeze(0)
            if self.norm_matching:
                in_state = in_state * self.adapter.norm_scale(in_state)
            q_corr, k_corr, v_corr = self._compute_attn_corrections(in_graph, layer_id)
            layer.head_wise_query_patch_hook(in_state + q_corr, patch_cache, post_scaling=self.norm_matching)
            layer.head_wise_key_patch_hook(in_state + k_corr, patch_cache, post_scaling=self.norm_matching)
            layer.head_wise_value_patch_hook(in_state + v_corr, patch_cache, post_scaling=self.norm_matching)
            self._update_source_cache(
                layer.head_wise_attn_out_hook(), self._c_source, type='attn', layer_id=layer_id)

            self._c_cache[f"{layer_id}.mid"] = layer.residual_mid_hook()
            mid_state = layer.residual_mid_hook()
            if self.norm_matching:
                mid_state = mid_state * self.adapter.norm_scale(mid_state)
            layer.mlp_patch_hook(
                mid_state + self._compute_patched_residual(in_graph, 'mlp', layer_id)[0], post_scaling= self.norm_matching
            )
            self._update_source_cache(
                layer.mlp_out_hook().reshape(1, B, S, d), self._c_source, type='mlp', layer_id=layer_id)
            
            self._c_cache[f"{layer_id}.out"] = layer.residual_out_hook()
            out_state = layer.residual_out_hook()
            if self.norm_matching:
                out_state = out_state * self.adapter.norm_scale(out_state)
            
        self.adapter.logit_patch_hook(
            out_state + self._compute_patched_residual(in_graph, 'lm_head', None)[0], post_scaling= self.norm_matching
        )
        
        return self._get_logit_diff(self.model.output['logits'])

    def get_valid_edge_mask(self):
        mask = torch.zeros(self.adapter.source_dims, self.adapter.grad_dims).bool()
        mask[0] = True # -> emb
        for i in range(self.adapter.n_layers):
            attn_src = self.adapter.get_src_slice(type='attn', layer_id=i)
            attn_tgt = self.adapter.get_tgt_slice(type='attn_v', layer_id=i)
            mask[attn_src, attn_tgt.stop:] = True

            mlp_src = self.adapter.get_src_slice(type='mlp', layer_id=i)
            mlp_tgt = self.adapter.get_tgt_slice(type='mlp', layer_id=i)
            mask[mlp_src, mlp_tgt.stop:] = True
        
        return mask

    def get_forward_to_backward(self):
        forward_to_backward = torch.zeros(self.adapter.source_dims, self.adapter.grad_dims).bool()
        for i in range(self.adapter.n_layers):
            attn_src = self.adapter.get_src_slice(type='attn', layer_id=i)
            for qkv_type in ('attn_q', 'attn_k', 'attn_v'):
                attn_tgt = self.adapter.get_tgt_slice(type=qkv_type, layer_id=i)
                forward_to_backward[attn_src, attn_tgt] = True

            mlp_src = self.adapter.get_src_slice(type='mlp', layer_id=i)
            mlp_tgt = self.adapter.get_tgt_slice(type='mlp', layer_id=i)
            forward_to_backward[mlp_src, mlp_tgt] = True
        
        return forward_to_backward
    
    def _prune(self, in_graph: torch.Tensor) -> torch.Tensor:
        nodes_in_graph = in_graph.any(dim=1)  # (n_forward,)

        changed = True
        while changed:
            nodes_with_outgoing = in_graph.any(dim=1)                                           # (n_forward,)
            nodes_with_ingoing = (in_graph.any(dim=0).float() @ self.forward_to_backward.float().T) > 0  # (n_forward,)
            nodes_with_ingoing[0] = True  # input node is always live

            new_nodes = nodes_with_outgoing & nodes_with_ingoing
            changed = not torch.equal(new_nodes, nodes_in_graph)
            nodes_in_graph = new_nodes

            backward_alive = (nodes_in_graph.float() @ self.forward_to_backward.float())  # (n_backward,)
            backward_alive[-1] = 1.0  # logits always live
            edge_mask = nodes_in_graph[:, None] & (backward_alive > 0)[None, :]      # (n_forward, n_backward)
            in_graph = in_graph & edge_mask

        return in_graph

    def _get_in_graph(self, percentage):
        k = int(self.n_edges * percentage)

        in_graph = torch.zeros_like(self.circuit_scores).bool()
        in_graph.view(-1)[self.sorted_score_idx[:k]] = True

        in_graph = self._prune(in_graph)

        # Flip boolean for all valid edges
        in_graph = ~in_graph
        in_graph[~self.valid_edge_mask] = False

        return in_graph.to(device=self.adapter.device)
    
    def _compute_attn_corrections(
        self,
        in_graph: torch.Tensor,
        layer_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute Q, K, V residual corrections in a single diff + einsum.
        Q/K/V share the same src_slice — diff is computed once, mask spans all 3*n_heads cols.
        Returns: (q_corr, k_corr, v_corr) each (n_heads, B, S, d)
        """
        src2d_slice = self.adapter.get_src_slice('attn_q', layer_id)
        src_slice = slice(None, src2d_slice.start)              # causal prefix
        q_tgt = self.adapter.get_tgt_slice('attn_q', layer_id)
        v_tgt = self.adapter.get_tgt_slice('attn_v', layer_id)
        qkv_tgt_slice = slice(q_tgt.start, v_tgt.stop)         # 3*n_heads consecutive cols

        base = self._c_base_source[src_slice]
        clean = self._c_source[src_slice]
        if self.norm_matching:
            base = base * self.adapter.norm_scale(self._c_base_cache[f"{layer_id}.in"])
            clean = clean * self.adapter.norm_scale(self._c_cache[f"{layer_id}.in"])

        diff = base - clean     # (n_src, B, S, d)
        mask = in_graph[src_slice, qkv_tgt_slice].to(dtype=diff.dtype)  # (n_src, 3*n_heads)
        corr = torch.einsum('sBSd,st->tBSd', diff, mask)       # (3*n_heads, B, S, d)
        n = self.adapter.n_heads
        return corr[:n], corr[n:2*n], corr[2*n:]

    def _compute_patched_residual(
        self,
        in_graph: torch.Tensor,
        type: str,
        layer_id: int | None,
    ) -> torch.Tensor:
        """Compute residual correction for a single-target node (mlp or lm_head). Returns: (1, B, S, d)"""
        src2d_slice = self.adapter.get_src_slice(type, layer_id)
        tgt_slice = self.adapter.get_tgt_slice(type, layer_id)
        src_slice = slice(None, src2d_slice.start)              # causal prefix

        base = self._c_base_source[src_slice]
        clean = self._c_source[src_slice]
        if self.norm_matching:
            if type == 'mlp':
                cache_key = f"{layer_id}.mid"
            elif type == 'lm_head':
                cache_key = f"{self.adapter.n_layers -1}.out"
            base = base * self.adapter.norm_scale(self._c_base_cache[cache_key])
            clean = clean * self.adapter.norm_scale(self._c_cache[cache_key])

        diff = base - clean     # (n_src, B, S, d)
        mask = in_graph[src_slice, tgt_slice].to(dtype=diff.dtype)  # (n_src, 1)
        return torch.einsum('sBSd,st->tBSd', diff, mask)