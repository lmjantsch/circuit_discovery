from abc import ABC, abstractmethod
from typing import List, Tuple

import torch
from torch import nn

from transformers import AutoModelForCausalLM

from modular_transformer.models import ArchAccessors
from adapters.utils import head_wise_forward

class ModelAdapter(ABC):

    def __init__(self, model: AutoModelForCausalLM, arch: ArchAccessors, frozen_norm: bool = False):
        self.model = model
        self.arch = arch
        self.frozen_norm = frozen_norm
        self.config = model.config
        self.device = model.device
        self.dtype = model.dtype

    @property
    def source_dims(self) -> int:
        return (1 + self.n_layers * (self.n_heads + 1))

    @property
    def grad_dims(self) -> int:
        return (self.n_layers * (3 * self.n_heads + 1) + 1)
    
    @property
    def uses_rotary_emb(self):
        return self.arch.uses_rotary_emb(self.model)

    @property
    def n_layers(self):
        return self.arch.n_layers(self.model)

    @property
    def n_heads(self):
        return self.arch.n_heads(self.model)

    @property
    def model_dim(self):
        return self.arch.model_dim(self.model)

    @property
    def head_dim(self):
        return self.arch.head_dim(self.model)

    @property
    def embed(self) -> nn.Module:
        return self.arch.embed(self.model)

    @property
    def pos_emb(self) -> nn.Module:
        return self.arch.pos_emb(self.model)

    @property
    def rotary_emb(self) -> nn.Module:
        return self.arch.rotary_emb(self.model)
    
    @property
    def ln(self) -> nn.Module:
        return self.arch.ln(self.model)

    @property
    def lm_head(self) -> nn.Module:
        return self.arch.lm_head(self.model)

    @property
    def layers(self) -> nn.Module:
        return self.arch.layers(self.model)
    
    @property
    @abstractmethod
    def wrapped_layers(self) -> List['LayerAdapter']:
        pass

    def get_src_slice(self, type: str, layer_id: int | None) -> slice:
        if type == 'emb':
            return slice(0, 1)
        if type.startswith('attn'):
            start = 1 + layer_id * (self.n_heads + 1)
            return slice(start, start + self.n_heads)
        if type == 'mlp':
            start = 1 + layer_id * (self.n_heads + 1) + self.n_heads
            return slice(start, start + 1)
        if type == 'lm_head': # needs to be included for '_update_score' lookup
            start = 1 + self.n_layers * (self.n_heads + 1) + self.n_heads
            return slice(start, None)
        raise NotImplementedError

    def get_tgt_slice(self, type: str, layer_id: int | None) -> slice:
        if type == 'lm_head':
            return slice(-1, None)
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
        
    def cache_rotary_emb(self, cache: dict):
        if self.uses_rotary_emb:
            cache['rotary_emb'] = self.rotary_emb.output
        else:
            cache['rotary_emb'] = None
        
    def cache_logits_for_grad(self, cache: dict):
        cache['logit_4grad'] = self.lm_head.input

    def embed_out_hook(self, detached: bool = True) -> torch.Tensor:
        if detached:
            return self.layers[0].input.detach()
        return self.layers[0].input

    def logit_grad_hook(self, cache: dict, pre_scaling: bool = False) -> torch.Tensor:
        grad = cache['logit_4grad'].grad
        return self._compute_norm_backward(grad, self.ln, cache['post'], pre_scaling)

    def embedding_patch_hook(self, hidden_state: torch.Tensor):
        """Patch embeddings for EAP-IG inputs to first layer input"""
        self.embed.output = hidden_state 

    def logit_patch_hook(self, hidden_state: torch.Tensor, post_scaling: bool = False): #NOTE reimplement for gemma
        if post_scaling:
            hidden_state = self._compute_norm_forward(hidden_state, self.ln, post_scaling)
            self.lm_head.input = hidden_state
        else:
            self.ln.input = hidden_state

    @abstractmethod
    def norm_scale(self, x: torch.Tensor) -> torch.Tensor:
        """Returns the norm scaling term rsqrt(variance + eps) over the last dim of x."""
        pass

    @abstractmethod
    def _compute_norm_forward(self, hidden_state: torch.Tensor, norm: nn.Module, post_scaling: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def _compute_norm_backward(self, grad: torch.Tensor, norm: nn.Module, forward_state: torch.Tensor, pre_scaling: bool = False) -> torch.Tensor:
        pass


class LayerAdapter(ABC):

    def __init__(self, layer: nn.Module, parent: ModelAdapter, layer_id: int = None):
        self.layer = layer
        self.parent = parent
        self.arch = parent.arch
        self.layer_id = layer_id
        self.device = parent.device
        self.dtype = parent.dtype
        self.n_heads = parent.n_heads
        self.head_dim = parent.head_dim

    @property
    def pre_ln1(self) -> nn.Module:
        return self.arch.pre_ln1(self.layer)

    @property
    def attn(self) -> nn.Module:
        return self.arch.attn(self.layer)

    @property
    def post_ln1(self) -> nn.Module:
        return self.arch.post_ln1(self.layer)

    @property
    def pre_ln2(self) -> nn.Module:
        return self.arch.pre_ln2(self.layer)

    @property
    def mlp(self) -> nn.Module:
        return self.arch.mlp(self.layer)

    @property
    def post_ln2(self) -> nn.Module:
        return self.arch.post_ln2(self.layer)

    @property
    def q_proj(self) -> nn.Module:
        return self.arch.q_proj(self.layer)

    @property
    def k_proj(self) -> nn.Module:
        return self.arch.k_proj(self.layer)

    @property
    def v_proj(self) -> nn.Module:
        return self.arch.v_proj(self.layer)

    @property
    def o_proj(self) -> nn.Module:
        return self.arch.o_proj(self.layer)

    @property
    def attention_interface(self) -> nn.Module:
        return self.arch.attention_interface(self.layer)

    @property
    def up_proj(self) -> nn.Module:
        return self.arch.up_proj(self.layer)

    @property
    def gate_proj(self) -> nn.Module:
        return self.arch.gate_proj(self.layer)

    @property
    def down_proj(self) -> nn.Module:
        return self.arch.down_proj(self.layer)
    
    def residual_in_hook(self, detached=True) -> torch.Tensor:
        if detached:
            return self.pre_ln1.input.detach()
        return self.pre_ln1.input
        
    def residual_mid_hook(self, detached=True) -> torch.Tensor:
        if detached:
            return self.pre_ln2.input.detach()
        return self.pre_ln2.input

    def residual_out_hook(self, detached=True) -> torch.Tensor:
        if detached:
            return self.layer.output.detach()
        return self.layer.output

    @abstractmethod
    def cache_attn_for_grad(self, cache: dict): #NOTE: implement for gemma and llama (GQA)
        pass

    def cache_mlp_for_grad(self, cache: dict):
        cache[f"{self.layer_id}.mlp_in_4grad"] = self.pre_ln2.output

    def head_wise_attn_out_hook(self, detached=True) -> torch.Tensor: #NOTE implement for gemma
        attn_out = self.attention_interface.output[0]
        attn_out = head_wise_forward(self.o_proj, attn_out, self.parent.head_dim)
        if detached:
            return attn_out.detach()
        return attn_out
    
    def mlp_out_hook(self, detached=True) -> torch.Tensor:#NOTE implement for gemma
        if detached:
            return self.mlp.output.detach()
        return self.mlp.output

    @abstractmethod
    def head_wise_query_grad_hook(self, cache: dict, pre_scaling: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def head_wise_key_grad_hook(self, cache: dict, pre_scaling: bool = False) -> torch.Tensor:
        pass

    @abstractmethod
    def head_wise_value_grad_hook(self, cache: dict, pre_scaling: bool = False) -> torch.Tensor:
        pass

    def mlp_grad_hook(self, cache: dict, pre_scaling: bool = False) -> torch.Tensor:
        grad = cache[f"{self.layer_id}.mlp_in_4grad"].grad
        return self.parent._compute_norm_backward(grad, self.pre_ln2, cache[f"{self.layer_id}.mid"], pre_scaling)
    
    @abstractmethod
    def head_wise_query_patch_hook(self, hidden_state: torch.Tensor, cache: dict, post_scaling: bool = False):
        pass

    @abstractmethod
    def head_wise_key_patch_hook(self, hidden_state: torch.Tensor, cache: dict, post_scaling: bool = False):
        pass

    @abstractmethod
    def head_wise_value_patch_hook(self, hidden_state: torch.Tensor, cache: dict, post_scaling: bool = False):
        pass

    def mlp_patch_hook(self, hidden_state: torch.Tensor, post_scaling: bool = False):
        if post_scaling:
            hidden_state = self.parent._compute_norm_forward(hidden_state, self.pre_ln2, post_scaling)
            self.mlp.input = hidden_state
        else:
            self.pre_ln2.input = hidden_state