from abc import ABC, abstractmethod

import torch
from torch import nn

from tracer.modeling_utils import apply_inverse_rope

class ModelAdapter(ABC):

    def __init__(self, model):
        self.model = model
        self.device = model.device
        self.dtype = model.dtype

        self.n_layers = model.config.num_hidden_layers
        self.n_heads = model.config.num_attention_heads
        self.model_dim = model.config.hidden_size

        if not hasattr(model.config, 'head_dim'):
            self.head_dim = self.model_dim // self.n_heads
        else:
            self.head_dim = model.config.head_dim

        self.uses_rotary_emb = False

    @property
    def source_dims(self) -> int:
        return (1 + self.n_layers * (self.n_heads + 1))

    @property
    def grad_dims(self) -> int:
        return (self.n_layers * (3 * self.n_heads + 1) + 1)
        
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
    
    @property
    @abstractmethod
    def embed_tokens(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def layers(self) -> nn.Module:
        pass
    
    @property
    @abstractmethod
    def lm_head(self):
        pass

    @abstractmethod
    def ln_1(self, layer: nn.Module):
        pass

    @abstractmethod
    def build_attn_source(self, layer):
        pass

    @abstractmethod
    def q_proj_out(self, layer: nn.Module):
        pass
    
    @abstractmethod
    def q_proj_weights(self, layer: nn.Module):
        pass

    @abstractmethod
    def k_proj_out(self, layer: nn.Module):
        pass
    
    @abstractmethod
    def k_proj_weights(self, layer: nn.Module):
        pass

    @abstractmethod
    def v_proj_out(self, layer: nn.Module):
        pass
    
    @abstractmethod
    def v_proj_weights(self, layer: nn.Module):
        pass
    
    @abstractmethod
    def attn_interface_out(self, layer: nn.Module):
        pass

    @abstractmethod
    def o_proj_weights(self, layer: nn.Module):
        pass


    @abstractmethod
    def ln_2(self, layer: nn.Module):
        pass

    @abstractmethod
    def mlp(self, layer: nn.Module):
        pass

    @abstractmethod
    def compute_per_head_attn_out(self, z: torch.Tensor, W_linear: torch.Tensor) -> torch.Tensor:
        pass

    @abstractmethod
    def compute_mlp_input_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module
    ):
        pass

    @abstractmethod
    def _compute_norm_input_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        norm: nn.Module,
    ):
        pass
    
    @abstractmethod
    def compute_headwise_query_input_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        pass
    
    @abstractmethod
    def compute_headwise_key_input_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        pass
    
    @abstractmethod
    def compute_headwise_value_input_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        pass

    @abstractmethod
    def _compute_headwise_attn_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        W_linear: torch.Tensor,
        norm: nn.Module,
        rot_embeds: torch.Tensor = None,
    ):
        pass







# ------------------------------------------------------------------
# Llama models
# ------------------------------------------------------------------


class Llama2ModelAdapter(ModelAdapter):

    def __init__(self, model):
        super().__init__(model)
        self.uses_rotary_emb = True
    
    @property
    def embed_tokens(self) -> nn.Module:
        return self.model.model.embed_tokens

    @property
    def rotary_emb(self):
        return self.model.model.rotary_emb

    @property
    def layers(self) -> nn.Module:
        return self.model.model.layers
    
    @property
    def lm_head(self):
        return self.model.lm_head

    def ln_1(self, layer: nn.Module):
        return layer.input_layernorm


    def build_attn_source(self, layer):
        layer.self_attn.source

    def q_proj_out(self, layer: nn.Module):
        return layer.self_attn.q_proj
    
    def q_proj_weights(self, layer: nn.Module):
        return layer.self_attn.q_proj.weight.data

    def k_proj_out(self, layer: nn.Module):
        return layer.self_attn.source.attention_interface_0.source.repeat_kv_0
    
    def k_proj_weights(self, layer: nn.Module):
        return layer.self_attn.k_proj.weight.data

    def v_proj_out(self, layer: nn.Module):
        return layer.self_attn.source.attention_interface_0.source.repeat_kv_1
    
    def v_proj_weights(self, layer: nn.Module):
        return layer.self_attn.v_proj.weight.data
    
    def attn_interface_out(self, layer: nn.Module):
        return layer.self_attn.source.attention_interface_0

    def o_proj_weights(self, layer: nn.Module):
        return layer.self_attn.o_proj.weight.data


    def ln_2(self, layer: nn.Module):
        return layer.post_attention_layernorm

    def mlp(self, layer: nn.Module):
        return layer.mlp


    def compute_per_head_attn_out(self, z: torch.Tensor, W_linear: torch.Tensor) -> torch.Tensor:
        _, _, H, d = z.shape
        W_linear = W_linear.T.reshape(H, d, -1).detach()
        return torch.einsum('BSHd, HdD -> HBSD', z, W_linear)

    def compute_mlp_input_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module
    ):
        return self._compute_norm_input_gradient(
            grad=grad, x_pre_norm=x_pre_norm, norm=self.ln_2(layer)
        )

    @torch.no_grad() 
    def _compute_norm_input_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        norm: nn.Module,
    ):
        input_dtype = grad.dtype
        x_pre_norm = x_pre_norm.float()

        if grad.dim() == x_pre_norm.dim() + 1: # accommodate head dimension
            x_pre_norm = x_pre_norm.unsqueeze(1)

        variance = (x_pre_norm ** 2).mean(dim=-1, keepdim=True)
        sigma = torch.sqrt(variance + norm.variance_epsilon)
        
        u = (grad * norm.weight.data).float()
        u_dot_x = (u * x_pre_norm).sum(dim=-1, keepdim=True)
        
        term2 = (u_dot_x / (self.model_dim * (variance + norm.variance_epsilon))) * x_pre_norm
        grad_pre_norm = (u - term2) / sigma
        
        return grad_pre_norm.to(input_dtype)
    
    def compute_headwise_query_input_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        B, S, _ = grad.shape
        grad = grad.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        return self._compute_headwise_attn_gradient(  # does not require inverse rot_embeds as cached at linear out
            grad=grad, x_pre_norm=x_pre_norm, W_linear=self.q_proj_weights(layer), norm=self.ln_1(layer)
        )
    
    def compute_headwise_key_input_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        return self._compute_headwise_attn_gradient(
            grad=grad, x_pre_norm=x_pre_norm, W_linear=self.k_proj_weights(layer), norm=self.ln_1(layer), rot_embeds=cache['rotary_emb']
        )
    
    def compute_headwise_value_input_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        return self._compute_headwise_attn_gradient( 
            grad=grad, x_pre_norm=x_pre_norm, W_linear=self.v_proj_weights(layer), norm=self.ln_1(layer)
        )

    @torch.no_grad() 
    def _compute_headwise_attn_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        W_linear: torch.Tensor,
        norm: nn.Module,
        rot_embeds: torch.Tensor = None,
    ):
        if rot_embeds is not None:
            grad = apply_inverse_rope(grad, rot_embeds)
        
        # STEP 1: Backprop through the projection
        W_linear = W_linear.view(-1, self.head_dim, self.model_dim)
        if W_linear.size(0) != self.n_heads: # repeat W_linear for grouped query attention (GQA)
            num_key_value_groups = self.n_heads // W_linear.size(0)
            W_linear = W_linear.repeat_interleave(num_key_value_groups, dim=0)

        grad_post_norm = torch.einsum('bhsd, hdD -> bhsD', grad, W_linear)
        
        # STEP 2: Backprop through the RMSNorm
        grad_pre_norm = self._compute_norm_input_gradient(grad_post_norm, x_pre_norm, norm)
        
        return grad_pre_norm.transpose(0, 1)
    


class FrozenModelAdapter(Llama2ModelAdapter):

    def k_proj_out(self, layer: nn.Module):
        return layer.self_attn.source.eager_attention_forward_0.source.repeat_kv_0

    def v_proj_out(self, layer: nn.Module):
        return layer.self_attn.source.eager_attention_forward_0.source.repeat_kv_1
    
    def attn_interface_out(self, layer: nn.Module):
        return layer.self_attn.source.eager_attention_forward_0

    @torch.no_grad() 
    def _compute_norm_input_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        norm: nn.Module,
    ):
        input_dtype = grad.dtype
        x_pre_norm = x_pre_norm.float()

        if grad.dim() == x_pre_norm.dim() + 1: # accommodate head dimension
            x_pre_norm = x_pre_norm.unsqueeze(1)

        variance = (x_pre_norm ** 2).mean(dim=-1, keepdim=True)
        sigma = torch.sqrt(variance + norm.variance_epsilon)
        
        u = (grad * norm.weight.data).float()
        grad_pre_norm = u / sigma
        
        return grad_pre_norm.to(input_dtype)
    




# ------------------------------------------------------------------
# Gemma2 
# ------------------------------------------------------------------

class Gemma2ModelAdapter(Llama2ModelAdapter):

    @torch.no_grad() 
    def _compute_norm_input_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        norm: nn.Module,
    ):
        input_dtype = grad.dtype
        grad = grad.float()
        x_pre_norm = x_pre_norm.float()

        if grad.dim() == x_pre_norm.dim() + 1: # accommodate head dimension
            x_pre_norm = x_pre_norm.unsqueeze(1)

        variance = (x_pre_norm ** 2).mean(dim=-1, keepdim=True)
        sigma = torch.sqrt(variance + norm.eps)
        
        u = grad * norm.weight.data.float()
        u_dot_x = (u * x_pre_norm).sum(dim=-1, keepdim=True)
        
        term2 = (u_dot_x / (self.model_dim * (variance + norm.eps))) * x_pre_norm
        grad_pre_norm = (u - term2) / sigma
        
        return grad_pre_norm.to(input_dtype)
    
class FrozenGemma2ModelAdapter(Llama2ModelAdapter):

    def k_proj_out(self, layer: nn.Module):
        return layer.self_attn.source.eager_attention_forward_0.source.repeat_kv_0

    def v_proj_out(self, layer: nn.Module):
        return layer.self_attn.source.eager_attention_forward_0.source.repeat_kv_1
    
    def attn_interface_out(self, layer: nn.Module):
        return layer.self_attn.source.eager_attention_forward_0

    @torch.no_grad() 
    def _compute_norm_input_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        norm: nn.Module,
    ):
        input_dtype = grad.dtype
        grad = grad.float()
        x_pre_norm = x_pre_norm.float()

        if grad.dim() == x_pre_norm.dim() + 1: # accommodate head dimension
            x_pre_norm = x_pre_norm.unsqueeze(1)

        variance = (x_pre_norm ** 2).mean(dim=-1, keepdim=True)
        sigma = torch.sqrt(variance + norm.eps)
        
        u = grad * norm.weight.data.float()
        grad_pre_norm = u / sigma
        
        return grad_pre_norm.to(input_dtype)