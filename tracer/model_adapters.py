from abc import ABC, abstractmethod

import torch
from torch import nn

from tracer.modeling_utils import apply_inverse_rope

class ModelAdapter(ABC):

    def __init__(self, model, ignore_norm: bool = False, frozen_norm: bool = False,
                 final_softcap_fn: str = 'tanh'):
        self.model = model
        self.config = model.config
        self.device = model.device
        self.dtype = model.dtype

        self.ignore_norm = ignore_norm
        self.frozen_norm = frozen_norm
        # Selects the activation used in Gemma2's final-logit softcap path.
        # 'tanh' = standard autograd (current default, applies (1 - tanh^2) in backward).
        # 'identity_tanh' = skips the softcap derivative (matches legacy eap_compat=False).
        self.final_softcap_fn = final_softcap_fn

    @property
    def source_dims(self) -> int:
        return (1 + self.n_layers * (self.n_heads + 1))

    @property
    def grad_dims(self) -> int:
        return (self.n_layers * (3 * self.n_heads + 1) + 1)
    
    @property
    @abstractmethod
    def uses_rotary_emb(self):
        pass

    @property
    @abstractmethod
    def n_layers(self):
        pass

    @property
    @abstractmethod
    def n_heads(self):
        pass

    @property
    @abstractmethod
    def model_dim(self):
        pass

    @property
    @abstractmethod
    def head_dim(self):
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

    @property
    @abstractmethod
    def emb(self) -> nn.Module:
        pass

    @property
    @abstractmethod
    def layers(self) -> nn.Module:
        pass
    
    # embeddings
    @abstractmethod
    def emb_in(self) -> torch.Tensor:
        pass

    @abstractmethod
    def emb_out(self) -> torch.Tensor:
        pass

    # residual positions
    @abstractmethod
    def residual_in(self, layer: nn.Module) -> torch.Tensor:
        pass

    @abstractmethod
    def residual_mid(self, layer: nn.Module) -> torch.Tensor:
        pass

    @abstractmethod
    def residual_out(self, layer: nn.Module) -> torch.Tensor:
        pass

    # attention helpers
    @abstractmethod
    def build_attn_source(self, layer) -> None:
        pass

    # attention
    @abstractmethod
    def query_out(self, layer: nn.Module) -> torch.Tensor:
        pass

    @abstractmethod
    def key_out(self, layer: nn.Module) -> torch.Tensor:
        pass

    @abstractmethod
    def value_out(self, layer: nn.Module) -> torch.Tensor:
        pass

    @abstractmethod
    def per_head_attn_out(self, layer: nn.Module) -> torch.Tensor:
        pass

    # mlp
    @abstractmethod
    def gate_out(self, layer: nn.Module) -> torch.Tensor:
        pass

    @abstractmethod
    def up_out(self, layer: nn.Module) -> torch.Tensor:
        pass

    @abstractmethod
    def mlp_out(self, layer: nn.Module) -> torch.Tensor:
        pass

    # logits
    @abstractmethod
    def logits(self):
        pass


    # graients 
    @abstractmethod
    def mlp_in_grad(
        self,
        gate_grad: torch.Tensor,
        up_grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module
    ):
        pass
    
    @abstractmethod
    def query_in_grad(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        pass
    
    @abstractmethod
    def key_in_grad(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        pass
    
    @abstractmethod
    def value_in_grad(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        pass

    # gradient helpers
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

    @abstractmethod
    def _compute_norm_input_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        norm: nn.Module,
    ):
        pass







# ------------------------------------------------------------------
# Llama models
# ------------------------------------------------------------------


class Llama2ModelAdapter(ModelAdapter):

    @property
    def uses_rotary_emb(self):
        return True
    
    @property
    def n_layers(self):
        return self.config.num_hidden_layers

    @property
    def n_heads(self):
        return self.config.num_attention_heads

    @property
    def model_dim(self):
        return self.config.hidden_size

    @property
    def head_dim(self):
        if not hasattr(self.config, 'head_dim'):
            return self.model_dim // self.n_heads
        return self.config.head_dim

    @property
    def emb(self) -> nn.Module:
        return self.model.model.embed_tokens

    @property
    def layers(self) -> nn.Module:
        return self.model.model.layers
    
    # embeddings
    def emb_in(self) -> torch.Tensor:
        return self.model.model.embed_tokens.input

    def emb_out(self) -> torch.Tensor:
        return self.model.model.embed_tokens.output

    def rotary_emb_out(self):
        rot_cos, rot_sin = self.model.model.rotary_emb.output
        return (rot_cos.detach(), rot_sin.detach())

    # residual positions
    def residual_in(self, layer: nn.Module) -> torch.Tensor:
        return layer.input_layernorm.input

    def residual_mid(self, layer: nn.Module) -> torch.Tensor:
        return layer.post_attention_layernorm.input

    def residual_out(self, layer: nn.Module) -> torch.Tensor:
        return layer.output

    # attention helpers
    def build_attn_source(self, layer: nn.Module) -> None:
        layer.self_attn.source

    # attention
    def query_out(self, layer: nn.Module) -> torch.Tensor:
        return layer.self_attn.q_proj.output

    def key_out(self, layer: nn.Module) -> torch.Tensor:
        return layer.self_attn.source.attention_interface_0.source.repeat_kv_0.output

    def value_out(self, layer: nn.Module) -> torch.Tensor:
        return layer.self_attn.source.attention_interface_0.source.repeat_kv_1.output

    @torch.no_grad() 
    def per_head_attn_out(self, layer: nn.Module) -> torch.Tensor:
        z = layer.self_attn.source.attention_interface_0.output[0].detach()
        _, _, H, d = z.shape

        W_linear = layer.self_attn.o_proj.weight.data
        W_linear = W_linear.T.reshape(H, d, -1).detach()
        return torch.einsum('BSHd, HdD -> HBSD', z, W_linear)

    # mlp
    def gate_out(self, layer: nn.Module) -> torch.Tensor:
        return layer.mlp.gate_proj.output

    def up_out(self, layer: nn.Module) -> torch.Tensor:
        return layer.mlp.up_proj.output

    def mlp_out(self, layer: nn.Module) -> torch.Tensor:
        return layer.mlp.output

    # logits
    def logits(self):
        return self.model.lm_head.output
    

    # graients 
    def mlp_in_grad(
        self,
        gate_grad: torch.Tensor,
        up_grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module
    ):
        
        # STEP 1: Backprop through the projection
        W_gate_linear = layer.mlp.gate_proj.weight.data
        gate_grad_post_norm = torch.einsum('bsd, dD -> bsD', gate_grad, W_gate_linear)

        W_up_linear = layer.mlp.up_proj.weight.data
        up_grad_post_norm = torch.einsum('bsd, dD -> bsD', up_grad, W_up_linear)

        grad = gate_grad_post_norm + up_grad_post_norm

        return self._compute_norm_input_gradient(
            grad=grad,
            x_pre_norm=x_pre_norm, 
            norm=layer.post_attention_layernorm
        )
    
    def query_in_grad(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        B, S, _ = grad.shape
        grad = grad.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        return self._compute_headwise_attn_gradient(  # does not require inverse rot_embeds as cached at linear out
            grad=grad, 
            x_pre_norm=x_pre_norm, 
            W_linear=layer.self_attn.q_proj.weight.data, 
            norm=layer.input_layernorm
        )
    
    def key_in_grad(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        return self._compute_headwise_attn_gradient(
            grad=grad, 
            x_pre_norm=x_pre_norm, 
            W_linear=layer.self_attn.k_proj.weight.data,
            norm=layer.input_layernorm, 
            rot_embeds=cache['rotary_emb']
        )
    
    def value_in_grad(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        return self._compute_headwise_attn_gradient( 
            grad=grad, 
            x_pre_norm=x_pre_norm, 
            W_linear=layer.self_attn.v_proj.weight.data,
            norm=layer.input_layernorm,
        )

    # gradient helpers
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

        u = (grad * norm.weight.data).float()

        # ignore all scaling and centering
        if self.ignore_norm:
            return u.to(input_dtype)

        variance = (x_pre_norm ** 2).mean(dim=-1, keepdim=True)
        sigma = torch.sqrt(variance + norm.variance_epsilon)
        
        # treat rms as detached adjoint transformation
        if self.frozen_norm == True:
            grad_pre_norm =  u / sigma
            return grad_pre_norm.to(input_dtype)
        
        # complete gradient through rms term
        u_dot_x = (u * x_pre_norm).sum(dim=-1, keepdim=True)
        term2 = (u_dot_x / (self.model_dim * (variance + norm.variance_epsilon))) * x_pre_norm
        grad_pre_norm = (u - term2) / sigma
        
        return grad_pre_norm.to(input_dtype)

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
    


# ------------------------------------------------------------------
# Gemma2 
# ------------------------------------------------------------------

class Gemma2ModelAdapter(Llama2ModelAdapter):

    # gemma has pre and post module layer norm
    def residual_mid(self, layer: nn.Module) -> torch.Tensor:
        return layer.pre_feedforward_layernorm.input

    @torch.no_grad() 
    def per_head_attn_out(self, layer: nn.Module) -> torch.Tensor:
        z = layer.self_attn.source.attention_interface_0.output[0].detach()
        _, _, H, d = z.shape

        W_linear = layer.self_attn.o_proj.weight.data
        W_linear = W_linear.T.reshape(H, d, -1).detach()
        per_head_attn_out = torch.einsum('BSHd, HdD -> HBSD', z, W_linear).float()

        # pass through norm
        W_norm = layer.post_attention_layernorm.weight.data
        eps = layer.post_attention_layernorm.eps

        rms_scalar = torch.rsqrt(per_head_attn_out.sum(0).pow(2).mean(-1, keepdim=True) + eps)
        per_head_attn_out = per_head_attn_out * rms_scalar
        per_head_attn_out = per_head_attn_out * (1.0 + W_norm.float())

        return per_head_attn_out.to(z.dtype)

    # mlp
    def mlp_out(self, layer: nn.Module) -> torch.Tensor:
        return layer.post_feedforward_layernorm.output

    # logits
    def logits(self):
        raw = self.model.lm_head.output
        cap = getattr(self.config, 'final_logit_softcapping', None)
        if cap is not None:
            from linear_transformer.modules import ACT_FN
            tanh_fn = ACT_FN.get(self.final_softcap_fn, torch.tanh)
            return tanh_fn(raw / cap) * cap
        return raw

    # graients
    def mlp_in_grad(
        self,
        gate_grad: torch.Tensor,
        up_grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module
    ):
        
        # STEP 1: Backprop through the projection
        W_gate_linear = layer.mlp.gate_proj.weight.data
        gate_grad_post_norm = torch.einsum('bsd, dD -> bsD', gate_grad, W_gate_linear)

        W_up_linear = layer.mlp.up_proj.weight.data
        up_grad_post_norm = torch.einsum('bsd, dD -> bsD', up_grad, W_up_linear)

        grad = gate_grad_post_norm + up_grad_post_norm

        return self._compute_norm_input_gradient(
            grad=grad,
            x_pre_norm=x_pre_norm, 
            norm=layer.pre_feedforward_layernorm
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

        u = grad.float() * (1 + norm.weight.data.float())

        # ignore all scaling and centering
        if self.ignore_norm:
            return u.to(input_dtype)

        variance = (x_pre_norm ** 2).mean(dim=-1, keepdim=True)
        sigma = torch.sqrt(variance + norm.eps)
        
        # treat rms as detached adjoint transformation
        if self.frozen_norm == True:
            grad_pre_norm =  u / sigma
            return grad_pre_norm.to(input_dtype)
        
        # complete gradient through rms term
        u_dot_x = (u * x_pre_norm).sum(dim=-1, keepdim=True)
        term2 = (u_dot_x / (self.model_dim * (variance + norm.eps))) * x_pre_norm
        grad_pre_norm = (u - term2) / sigma
        
        return grad_pre_norm.to(input_dtype)



# ------------------------------------------------------------------
# GPT2 
# ------------------------------------------------------------------

class GPT2ModelAdapter(ModelAdapter):
    
    @property
    def uses_rotary_emb(self):
        return False

    @property
    def n_layers(self):
        return self.config.n_layer

    @property
    def n_heads(self):
        return self.config.n_head

    @property
    def model_dim(self):
        return self.config.n_embd

    @property
    def head_dim(self):
        return self.model_dim // self.n_heads
    
    @property
    def emb(self) -> nn.Module:
        return self.model.transformer.wte
    
    @property
    def layers(self) -> nn.Module:
        return self.model.transformer.h
    
    # embeddings
    def emb_in(self) -> torch.Tensor:
        return self.model.transformer.wte.input

    def emb_out(self) -> torch.Tensor:
        return self.model.transformer.wte.output

    # residual positions
    def residual_in(self, layer: nn.Module) -> torch.Tensor:
        return layer.ln_1.input

    def residual_mid(self, layer: nn.Module) -> torch.Tensor:
        return layer.ln_2.input

    def residual_out(self, layer: nn.Module) -> torch.Tensor:
        return layer.output

    # attention helpers
    def build_attn_source(self, layer: nn.Module) -> None:
        layer.attn.source

    # attention
    def query_out(self, layer: nn.Module) -> torch.Tensor:
        return layer.attn.source.split_1.output[0]

    def key_out(self, layer: nn.Module) -> torch.Tensor:
        return layer.attn.source.transpose_2.output

    def value_out(self, layer: nn.Module) -> torch.Tensor:
        return layer.attn.source.transpose_3.output

    @torch.no_grad() 
    def per_head_attn_out(self, layer: nn.Module) -> torch.Tensor:
        z = layer.attn.source.attention_interface_0.output[0].detach()
        _, _, H, d = z.shape

        W_linear = layer.attn.c_proj.weight.data
        # Conv_1D stores weights in (n_in, n_out)
        W_linear = W_linear.reshape(H, d, -1).detach()
        return torch.einsum('BSHd, HdD -> HBSD', z, W_linear)

    # mlp
    def gate_out(self, layer: nn.Module) -> torch.Tensor:
        return None

    def up_out(self, layer: nn.Module) -> torch.Tensor:
        return layer.mlp.c_fc.output

    def mlp_in(self, layer: nn.Module) -> torch.Tensor:
        return layer.mlp.input

    def mlp_out(self, layer: nn.Module) -> torch.Tensor:
        return layer.mlp.output

    # logits
    def logits(self):
        return self.model.lm_head.output 

    # graients 
    def mlp_in_grad(
        self,
        gate_grad: torch.Tensor,
        up_grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module
    ):  
        # STEP 1: Backprop through the projection
        W_up_linear = layer.mlp.c_fc.weight.data
        grad = torch.einsum('bsd, Dd -> bsD', up_grad, W_up_linear) # -> weights have shape (input, output)

        return self._compute_norm_input_gradient(
            grad=grad,
            x_pre_norm=x_pre_norm, 
            norm=layer.ln_2
        )
    
    def query_in_grad(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        B, S, _ = grad.shape
        grad = grad.reshape(B, S, self.n_heads, self.head_dim).transpose(1, 2)
        return self._compute_headwise_attn_gradient(
            grad=grad, 
            x_pre_norm=x_pre_norm, 
            W_linear=layer.attn.c_attn.weight.data.reshape(self.model_dim, 3, -1)[:, 0, :],
            norm=layer.ln_1
        )
    
    def key_in_grad(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        return self._compute_headwise_attn_gradient(
            grad=grad, 
            x_pre_norm=x_pre_norm, 
            W_linear=layer.attn.c_attn.weight.data.reshape(self.model_dim, 3, -1)[:, 1, :],
            norm=layer.ln_1
        )
    
    def value_in_grad(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        layer: nn.Module,
        cache: dict,
    ):
        return self._compute_headwise_attn_gradient( 
            grad=grad, 
            x_pre_norm=x_pre_norm, 
            W_linear=layer.attn.c_attn.weight.data.reshape(self.model_dim, 3, -1)[:, 2, :], 
            norm=layer.ln_1
        )

    
    # gradient helpers
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

        u = grad.float() * norm.weight.data.float()

        # ignore all scaling and centering
        if self.ignore_norm:
            return u.to(input_dtype)

        mean = x_pre_norm.mean(dim=-1, keepdim=True)
        x_centered = x_pre_norm - mean
        variance = (x_centered ** 2).mean(dim=-1, keepdim=True)
        sigma = torch.sqrt(variance + norm.eps)

        # treat layernorm as detached adjoint transformation
        if self.frozen_norm == True:
            grad_pre_norm =  u / sigma
            return grad_pre_norm.to(input_dtype)

        # complete gradient through layernorm
        u_mean = u.mean(dim=-1, keepdim=True)
        u_dot_xc = (u * x_centered).sum(dim=-1, keepdim=True)
        term2 = (u_dot_xc / (self.model_dim * (variance + norm.eps))) * x_centered
        grad_pre_norm = (u - u_mean - term2) / sigma
        
        return grad_pre_norm.to(input_dtype)

    @torch.no_grad() 
    def _compute_headwise_attn_gradient(
        self,
        grad: torch.Tensor,
        x_pre_norm: torch.Tensor,
        W_linear: torch.Tensor,
        norm: nn.Module,
        rot_embeds: torch.Tensor = None,
    ):  
        # STEP 1: Backprop through the projection
        W_linear = W_linear.T.reshape(-1, self.head_dim, self.model_dim)  # Conv_1D stores weights in (n_in, n_out)
        grad_post_norm = torch.einsum('bhsd, hdD -> bhsD', grad, W_linear)
        
        # STEP 2: Backprop through the LayerNorm
        grad_pre_norm = self._compute_norm_input_gradient(grad_post_norm, x_pre_norm, norm)
        
        return grad_pre_norm.transpose(0, 1)
        
