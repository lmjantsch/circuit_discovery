import torch

from modeling_utils import apply_inverse_rope

def compute_rmsnorm_input_gradients(
    grad: torch.Tensor,
    x_pre_norm: torch.Tensor,
    rmsnorm_weight: torch.Tensor,
    eps: float = 1e-6
):
    D = x_pre_norm.shape[-1]
    if grad.dim() == x_pre_norm.dim() + 1: # accommodate head dimension
        x_pre_norm = x_pre_norm.unsqueeze(1)
    variance = (x_pre_norm ** 2).mean(dim=-1, keepdim=True)
    sigma = torch.sqrt(variance + eps)
    
    u = grad * rmsnorm_weight
    u_dot_x = (u * x_pre_norm).sum(dim=-1, keepdim=True)
    
    term2 = (u_dot_x / (D * (variance + eps))) * x_pre_norm
    grad_pre_norm = (u - term2) / sigma
    
    return grad_pre_norm

def compute_headwise_input_gradients(
    grad_heads: torch.Tensor, 
    W_linear: torch.Tensor, 
    x_pre_norm: torch.Tensor, 
    rmsnorm_weight: torch.Tensor,
    rot_embeds: tuple = None, 
    eps: float = 1e-6
):
    B, H, S, d = grad_heads.shape
    D = x_pre_norm.size(-1)

    if rot_embeds:
        grad_heads = apply_inverse_rope(grad_heads, rot_embeds)
    
    # STEP 1: Backprop through the projection
    W_linear = W_linear.view(-1, d, D)
    if W_linear.size(0) != H: # repeat W_linear for grouped query attention (GQA)
        num_key_value_groups = H // W_linear.size(0)
        W_linear = W_linear.repeat_interleave(num_key_value_groups, dim=0)

    grad_post_norm = torch.einsum('bhsd, hdD -> bhsD', grad_heads, W_linear)
    
    # STEP 2: Backprop through the RMSNorm
    grad_pre_norm = compute_rmsnorm_input_gradients(grad_post_norm, x_pre_norm, rmsnorm_weight, eps)
    
    return grad_pre_norm.transpose(0, 1)