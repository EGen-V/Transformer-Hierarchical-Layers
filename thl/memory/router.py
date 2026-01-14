import torch
import torch.nn as nn
from thl.config import THLConfig
from thl.utils.numerical import masked_stable_softmax
from typing import Tuple, Optional, Dict, Any, Union
import math

class SparseRouter(nn.Module):
    """
    Multi-Head Sparse router for THL memory read.
    """
    def __init__(self, config: THLConfig):
        super().__init__()
        self.num_heads = config.num_heads
        self.head_dim = config.head_dim
        self.d_q = self.num_heads * self.head_dim # Total query dim
        
        # d_v is total value dim. We assume d_v is split across heads OR projected per head?
        # Usually distinct heads read distinct content.
        # Let's assume output r_t is concatenation of H heads.
        self.d_v = config.value_dim 
        if self.d_v % self.num_heads != 0:
            from thl.errors import THLConfigError
            raise THLConfigError(
                f"Value dimension {self.d_v} must be divisible by num_heads {self.num_heads}.",
                hint="Ensure 'value_dim' is a multiple of 'num_heads'."
            )
        self.value_head_dim = self.d_v // self.num_heads
        
        self.d_m = config.memory_dim
        self.kappa = config.read_topk
        self.load_penalty = config.load_penalty
        self.read_slot_capacity = getattr(config, "read_slot_capacity", None)
        
        # Dimensions for input 'u_t'. 
        tier_dims = config.tier_dims
        input_dim = config.embedding_dim + tier_dims[0] + tier_dims[-1]
        
        self.W_q = nn.Linear(input_dim, self.d_q, bias=False)
        self.W_k = nn.Linear(self.d_m, self.d_q, bias=False)
        self.W_v = nn.Linear(self.d_m, self.d_v, bias=False)

    def _apply_slot_capacity(self, alpha: torch.Tensor, indices: torch.Tensor, slot_capacity: int) -> torch.Tensor:
        B, H, K = indices.shape
        allowed = torch.ones(B, H, K, device=indices.device, dtype=alpha.dtype)

        for b in range(B):
            counts: Dict[int, int] = {}
            for h in range(H):
                for k in range(K):
                    slot = int(indices[b, h, k].item())
                    c = counts.get(slot, 0)
                    if c >= slot_capacity:
                        allowed[b, h, k] = 0.0
                    else:
                        counts[slot] = c + 1

        alpha_capped = alpha * allowed
        denom = alpha_capped.sum(dim=-1, keepdim=True)
        alpha_capped = torch.where(denom > 0, alpha_capped / denom.clamp_min(1e-9), alpha)
        return alpha_capped

    def _load_balance_aux(self, alpha: torch.Tensor, indices: torch.Tensor, num_slots: int) -> Dict[str, Any]:
        B, H, K = indices.shape
        idx = indices.reshape(B, H * K)
        w = alpha.reshape(B, H * K)

        usage = torch.zeros(B, num_slots, device=alpha.device, dtype=alpha.dtype)
        usage.scatter_add_(1, idx.to(torch.long), w)

        p = usage / usage.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        eps = 1e-9
        kl = (p * (torch.log(p + eps) + math.log(num_slots))).sum(dim=-1)

        return {
            "usage": usage,
            "usage_prob": p,
            "load_balance_kl": kl.mean(),
        }
        
    def forward(self, 
                u_t: torch.Tensor, 
                memory_state: torch.Tensor, 
                read_ema: torch.Tensor,
                slot_capacity: Optional[int] = None,
                return_aux: bool = False
                ) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Dict[str, Any]]]:
        """
        Args:
            u_t: [B, input_dim]
            memory_state: [B, J, d_m]
            read_ema: [B, J]
            
        Returns:
            r_t: [B, d_v] - read vector (concatenated heads)
            alpha: [B, H, kappa] - attention weights for topk per head
            indices: [B, H, kappa] - selected slot indices
        """
        B = u_t.size(0)
        J = memory_state.size(1)
        
        # 1. Project Query [B, H*hd_q] -> [B, H, hd_q]
        q_t = self.W_q(u_t).view(B, self.num_heads, self.head_dim)
        
        # 2. Project Keys [B, J, H*hd_q] -> [B, J, H, hd_q] -> [B, H, J, hd_q] (permute for BMM)
        k = self.W_k(memory_state).view(B, J, self.num_heads, self.head_dim).permute(0, 2, 1, 3)
        
        # 3. Affinity Scores [B, H, J]
        # (q . k)
        scores = torch.matmul(k, q_t.unsqueeze(-1)).squeeze(-1) # [B, H, J]
        scores = scores / (self.head_dim ** 0.5)
        
        # Apply load penalty (broadcast across heads?)
        # read_ema is [B, J]. We expand to [B, 1, J]
        scores = scores - self.load_penalty * read_ema.unsqueeze(1)
        
        # 4. Routing (TopK vs Gumbel)
        if self.training:
            # Gumbel-Softmax for differentiable TopK
            # "Perturb-and-MAP" equivalent for TopK
            # We add Gumbel noise and take TopK indices
            gumbel_noise = -torch.log(-torch.log(torch.rand_like(scores) + 1e-9) + 1e-9)
            perturbed_scores = scores + gumbel_noise
            topk_scores, indices = torch.topk(perturbed_scores, k=self.kappa, dim=-1)
            
            # For gradient flow, we need a soft relaxation on the weights
            # We can use the scores for the selected indices to compute alpha
            # alpha = softmax(scores[on_indices] / tau)
            # Standard Gumbel-Softmax is usually for 1-hot. For TopK, we can just use softmax on the topk items.
            # Straight-Through: Use hard indices for lookup, but soft alpha for weighting.
            
            # Re-compute alpha purely from un-perturbed scores (or perturbed? usually unperturbed for weight)
            # Actually, standard attention uses softmax.
            # Let's use the actual topk scores from the perturbed version to select, 
            # then softmax the original scores for those indices? 
            # Or softmax the topk_scores?
            # Softmax on topk_scores allows gradients to flow into scores.
            alpha = torch.softmax(topk_scores, dim=-1)
            
        else:
            # Inference: Hard TopK
            topk_scores, indices = torch.topk(scores, k=self.kappa, dim=-1) # [B, H, kappa]
            alpha = masked_stable_softmax(topk_scores, mask=torch.ones_like(topk_scores), dim=-1)

        effective_capacity = self.read_slot_capacity if slot_capacity is None else slot_capacity
        if effective_capacity is not None:
            alpha = self._apply_slot_capacity(alpha, indices, int(effective_capacity))
        
        # 6. Readout
        # Values: [B, J, d_v] -> [B, J, H, hd_v] -> [B, H, J, hd_v]
        v = self.W_v(memory_state).view(B, J, self.num_heads, self.value_head_dim).permute(0, 2, 1, 3)
        
        # Gather selected values: [B, H, kappa, hd_v]
        # indices is [B, H, kappa]
        # expand indices to [B, H, kappa, hd_v]
        expanded_indices = indices.unsqueeze(-1).expand(-1, -1, -1, self.value_head_dim)
        v_selected = torch.gather(v, 2, expanded_indices)
        
        # Weighted sum: alpha [B, H, kappa] -> [B, H, 1, kappa]
        # v_select [B, H, kappa, hd_v]
        # r_head = alpha * v
        r_heads = torch.matmul(alpha.unsqueeze(2), v_selected).squeeze(2) # [B, H, hd_v]
        
        # Returns [B, d_v]
        r_t = r_heads.view(B, self.d_v)

        if return_aux:
            aux = self._load_balance_aux(alpha, indices, J)
            return r_t, alpha, indices, aux

        return r_t, alpha, indices
