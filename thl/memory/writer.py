import torch
import torch.nn as nn
from thl.config import THLConfig
from thl.utils.numerical import norm_clip_vector
from typing import Tuple

class MemoryWriter(nn.Module):
    """
    Handles memory writing: candidate generation, novelty gating, and slot selection.
    """
    def __init__(self, config: THLConfig):
        super().__init__()
        self.d_m = config.memory_dim
        self.d_v = config.value_dim
        self.write_slots = config.write_slots
        self.beta = config.novelty_beta
        self.theta = config.novelty_theta
        self.delta_max = config.write_clip_max
        self.lambda_s = config.stale_weight
        self.lambda_u = config.underuse_weight
        
        # Input: [s_t^0; s_t^(K-1)]
        tier_dims = config.tier_dims
        input_dim = tier_dims[0] + tier_dims[-1]
        
        self.W_w = nn.Linear(input_dim, self.d_m, bias=True)
        self.W_r = nn.Linear(self.d_v, self.d_m, bias=False) # Project reading r_t to memory space for comparison
        
    def forward(self, 
                s_0: torch.Tensor, 
                s_K: torch.Tensor, 
                r_t: torch.Tensor, 
                staleness: torch.Tensor, 
                underuse: torch.Tensor,
                memory_state: torch.Tensor = None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Args:
            s_0, s_K: Tiers states [B, d_k]
            r_t: Current read vector [B, d_v]
            staleness: [B, J]
            underuse: [B, J]
            
        Returns:
            w_t: [B, d_m] - clipped candidate vector
            g_t: [B, 1] - novelty gate
            write_indices: [B, W] - chosen slots
        """
        # 1. Candidate Write Vector
        concat_s = torch.cat([s_0, s_K], dim=-1)
        w_raw = self.W_w(concat_s) # [B, d_m]
        w_t = norm_clip_vector(w_raw, self.delta_max)
        
        # 2. Novelty Gate
        eps = 1e-8
        if memory_state is None:
            r_hat = self.W_r(r_t) # [B, d_m]
            w_norm = w_t.norm(p=2, dim=-1, keepdim=True) + eps
            r_hat_norm = r_hat.norm(p=2, dim=-1, keepdim=True) + eps
            cos_sim = (w_t * r_hat).sum(dim=-1, keepdim=True) / (w_norm * r_hat_norm)
            novelty = 1.0 - cos_sim
            g_t = torch.sigmoid(self.beta * (novelty - self.theta))

            score = self.lambda_s * staleness + self.lambda_u * underuse
            _, write_indices = torch.topk(score, k=self.write_slots, dim=-1)
            return w_t, g_t, write_indices

        w_norm = w_t.norm(p=2, dim=-1, keepdim=True) + eps
        m_norm = memory_state.norm(p=2, dim=-1) + eps
        dot = (memory_state * w_t.unsqueeze(1)).sum(dim=-1)
        affinity = dot / (w_norm * m_norm)
        max_affinity, _ = affinity.max(dim=-1)
        novelty = (1.0 - max_affinity).unsqueeze(-1)
        g_t = torch.sigmoid(self.beta * (novelty - self.theta))
        
        # 3. Write Target Scoring
        # u_{t,j} = lambda_s * staleness + lambda_u * underuse
        # staleness and underuse should be normalized? 
        # Spec says: staleness is raw time diff. underuse is 1/ema.
        # Scales might differ significantly. Lambda_s/u balance this.
        
        score = self.lambda_s * staleness + self.lambda_u * underuse

        empty_mask = (memory_state.norm(p=2, dim=-1) <= 1e-6)
        has_empty = empty_mask.any(dim=-1)

        slot_idx = torch.arange(memory_state.size(1), device=memory_state.device, dtype=score.dtype)
        empty_score = empty_mask.to(score.dtype) * 1e4 + (-slot_idx.unsqueeze(0).expand_as(score))

        _, empty_choice = torch.topk(empty_score, k=self.write_slots, dim=-1)
        _, lru_choice = torch.topk(score, k=self.write_slots, dim=-1)
        _, update_choice = torch.topk(affinity.to(score.dtype), k=self.write_slots, dim=-1)

        novel_flag = (novelty.squeeze(-1) > self.theta)
        use_empty = novel_flag & has_empty
        use_lru = novel_flag & (~has_empty)

        g_t = torch.where(novel_flag.unsqueeze(-1), g_t, torch.ones_like(g_t))

        write_indices = update_choice
        write_indices = torch.where(use_lru.unsqueeze(1), lru_choice, write_indices)
        write_indices = torch.where(use_empty.unsqueeze(1), empty_choice, write_indices)
        
        return w_t, g_t, write_indices
