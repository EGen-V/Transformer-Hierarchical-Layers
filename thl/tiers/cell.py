import torch
import torch.nn as nn
from thl.config import THLConfig

class TierCell(nn.Module):
    """
    GRU-based tier update unit.
    Implements:
    z_t = sigmoid(W_z [s_prev; x_t])
    r_t = sigmoid(W_r [s_prev; x_t])
    s_tilde = tanh(W_h [r_t * s_prev; x_t])
    s_t = (1-z_t)*s_prev + z_t*s_tilde
    """
    def __init__(self, tier_idx: int, config: THLConfig):
        super().__init__()
        self.tier_idx = tier_idx
        self.hidden_dim = config.tier_dims[tier_idx]
        self.value_dim = config.value_dim
        
        # Determine input dimension
        # x_t^{(0)} = [e_t; r_t]
        # x_t^{(k)} = [s_t^{(k-1)}; r_t] for k >= 1
        
        if tier_idx == 0:
            input_dim = config.embedding_dim + self.value_dim
        else:
            prev_dim = config.tier_dims[tier_idx - 1]
            input_dim = prev_dim + self.value_dim
            
        # Standard GRU has linear maps for update (z), reset (r), and new (h)
        # We concatenate [s_prev; x_t] for gates
        total_input_dim = self.hidden_dim + input_dim
        
        self.W_z = nn.Linear(total_input_dim, self.hidden_dim)
        self.W_r = nn.Linear(total_input_dim, self.hidden_dim)
        
        # For candidate s_tilde, input is [r_t * s_prev; x_t]
        # Dimension is same: hidden + input
        self.W_h = nn.Linear(total_input_dim, self.hidden_dim)
        
    def forward(self, 
                s_prev: torch.Tensor, 
                lower_input: torch.Tensor, 
                memory_read: torch.Tensor, 
                active: bool = True) -> torch.Tensor:
        """
        Args:
            s_prev: [B, d_k]
            lower_input: [B, d_{lower}] (or embedding for tier 0)
            memory_read: [B, d_v]
            active: boolean flag. If False, returns s_prev unchanged (clocked gating).
            
        Returns:
            s_t: [B, d_k]
        """
        if not active:
            return s_prev
            
        # Construct x_t^{(k)}
        x_t = torch.cat([lower_input, memory_read], dim=-1)
        
        # 1. Update/Reset Gates
        # [s_prev; x_t]
        gate_input = torch.cat([s_prev, x_t], dim=-1)
        z_t = torch.sigmoid(self.W_z(gate_input))
        r_t = torch.sigmoid(self.W_r(gate_input))
        
        # 2. Candidate State
        # [r_t * s_prev; x_t]
        seeded_state = r_t * s_prev
        candidate_input = torch.cat([seeded_state, x_t], dim=-1)
        s_tilde = torch.tanh(self.W_h(candidate_input))
        
        # 3. Final Update
        s_t = (1 - z_t) * s_prev + z_t * s_tilde
        
        return s_t

    def init_state(self, batch_size: int, device: str) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)
