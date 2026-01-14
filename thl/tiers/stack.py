import torch
import torch.nn as nn
from typing import List, Tuple
from thl.config import THLConfig
from thl.tiers.cell import TierCell
from thl.tiers.clock import TierClock

class HierarchicalTierStack(nn.Module):
    """
    Manages the stack of K TierCells.
    """
    def __init__(self, config: THLConfig):
        super().__init__()
        self.num_tiers = config.num_tiers
        
        # Use ModuleList to register sub-modules properly
        self.cells = nn.ModuleList([
            TierCell(k, config) for k in range(self.num_tiers)
        ])
        
        self.clock = TierClock(config)
        self.config = config
        
    def forward(self, 
                embedding: torch.Tensor, 
                memory_read: torch.Tensor, 
                prev_states: List[torch.Tensor], 
                timestep: int,
                tier0_memory_read: torch.Tensor = None) -> List[torch.Tensor]:
        """
        Args:
            embedding: [B, d_e]
            memory_read: [B, d_v]
            prev_states: List of [B, d_k] for each tier k
            timestep: current time t
            
        Returns:
            new_states: List of updated states
        """
        active_flags = self.clock.get_active_tiers(timestep)
        new_states = []
        
        current_lower_input = embedding
        
        for k in range(self.num_tiers):
            cell = self.cells[k]
            s_prev = prev_states[k]
            active = active_flags[k]

            r_k = tier0_memory_read if (k == 0 and tier0_memory_read is not None) else memory_read
            
            # Update tier
            s_new = cell(s_prev, current_lower_input, r_k, active=active)
            new_states.append(s_new)
            
            # The input to the next tier is strictly the state of this tier?
            # Tier spec: x_t^{(k)} = [s_t^{(k-1)}; r_t]
            # Wait, s_t^{(k-1)} is the iterate we just computed?
            # "Tier update F_k ... x_t^{(k)} = [s_t^{(k-1)}; r_t]"
            # BUT: "Clocked update ... s_t^{(k)} = ... if clock else s_{t-1}^{(k)}"
            # If tier k-1 did NOT update (clock off), its state s_t is s_{t-1}.
            # So yes, we pass the current s_new (which might be old value) as input to next tier.
            
            current_lower_input = s_new
            
        return new_states

    def init_states(self, batch_size: int) -> List[torch.Tensor]:
        return [cell.init_state(batch_size, self.config.device) for cell in self.cells]
