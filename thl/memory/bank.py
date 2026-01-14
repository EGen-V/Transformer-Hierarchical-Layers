import torch
import torch.nn as nn
from thl.config import THLConfig
from thl.utils.numerical import norm_clip_vector

class MemoryBank(nn.Module):
    """
    Bounded external memory bank M_t in R^{B x J x d_m}.
    """
    def __init__(self, config: THLConfig):
        super().__init__()
        self.num_slots = config.memory_slots
        self.dim = config.memory_dim
        self.decay = config.memory_decay
        self.device = config.device
        self.dtype = getattr(torch, config.dtype)
        
        # Note: We prefer to keep memory state explicitly passed in the forward loop 
        # rather than storing it as a module attribute (stateless design for layered inference).
        # But for initialization convenience, we can provide a helper.
    
    def init_state(self, batch_size: int) -> torch.Tensor:
        return torch.zeros(batch_size, self.num_slots, self.dim, 
                          dtype=self.dtype, device=self.device)

    def forward(self, 
                memory_state: torch.Tensor, 
                write_indices: torch.Tensor, 
                write_content: torch.Tensor, 
                gate: torch.Tensor) -> torch.Tensor:
        """
        Updates memory state:
        m_{t,j} = gamma * m_{t-1,j}  (global decay)
        if j in write_set:
           m_{t,j} += eta * g_t * Clip(w_t)
           
        memory_state: [B, J, d_m]
        write_indices: [B, W]
        write_content: [B, d_m] - same vector w_t proposed for all slots in the batch item?
                       The spec says w_t = W_w [...] is a single vector (per batch item).
                       So yes, we broadcast w_t to the selected slots.
        gate: [B, 1] - g_t novelty gate
        """
        # 1. Apply global decay
        next_memory = memory_state * self.decay
        
        # 2. Sparse Update
        # We need to add (eta * g_t * w_t) to specific slots
        # write_content is already projected but NOT yet scaled by eta/gate/clip?
        # Let's assume write_content passed here is already Pre-Clipped w_t.
        # We need to apply gate and eta here or caller does it?
        # Caller usually does the heavy lifting, but let's do it here for clarity if args allow.
        # Arguments: write_content is 'Clip(w_t)'. 
        # actual_update = eta * g_t * Clip(w_t)
        
        # But wait, 'write_content' is [B, d_m], but we might write to W slots.
        # Do we write the SAME vector to multiple slots?
        # Spec: w_t is derived from system state. The slots are selected by 'underuse/staleness'.
        # Yes, w_t is global for the timestep.
        
        # Prepare update vector: [B, d_m]
        # We shouldn't strictly hardcode the float(eta) here if not passed, but let's assume it's pre-scaled?
        # No, better explicit. let's assume this function just applies the sparse add.
        # So we expect `write_value` which is (eta * g_t * Clip(w_t)).
        
        # Let's rename arg to `write_value` [B, d_m]
        write_value = write_content * gate # [B, d_m] (broadcasting gate)
        
        # We need to scatter add this to the memory.
        # write_indices [B, W]
        # write_value [B, d_m] -> needs to be broadcast to [B, W, d_m]
        
        B, W = write_indices.shape
        sparse_update = write_value.unsqueeze(1).expand(B, W, self.dim) # [B, W, d_m]
        
        # Scatter add
        # next_memory.scatter_add_(dim=1, index=..., src=sparse_update)
        # index needs to be [B, W, d_m] repeating the indices along last dim
        expanded_indices = write_indices.unsqueeze(-1).expand(B, W, self.dim)
        
        next_memory.scatter_add_(1, expanded_indices, sparse_update.to(next_memory.dtype))
        
        return next_memory
