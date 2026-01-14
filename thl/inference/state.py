from dataclasses import dataclass
from typing import List, Optional
import torch
from thl.config import THLConfig
from thl.memory.metadata import BatchedMemoryMetadata

@dataclass
class InferenceState:
    """
    Bounded inference state for THL.
    Contains strictly the recurrent state needed for the next step.
    Does NOT grow with sequence length.
    """
    timestep: int
    tier_states: List[torch.Tensor]       # List of [B, d_k]
    memory_state: torch.Tensor            # [B, J, d_m]
    memory_metadata: BatchedMemoryMetadata
    local_buffer: torch.Tensor            # [B, W, d_e]
    local_valid: int
    
    @classmethod
    def init(cls, batch_size: int, config: THLConfig, 
             tier_stack, memory_bank) -> 'InferenceState':
        local_window = int(getattr(config, "local_window", 0))
        return cls(
            timestep=0,
            tier_states=tier_stack.init_states(batch_size),
            memory_state=memory_bank.init_state(batch_size),
            memory_metadata=BatchedMemoryMetadata(batch_size, config),
            local_buffer=torch.zeros(batch_size, local_window, config.embedding_dim, device=config.device),
            local_valid=0
        )
    
    def detach(self):
        """Detach all tensors from computation graph (for inference loop)"""
        self.tier_states = [s.detach() for s in self.tier_states]
        self.memory_state = self.memory_state.detach()
        self.local_buffer = self.local_buffer.detach()
        # Metadata tensors are inside the object, need to handle carefully or 
        # assume they are manually managed. 
        # Metadata class uses in-place operations mostly.
