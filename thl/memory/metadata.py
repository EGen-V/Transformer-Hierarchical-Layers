import torch
from thl.config import THLConfig

class MemoryMetadata:
    """
    Manages metadata for memory slots:
    - last_read: timestamp of last read
    - last_write: timestamp of last write
    - read_ema: exponential moving average of read attention
    - write_ema: exponential moving average of write frequency
    """
    def __init__(self, config: THLConfig):
        self.num_slots = config.memory_slots
        self.device = config.device
        self.dtype = getattr(torch, config.dtype)
        
        # J x 1 tensors
        self.last_read = torch.zeros(self.num_slots, dtype=torch.long, device=self.device)
        self.last_write = torch.zeros(self.num_slots, dtype=torch.long, device=self.device)
        self.read_ema = torch.zeros(self.num_slots, dtype=self.dtype, device=self.device)
        self.write_ema = torch.zeros(self.num_slots, dtype=self.dtype, device=self.device)
        
        self.read_decay = config.read_ema_decay
        self.write_decay = config.write_ema_decay
        self.epsilon = 1e-8

    def update_read(self, timestep: int, alpha: torch.Tensor, indices: torch.Tensor):
        """
        Update read metadata based on routing weights alpha and selected indices.
        alpha: [batch_size, kappa]
        indices: [batch_size, kappa]
        """
        batch_size, kappa = indices.shape
        
        # Flatten for scatter operations
        flat_indices = indices.view(-1)
        flat_alpha = alpha.view(-1)
        
        # 1. Update last_read
        # We need to scatter the timestep into last_read at the indices
        # Since multiple batch items might read the same slot, we take the max (latest) timestep
        # But here timestep is constant for the batch, so just setting it is enough.
        # However, scatter requires src to be same size.
        
        # Optimization: Create a mask of "touched" slots in this step
        touched_mask = torch.zeros_like(self.last_read, dtype=torch.bool)
        touched_mask.scatter_(0, flat_indices, True)
        
        self.last_read = torch.where(touched_mask, timestep, self.last_read)
        
        # 2. Update read_ema
        # read_ema_t = rho * read_ema_{t-1} + (1-rho) * alpha_sum
        # We accumulate alpha for each slot across the batch
        alpha_sum = torch.zeros_like(self.read_ema)
        alpha_sum.scatter_add_(0, flat_indices, flat_alpha)
        
        # Since typical EMA is per-step, we decay ALL slots first, then add the new alpha
        # Note: The spec says: read_ema_{t,j} = rho * read_ema_{t-1,j} + (1-rho) * alpha_{t,j}
        # If a slot is NOT read, alpha is 0. So it just decays.
        
        # However, if batch_size > 1, "alpha_{t,j}" is effectively the aggregate usage in this step?
        # Or is metadata per-sequence?
        # The spec implies a global memory or per-sequence memory. 
        # "External memory M_t ... with slots m_{t,j}"
        # Usually for LLMs, memory is per-sequence (KV cache equivalent).
        # IF memory is PER-SEQUENCE, then self.read_ema should be [batch_size, J].
        # Let's re-read carefully. "Memory footprint scales... KV cache footprint scales... 
        # THL architecture... bounded external memory (slot-indexed)".
        # Usually recurrent state is per-sequence.
        # So MemoryMetadata should probably handle [batch_size, J] if we want valid batching.
        
        # Implementation Decision: Make metadata [batch_size, J] to support batching correctly.
        pass

class BatchedMemoryMetadata:
    """
    Batched version of metadata tracking.
    """
    def __init__(self, batch_size: int, config: THLConfig):
        self.batch_size = batch_size
        self.num_slots = config.memory_slots
        self.device = config.device
        self.dtype = getattr(torch, config.dtype)
        
        # [batch_size, J]
        self.last_read = torch.zeros(batch_size, self.num_slots, dtype=torch.long, device=self.device)
        self.last_write = torch.zeros(batch_size, self.num_slots, dtype=torch.long, device=self.device)
        self.read_ema = torch.zeros(batch_size, self.num_slots, dtype=self.dtype, device=self.device)
        self.write_ema = torch.zeros(batch_size, self.num_slots, dtype=self.dtype, device=self.device)
        
        self.read_decay = config.read_ema_decay
        self.write_decay = config.write_ema_decay

    def update_read(self, timestep: int, alpha: torch.Tensor, indices: torch.Tensor):
        """
        alpha: [batch_size, kappa]
        indices: [batch_size, kappa]
        """
        if alpha.dim() == 3:
            alpha = alpha.reshape(alpha.size(0), -1)
        if indices.dim() == 3:
            indices = indices.reshape(indices.size(0), -1)

        alpha = alpha.to(device=self.read_ema.device, dtype=self.read_ema.dtype)
        indices = indices.to(device=self.read_ema.device, dtype=torch.long)

        # Create a sparse update tensor
        # We want to add (1-rho)*alpha to the slots at 'indices'
        # And multiply everything by rho first? No, that's: rho*old + (1-rho)*new
        # So: new_ema = rho * old_ema + (1-rho) * scattered_alpha
        
        scattered_alpha = torch.zeros_like(self.read_ema)
        scattered_alpha.scatter_add_(1, indices, alpha)
        
        self.read_ema = self.read_decay * self.read_ema + (1 - self.read_decay) * scattered_alpha
        
        # Update last_read
        # Create a mask of accessed slots
        access_mask = torch.zeros_like(self.last_read, dtype=torch.bool)
        # We can't scatter boolean easily with just 'True', need source tensor
        ones = torch.ones_like(indices, dtype=torch.bool)
        access_mask.scatter_(1, indices, ones)
        
        self.last_read = torch.where(access_mask, timestep, self.last_read)

    def update_write(self, timestep: int, write_indices: torch.Tensor):
        """
        write_indices: [batch_size, W]
        """
        # Delta write is 1 if written, 0 otherwise
        delta_write = torch.zeros_like(self.write_ema)
        ones = torch.ones_like(write_indices, dtype=self.dtype)
        delta_write.scatter_add_(1, write_indices, ones)
        
        self.write_ema = self.write_decay * self.write_ema + (1 - self.write_decay) * delta_write
        
        # Update last_write
        access_mask = torch.zeros_like(self.last_write, dtype=torch.bool)
        bool_ones = torch.ones_like(write_indices, dtype=torch.bool)
        access_mask.scatter_(1, write_indices, bool_ones)
        
        self.last_write = torch.where(access_mask, timestep, self.last_write)

    def get_staleness(self, timestep: int) -> torch.Tensor:
        """
        Returns staleness: t - last_write
        """
        return timestep - self.last_write

    def get_underuse(self, epsilon: float = 1e-8) -> torch.Tensor:
        """
        Returns underuse: 1 / (epsilon + read_ema)
        """
        return 1.0 / (epsilon + self.read_ema)
