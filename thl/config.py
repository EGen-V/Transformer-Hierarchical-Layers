from dataclasses import dataclass, field
from typing import List, Optional

@dataclass
class THLConfig:
    """
    Configuration for Transformer Hierarchical Layers (THL) model.
    """
    # Model Dimensions
    vocab_size: int = 50257  # GPT-2 default
    embedding_dim: int = 768
    hidden_dim: int = 768
    
    # Tier Hierarchy
    num_tiers: int = 3
    tier_dims: List[int] = field(default_factory=lambda: [768, 768, 768])
    tier_timescales: List[int] = field(default_factory=lambda: [1, 8, 64])

    local_window: int = 64
    
    # Memory System
    memory_slots: int = 1024  # J
    memory_dim: int = 768     # d_m
    query_dim: int = 64      # d_q
    value_dim: int = 768     # d_v
    pad_token_id: Optional[int] = None
    
    # Routing (Read)
    num_heads: int = 4       # H
    head_dim: int = 64       # d_h (usually query_dim / num_heads or independent)
    read_topk: int = 4       # kappa (PER HEAD)
    load_penalty: float = 0.1  # lambda_load
    read_ema_decay: float = 0.99  # rho (for read metadata)
    read_slot_capacity: Optional[int] = None
    router_load_balance_lambda: float = 0.0
    
    # Writing
    write_slots: int = 1     # W
    memory_decay: float = 0.9995  # gamma
    write_rate: float = 0.05    # eta
    write_clip_max: float = 1.0 # Delta_max
    write_ema_decay: float = 0.99 # rho (for write metadata)
    
    # Novelty Gate
    novelty_beta: float = 8.0
    novelty_theta: float = 0.3
    stale_weight: float = 1.0   # lambda_s (placeholder, not strictly defined in spec but implied for scoring)
    underuse_weight: float = 1.0 # lambda_u (placeholder)

    # Output
    # output_dim removed as field, using property
    
    # System
    dtype: str = "float32"
    device: str = "cpu"

    @property
    def output_dim(self) -> int:
        return self.vocab_size

    def __post_init__(self):
        from thl.errors import THLConfigError
        
        if len(self.tier_dims) != self.num_tiers:
            raise THLConfigError(
                f"Mismatch in tier dimensions: defined {self.num_tiers} tiers but provided {len(self.tier_dims)} dimensions.",
                hint="Ensure 'tier_dims' list length matches 'num_tiers'."
            )
            
        if len(self.tier_timescales) != self.num_tiers:
            raise THLConfigError(
                f"Mismatch in tier timescales: defined {self.num_tiers} tiers but provided {len(self.tier_timescales)} timescales.",
                hint="Ensure 'tier_timescales' list length matches 'num_tiers'."
            )
            
        if self.tier_timescales[0] != 1:
            raise THLConfigError(
                f"Invalid base tier timescale: {self.tier_timescales[0]}.",
                hint="The first tier (Tier 0) must always have a timescale of 1 (token-level)."
            )

