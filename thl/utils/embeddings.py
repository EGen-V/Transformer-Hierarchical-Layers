import torch

def precompute_rope_freqs(dim: int, max_seq_len: int = 10000, base: float = 10000.0, device: str = "cpu") -> torch.Tensor:
    """
    Precompute theta parameters for RoPE.
    Returns: [max_seq_len, dim // 2]
    """
    theta = 1.0 / (base ** (torch.arange(0, dim, 2, dtype=torch.float32, device=device) / dim))
    m = torch.arange(max_seq_len, device=device, dtype=torch.float32)
    freqs = torch.outer(m, theta)
    # Return in complex form often cleaner, but here we stick to real ops for compatibility
    # shape [max_seq_len, dim/2]
    return freqs

def apply_rope(x: torch.Tensor, freqs: torch.Tensor) -> torch.Tensor:
    """
    Applies Rotary Positional Embeddings to input x.
    x: [B, H, T, D] or [B, T, D] or [B, D]
    freqs: [T, D/2] or just [D/2] for single step
    
    Note: THL is non-Transformer. 
    1. Query q_t usually depends on current step 't'. So we need freq for 't'.
    2. Memory slots are NOT strictly ordered by time 't', they are content-addressable slots.
       However, spec mentions "Sequence Order Inductive Biases" via RoPE for "both tokens and slot interactions".
       
    Interpretation:
    - If slots effectively represent positions (e.g. slot `j` has associated `pos_j`), we apply RoPE.
    - If slots are semantic buckets, RoPE might not apply to Key `k_j`.
    
    BUT, Spec says: "RoPE ... for both tokens and slot interactions".
    This implies we might encode relative distance between current query time `t` and memory write time `t_write`?
    
    If we stick to standard RoPE:
    - q at pos t gets rotated by R_t
    - k at pos t' gets rotated by R_{t'}
    - q.T @ k = cos(t-t') ...
    
    For THL:
    - q_t is at time t.
    - k_j is static weight matrix projection of m_{t-1, j}.
    
    If we want RoPE, we need to associate a position with each slot?
    Or maybe we just apply it to the Tier states?
    
    Let's implement the generic apply function first.
    Assuming x is [B, ..., D].
    """
    # Assuming standard half-rotation implementation
    d = x.shape[-1]
    x_real = x[..., :d//2]
    x_imag = x[..., d//2:]
    
    # freqs usually [..., D/2]
    # We broaden freqs to match x batch dims
    sin = torch.sin(freqs)
    cos = torch.cos(freqs)
    
    # Broadcating
    # x: [..., D] -> pair
    # freqs need compatible shape
    
    # x_out = [real*cos - imag*sin, real*sin + imag*cos]
    out_real = x_real * cos - x_imag * sin
    out_imag = x_real * sin + x_imag * cos
    return torch.cat([out_real, out_imag], dim=-1)
