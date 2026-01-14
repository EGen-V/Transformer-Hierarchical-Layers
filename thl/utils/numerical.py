import torch
import torch.nn.functional as F

def log_sum_exp(x: torch.Tensor, dim: int = -1, keepdim: bool = False) -> torch.Tensor:
    """
    Numerically stable log-sum-exp: log(sum(exp(x)))
    max_x + log(sum(exp(x - max_x)))
    """
    max_x = x.max(dim=dim, keepdim=True)[0]
    return max_x + torch.log(torch.exp(x - max_x).sum(dim=dim, keepdim=True))

def stable_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Numerically stable softmax using the log-sum-exp trick internally by PyTorch,
    but we expose explicit stable implementation if needed. 
    PyTorch's F.softmax is generally stable, but for masked operations we might need care.
    Here we just wrap F.softmax for consistency with the spec mentions.
    """
    # Simply using F.softmax is standard stable implementation
    return F.softmax(x, dim=dim)

def masked_stable_softmax(x: torch.Tensor, mask: torch.Tensor, dim: int = -1, epsilon: float = 1e-10) -> torch.Tensor:
    """
    Computes masked softmax by setting masked values to -inf before softmax.
    """
    # mask: 1 for keep, 0 for discard
    masked_input = x.masked_fill(mask == 0, float('-inf'))
    return F.softmax(masked_input, dim=dim)

def norm_clip_vector(x: torch.Tensor, max_norm: float, epsilon: float = 1e-6) -> torch.Tensor:
    """
    Clips vector norm to max_norm.
    Clip(w) = w * min(1, max_norm / (||w|| + epsilon))
    """
    norm = x.norm(p=2, dim=-1, keepdim=True)
    scale = torch.clamp(max_norm / (norm + epsilon), max=1.0)
    return x * scale
