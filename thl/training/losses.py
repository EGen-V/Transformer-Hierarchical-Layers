import torch
import torch.nn as nn
from thl.utils.numerical import log_sum_exp
import math

class THLLoss(nn.Module):
    """
    Composite loss for THL training.
    """
    def __init__(self, load_balance_weight: float = 0.0):
        super().__init__()
        self.ce_loss = nn.CrossEntropyLoss()
        self.load_balance_weight = load_balance_weight
        
    def forward(self, logits: torch.Tensor, targets: torch.Tensor, 
                router_alpha: torch.Tensor = None) -> torch.Tensor:
        """
        Args:
            logits: [B, T, vocab_size]
            targets: [B, T]
            router_alpha: [B, T, J] optional for auxiliary loss
        """
        # 1. Main Task Loss
        # Flatten for CE
        loss = self.ce_loss(logits.view(-1, logits.size(-1)), targets.view(-1))
        
        # 2. Auxiliary Loss (Load Balancing)
        # If router_alpha provided (full distribution or soft approximation)
        # Spec says "THL uses hard TopK", so load balancing is handled via 
        # "load penalty" in the forward pass, not necessarily an aux loss.
        # But commonly MoE uses aux loss: LoadBalance = J * sum(f_j * P_j)
        # Here we leave it as placeholder if needed.
        
        if self.load_balance_weight > 0 and router_alpha is not None:
            if isinstance(router_alpha, dict) and "load_balance_kl" in router_alpha:
                load_balance_kl = router_alpha["load_balance_kl"]
                if not torch.is_tensor(load_balance_kl):
                    load_balance_kl = torch.tensor(load_balance_kl, device=loss.device, dtype=loss.dtype)
                load_balance_kl = load_balance_kl.to(device=loss.device, dtype=loss.dtype)
                loss = loss + self.load_balance_weight * load_balance_kl
            elif torch.is_tensor(router_alpha) and router_alpha.dim() >= 1:
                if router_alpha.dim() >= 3:
                    reduce_dims = tuple(range(router_alpha.dim() - 1))
                    p = router_alpha.mean(dim=reduce_dims)
                else:
                    p = router_alpha
                p = p.to(device=loss.device, dtype=loss.dtype)
                p = p / p.sum().clamp_min(1e-9)
                kl = (p * (torch.log(p.clamp_min(1e-9)) + math.log(p.numel()))).sum()
                loss = loss + self.load_balance_weight * kl
             
        return loss
