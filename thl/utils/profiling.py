import torch
import time
from collections import defaultdict
from typing import Dict, Any, Optional

class Profiler:
    """
    Simple profiler to track memory usage and execution time.
    """
    def __init__(self):
        self.metrics: Dict[str, list] = defaultdict(list)
        self.start_times: Dict[str, float] = {}

    def start(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        self.start_times[name] = time.time()

    def stop(self, name: str):
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        if name in self.start_times:
            elapsed = time.time() - self.start_times[name]
            self.metrics[f"{name}_time_sec"].append(elapsed)
            del self.start_times[name]

    def log_memory(self, step: int):
        if torch.cuda.is_available():
            mem = torch.cuda.memory_allocated() / 1024**2 # MB
            max_mem = torch.cuda.max_memory_allocated() / 1024**2
            self.metrics["memory_allocated_mb"].append(mem)
            self.metrics["max_memory_allocated_mb"].append(max_mem)
            self.metrics["step"].append(step)

    def get_summary(self) -> Dict[str, float]:
        summary = {}
        for key, values in self.metrics.items():
            if values:
                summary[f"avg_{key}"] = sum(values) / len(values)
                summary[f"max_{key}"] = max(values)
        return summary

class RoutingDiagnostics:
    """
    diagnostic collector for router stats.
    Logs entropy, slot usage, etc.
    """
    def __init__(self):
        self.slot_usage = defaultdict(int) 
        self.entropy_history = []
        self.top1_mass_history = []
        
    def update(self, alpha: torch.Tensor, indices: torch.Tensor):
        """
        alpha: [batch_size, kappa] - attention weights for topk
        indices: [batch_size, kappa] - slot indices
        """
        # Track slot usage
        flat_indices = indices.flatten().tolist()
        for idx in flat_indices:
            self.slot_usage[idx] += 1
            
        # Calculate entropy of the categorical distribution over selected slots
        # For simplicity, we just look at the entropy of the alpha weights themselves 
        # (though this is local entropy, global entropy requires full distribution)
        # Spec says: "Routing entropy, effective slots used, top-1 mass"
        
        # Local entropy of the prediction
        # H(p) = -sum p log p
        entropy = -(alpha * torch.log(alpha + 1e-10)).sum(dim=-1)
        self.entropy_history.extend(entropy.detach().cpu().numpy().tolist())
        
        # Top-1 mass
        top1, _ = alpha.max(dim=-1)
        self.top1_mass_history.extend(top1.detach().cpu().numpy().tolist())

    def get_stats(self):
        return {
            "avg_entropy": sum(self.entropy_history) / max(1, len(self.entropy_history)),
            "avg_top1_mass": sum(self.top1_mass_history) / max(1, len(self.top1_mass_history)),
            "unique_slots_used": len(self.slot_usage)
        }
