import torch
import torch.nn as nn
from contextlib import contextmanager

class ModuleStreamer:
    """
    Helper to manage module device placement for layered inference.
    """
    def __init__(self, device: str = "cuda"):
        self.device = device
        
    @contextmanager
    def use_module(self, module: nn.Module):
        """
        Context manager that moves a module to the target device, 
        and moves it back to CPU upon exit.
        """
        # Optimization: Only move if not already on device
        original_device = next(module.parameters()).device
        
        if str(original_device) != self.device:
            module.to(self.device)
            
        try:
            yield module
        finally:
            if str(original_device) != self.device:
                module.to("cpu")
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
