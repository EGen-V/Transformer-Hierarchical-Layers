import torch
import torch.nn as nn
from thl.inference.streaming import ModuleStreamer

def test_module_streamer(device):
    if not torch.cuda.is_available() and device == "cuda":
        pytest.skip("CUDA not available")
        
    streamer = ModuleStreamer(device)
    module = nn.Linear(10, 10)
    
    # Initially on CPU
    assert next(module.parameters()).device.type == "cpu"
    
    with streamer.use_module(module):
        # Should be on target device
        if device == "cuda":
            assert next(module.parameters()).device.type == "cuda"
        else:
            assert next(module.parameters()).device.type == "cpu"
            
    # Should be back on CPU
    assert next(module.parameters()).device.type == "cpu"
