import torch
import time
from typing import List
from thl.config import THLConfig
from thl.inference.layered import LayeredInferenceEngine
from thl.inference.state import InferenceState
from thl.model import THLModel
from thl.utils.profiling import Profiler

class MemoryBenchmark:
    def __init__(self, device="cuda"):
        self.device = device
        self.config = THLConfig(
            num_tiers=3,
            memory_slots=1024,
            embedding_dim=256,
            hidden_dim=256,
            device=device
        )
        self.model = THLModel(self.config)
        self.engine = LayeredInferenceEngine(self.model, device=device)
        
    def profile_sequence(self, length: int) -> float:
        """
        Runs a sequence of 'length' and returns peak memory usage (MB).
        """
        if self.device == "cuda":
            torch.cuda.reset_peak_memory_stats()
            torch.cuda.empty_cache()
            
        state = InferenceState.init(1, self.config, self.model.tiers, self.model.memory_bank)
        
        # Warmup
        token = torch.tensor([1], device=self.device)
        
        # Run sequence
        for _ in range(length):
            # We don't detach here to simulate real inference accumulation? 
            # No, inference state IS detached naturally in THL (it's updated in place or replaced).
            # But standard Autograd accumulates graph.
            # LayeredInferenceEngine typically separates graph per step or uses no_grad.
            # Use no_grad for inference benchmark.
            with torch.no_grad():
                logits, state = self.engine.step(token, state)
                
        if self.device == "cuda":
            peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            # CPU memory profiling is harder, use dummy value or psutil if available.
            # For this script, we'll return 0 if CPU, but THL is targeted at 4GB VRAM GPU.
            peak_mem = 0.0
            
        return peak_mem

def main():
    if not torch.cuda.is_available():
        print("CUDA not available. Skipping memory benchmark.")
        return

    benchmark = MemoryBenchmark(device="cuda")
    lengths = [32, 64, 128, 256, 512, 1024]
    mem_usages = []
    
    print(f"{'Length':<10} | {'Peak Memory (MB)':<15}")
    print("-" * 30)
    
    for length in lengths:
        mem = benchmark.profile_sequence(length)
        mem_usages.append(mem)
        print(f"{length:<10} | {mem:<15.2f}")

    # Verify Boundedness: Last memory should be close to First memory (or at least not linear)
    # Allows for some fragmentation overhead.
    # Typically THL memory: J*dm is constant. Tiers constant. 
    # Only input history might grow if we kept it, but we don't.
    
    # Save graph
    try:
        from tests.graphs.graph_generator import GraphGenerator
        gen = GraphGenerator("results/")
        gen.plot_memory_scaling(lengths, mem_usages, label="THL-Layered")
        print("Graph saved to results/memory_scaling.png")
    except ImportError:
        print("Graph generator not found or matplotlib missing.")

if __name__ == "__main__":
    main()
