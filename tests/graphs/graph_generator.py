import os
import matplotlib.pyplot as plt
import torch
from typing import List, Dict

class GraphGenerator:
    """
    Utility to generate graphs from collected metrics.
    """
    def __init__(self, output_dir: str):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def plot_memory_scaling(self, seq_lengths: List[int], memory_mb: List[float], label: str = "THL"):
        """
        Plot memory usage vs sequence length.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(seq_lengths, memory_mb, marker='o', label=label)
        plt.xlabel("Sequence Length")
        plt.ylabel("Memory Usage (MB)")
        plt.title("Memory Scaling vs Sequence Length")
        plt.grid(True)
        plt.legend()
        plt.savefig(os.path.join(self.output_dir, "memory_scaling.png"))
        plt.close()

    def plot_routing_entropy(self, steps: List[int], entropy: List[float]):
        """
        Plot routing entropy over time.
        """
        plt.figure(figsize=(10, 6))
        plt.plot(steps, entropy, label="Routing Entropy")
        plt.xlabel("Step")
        plt.ylabel("Entropy")
        plt.title("Router Entropy Stability")
        plt.grid(True)
        plt.savefig(os.path.join(self.output_dir, "routing_entropy.png"))
        plt.close()
