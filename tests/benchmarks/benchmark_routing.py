import torch
from thl.config import THLConfig
from thl.inference.layered import LayeredInferenceEngine
from thl.inference.state import InferenceState
from thl.model import THLModel
from thl.utils.profiling import RoutingDiagnostics

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    config = THLConfig(num_tiers=2, memory_slots=128, device=device)
    model = THLModel(config)
    engine = LayeredInferenceEngine(model, device=device)
    
    state = InferenceState.init(1, config, model.tiers, model.memory_bank)
    diagnostics = RoutingDiagnostics()
    
    token = torch.tensor([1])
    seq_len = 50
    steps = list(range(seq_len))
    
    print("Running Routing Benchmark...")
    with torch.no_grad():
        for t in range(seq_len):
            _, state = engine.step(token, state, diagnostics=diagnostics)
            
    stats = diagnostics.get_stats()
    print("Routing Statistics:")
    print(f"Avg Entropy: {stats['avg_entropy']:.4f}")
    print(f"Avg Top1 Mass: {stats['avg_top1_mass']:.4f}")
    print(f"Unique Slots Used: {stats['unique_slots_used']}/{config.memory_slots}")
    
    try:
        from tests.graphs.graph_generator import GraphGenerator
        gen = GraphGenerator("results/")
        gen.plot_routing_entropy(steps, diagnostics.entropy_history)
        print("Graph saved to results/routing_entropy.png")
    except Exception as e:
        print(f"Graph generation failed: {e}")

if __name__ == "__main__":
    main()
