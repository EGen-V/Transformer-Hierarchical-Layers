import torch
from thl.memory.router import SparseRouter

def test_router_forward(config):
    router = SparseRouter(config)
    batch_size = 2
    
    # Input u_t: embedding + s_0 + s_K
    input_dim = config.embedding_dim + config.tier_dims[0] + config.tier_dims[-1]
    u_t = torch.randn(batch_size, input_dim)
    
    mem_state = torch.randn(batch_size, config.memory_slots, config.memory_dim)
    
    # read_ema for load penalty
    read_ema = torch.zeros(batch_size, config.memory_slots)
    
    r_t, alpha, indices = router(u_t, mem_state, read_ema)
    
    # Check shapes
    assert r_t.shape == (batch_size, config.value_dim)
    assert alpha.shape == (batch_size, config.num_heads, config.read_topk)
    assert indices.shape == (batch_size, config.num_heads, config.read_topk)
    
    # Check probabilities sum to 1
    assert torch.allclose(alpha.sum(dim=-1), torch.ones(batch_size, config.num_heads))
