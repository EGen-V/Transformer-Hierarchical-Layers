import torch
from thl.tiers.cell import TierCell

def test_tier_cell_update(config):
    # Tier 0 has different input dim (embedding vs prev tier)
    cell = TierCell(0, config)
    batch_size = 2
    
    s_prev = cell.init_state(batch_size, config.device)
    lower_input = torch.randn(batch_size, config.embedding_dim) # Tier 0 input is embedding
    memory_read = torch.randn(batch_size, config.value_dim)
    
    # Test Active Update
    s_new = cell(s_prev, lower_input, memory_read, active=True)
    assert s_new.shape == s_prev.shape
    assert not torch.allclose(s_new, s_prev) # Should have changed
    
    # Test Inactive (Clocked) Update
    s_skipped = cell(s_new, lower_input, memory_read, active=False)
    assert torch.allclose(s_skipped, s_new) # Should be identical
