import torch
from thl.inference.state import InferenceState
from thl.memory.bank import MemoryBank
from thl.tiers.stack import HierarchicalTierStack

def test_inference_state_init(config):
    tiers = HierarchicalTierStack(config)
    bank = MemoryBank(config)
    
    batch_size = 2
    state = InferenceState.init(batch_size, config, tiers, bank)
    
    assert state.timestep == 0
    assert state.memory_state.shape == (batch_size, config.memory_slots, config.memory_dim)
    assert len(state.tier_states) == config.num_tiers
    assert state.local_buffer.shape == (batch_size, config.local_window, config.embedding_dim)
    assert state.local_valid == 0
    
    # Check bounding: state size should not change with logical timestep updates
    # (Functional check done via usage, here just static props)
    pass
