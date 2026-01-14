import torch
from thl.memory.bank import MemoryBank

def test_memory_bank_update(config):
    bank = MemoryBank(config)
    batch_size = 2
    
    mem_state = bank.init_state(batch_size)
    assert mem_state.shape == (batch_size, config.memory_slots, config.memory_dim)
    
    # Fake write inputs
    write_indices = torch.tensor([[0], [1]]) # Batch 0 writes to slot 0, Batch 1 to slot 1
    
    # Write value will be added (after decay)
    # Let's say we write '1.0' vector.
    # Clip(w_t) assumed passed in scaled by gate/eta?
    # bank.forward takes (mem, indices, write_value, gate)
    # wait signature was: forward(memory_state, write_indices, write_content, gate)
    # where write_content is applied with gate: write_content * gate
    
    write_content = torch.ones(batch_size, config.memory_dim)
    gate = torch.ones(batch_size, 1)
    
    # Apply update
    # Initial state is 0. After decay: 0. Add update: 1.0 (if gate=1) at target index.
    new_mem = bank(mem_state, write_indices, write_content, gate)
    
    # Check slots
    # Batch 0, Slot 0 should be 1.0
    assert torch.allclose(new_mem[0, 0], torch.tensor(1.0))
    # Batch 0, Slot 1 was 0, decayed 0 -> 0
    assert torch.allclose(new_mem[0, 1], torch.tensor(0.0))
    
    # Batch 1, Slot 1 should be 1.0
    assert torch.allclose(new_mem[1, 1], torch.tensor(1.0))
    
    # Test second step decay
    # If we pass new_mem back with NO write (gate=0), it should just decay
    zero_gate = torch.zeros(batch_size, 1)
    decayed_mem = bank(new_mem, write_indices, write_content, zero_gate)
    
    expected = 1.0 * config.memory_decay
    assert torch.allclose(decayed_mem[0, 0], torch.tensor(expected))
