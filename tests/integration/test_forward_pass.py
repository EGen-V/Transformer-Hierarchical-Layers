import torch
from thl.model import THLModel
from thl.inference.state import InferenceState

def test_full_model_forward(config):
    model = THLModel(config)
    batch_size = 2
    seq_len = 5
    
    input_ids = torch.randint(0, config.vocab_size, (batch_size, seq_len))
    
    # Test Sequence Forward
    logits, state = model(input_ids)
    
    assert logits.shape == (batch_size, seq_len, config.output_dim)
    assert state.timestep == seq_len
    assert state.local_buffer.shape == (batch_size, config.local_window, config.embedding_dim)
    assert state.local_valid == min(seq_len, config.local_window)

def test_step_forward(config):
    model = THLModel(config)
    batch_size = 1
    
    state = InferenceState.init(batch_size, config, model.tiers, model.memory_bank)
    token_id = torch.tensor([config.vocab_size - 1])
    
    # Single step
    logits, state = model.forward_step(token_id, state)
    
    assert logits.shape == (batch_size, config.output_dim)
    assert state.timestep == 1
    assert state.local_buffer.shape == (batch_size, config.local_window, config.embedding_dim)
    assert state.local_valid == 1
