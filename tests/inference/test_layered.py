import torch
import pytest
from thl.model import THLModel
from thl.inference.layered import LayeredInferenceEngine
from thl.inference.state import InferenceState

def test_layered_inference_step(config):
    model = THLModel(config)
    engine = LayeredInferenceEngine(model, device=config.device)
    state = InferenceState.init(1, config, model.tiers, model.memory_bank)
    
    token = torch.tensor([50], device=config.device)
    
    # Step
    logits, new_state = engine.step(token, state)
    
    # Check shapes
    assert logits.shape == (1, config.vocab_size)
    assert new_state.timestep == 1
    assert len(new_state.tier_states) == config.num_tiers
    assert new_state.local_buffer.shape == (1, config.local_window, config.embedding_dim)
    assert new_state.local_valid == 1
