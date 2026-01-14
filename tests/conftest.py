import pytest
import torch
from thl.config import THLConfig

@pytest.fixture
def config():
    """
    Default small config for faster testing.
    """
    return THLConfig(
        vocab_size=100,
        embedding_dim=32,
        hidden_dim=32,
        num_tiers=2,
        tier_dims=[32, 32],
        tier_timescales=[1, 4],
        memory_slots=64,
        memory_dim=32,
        query_dim=16,
        value_dim=32,
        read_topk=4,
        write_slots=1,
        device="cpu"
    )

@pytest.fixture
def device():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"
