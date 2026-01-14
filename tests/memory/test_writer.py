import torch
from thl.config import THLConfig
from thl.memory.writer import MemoryWriter


def _make_config() -> THLConfig:
    return THLConfig(
        vocab_size=100,
        embedding_dim=8,
        hidden_dim=8,
        num_tiers=2,
        tier_dims=[8, 8],
        tier_timescales=[1, 2],
        memory_slots=8,
        memory_dim=8,
        query_dim=8,
        value_dim=8,
        read_topk=2,
        write_slots=1,
        novelty_beta=50.0,
        novelty_theta=0.3,
        device="cpu",
    )


def test_writer_writes_to_empty_on_novel():
    config = _make_config()
    writer = MemoryWriter(config)

    writer.W_w.weight.data.zero_()
    writer.W_w.bias.data.fill_(1.0)

    batch_size = 1
    s_0 = torch.zeros(batch_size, config.tier_dims[0])
    s_K = torch.zeros(batch_size, config.tier_dims[-1])
    r_t = torch.zeros(batch_size, config.value_dim)

    staleness = torch.zeros(batch_size, config.memory_slots)
    underuse = torch.zeros(batch_size, config.memory_slots)
    memory_state = torch.zeros(batch_size, config.memory_slots, config.memory_dim)

    _, g_t, write_indices = writer(s_0, s_K, r_t, staleness, underuse, memory_state)

    assert write_indices.item() == 0
    assert g_t.item() > 0.5


def test_writer_updates_most_relevant_on_not_novel():
    config = _make_config()
    writer = MemoryWriter(config)

    writer.W_w.weight.data.zero_()
    writer.W_w.bias.data.fill_(1.0)

    batch_size = 1
    s_0 = torch.zeros(batch_size, config.tier_dims[0])
    s_K = torch.zeros(batch_size, config.tier_dims[-1])
    r_t = torch.zeros(batch_size, config.value_dim)

    staleness = torch.zeros(batch_size, config.memory_slots)
    underuse = torch.zeros(batch_size, config.memory_slots)

    memory_state = torch.zeros(batch_size, config.memory_slots, config.memory_dim)
    w_t_ref = writer.W_w.bias.detach().clone()
    w_t_ref = w_t_ref / w_t_ref.norm(p=2).clamp_min(1e-9)
    memory_state[0, 5] = w_t_ref

    _, g_t, write_indices = writer(s_0, s_K, r_t, staleness, underuse, memory_state)

    assert write_indices.item() == 5
    assert torch.allclose(g_t, torch.ones_like(g_t))


def test_writer_uses_lru_when_novel_and_no_empty():
    config = _make_config()
    writer = MemoryWriter(config)

    writer.W_w.weight.data.zero_()
    writer.W_w.bias.data.fill_(1.0)

    batch_size = 1
    s_0 = torch.zeros(batch_size, config.tier_dims[0])
    s_K = torch.zeros(batch_size, config.tier_dims[-1])
    r_t = torch.zeros(batch_size, config.value_dim)

    staleness = torch.zeros(batch_size, config.memory_slots)
    staleness[0, 7] = 10.0
    underuse = torch.zeros(batch_size, config.memory_slots)

    memory_state = torch.eye(config.memory_slots).unsqueeze(0)

    _, g_t, write_indices = writer(s_0, s_K, r_t, staleness, underuse, memory_state)

    assert write_indices.item() == 7
    assert g_t.item() > 0.5
