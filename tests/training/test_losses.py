import torch
from thl.training.losses import THLLoss

def test_thl_loss(config):
    loss_fn = THLLoss()
    batch_size = 2
    seq_len = 5
    vocab_size = config.output_dim
    
    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))
    
    loss = loss_fn(logits, targets)
    
    assert loss.dim() == 0
    assert loss.item() > 0


def test_thl_loss_with_load_balance_aux(config):
    batch_size = 2
    seq_len = 5
    vocab_size = config.output_dim

    logits = torch.randn(batch_size, seq_len, vocab_size)
    targets = torch.randint(0, vocab_size, (batch_size, seq_len))

    base_loss = THLLoss(load_balance_weight=0.0)(logits, targets)
    aux_loss = THLLoss(load_balance_weight=1.0)(logits, targets, router_alpha={"load_balance_kl": torch.tensor(0.5)})

    assert aux_loss > base_loss
