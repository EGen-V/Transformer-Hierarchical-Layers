import torch
from thl.training.straight_through import StraightThroughTopK

def test_ste_topk_forward():
    scores = torch.tensor([[10.0, 2.0, 5.0], [1.0, 8.0, 3.0]])
    k = 1
    
    # Forward pass
    mask = StraightThroughTopK.apply(scores, k)
    
    # Check shape
    assert mask.shape == scores.shape
    
    # Check correctness (Hard Top1)
    # Row 0: 10.0 is max -> index 0
    assert torch.allclose(mask[0], torch.tensor([1.0, 0.0, 0.0]))
    # Row 1: 8.0 is max -> index 1
    assert torch.allclose(mask[1], torch.tensor([0.0, 1.0, 0.0]))

def test_ste_topk_backward():
    scores = torch.tensor([10.0, 5.0, 2.0], requires_grad=True)
    k = 1
    
    mask = StraightThroughTopK.apply(scores, k)
    
    # Fake loss: sum(mask * scores) -> should behave like selecting the top value
    loss = (mask * scores).sum()
    loss.backward()
    
    # Gradient should exist for scores
    assert scores.grad is not None
    # With simple pass-through or re-weighted, we expect non-zero grads.
