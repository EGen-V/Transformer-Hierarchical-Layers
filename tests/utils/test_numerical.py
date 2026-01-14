import torch
import pytest
from thl.utils.numerical import log_sum_exp, masked_stable_softmax, norm_clip_vector

def test_log_sum_exp():
    x = torch.tensor([1.0, 2.0, 3.0])
    lse = log_sum_exp(x, dim=0)
    expected = torch.log(torch.exp(x).sum())
    assert torch.allclose(lse, expected)

def test_masked_stable_softmax():
    x = torch.tensor([10.0, 10.0, 50.0])
    mask = torch.tensor([1.0, 1.0, 0.0]) # Mask the large value
    
    probs = masked_stable_softmax(x, mask, dim=0)
    
    # Outcome should split prob between first two indices equally
    assert torch.allclose(probs[0], torch.tensor(0.5))
    assert torch.allclose(probs[1], torch.tensor(0.5))
    assert torch.allclose(probs[2], torch.tensor(0.0))

def test_norm_clip_vector():
    x = torch.tensor([3.0, 4.0]) # Norm is 5.0
    
    # Clip to max=1.0
    clipped = norm_clip_vector(x, max_norm=1.0)
    assert torch.allclose(clipped.norm(p=2), torch.tensor(1.0))
    
    # Clip to max=10.0 (should be unchanged)
    clipped_large = norm_clip_vector(x, max_norm=10.0)
    assert torch.allclose(clipped_large, x)
