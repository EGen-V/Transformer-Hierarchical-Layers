import torch
import pytest
from thl.config import THLConfig
from thl.model import THLForSequenceClassification, THLForMultipleChoice, THLForTokenClassification

def test_sequence_classification(config):
    model = THLForSequenceClassification(config, num_labels=3)
    input_ids = torch.randint(0, config.vocab_size, (2, 5))
    
    # Forward without labels
    logits, loss = model(input_ids)
    assert logits.shape == (2, 3)
    assert loss is None
    
    # Forward with labels
    labels = torch.tensor([0, 2])
    logits, loss = model(input_ids, labels=labels)
    assert loss is not None
    assert loss.item() > 0

def test_multiple_choice(config):
    model = THLForMultipleChoice(config)
    num_choices = 4
    input_ids = torch.randint(0, config.vocab_size, (2, num_choices, 5))
    
    # Forward without labels
    logits, loss = model(input_ids)
    assert logits.shape == (2, 4)
    assert loss is None
    
    # Forward with labels
    labels = torch.tensor([0, 3])
    logits, loss = model(input_ids, labels=labels)
    assert loss is not None

def test_token_classification(config):
    model = THLForTokenClassification(config, num_labels=5)
    input_ids = torch.randint(0, config.vocab_size, (2, 10))
    
    # Forward without labels
    logits, loss = model(input_ids)
    assert logits.shape == (2, 10, 5)
    assert loss is None
    
    # Forward with labels (with ignore index)
    labels = torch.randint(0, 5, (2, 10))
    labels[0, 5:] = -100 # Ignore masking
    logits, loss = model(input_ids, labels=labels)
    assert loss is not None
