from thl.training.diagnostics import TrainingDiagnostics

def test_training_diagnostics():
    diag = TrainingDiagnostics()
    diag.log_loss(0.5)
    diag.log_loss(0.3)
    
    assert len(diag.loss_history) == 2
    assert diag.loss_history == [0.5, 0.3]
