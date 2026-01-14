from thl.utils.profiling import RoutingDiagnostics

class TrainingDiagnostics(RoutingDiagnostics):
    """
    Extended diagnostics for training loop.
    """
    def __init__(self):
        super().__init__()
        self.loss_history = []
        
    def log_loss(self, loss_val: float):
        self.loss_history.append(loss_val)
