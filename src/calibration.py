# src/calibration.py
import torch, torch.nn as nn

class TemperatureScaling(nn.Module):
    def __init__(self):
        super().__init__()
        self.T = nn.Parameter(torch.ones(1))

    def forward(self, logits):
        return logits / self.T

def fit_temperature(logits_val, y_val, max_iter=500, lr=0.01):
    """logits_val, y_val are torch tensors (no sigmoid yet)."""
    ts = TemperatureScaling()
    opt = torch.optim.LBFGS(ts.parameters(), lr=lr, max_iter=max_iter)
    bce = nn.BCEWithLogitsLoss()
    def closure():
        opt.zero_grad()
        loss = bce(ts(logits_val), y_val)
        loss.backward()
        return loss
    opt.step(closure)
    return ts
