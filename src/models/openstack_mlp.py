# src/models/openstack_mlp.py
import torch
import torch.nn as nn
import torch.nn.functional as F

class OpenStackMLP(nn.Module):
    """
    Simple MLP for binary failure prediction on the OpenStack dataset.

    Input:  X [batch_size, D]   (feature vector per experiment)
    Output: p [batch_size]      (probability of failure)
    """
    def __init__(self, d_in: int, hidden: int = 64, dropout: float = 0.2):
        super().__init__()
        self.fc1 = nn.Linear(d_in, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        """
        x: [B, D]
        returns: p in [0,1], shape [B]
        """
        h = F.relu(self.fc1(x))
        h = self.dropout(h)
        h = F.relu(self.fc2(h))
        h = self.dropout(h)
        logit = self.fc3(h).squeeze(-1)  # [B]
        p = torch.sigmoid(logit)
        return p, logit
