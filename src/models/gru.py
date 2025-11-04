import torch
import torch.nn as nn

class OutageGRU(nn.Module):
    def __init__(self, fm: int, fe: int, dh: int = 128, de: int = 32):
        super().__init__()
        self.gru_m = nn.GRU(input_size=fm, hidden_size=dh, num_layers=2, batch_first=True, dropout=0.1)
        self.gru_e = nn.GRU(input_size=fe, hidden_size=de, num_layers=1, batch_first=True)
        self.head  = nn.Sequential(nn.Linear(dh+de, 128), nn.ReLU(), nn.Linear(128, 1))

    def forward(self, xm, xe):
        # xm, xe: [B, T, F]
        _, hm = self.gru_m(xm)  # [layers,B,dh]
        _, he = self.gru_e(xe)  # [layers,B,de]
        h = torch.cat([hm[-1], he[-1]], dim=1)
        return torch.sigmoid(self.head(h)).squeeze(1)
