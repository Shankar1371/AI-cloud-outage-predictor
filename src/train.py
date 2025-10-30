import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from src.models.gru import OutageGRU
from src.data.window import SlidingWindowDataset
from src.evaluate import pr_auc, roc_auc, f1_at

class WeightedBCELoss(nn.Module):
    """
    Weighted BCELoss is a custom losss function used for model training
    ///
    In this class the problems that are addressed aare imbalanced dataset where the positive class is rare and compared to the negative class
    this class helps is to be dominated by the negatuve class that leading  the model to predict 0 most of the times
    """
    def __init__(self, pos_weight: float = 1.0):
        super().__init__()
        self.bce = nn.BCELoss(reduction='none')
        self.pos_weight = pos_weight

    def forward(self, preds, targets):
        w = torch.ones_like(targets)
        w[targets==1] = self.pos_weight
        return (self.bce(preds, targets)*w).mean()


class TempScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.t = nn.Parameter(torch.ones(1))

    def forward(self, p):
        # p in (0,1), map to logit, scale, back to prob
        eps = 1e-6
        logit = torch.log(p.clamp(eps,1-eps)) - torch.log(1-p.clamp(eps,1-eps))
        logit = logit / self.t
        return torch.sigmoid(logit)


def train_model(train_ds, val_ds, fm, fe, pos_weight=10.0, epochs=10, lr=1e-3, bs=256, device='cuda'):
    model = OutageGRU(fm, fe).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    crit = WeightedBCELoss(pos_weight)


    train_loader = DataLoader(train_ds, batch_size=bs, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=4)


    best_auc, best_state = 0.0, None
    for ep in range(1, epochs+1):
        model.train()
        for xm, xe, y in train_loader:
            xm, xe, y = xm.to(device), xe.to(device), y.to(device)
            p = model(xm, xe)
            loss = crit(p, y)
            opt.zero_grad(); loss.backward(); opt.step()
# validation
        model.eval()
        preds, targs = [], []
        with torch.no_grad():
            for xm, xe, y in val_loader:
                xm, xe = xm.to(device), xe.to(device)
                p = model(xm, xe)
                preds.append(p.cpu()); targs.append(y)
        import torch
        preds = torch.cat(preds).numpy(); targs = torch.cat(targs).numpy()
        auc = pr_auc(targs, preds)
        print(f"[ep {ep}] PR-AUC={auc:.4f}")
        if auc > best_auc:
            best_auc, best_state = auc, model.state_dict()
    if best_state is not None:
        model.load_state_dict(best_state)
# temperature scaling on val set
    scaler = TempScaler().to(device)
    opt_t = torch.optim.LBFGS(scaler.parameters(), lr=0.1, max_iter=50)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False)


    def _closure():
        opt_t.zero_grad();
        loss = 0.0
        for xm, xe, y in val_loader:
            with torch.no_grad():
                p = model(xm.to(device), xe.to(device))
            p_cal = scaler(p)
            loss += nn.BCELoss()(p_cal, y.to(device))
        loss.backward(); return loss
    opt_t.step(_closure)
    return model, scaler