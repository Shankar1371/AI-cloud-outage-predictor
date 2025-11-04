import argparse, torch
from torch.utils.data import DataLoader, TensorDataset

from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from src.models.gru import OutageGRU


def load_split(path):
    pkg = torch.load(path, map_location="cpu")
    ds = TensorDataset(pkg["X_metrics"], pkg["X_events"], pkg["y"])
    return ds, pkg

def train_one_epoch(model, loader, opt, device, pos_weight=10.0):
    bce = torch.nn.BCELoss(reduction="none")
    model.train()
    for xm, xe, y in loader:
        xm, xe, y = xm.to(device), xe.to(device), y.to(device)
        p = model(xm, xe)
        w = torch.ones_like(y); w[y==1] = pos_weight
        loss = (bce(p, y) * w).mean()
        opt.zero_grad(); loss.backward(); opt.step()

def evaluate(model, loader, device):
    import numpy as np
    from sklearn.metrics import average_precision_score, roc_auc_score, f1_score
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for xm, xe, y in loader:
            p = model(xm.to(device), xe.to(device)).cpu().numpy()
            P.append(p); Y.append(y.numpy())
    P = np.concatenate(P); Y = np.concatenate(Y)
    pr = float(average_precision_score(Y, P))
    roc = float(roc_auc_score(Y, P))
    f1 = float(f1_score(Y, (P>=0.5).astype(int)))
    return pr, roc, f1

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)  # folder containing train.pt/val.pt/test.pt
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=10.0)
    args = ap.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    train_ds, meta = load_split(f"{args.data}/train.pt")
    val_ds, _      = load_split(f"{args.data}/val.pt")

    fm = train_ds.tensors[0].shape[-1]
    fe = train_ds.tensors[1].shape[-1]

    model = OutageGRU(fm=fm, fe=fe).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch, shuffle=False)

    best_pr = -1.0
    for ep in range(1, args.epochs+1):
        train_one_epoch(model, train_loader, opt, device, pos_weight=args.pos_weight)
        pr, roc, f1 = evaluate(model, val_loader, device)
        print(f"[epoch {ep}] PR-AUC={pr:.4f} ROC-AUC={roc:.4f} F1={f1:.4f}")
        if pr > best_pr:
            best_pr = pr
            torch.save({"state_dict": model.state_dict(), "fm": fm, "fe": fe}, f"{args.data}/best_gru.pt")

if __name__ == "__main__":
    main()
