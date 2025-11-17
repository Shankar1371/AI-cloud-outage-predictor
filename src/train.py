import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

# ensure we can import src.models.gru
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.models.gru import OutageGRU


def load_split(path: Path):
    pkg = torch.load(path, map_location="cpu")
    ds = TensorDataset(pkg["X_metrics"], pkg["X_events"], pkg["y"])
    return ds, pkg


def compute_basic_metrics(y_true, y_prob, threshold=0.5):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    y_pred = (y_prob >= threshold).astype(int)

    tp = int(((y_pred == 1) & (y_true == 1)).sum())
    tn = int(((y_pred == 0) & (y_true == 0)).sum())
    fp = int(((y_pred == 1) & (y_true == 0)).sum())
    fn = int(((y_pred == 0) & (y_true == 1)).sum())
    total = len(y_true)

    acc = (tp + tn) / max(total, 1)
    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    f1 = 0.0 if (prec + rec) == 0 else 2 * prec * rec / (prec + rec)

    return {
        "acc": float(acc),
        "precision": float(prec),
        "recall": float(rec),
        "f1": float(f1),
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


def pr_curve(y_true, y_prob):
    order = np.argsort(-y_prob)
    y_true = y_true[order]
    y_prob = y_prob[order]

    tp = 0
    fp = 0
    tps = []
    fps = []

    for i in range(len(y_true)):
        if y_true[i] == 1:
            tp += 1
        else:
            fp += 1
        tps.append(tp)
        fps.append(fp)

    tps = np.array(tps, dtype=float)
    fps = np.array(fps, dtype=float)
    precision = tps / np.maximum(tps + fps, 1)
    recall = tps / max((y_true == 1).sum(), 1)

    return precision, recall


def roc_curve(y_true, y_prob):
    thresholds = np.unique(y_prob)[::-1]  # descending
    P = max((y_true == 1).sum(), 1)
    N = max((y_true == 0).sum(), 1)

    fprs = []
    tprs = []

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tprs.append(tp / P)
        fprs.append(fp / N)

    return np.array(fprs), np.array(tprs)


def auc(x, y):
    # trapezoidal rule
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return float(np.trapz(y, x))


def evaluate(model, loader, device):
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for xm, xe, y in loader:
            xm = xm.to(device)
            xe = xe.to(device)
            p = model(xm, xe).cpu().numpy()
            P.append(p)
            Y.append(y.numpy())
    P = np.concatenate(P)
    Y = np.concatenate(Y).astype(int)

    metrics = compute_basic_metrics(Y, P, threshold=0.5)
    # PR-AUC & ROC-AUC
    prec, rec = pr_curve(Y, P)
    fpr, tpr = roc_curve(Y, P)
    pr_auc = auc(rec, prec) if len(rec) > 1 else 0.0
    roc_auc = auc(fpr, tpr) if len(fpr) > 1 else 0.5

    metrics["pr_auc"] = pr_auc
    metrics["roc_auc"] = roc_auc
    return metrics, P, Y


def train_one_epoch(model, loader, opt, device, pos_weight=10.0):
    bce = torch.nn.BCELoss(reduction="none")
    model.train()
    running_loss = 0.0
    n_samples = 0

    for xm, xe, y in loader:
        xm, xe, y = xm.to(device), xe.to(device), y.to(device)
        p = model(xm, xe)
        w = torch.ones_like(y)
        w[y == 1] = pos_weight
        loss = (bce(p, y) * w).mean()

        opt.zero_grad()
        loss.backward()
        opt.step()

        running_loss += float(loss.item()) * y.size(0)
        n_samples += y.size(0)

    return running_loss / max(n_samples, 1)


def main():
    import csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True)
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--batch", type=int, default=256)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--pos-weight", type=float, default=10.0)
    args = ap.parse_args()

    data_dir = Path(args.data)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    train_ds, _ = load_split(data_dir / "train.pt")
    val_ds, _ = load_split(data_dir / "val.pt")
    test_ds, _ = load_split(data_dir / "test.pt")

    fm = train_ds.tensors[0].shape[-1]
    fe = train_ds.tensors[1].shape[-1]
    print(f"feature dims -> fm: {fm} fe: {fe}")

    model = OutageGRU(fm=fm, fe=fe).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    log_rows = []
    best_f1 = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        train_loss = train_one_epoch(model, train_loader, opt, device, pos_weight=args.pos_weight)
        val_metrics, _, _ = evaluate(model, val_loader, device)

        print(
            f"[epoch {ep}] PR-AUC={val_metrics['pr_auc']:.4f} "
            f"ROC-AUC={val_metrics['roc_auc']:.4f} "
            f"F1={val_metrics['f1']:.4f}"
        )

        log_rows.append({
            "epoch": ep,
            "train_loss": train_loss,
            "val_pr_auc": val_metrics["pr_auc"],
            "val_roc_auc": val_metrics["roc_auc"],
            "val_acc": val_metrics["acc"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
        })

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {
                "state_dict": model.state_dict(),
                "fm": fm,
                "fe": fe,
            }

    # save best model
    if best_state is not None:
        best_path = data_dir / "best_gru.pt"
        torch.save(best_state, best_path)
        print(f"[save] best GRU model -> {best_path} (best val F1={best_f1:.4f})")

    # save training log
    log_path = data_dir / "gru_training_log.csv"
    with open(log_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "epoch",
                "train_loss",
                "val_pr_auc",
                "val_roc_auc",
                "val_acc",
                "val_precision",
                "val_recall",
                "val_f1",
            ],
        )
        writer.writeheader()
        for row in log_rows:
            writer.writerow(row)
    print(f"[save] training curve -> {log_path}")

    # test eval
    if best_state is not None:
        model.load_state_dict(best_state["state_dict"])

    test_metrics, P_test, Y_test = evaluate(model, test_loader, device)
    print(
        f"[TEST] PR-AUC={test_metrics['pr_auc']:.4f} "
        f"ROC-AUC={test_metrics['roc_auc']:.4f} "
        f"F1={test_metrics['f1']:.4f}"
    )

    # save preds
    preds_path = data_dir / "gru_test_preds.npz"
    np.savez(preds_path, y_true=Y_test, y_prob=P_test)
    print(f"[save] test predictions -> {preds_path}")


if __name__ == "__main__":
    main()
