# src/openstack_train.py

import argparse
from pathlib import Path
import sys

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


def load_split(path: Path):
    """
    Expect .npz file with keys: X (N, d_in), y (N,)
    """
    data = np.load(path)
    X = torch.tensor(data["X"], dtype=torch.float32)
    y = torch.tensor(data["y"], dtype=torch.float32)
    ds = TensorDataset(X, y)
    return ds, X.shape[1]


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
    tps, fps = [], []
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

    fprs, tprs = [], []
    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tprs.append(tp / P)
        fprs.append(fp / N)
    return np.array(fprs), np.array(tprs)


def auc(x, y):
    order = np.argsort(x)
    x = x[order]
    y = y[order]
    return float(np.trapz(y, x))


def evaluate(model, loader, device):
    model.eval()
    P, Y = [], []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            p = model(X).cpu().numpy().ravel()
            P.append(p)
            Y.append(y.numpy())
    P = np.concatenate(P)
    Y = np.concatenate(Y).astype(int)

    metrics = compute_basic_metrics(Y, P, threshold=0.5)
    prec, rec = pr_curve(Y, P)
    fpr, tpr = roc_curve(Y, P)
    metrics["pr_auc"] = auc(rec, prec) if len(rec) > 1 else 0.0
    metrics["roc_auc"] = auc(fpr, tpr) if len(fpr) > 1 else 0.5
    return metrics, P, Y


def build_mlp(d_in, hidden=128):
    return torch.nn.Sequential(
        torch.nn.Linear(d_in, hidden),
        torch.nn.ReLU(),
        torch.nn.Dropout(0.2),
        torch.nn.Linear(hidden, hidden // 2),
        torch.nn.ReLU(),
        torch.nn.Linear(hidden // 2, 1),
        torch.nn.Sigmoid(),
    )


def main():
    import csv

    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Folder with train.npz / val.npz / test.npz")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    data_dir = Path(args.data)
    print(f"[config] data_dir={data_dir}")

    train_ds, d_in = load_split(data_dir / "train.npz")
    val_ds, _ = load_split(data_dir / "val.npz")
    test_ds, _ = load_split(data_dir / "test.npz")

    print(f"[config] d_in={d_in}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[config] device={device}")

    # compute pos_weight for BCE
    y_all = torch.cat([train_ds.tensors[1], val_ds.tensors[1], test_ds.tensors[1]], dim=0)
    n_pos = float((y_all == 1).sum().item())
    n_neg = float((y_all == 0).sum().item())
    if n_pos > 0 and n_neg > 0:
        pos_weight = torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float32).to(device)
    print(f"[info] pos_weight={pos_weight.item():.3f}")

    model = build_mlp(d_in).to(device)
    bce = torch.nn.BCELoss(weight=None, reduction="none")
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    train_loader = DataLoader(train_ds, batch_size=args.batch, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=args.batch, shuffle=False)

    log_rows = []
    best_f1 = -1.0
    best_state = None

    for ep in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_samples = 0

        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            p = model(X).view(-1)
            w = torch.ones_like(y)
            w[y == 1] = pos_weight.item()
            loss = (bce(p, y) * w).mean()

            opt.zero_grad()
            loss.backward()
            opt.step()

            total_loss += float(loss.item()) * y.size(0)
            n_samples += y.size(0)

        train_loss = total_loss / max(n_samples, 1)
        val_metrics, _, _ = evaluate(model, val_loader, device)

        print(
            f"[epoch {ep:02d}] loss={train_loss:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"(P={val_metrics['precision']:.3f}, R={val_metrics['recall']:.3f})"
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
                "d_in": d_in,
            }

    # save best model
    if best_state is not None:
        best_path = data_dir / "best_openstack_mlp.pt"
        torch.save(best_state, best_path)
        print(f"[save] best model -> {best_path} (best val F1={best_f1:.4f})")

    # save training log
    log_path = data_dir / "openstack_train_log.csv"
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
    print(f"[save] training log -> {log_path}")

    # final test evaluation
    if best_state is not None:
        model.load_state_dict(best_state["state_dict"])

    test_metrics, P_test, Y_test = evaluate(model, test_loader, device)
    print(
        f"[TEST] acc={test_metrics['acc']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"P={test_metrics['precision']:.4f} "
        f"R={test_metrics['recall']:.4f} "
        f"tp={test_metrics['tp']} fp={test_metrics['fp']} "
        f"fn={test_metrics['fn']} tn={test_metrics['tn']}"
    )

    preds_path = data_dir / "openstack_test_preds.npz"
    np.savez(preds_path, y_true=Y_test, y_prob=P_test)
    print(f"[save] test predictions -> {preds_path}")


if __name__ == "__main__":
    main()
