# src/openstack_train.py
import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import TensorDataset, DataLoader

from src.models.openstack_mlp import OpenStackMLP


def load_split(path: Path):
    """
    Each .pt file from prepare_openstack.py contains:
      - X: [N, D] float32
      - y: [N]    float32 (0 or 1)
      - features: list[str]
      - workload: str
    """
    pkg = torch.load(path, map_location="cpu")
    X = pkg["X"]
    y = pkg["y"]
    features = pkg.get("features", None)
    workload = pkg.get("workload", None)
    ds = TensorDataset(X, y)
    return ds, features, workload


def compute_class_weights(y_train: torch.Tensor):
    """
    Compute pos_weight for BCEWithLogitsLoss to handle imbalance.
      pos_weight = (N_neg / N_pos)
    """
    y_np = y_train.numpy()
    n_pos = float((y_np == 1).sum())
    n_neg = float((y_np == 0).sum())
    if n_pos == 0:
        pos_weight = 1.0
    else:
        pos_weight = n_neg / max(n_pos, 1.0)
    return torch.tensor([pos_weight], dtype=torch.float32)


def evaluate(model, loader, device):
    """
    Evaluate using accuracy, precision, recall, F1.
    Implemented with pure numpy (no sklearn) to stay rubric-safe.
    """
    model.eval()
    all_p = []
    all_y = []
    with torch.no_grad():
        for X, y in loader:
            X = X.to(device)
            p, _ = model(X)
            all_p.append(p.cpu().numpy())
            all_y.append(y.cpu().numpy())

    p = np.concatenate(all_p)  # probabilities
    y = np.concatenate(all_y).astype(int)  # 0/1

    # Threshold at 0.5
    y_hat = (p >= 0.5).astype(int)

    # Confusion matrix components
    tp = int(((y_hat == 1) & (y == 1)).sum())
    tn = int(((y_hat == 0) & (y == 0)).sum())
    fp = int(((y_hat == 1) & (y == 0)).sum())
    fn = int(((y_hat == 0) & (y == 1)).sum())

    total = len(y)
    acc = (tp + tn) / max(total, 1)

    prec = tp / max(tp + fp, 1)
    rec = tp / max(tp + fn, 1)
    if prec + rec == 0:
        f1 = 0.0
    else:
        f1 = 2 * prec * rec / (prec + rec)

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


def train_openstack(
    data_dir: str,
    epochs: int = 20,
    batch_size: int = 64,
    lr: float = 1e-3,
    device: str = None,
):
    data_dir = Path(data_dir)

    train_ds, features, workload = load_split(data_dir / "train.pt")
    val_ds, _, _ = load_split(data_dir / "val.pt")
    test_ds, _, _ = load_split(data_dir / "test.pt")

    d_in = train_ds.tensors[0].shape[1]

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"[config] data_dir={data_dir}")
    print(f"[config] workload={workload}, d_in={d_in}, device={device}")
    if features is not None:
        print(f"[info] {len(features)} features")

    model = OpenStackMLP(d_in=d_in, hidden=64, dropout=0.2).to(device)

    # Compute pos_weight from training labels
    y_train = train_ds.tensors[1]
    pos_weight = compute_class_weights(y_train)
    print(f"[info] pos_weight={pos_weight.item():.3f}")

    criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight.to(device))
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False)

    best_f1 = -1.0
    best_state = None

    for ep in range(1, epochs + 1):
        model.train()
        running_loss = 0.0
        for X, y in train_loader:
            X = X.to(device)
            y = y.to(device)
            p, logit = model(X)
            loss = criterion(logit, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item()) * X.size(0)

        train_loss = running_loss / max(len(train_ds), 1)
        val_metrics = evaluate(model, val_loader, device)
        print(
            f"[epoch {ep:02d}] "
            f"loss={train_loss:.4f} "
            f"val_acc={val_metrics['acc']:.4f} "
            f"val_f1={val_metrics['f1']:.4f} "
            f"(P={val_metrics['precision']:.3f}, R={val_metrics['recall']:.3f})"
        )

        if val_metrics["f1"] > best_f1:
            best_f1 = val_metrics["f1"]
            best_state = {
                "model_state": model.state_dict(),
                "d_in": d_in,
                "features": features,
                "workload": workload,
            }

    # Save best model (optional)
    if best_state is not None:
        out_path = data_dir / "best_openstack_mlp.pt"
        torch.save(best_state, out_path)
        print(f"[save] best model -> {out_path} (best val F1={best_f1:.4f})")

    # Evaluate on test set
    if best_state is not None:
        model.load_state_dict(best_state["model_state"])
    test_metrics = evaluate(model, test_loader, device)
    print(
        "[TEST] "
        f"acc={test_metrics['acc']:.4f} "
        f"f1={test_metrics['f1']:.4f} "
        f"P={test_metrics['precision']:.4f} "
        f"R={test_metrics['recall']:.4f} "
        f"tp={test_metrics['tp']} "
        f"fp={test_metrics['fp']} "
        f"fn={test_metrics['fn']} "
        f"tn={test_metrics['tn']}"
    )

    return test_metrics


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--data",
        type=str,
        default="data/processed/openstack_DEPL",
        help="Folder containing train.pt/val.pt/test.pt from prepare_openstack.py",
    )
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch", type=int, default=64)
    ap.add_argument("--lr", type=float, default=1e-3)
    args = ap.parse_args()

    train_openstack(
        data_dir=args.data,
        epochs=args.epochs,
        batch_size=args.batch,
        lr=args.lr,
    )


if __name__ == "__main__":
    main()
