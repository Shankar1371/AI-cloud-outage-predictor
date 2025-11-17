import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_curves(log_path: Path, out_dir: Path):
    df = pd.read_csv(log_path)

    # Loss curve
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("GRU Training Loss")
    plt.grid(True)
    plt.tight_layout()
    (out_dir / "gru_train_loss.png").parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_dir / "gru_train_loss.png", dpi=200)

    # Val F1 curve
    plt.figure()
    plt.plot(df["epoch"], df["val_f1"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation F1")
    plt.title("GRU Validation F1")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "gru_val_f1.png", dpi=200)


def compute_pr_curve(y_true, y_prob):
    # sort by probability descending
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


def compute_roc_curve(y_true, y_prob):
    # thresholds = sorted unique probabilities
    thresholds = np.unique(y_prob)
    thresholds = thresholds[::-1]  # descending

    tprs = []
    fprs = []
    P = max((y_true == 1).sum(), 1)
    N = max((y_true == 0).sum(), 1)

    for thr in thresholds:
        y_pred = (y_prob >= thr).astype(int)
        tp = ((y_pred == 1) & (y_true == 1)).sum()
        fp = ((y_pred == 1) & (y_true == 0)).sum()
        tpr = tp / P
        fpr = fp / N
        tprs.append(tpr)
        fprs.append(fpr)

    return np.array(fprs), np.array(tprs)


def plot_curves_from_preds(preds_path: Path, out_dir: Path):
    data = np.load(preds_path)
    y_true = data["y_true"].astype(int)
    y_prob = data["y_prob"].astype(float)

    # PR curve
    prec, rec = compute_pr_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec)
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("GRU Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "gru_pr_curve.png", dpi=200)

    # ROC curve
    fpr, tpr = compute_roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr)
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("GRU ROC Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "gru_roc_curve.png", dpi=200)

    # Confusion matrix at threshold 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    cm = np.array([[tn, fp],
                   [fn, tp]])

    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("GRU Confusion Matrix (thr=0.5)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]),
                     ha="center", va="center")
    plt.tight_layout()
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.savefig(out_dir / "gru_confusion_matrix.png", dpi=200)


def main():
    base = Path("data/processed/google2019")
    log_path = base / "gru_training_log.csv"
    preds_path = base / "gru_test_preds.npz"
    out_dir = base / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] reading log from {log_path}")
    plot_training_curves(log_path, out_dir)

    print(f"[info] reading preds from {preds_path}")
    plot_curves_from_preds(preds_path, out_dir)

    print(f"[done] plots saved under {out_dir}")


if __name__ == "__main__":
    main()
