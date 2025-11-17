# scripts/plot_openstack_results.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


def plot_training_curves(log_path: Path, out_dir: Path):
    df = pd.read_csv(log_path)

    # loss
    plt.figure()
    plt.plot(df["epoch"], df["train_loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Train Loss")
    plt.title("OpenStack MLP Training Loss")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "openstack_train_loss.png", dpi=200)

    # F1
    plt.figure()
    plt.plot(df["epoch"], df["val_f1"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Validation F1")
    plt.title("OpenStack MLP Validation F1")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "openstack_val_f1.png", dpi=200)

    # ROC-AUC per epoch
    plt.figure()
    plt.plot(df["epoch"], df["val_roc_auc"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Val ROC-AUC")
    plt.title("OpenStack MLP Validation ROC-AUC")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "openstack_val_roc_auc.png", dpi=200)


def confusion_matrix(y_true, y_pred):
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)

    tp = ((y_pred == 1) & (y_true == 1)).sum()
    tn = ((y_pred == 0) & (y_true == 0)).sum()
    fp = ((y_pred == 1) & (y_true == 0)).sum()
    fn = ((y_pred == 0) & (y_true == 1)).sum()

    return np.array([[tn, fp],
                     [fn, tp]])


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
    thresholds = np.unique(y_prob)[::-1]
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


def calibration_curve(y_true, y_prob, n_bins=10):
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob)
    bins = np.linspace(0.0, 1.0, n_bins + 1)

    bin_centers = []
    frac_positives = []

    for i in range(n_bins):
        left, right = bins[i], bins[i + 1]
        mask = (y_prob >= left) & (y_prob < right)
        if mask.sum() == 0:
            continue
        bin_centers.append((left + right) / 2.0)
        frac_positives.append(y_true[mask].mean())

    return np.array(bin_centers), np.array(frac_positives)


def plot_from_preds(preds_path: Path, out_dir: Path):
    data = np.load(preds_path)
    y_true = data["y_true"].astype(int)
    y_prob = data["y_prob"].astype(float)

    # label distribution
    counts = pd.Series(y_true).value_counts().sort_index()
    plt.figure()
    counts.plot(kind="bar")
    plt.xticks([0, 1], ["Normal (0)", "Failure (1)"], rotation=0)
    plt.ylabel("Count")
    plt.title("OpenStack Label Distribution")
    plt.tight_layout()
    plt.savefig(out_dir / "openstack_label_distribution.png", dpi=200)

    # ROC
    fpr, tpr = roc_curve(y_true, y_prob)
    plt.figure()
    plt.plot(fpr, tpr, marker=".")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("OpenStack ROC Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "openstack_roc_curve.png", dpi=200)

    # PR
    prec, rec = pr_curve(y_true, y_prob)
    plt.figure()
    plt.plot(rec, prec, marker=".")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("OpenStack Precision-Recall Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "openstack_pr_curve.png", dpi=200)

    # confusion matrix at 0.5
    y_pred = (y_prob >= 0.5).astype(int)
    cm = confusion_matrix(y_true, y_pred)
    plt.figure()
    plt.imshow(cm, interpolation="nearest")
    plt.title("OpenStack Confusion Matrix (thr=0.5)")
    plt.colorbar()
    tick_marks = np.arange(2)
    plt.xticks(tick_marks, ["Pred 0", "Pred 1"])
    plt.yticks(tick_marks, ["True 0", "True 1"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, int(cm[i, j]),
                     ha="center", va="center")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(out_dir / "openstack_confusion_matrix.png", dpi=200)

    # calibration curve
    centers, frac_pos = calibration_curve(y_true, y_prob, n_bins=10)
    plt.figure()
    plt.plot(centers, frac_pos, marker="o")
    plt.plot([0, 1], [0, 1], linestyle="--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.title("OpenStack Calibration Curve")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_dir / "openstack_calibration_curve.png", dpi=200)


def main():
    base = Path("data/processed/openstack_DEPL")
    log_path = base / "openstack_train_log.csv"
    preds_path = base / "openstack_test_preds.npz"
    out_dir = base / "plots"
    out_dir.mkdir(parents=True, exist_ok=True)

    print(f"[info] reading log from {log_path}")
    plot_training_curves(log_path, out_dir)

    print(f"[info] reading predictions from {preds_path}")
    plot_from_preds(preds_path, out_dir)

    print(f"[done] plots saved to {out_dir}")


if __name__ == "__main__":
    main()
