# scripts/plot_pr_roc.py
import argparse, numpy as np
from sklearn.metrics import precision_recall_curve, roc_curve, auc
import matplotlib.pyplot as plt

ap = argparse.ArgumentParser()
ap.add_argument("--npz", required=True)  # file with y,y_prob np arrays
ap.add_argument("--outdir", required=True)
args = ap.parse_args()
d = np.load(args.npz)
y, p = d["y"], d["p"]

pr, rc, _ = precision_recall_curve(y, p)
fpr, tpr, _ = roc_curve(y, p)
with plt.style.context(None):
    plt.figure(); plt.plot(rc, pr); plt.xlabel("Recall"); plt.ylabel("Precision"); plt.grid(); plt.savefig(f"{args.outdir}/pr_curve.png", dpi=160)
    plt.figure(); plt.plot(fpr, tpr); plt.xlabel("FPR"); plt.ylabel("TPR"); plt.grid(); plt.savefig(f"{args.outdir}/roc_curve.png", dpi=160)
print("Saved PR/ROC curves.")
