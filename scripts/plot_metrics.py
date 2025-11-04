import argparse, pandas as pd, matplotlib.pyplot as plt
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, help="path to train_log.csv")
    ap.add_argument("--out", default=None, help="optional png path to save")
    args = ap.parse_args()

    df = pd.read_csv(args.log)
    if "epoch" not in df.columns:
        df.columns = ["epoch","pr_auc","roc_auc","f1"]  # fallback if no header
    plt.figure()
    plt.plot(df["epoch"], df["pr_auc"], label="PR-AUC")
    plt.plot(df["epoch"], df["roc_auc"], label="ROC-AUC")
    plt.plot(df["epoch"], df["f1"], label="F1")
    plt.xlabel("Epoch"); plt.ylabel("Score"); plt.legend(); plt.grid(True)
    if args.out:
        plt.savefig(args.out, bbox_inches="tight", dpi=160)
    else:
        plt.show()

if __name__ == "__main__":
    main()
