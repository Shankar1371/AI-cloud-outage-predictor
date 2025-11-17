# scripts/prepare_openstack.py
"""
Prepare the Failure-Dataset-OpenStack data for ML training.

We take one workload folder (DEPL / NET / STO), read:
  - SEQ.tsv              : features per experiment
  - Failure_Labels.txt   : failure type per experiment

We then build:
  - X: feature matrix [N, D]
  - y: binary labels [N], 1 = failure, 0 = no failure

and save train/val/test splits as .pt files:
  out_dir/train.pt
  out_dir/val.pt
  out_dir/test.pt

Each .pt file is a dict:
  {
    "X": torch.FloatTensor [N, D],
    "y": torch.FloatTensor [N],
    "features": list[str]   # feature column names
  }
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def to_binary_label(raw):
    """
    Map raw failure label to binary:
      0 -> no failure
      1 -> any failure
    We handle both string labels ("No Failure") and numeric codes (e.g., 6).
    """
    s = str(raw).strip().lower()

    # String-style: "No Failure", "no_failure", etc.
    if "no" in s and "failure" in s:
        return 0

    # Numeric-style: in the original paper, class '6' is 'No Failure'
    try:
        v = int(raw)
        if v == 6:
            return 0
        else:
            return 1
    except ValueError:
        # Anything else we treat as some kind of failure
        return 1


def prepare_workload(root: Path, workload: str, out_dir: Path):
    """
    root/workload should contain:
      - SEQ.tsv
      - Failure_Labels.txt
    """
    wl_dir = root / workload
    seq_path = wl_dir / "SEQ.tsv"
    lab_path = wl_dir / "Failure_Labels.txt"

    if not seq_path.exists():
        raise FileNotFoundError(f"Missing {seq_path}")
    if not lab_path.exists():
        raise FileNotFoundError(f"Missing {lab_path}")

    print(f"[load] SEQ from {seq_path}")
    # SEQ.tsv: tab-separated, first column is experiment/sequence id
    seq = pd.read_csv(seq_path, sep="\t")

    print(f"[load] Labels from {lab_path}")
    # Labels: two columns (id, label) or similar; we force 2 columns.
    labels = pd.read_csv(
        lab_path,
        sep="\t",
        header=None,
        names=[seq.columns[0], "raw_label"]
    )

    # Merge features + labels on the id column (first column of SEQ)
    id_col = seq.columns[0]
    df = seq.merge(labels, on=id_col, how="inner")

    # Build binary label
    df["label"] = df["raw_label"].apply(to_binary_label).astype("int64")

    # Feature columns: everything except id, raw_label, label
    feature_cols = [c for c in df.columns if c not in (id_col, "raw_label", "label")]
    X = df[feature_cols].values.astype("float32")
    y = df["label"].values.astype("float32")

    print(f"[info] workload={workload} -> N={X.shape[0]} samples, D={X.shape[1]} features")

    # Train/val/test split: 70 / 15 / 15
    rng = np.random.RandomState(1371)
    idx = np.arange(len(X))
    rng.shuffle(idx)

    n = len(X)
    n_train = int(round(0.7 * n))
    n_val = int(round(0.15 * n))
    n_test = n - n_train - n_val
    if n_test <= 0:
        n_test = 1
        n_val = max(1, n - n_train - n_test)

    train_idx = idx[:n_train]
    val_idx = idx[n_train:n_train + n_val]
    test_idx = idx[n_train + n_val:]

    splits = {
        "train": (X[train_idx], y[train_idx]),
        "val":   (X[val_idx],   y[val_idx]),
        "test":  (X[test_idx],  y[test_idx]),
    }

    out_dir.mkdir(parents=True, exist_ok=True)
    for name, (Xs, ys) in splits.items():
        pkg = {
            "X": torch.tensor(Xs, dtype=torch.float32),
            "y": torch.tensor(ys, dtype=torch.float32),
            "features": feature_cols,
            "workload": workload,
        }
        out_path = out_dir / f"{name}.pt"
        torch.save(pkg, out_path)
        print(f"[save] {name}: {out_path}  (N={Xs.shape[0]})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--root",
        type=str,
        default="data/openstack_raw/Failure-Dataset-OpenStack",
        help="Path to Failure-Dataset-OpenStack root folder",
    )
    ap.add_argument(
        "--workload",
        type=str,
        default="DEPL",
        choices=["DEPL", "NET", "STO"],
        help="Which workload folder to use",
    )
    ap.add_argument(
        "--outdir",
        type=str,
        default="data/processed/openstack_DEPL",
        help="Output folder for processed tensors",
    )
    args = ap.parse_args()

    root = Path(args.root)
    out_dir = Path(args.outdir)

    print(f"[config] root={root}, workload={args.workload}, outdir={out_dir}")
    prepare_workload(root, args.workload, out_dir)
    print("[done] OpenStack workload tensors ready.")


if __name__ == "__main__":
    main()
