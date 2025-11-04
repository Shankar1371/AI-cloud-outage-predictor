# scripts/make_tensors.py
import argparse
from pathlib import Path
import numpy as np
import pandas as pd
import torch


def zscore_fit(X: np.ndarray):
    mu = X.mean(axis=0)
    sd = X.std(axis=0) + 1e-6
    return mu, sd


def zscore_apply(X: np.ndarray, mu: np.ndarray, sd: np.ndarray):
    return (X - mu) / sd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, help="e.g., google2019")
    ap.add_argument("--T", type=int, default=30, help="history window length")
    ap.add_argument(
        "--features",
        default="cpu,mem,disk,net,throttle,tasks_running",
        help="comma-separated metric feature columns",
    )
    ap.add_argument(
        "--event-features",
        dest="event_features",
        default="fail_burst",
        help="comma-separated event feature columns",
    )
    ap.add_argument("--outdir", required=True, help="output dir for .pt files")
    ap.add_argument("--seed", type=int, default=1371, help="split seed")
    ap.add_argument("--train-frac", type=float, default=0.70)
    ap.add_argument("--val-frac", type=float, default=0.15)
    args = ap.parse_args()

    INT = Path("data/interim") / args.provider
    OUT = Path(args.outdir)
    OUT.mkdir(parents=True, exist_ok=True)

    # ---- load
    metrics = pd.read_parquet(INT / "metrics_train.parquet")
    labels = pd.read_parquet(INT / "labels_train.parquet")
    df = (
        metrics.merge(labels, on=["time", "machine_id"], how="left")
        .sort_values(["machine_id", "time"])
        .reset_index(drop=True)
    )
    df["label"] = df["label"].fillna(0).astype("int8")

    feat_cols = [c.strip() for c in args.features.split(",") if c.strip()]
    evt_cols = [c.strip() for c in args.event_features.split(",") if c.strip()]

    # ---- split by machine (prevents leakage)
    machines = df["machine_id"].unique().tolist()
    rng = np.random.RandomState(args.seed)
    rng.shuffle(machines)

    n = len(machines)
    train_frac = min(max(args.train_frac, 0.0), 1.0)
    val_frac = min(max(args.val_frac, 0.0), 1.0)
    # Ensure fractions are sane
    if train_frac + val_frac > 0.95:
        val_frac = max(0.05, 0.95 - train_frac)

    n_train = max(1, int(round(train_frac * n)))
    n_val = max(1, int(round(val_frac * n)))
    n_test = max(1, n - n_train - n_val)

    # Re-balance if rounding overflow/underflow
    while n_train + n_val + n_test > n:
        if n_train > 1:
            n_train -= 1
        elif n_val > 1:
            n_val -= 1
        else:
            n_test -= 1
    while n_train + n_val + n_test < n:
        n_test += 1

    train_m = set(machines[:n_train])
    val_m = set(machines[n_train : n_train + n_val])
    test_m = set(machines[n_train + n_val :])

    # Guarantee non-empty val/test (tiny data safety)
    if not val_m and train_m:
        val_m.add(train_m.pop())
    if not test_m and train_m:
        test_m.add(train_m.pop())

    print(
        f"machines={n}  train={len(train_m)}  val={len(val_m)}  test={len(test_m)}"
    )

    # ---- fit zscore on TRAIN metrics only; if empty fallback to all
    m_train_df = df[df["machine_id"].isin(train_m)]
    if m_train_df.empty:
        print("[WARN] train metrics empty for zscore; using all data for stats.")
        m_train_df = df

    mu, sd = zscore_fit(m_train_df[feat_cols].values.astype("float32"))

    def make_windows(df_split: pd.DataFrame, T: int):
        Xm, Xe, Y = [], [], []
        for mid, g in df_split.groupby("machine_id", sort=True):
            g = g.reset_index(drop=True)
            if len(g) < T:
                continue
            xm = zscore_apply(g[feat_cols].values.astype("float32"), mu, sd)
            xe = g[evt_cols].values.astype("float32")
            y = g["label"].values.astype("float32")

            for t in range(T - 1, len(g)):
                Xm.append(xm[t - T + 1 : t + 1])
                Xe.append(xe[t - T + 1 : t + 1])
                Y.append(y[t])

        if not Xm:
            return None
        pkg = {
            "X_metrics": torch.tensor(np.stack(Xm), dtype=torch.float32),
            "X_events": torch.tensor(np.stack(Xe), dtype=torch.float32),
            "y": torch.tensor(np.array(Y), dtype=torch.float32),
            "features": feat_cols,
            "event_features": evt_cols,
            "T": T,
        }
        print(
            f"windows={pkg['X_metrics'].shape[0]}  T={pkg['T']}  fm={pkg['X_metrics'].shape[-1]}  fe={pkg['X_events'].shape[-1]}"
        )
        return pkg

    splits = {
        "train": train_m,
        "val": val_m,
        "test": test_m,
    }

    for split_name, mids in splits.items():
        pkg = make_windows(df[df["machine_id"].isin(mids)], args.T)
        if pkg is None:
            print(f"[WARN] No windows for split {split_name}")
            continue
        out_file = OUT / f"{split_name}.pt"
        torch.save(pkg, out_file)
        print("Wrote", out_file)


if __name__ == "__main__":
    main()
