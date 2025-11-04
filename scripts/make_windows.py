"""
1) Read raw provider traces and normalize to a common schema.
2) Resample to 1-min; compute event counts per minute.
3) Build labels for horizon H.
4) Save interim parquet and (optionally) processed torch tensors.
"""

import argparse
from pathlib import Path
import pandas as pd
import sys

ROOT = Path(__file__).resolve().parents[1]  # project root (the folder that contains 'src')
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Optional tensor export (only imported if --save-tensors)
try:
    import torch
except Exception:
    torch = None

# ----- helpers ---------------------------------------------------------------

def _to_utc(df, cols=("time",)):
    """Ensure datetime64[ns, UTC] for given columns."""
    for c in cols:
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df

def _ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

# ----- main ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True, help="e.g., google2019")
    ap.add_argument("--horizon", type=int, default=15, help="future minutes for label window (H)")
    ap.add_argument("--T", type=int, default=30, help="history window length for tensors")
    ap.add_argument("--out", required=True, help="dir to write processed tensors (if --save-tensors)")
    ap.add_argument("--save-tensors", action="store_true", help="also emit processed tensors to --out")
    ap.add_argument("--features", default="cpu,mem,disk,net,throttle,tasks_running",
                    help="comma-separated metric feature columns for model input")
    ap.add_argument("--event-features", default="fail_burst",
                    help="comma-separated event feature columns for model input")
    args = ap.parse_args()

    RAW = Path("data/raw") / args.provider
    INT = Path("data/interim") / args.provider
    OUT = Path(args.out)

    _ensure_dirs(INT, OUT)

    # ---- 1) Load raw (synthetic or real parquet) ---------------------------
    usage = pd.read_parquet(RAW / "usage.parquet")
    task_events = pd.read_parquet(RAW / "task_events.parquet")
    machine_events = pd.read_parquet(RAW / "machine_events.parquet")

    # ---- 2) Normalize to UTC & minute cadence ------------------------------
    usage = _to_utc(usage.copy())
    task_events = _to_utc(task_events.copy())
    machine_events = _to_utc(machine_events.copy())

    # Minute grid (if usage is already minute-level, this keeps it tidy)
    minutes = (
        usage[["time", "machine_id"]]
        .drop_duplicates()
        .sort_values(["machine_id", "time"])
        .reset_index(drop=True)
    )

    # Per-minute event counts (FAIL/EVICT/LOST)
    fail_like = task_events[task_events["event_type"].isin(["FAIL", "EVICT", "LOST"])].copy()
    ev = (
        fail_like
        .groupby(["machine_id", pd.Grouper(key="time", freq="1min")])
        .size()
        .rename("fail_burst")
        .reset_index()
    )

    # Merge metrics + counts
    metrics = (
        minutes
        .merge(usage, on=["time", "machine_id"], how="left")
        .merge(ev, on=["time", "machine_id"], how="left")
        .sort_values(["machine_id", "time"])
        .reset_index(drop=True)
    )
    metrics["fail_burst"] = metrics["fail_burst"].fillna(0).astype("int16")

    # Machine REMOVE flag
    rem = machine_events.loc[machine_events["event_type"] == "REMOVE", ["time", "machine_id"]].copy()
    rem["remove"] = 1
    metrics = metrics.merge(rem, on=["time", "machine_id"], how="left")
    metrics["remove"] = metrics["remove"].fillna(0).astype("int8")

    # ---- 3) Build labels using provider-specific builder -------------------
    # Import here to avoid circulars and keep script generic
    from src.labels.google2019 import build_labels, H as DEFAULT_H

    # Respect CLI --horizon if provided; temporarily override the module constant
    H_cli = args.horizon if args.horizon is not None else DEFAULT_H
    labels = build_labels(
        machine_events=machine_events,
        task_events=task_events,
        usage=usage
    )
    # (labels already encodes a horizon inside build_labels; if you later want
    # to pass H_cli into build_labels, expose a parameter in that function.)

    # ---- 4) Save interim parquet ------------------------------------------
    metrics_file = INT / "metrics_train.parquet"
    labels_file = INT / "labels_train.parquet"
    metrics.to_parquet(metrics_file)
    labels.to_parquet(labels_file)

    print("Wrote", metrics_file)
    print("Wrote", labels_file)

    # ---- 5) (Optional) Save processed tensors for training -----------------
    if args.save_tensors:
        if torch is None:
            raise RuntimeError("PyTorch is not available. Install torch or omit --save-tensors.")

        feat_cols = [c.strip() for c in args.features.split(",") if c.strip()]
        evt_cols  = [c.strip() for c in args.event_features.split(",") if c.strip()]

        # Join metrics + labels; enforce strict ordering
        df = (
            metrics.merge(labels, on=["time", "machine_id"], how="left")
                   .sort_values(["machine_id", "time"])
                   .reset_index(drop=True)
        )
        df["label"] = df["label"].fillna(0).astype("int8")

        # Group by machine and build sliding windows
        X_m_list, X_e_list, y_list = [], [], []
        T = int(args.T)

        # simple z-score normalization using train-wide stats (prototype)
        # (For production, prefer per-machine stats computed on a train split.)
        m_stats = df[feat_cols].agg(["mean", "std"])
        mean_vec = m_stats.loc["mean"].values
        std_vec  = (m_stats.loc["std"].values + 1e-6)

        for mid, g in df.groupby("machine_id", sort=True):
            g = g.reset_index(drop=True)
            xm = g[feat_cols].values.astype("float32")
            xe = g[evt_cols].values.astype("float32")
            y  = g["label"].values.astype("float32")

            # normalize metrics globally (prototype)
            xm = (xm - mean_vec) / std_vec

            # build sliding windows
            if len(g) < T:
                continue
            for t in range(T - 1, len(g)):
                X_m_list.append(xm[t - T + 1:t + 1])
                X_e_list.append(xe[t - T + 1:t + 1])
                y_list.append(y[t])

        if not X_m_list:
            raise RuntimeError("No windows were created. Check data size and T.")

        X_m = torch.tensor(X_m_list, dtype=torch.float32)
        X_e = torch.tensor(X_e_list, dtype=torch.float32)
        y_t = torch.tensor(y_list,   dtype=torch.float32)

        out_file = OUT / "dataset.pt"
        torch.save({"X_metrics": X_m, "X_events": X_e, "y": y_t,
                    "features": feat_cols, "event_features": evt_cols, "T": T},
                   out_file)
        print("Wrote", out_file)
    else:
        print("Note: tensors were not written. Use --save-tensors to emit processed tensors.")

if __name__ == "__main__":
    main()
