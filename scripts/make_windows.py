"""
1) Read raw provider traces and normalize to a common schema.
2) Resample to 1-min; compute event counts per minute.
3) Build labels for horizon H.
4) Save interim parquet and processed torch tensors.
"""
import argparse
from pathlib import Path
import pandas as pd

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--provider", required=True)
    ap.add_argument("--horizon", type=int, default=15)
    ap.add_argument("--T", type=int, default=30)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    RAW = Path("data/raw")/args.provider
    OUT = Path("data/interim")/args.provider
    OUT.mkdir(parents=True, exist_ok=True)

    # Load raw (for now, synthetic parquet files)
    usage = pd.read_parquet(RAW/"usage.parquet")
    task_events = pd.read_parquet(RAW/"task_events.parquet")
    machine_events = pd.read_parquet(RAW/"machine_events.parquet")

    # Ensure UTC timestamps
    to_dt = lambda s: pd.to_datetime(s, utc=True)
    usage["time"] = to_dt(usage["time"])
    task_events["time"] = to_dt(task_events["time"])
    machine_events["time"] = to_dt(machine_events["time"])

    # Minute grid (already minute-level in our synthetic)
    minutes = usage[["time","machine_id"]].drop_duplicates().sort_values(["machine_id","time"])

    # Per-minute event counts
    fail_like = task_events[task_events.event_type.isin(["FAIL","EVICT","LOST"])]
    ev = (
        fail_like
        .groupby(["machine_id", pd.Grouper(key="time", freq="1min")])
        .size()
        .rename("fail_burst")
        .reset_index()
    )

    # Merge metrics + counts
    metrics = minutes.merge(usage, on=["time","machine_id"], how="left")
    metrics = metrics.merge(ev, on=["time","machine_id"], how="left")
    metrics["fail_burst"] = metrics["fail_burst"].fillna(0)

    # Machine REMOVE flag
    rem = machine_events[machine_events.event_type == "REMOVE"][["time","machine_id"]].copy()
    rem["remove"] = 1
    metrics = metrics.merge(rem, on=["time","machine_id"], how="left")
    metrics["remove"] = metrics["remove"].fillna(0)

    # Save interim parquet
    OUT.mkdir(parents=True, exist_ok=True)
    outfile = OUT/"metrics_train.parquet"
    metrics.to_parquet(outfile)
    print("Wrote", outfile)

    # For now we won't use --out; that will be used when saving processed tensors later.
    print("Note: --out will be used in a later step for processed tensors.")

if __name__ == "__main__":
    main()
