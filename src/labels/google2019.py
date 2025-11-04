# src/labels/google2019.py
import pandas as pd
import numpy as np

H = 15  # horizon minutes

def _to_utc(df, cols=("time",)):
    for c in cols:
        if not pd.api.types.is_datetime64_any_dtype(df[c]):
            df[c] = pd.to_datetime(df[c], utc=True, errors="coerce")
    return df

def build_labels(machine_events: pd.DataFrame,
                 task_events: pd.DataFrame,
                 usage: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a dataframe with columns: ['time','machine_id','label']
    label==1 if (t, t+H] contains a machine REMOVE or an extreme fail_burst.
    """
    # --- 0) Clean inputs ---
    machine_events = _to_utc(machine_events.copy())
    task_events    = _to_utc(task_events.copy())
    usage          = _to_utc(usage.copy())

    # --- 1) Minute grid per machine (from usage cadence) ---
    minutes = (
        usage[["time", "machine_id"]]
        .drop_duplicates()
        .sort_values(["machine_id", "time"])
        .reset_index(drop=True)
    )

    # --- 2) Machine down anchors (REMOVE = outage anchor) ---
    down = machine_events.loc[machine_events["event_type"] == "REMOVE", ["time", "machine_id"]].copy()
    down["down_flag"] = 1

    # --- 3) Per-minute failure-like bursts ---
    te = task_events.loc[task_events["event_type"].isin(["FAIL", "EVICT", "LOST"])].copy()
    ev = (
        te.groupby(["machine_id", pd.Grouper(key="time", freq="1min")])
          .size()
          .rename("fail_burst")
          .reset_index()
    )

    # --- 4) Merge into one minute-level table ---
    df = (
        minutes
        .merge(usage, on=["time", "machine_id"], how="left")
        .merge(ev, on=["time", "machine_id"], how="left")
        .merge(down, on=["time", "machine_id"], how="left")
        .sort_values(["machine_id", "time"])
        .reset_index(drop=True)
    )
    df["fail_burst"] = df["fail_burst"].fillna(0).astype("int16")
    df["down_flag"]  = df["down_flag"].fillna(0).astype("int8")

    # --- 5) Per-machine dynamic burst threshold (e.g., 99th pct) ---
    thr = (
        df.groupby("machine_id")["fail_burst"]
          .quantile(0.99)
          .rename("thr")
          .reset_index()
    )
    df = df.merge(thr, on="machine_id", how="left")
    # fallback when a machine has all zeros / NaN threshold
    df["thr"] = df["thr"].fillna(0)

    # --- 6) Future window maxima over next H minutes (exclude current minute) ---
    # shift(-1) moves future into alignment with the current row; rolling(H) takes the next H minutes
    df["future_down"]  = (
        df.groupby("machine_id")["down_flag"]
          .transform(lambda s: s.shift(-1).rolling(H, min_periods=1).max())
          .fillna(0)
    )
    df["future_burst"] = (
        df.groupby("machine_id")["fail_burst"]
          .transform(lambda s: s.shift(-1).rolling(H, min_periods=1).max())
          .fillna(0)
    )

    # --- 7) Label: future REMOVE or future extreme burst ---
    df["label"] = ((df["future_down"] >= 1) | (df["future_burst"] > df["thr"])).astype("int8")

    return df[["time", "machine_id", "label"]]
