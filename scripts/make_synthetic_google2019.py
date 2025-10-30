from pathlib import Path
import numpy as np, pandas as pd

base = Path("data/raw/google2019")
base.mkdir(parents=True, exist_ok=True)

rng = pd.date_range("2019-05-01", periods=60*12, freq="1min")  # 12 hours
mids = ["m1","m2","m3"]

# usage (already minute-level)
rows = []
for m in mids:
    cpu = np.clip(np.random.normal(0.5, 0.15, len(rng)), 0, 1)
    mem = np.clip(np.random.normal(0.6, 0.10, len(rng)), 0, 1)
    disk= np.clip(np.random.normal(0.4, 0.10, len(rng)), 0, 1)
    net = np.clip(np.random.normal(0.3, 0.10, len(rng)), 0, 1)
    thr = (cpu > 0.9).astype(int)
    run = (10 + (cpu*10).astype(int))
    rows.append(pd.DataFrame({
        "time": rng, "machine_id": m, "cpu": cpu, "mem": mem,
        "disk": disk, "net": net, "throttle": thr, "tasks_running": run
    }))
usage = pd.concat(rows, ignore_index=True)
usage.to_parquet(base/"usage.parquet")

# task_events (irregular)
ev_rows = []
for m in mids:
    t = np.random.choice(rng, size=50, replace=False)
    types = np.random.choice(["FAIL","EVICT","LOST","FINISH"], size=len(t), p=[0.2,0.2,0.1,0.5])
    ev_rows.append(pd.DataFrame({"time": t, "machine_id": m, "task_id": range(len(t)), "event_type": types}))
task_events = pd.concat(ev_rows).sort_values("time")
task_events.to_parquet(base/"task_events.parquet")

# machine_events (rare REMOVE)
machine_events = pd.DataFrame([
    {"time": rng[int(len(rng)*0.75)], "machine_id": m, "event_type": "REMOVE"} for m in mids
])
machine_events.to_parquet(base/"machine_events.parquet")

print("Synthetic data written to data/raw/google2019/*.parquet")
