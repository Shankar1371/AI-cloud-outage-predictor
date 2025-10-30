#label builders for google cluster data 2019


import pandas as pd
import numpy as np


# Expected raw tables after basic conversion/cleaning:
# machine_events: [time, machine_id, event_type] # event_type in {ADD, REMOVE, UPDATE}
# task_events: [time, machine_id, task_id, event_type] # FAIL, EVICT, LOST, FINISH, ...
# usage: [time, machine_id, cpu, mem, disk, net, throttle, tasks_running]


H = 15 # horizon minutes


def build_labels(machine_events: pd.DataFrame, task_events: pd.DataFrame, usage: pd.DataFrame) -> pd.DataFrame:
# 1) Minute grid per machine
    minutes = usage[["time","machine_id"]].drop_duplicates().sort_values(["machine_id","time"])\
.reset_index(drop=True)
#the above line creates a skeleton data frame for every unique (time, machine_id) pair present in the usage data


# 2) Machine down anchors
    down = machine_events[machine_events.event_type == "REMOVE"][["time","machine_id"]]
    down["down_flag"] = 1
#this filters the mcahine_events to find only machine removal events which is treated as hard outages or shutdown


# 3) Event bursts per minute
    te = task_events[task_events.event_type.isin(["FAIL","EVICT","LOST"])]
    ev = te.groupby(["machine_id", pd.Grouper(key="time", freq="1min")]).size().rename("fail_burst").reset_index()
#this is used for filtering the task_events to include only major negative tasjs events that are failed evicted or lost



# 4) Merge
    df = minutes.merge(usage, on=["time","machine_id"], how="left")\
    .merge(ev, on=["time","machine_id"], how="left")\
    .merge(down, on=["time","machine_id"], how="left")
    df["fail_burst"] = df["fail_burst"].fillna(0)
    df["down_flag"] = df["down_flag"].fillna(0)
#merges the machine grid

# 5) Outage proxy in future window (t, t+H]
# For each row, check if any future minute within H has down_flag==1 OR extreme fail_burst
    df = df.sort_values(["machine_id","time"]).reset_index(drop=True)
    df["label"] = 0
#this is used to calculate the threshold of the failed task failures or the percintile of the fail burst


# Determine dynamic threshold for bursts (e.g., top 1% per machine)
    thr = df.groupby("machine_id")["fail_burst"].quantile(0.99).rename("thr").reset_index()
    df = df.merge(thr, on="machine_id", how="left")


# Rolling future max over next H minutes
    df["future_down"] = df.groupby("machine_id")["down_flag"].transform(lambda x: x.shift(-1).rolling(H, min_periods=1).max())
    df["future_burst"] = df.groupby("machine_id")["fail_burst"].transform(lambda x: x.shift(-1).rolling(H, min_periods=1).max())


    df["label"] = ((df["future_down"]>=1) | (df["future_burst"]>df["thr"])) .astype(int)
    return df[["time","machine_id","label"]]


#the goal is making predictive labelling of three raw data that is machine events,task events and usage
