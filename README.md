#### AI-Powered Cloud Outage Prediction ####

This project predicts cloud system failures before they happen, using a PyTorch model built entirely from scratch.
The goal is simple:
Use time-series server metrics and system events to detect outages 10–15 minutes before they occur.

# I built two complete pipelines:

1. Synthetic Google-style cluster outage data
(This simulates how large services like Google Borg/AWS/GCP behave under heavy workloads.)

2. Real OpenStack cloud failure dataset
(This contains actual cloud failures such as VM Hang, Network Failure, Volume Failure, etc.)

Both are trained with my own PyTorch models (GRU for time-series, MLP for real failures), following the academic rubric:
✔ custom PyTorch models
✔ full training & evaluation
✔ classical ML baselines
✔ comparative analysis


#### 
1. Why This Project Matters

Cloud outages are extremely costly:

- AWS outage → companies lose millions per hour

- Kubernetes node failure → disrupts entire workloads

- VM failure or network outage → impacts SRE, DevOps, financial services, hospitals, universities

Cloud platforms generate massive logs & metrics every minute:

- CPU %, Memory %, Disk I/O, Network throughput

-Task failures, evictions, reschedules

Node add/remove events

VM/instance-level failures

Today most alerts are reactive (after failure).
This project builds a proactive outage predictor.

If we can predict a failure 10 minutes early, SRE teams can:

Migrate workloads

Scale replicas

Evict unhealthy nodes

Restart services

Prevent downtime

Even a single prevented outage can save thousands.