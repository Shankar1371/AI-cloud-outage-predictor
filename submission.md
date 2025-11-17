# AI-Powered Cloud Outage Prediction (Submission – 1 page)

**Problem.** Cloud providers (AWS/Azure/GCP) operate thousands of servers streaming metrics (CPU, memory, disk I/O, network). Outages cost thousands to millions per hour. We predict an outage **10–15 min ahead** from recent multivariate time-series windows.

**Input / Output.**
- Input: 30-minute windows of 1-min metrics per machine + event counts (e.g., FAIL/EVICT/LOST).
- Output: P(outage in next 10–15 min).

**Proposed Method (PyTorch, own code).**  
A lightweight **GRU** model (PyTorch only; no high-level ML libs for the main model) learns temporal dependencies:
- Metrics encoder: 2-layer GRU (dh=128) over CPU/mem/disk/net/throttle/tasks_running.
- Events encoder: 1-layer GRU (de=32) over event-count sequences (fail_burst).
- Fusion head: MLP(128→1, Sigmoid) → outage probability.  
Training uses **weighted BCE** (class imbalance) and AdamW (lr=1e-3).  
We implement the full training loop, evaluation, and checkpointing ourselves.

**Label Construction (Google 2019 style).**  
Minute grid per machine. Positive label if **(t, t+H]** contains a machine **REMOVE** or an extreme **fail_burst** (≥ per-machine 99th percentile). H=15 min.  

**Datasets.**  
We prototype with a synthetic Google-style trace (minute cadence), then plug-compatible with Google ClusterData 2019. (Other public traces like Azure 2019 / Alibaba 2018 can be added.)

**Baselines (libraries allowed).**
- **Logistic Regression** (balanced class weight) on window aggregates (mean, std, last, slope).
- **Random Forest** (balanced_subsample).  
(Implemented with scikit-learn; allowed by rubric.)

**Evaluation.**
Primary metrics: **PR-AUC**, **ROC-AUC**, **F1@0.5** on a **held-out test split** (split by machine, not by row).  
We also track training PR-AUC/ROC-AUC/F1 per epoch and save **best_gru.pt** by val PR-AUC.

**Results (Test).**
| Model | PR-AUC | ROC-AUC | F1 |
|---|---:|---:|---:|
| Logistic Regression | 0.XXX | 0.XXX | 0.XXX |
| Random Forest | 0.XXX | 0.XXX | 0.XXX |
| **GRU (ours)** | **0.YYY** | **0.YYY** | **0.YYY** |

**Why it matters.** Proactively detecting outages enables **self-healing** actions (reschedule, scale-out) that reduce SLO violations and downtime.

**Rubric compliance.**  
• PyTorch only for the proposed model; our model and training loop are **own code**.  
• scikit-learn used **only** for preprocessing/baselines (allowed).  
• Includes at least one baseline and a comprehensive evaluation.
