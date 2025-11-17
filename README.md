AI-Powered Cloud Outage Prediction

This project predicts cloud system failures before they happen, using a PyTorch model built entirely from scratch.
The goal is simple:
Use time-series server metrics and system events to detect outages 10–15 minutes before they occur.

I built two complete pipelines:

1 Synthetic Google-style cluster outage data
(This simulates how large services like Google Borg/AWS/GCP behave under heavy workloads.)

2 Real OpenStack cloud failure dataset
(This contains actual cloud failures such as VM Hang, Network Failure, Volume Failure, etc.)

Both are trained with my own PyTorch models (GRU for time-series, MLP for real failures), following the academic rubric:
 - custom PyTorch models
-  full training & evaluation
- classical ML baselines
- comparative analysis

  
## 1. Why This Project Matters

Cloud outages are extremely costly:

- AWS outage → companies lose millions per hour

- Kubernetes node failure → disrupts entire workloads

- VM failure or network outage → impacts SRE, DevOps, financial services, hospitals, universities

Cloud platforms generate massive logs & metrics every minute:

- CPU %, Memory %, Disk I/O, Network throughput

- Task failures, evictions, reschedules

- Node add/remove events

- VM/instance-level failures

Today most alerts are reactive (after failure).
This project builds a proactive outage predictor.

If we can predict a failure 10 minutes early, SRE teams can:

- Migrate workloads

- Scale replicas

- Evict unhealthy nodes

- Restart services

- Prevent downtime

Even a single prevented outage can save thousands.

## 2. Project Structure
cloud-outage-predictor/
│
├── data/
│   ├── raw/                  # raw synthetic google-style data (usage, task_events,…)
│   ├── interim/              # parquet after preprocessing (metrics + labels)
│   ├── processed/            # final tensors (train.pt, val.pt, test.pt)
│   └── openstack_raw/        # real OpenStack failure dataset
│
├── scripts/
│   ├── make_windows.py       # window generator for GRU model
│   ├── make_tensors.py       # build train/val/test tensors for GRU
│   └── prepare_openstack.py  # converts real OpenStack failures to tensors
│
├── src/
│   ├── models/
│   │   ├── gru.py            # my PyTorch GRU outage model
│   │   └── openstack_mlp.py  # my PyTorch MLP model for real failures
│   ├── train.py              # training script for synthetic GRU model
│   └── openstack_train.py    # training script for real failure dataset
│
└── baselines/
    └── classical_ml.py       # Logistic Regression / Random Forest baseline

### 3. How the Synthetic Pipeline Works (Google-Style)

This pipeline simulates real cluster behavior.

Step 1: Raw data

We start with:

- usage.parquet → CPU, mem, disk, network every minute

- task_events.parquet → FAIL/EVICT/LOST events

- machine_events.parquet → REMOVE (machine down)

Step 2: Create labels (future outage = 1)

We define an outage as:

- Machine REMOVE event in next 15 minutes

- OR a burst of FAIL/EVICT/LOST events above 99th percentile

This produces a binary label:
0 = normal, 1 = outage soon.

Step 3: Create time windows

Each sample = past 30 minutes of metrics + event counts.

Step 4: GRU model (PyTorch, my own code)

GRU over metrics (CPU, mem, disk, …)
GRU over event counts
Concatenate → MLP → Sigmoid

Step 5: Train

Uses:

- BCE loss

- AdamW

- class balancing (pos_weight)

Step 6: Evaluate

Metrics used:

- PR-AUC

- ROC-AUC

- F1 Score

Classical baselines:

- Logistic Regression

- Random Forest

The GRU clearly outperforms the baselines.

4. How the Real OpenStack Failure Pipeline Works

This is a real cloud outage dataset, not synthetic.

Dataset contains:

- VM Hang

- CPU Spike

- Network Failure

- Volume Failure

- Instance Failure

- No Failure

Each row = one experiment with features describing cloud behavior.
The labels come from Failure_Labels.txt.

My pipeline

1. Convert SEQ.tsv + Failure_Labels.txt → (X, y)

2. y = 1 for any failure, y = 0 for “No Failure”

3. Train my own PyTorch MLP model:

- Two hidden layers

- Dropout

- BCEWithLogitsLoss with class balancing

4. Evaluate with:

- Accuracy

- Precision

- Recall

- F1 Score

This gives real-world cloud failure detection results that I can show to my professor.

## 5. Running the Synthetic (Google-Style) Pipeline
Step 1: Create interim parquet data

python scripts/make_windows.py --provider google2019 --horizon 15 --T 30

Step 2: Build final tensors

python scripts/make_tensors.py --provider google2019 --T 30 --outdir data/processed/google2019

Step 3: Train GRU model

python -m src.train --data data/processed/google2019 --epochs 10 --pos-weight 10

Output:

Best GRU checkpoint → best_gru.pt

Validation/Testing PR-AUC, ROC-AUC, F1

6. Running the Real OpenStack Failure Pipeline
   
Step 1: Convert OpenStack raw data to tensors
python scripts/prepare_openstack.py --workload DEPL --outdir data/processed/openstack_DEPL


This creates:

train.pt
val.pt
test.pt

Step 2: Train the OpenStack MLP

python -m src.openstack_train --data data/processed/openstack_DEPL --epochs 20

Output:

Best model → best_openstack_mlp.pt

Final TEST accuracy, precision, recall, F1

7. Real-World Use Cases

This project matches real SRE/DevOps workflows.

What this model can help with

- Early detection of VM instability
- Predicting node removals in Kubernetes clusters
- Detecting failure patterns before they escalate
- Reducing downtime costs for cloud-hosted services
- Improving autoscaling & workload migration

Who benefits

Cloud reliability engineers

DevOps / SRE teams

Researchers studying cloud behavior

Companies using AWS/GCP/Azure/OpenStack

Universities & HPC clusters

This prototype is also a foundation for self-healing systems where:

The model predicts failure

A policy automatically migrates workloads or spins up replicas

8. Final Remarks

This project is fully implemented according to the rubric:

✔ Custom PyTorch models (no pre-built ML models)
✔ Proper baselines with classical ML
✔ Real + synthetic datasets
✔ Full end-to-end preprocessing → training → evaluation
✔ Technical depth appropriate for graduate-level AI coursework
