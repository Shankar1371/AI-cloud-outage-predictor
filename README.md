# AI-Powered Cloud Outage Prediction

This project builds an early-warning system that predicts cloud service outages 15 minutes before they occur**, using the last 30 minutes of system metrics and event bursts.

The main contributions are:

- A full data pipeline: raw traces → minute-level metrics → outage labels → sliding windows → tensors  
- A GRU-based temporal model implemented from scratch in PyTorch (no high-level ML libraries)  
- Baseline models (Logistic Regression / Random Forest) using scikit-learn  
- Experiments on:
  - A synthetic Google-style cluster dataset 
  - A real OpenStack failure dataset (DEPL workload) 


---

## 1. Repository Structure

```text
cloud-outage-predictor/
│
├── data/
│   ├── raw/
│   │   └── google2019/           # synthetic "Google-style" traces (parquet)
│   ├── openstack_raw/            # raw OpenStack Failure Dataset (DEPL)
│   ├── interim/
│   │   └── google2019/           # minute metrics, labels (parquet)
│   └── processed/
│       ├── google2019/           # train/val/test tensors + plots for GRU
│       └── openstack_DEPL/       # processed OpenStack splits + plots
│
├── scripts/
│   ├── make_synthetic_google2019.py  # generate synthetic google-style traces
│   ├── make_windows.py               # build minute metrics + outage labels
│   ├── make_tensors.py               # build sliding-window tensors for GRU
│   ├── prepare_openstack.py          # process OpenStack Failure Dataset (DEPL)
│   ├── plot_gru_results.py           # training curves + ROC/PR for GRU
│   └── plot_openstack_results.py     # plots for OpenStack classifier
│
├── src/
│   ├── labels/
│   │   └── google2019.py             # label builder for synthetic dataset
│   ├── models/
│   │   └── gru.py                    # OutageGRU model (PyTorch)
│   ├── train.py                      # GRU training & evaluation on google2019
│   └── openstack_train.py            # MLP classifier on OpenStack DEPL
│
├── README.md
└── requirements.txt

```
# 2 Environement Setup

## 2.1 Create and activate virtual environment (Windows)

python -m venv .venv
.\.venv\Scripts\activate

## 2.2 Install dependencies
pip install -r requirements.txt

### requiremnets include

torch
pandas
numpy
pyarrow
scikit-learn
matplotlib


# 3 Synthetic Google-Style Experiment (GRU Model)
This experiment trains a GRU-based temporal model to predict whether a machine will experience an outage in the next 15 minutes, given the previous 30 minutes of telemetry.

## 3.1 Step 1 – Generate synthetic raw traces
python .\scripts\make_synthetic_google2019.py

### this creates
data/raw/google2019/usage.parquet
data/raw/google2019/task_events.parquet
data/raw/google2019/machine_events.parquet


## 3.2 Step 2 – Build minute metrics and outage labels
python .\scripts\make_windows.py ^
  --provider google2019 ^
  --horizon 15 ^
  --T 30 ^
  --out ".\data\processed\google2019"


### This step:

Converts raw traces to minute-level metrics

Aggregates failure/eviction bursts

Joins machine remove events

Writes:
data/interim/google2019/metrics_train.parquet
data/interim/google2019/labels_train.parquet

#### you can inspect the label balance
import pandas as pd
labels = pd.read_parquet("data/interim/google2019/labels_train.parquet")
print(labels["label"].value_counts())

## 3.3 Step 3 – Build sliding-window tensors for GRU
python .\scripts\make_tensors.py ^
  --provider google2019 ^
  --T 30 ^
  --outdir ".\data\processed\google2019"

### this creates.
data/processed/google2019/train.pt
data/processed/google2019/val.pt
data/processed/google2019/test.pt

Each .pt package contains:

X_metrics → [N, T, fm] (time windows of metrics)

X_events → [N, T, fe] (time windows of event counts)

y → [N] (labels: 0 = normal, 1 = outage in next 15 min)

features, event_features, T

## 3.4 Step 4 – Train the GRU outage predictor
python -m src.train ^
  --data ".\data\processed\google2019" ^
  --epochs 30 ^
  --pos-weight 50

### this will:
Load train.pt / val.pt

Build OutageGRU with the appropriate input feature sizes

Train with weighted BCE loss to handle class imbalance

Evaluate on the validation set every epoch

Save the best model to:
data/processed/google2019/best_gru.pt


At the end it also reports test metrics on test.pt, including:

PR-AUC (precision–recall area under curve)

ROC-AUC

F1-score

## 3.5 Step 5 – Plot GRU training curves and metrics
python .\scripts\plot_gru_results.py

### this script:
This script:

Reads data/processed/google2019/gru_training_log.csv

Produces plots under data/processed/google2019/plots/, such as:

gru_train_loss.png

gru_val_f1.png

gru_roc_curve.png

gru_pr_curve.png

These plots are the ones included in the report and slides.

# 4. OpenStack DEPL Experiment (Real Failure Logs)
This experiment uses the OpenStack Failure Dataset (DEPL workload) to show the pipeline on real logs.
DEPL contains only failure sequences (all labels = 1), so metrics are trivial but useful to verify integration.

## 4.1 Step 1 – Download the dataset
Download the OpenStack Failure Dataset (DEPL) from the official GitHub repository.

Place the extracted folder under:
data/openstack_raw/Failure-Dataset-OpenStack/

Ensure that structure looks like:
data/openstack_raw/Failure-Dataset-OpenStack/DEPL/SEQ.tsv
data/openstack_raw/Failure-Dataset-OpenStack/DEPL/LABEL.tsv
...


## 4.2 Step 2 – Prepare the OpenStack DEPL workload
python .\scripts\prepare_openstack.py

### this script:
Reads the DEPL workload from data/openstack_raw/Failure-Dataset-OpenStack

Cleans, encodes, and normalizes features

Splits into train/val/test sets

Saves processed data under:
data/processed/openstack_DEPL/

(e.g., train.pt / val.pt / test.pt or their equivalent).

## 4.3 Step 3 – Train the OpenStack classifier
python -m src.openstack_train ^
  --data ".\data\processed\openstack_DEPL" ^
  --epochs 20

### this will
Train a feed-forward MLP on the processed OpenStack features

Report accuracy, F1, precision, recall on validation and test sets

Save the best model (by F1) to:
data/processed/openstack_DEPL/best_openstack_mlp.pt

\Because DEPL contains only positive samples, you should see:

Accuracy ≈ 1.0

F1 ≈ 1.0

Confusion matrix with only TP populated

This behavior is explained in the report.

## 4.4 Step 4 – Plot OpenStack evaluation metrics
python .\scripts\plot_openstack_results.py

### This script reads the saved logs/predictions and generates:

Training loss curve

Validation F1 curve

ROC curve

Precision–Recall curve

Calibration curve

Confusion matrix plot

Label distribution plot

All saved under data/processed/openstack_DEPL/plots/.

# 5 Reproducing Reported Results

To reproduce the main results from the report:

1. Synthetic GRU experiment

Run:
python .\scripts\make_synthetic_google2019.py
python .\scripts\make_windows.py --provider google2019 --horizon 15 --T 30 --out ".\data\processed\google2019"
python .\scripts\make_tensors.py --provider google2019 --T 30 --outdir ".\data\processed\google2019"
python -m src.train --data ".\data\processed\google2019" --epochs 30 --pos-weight 50
python .\scripts\plot_gru_results.py

. You should obtain similar PR-AUC, ROC-AUC, and F1 scores on the test set, and the same style of training and ROC/PR curves as in the report.

2. OpenStack DEPL experiment

Download and place the dataset under data/openstack_raw/Failure-Dataset-OpenStack/

Run:

python .\scripts\prepare_openstack.py
python -m src.openstack_train --data ".\data\processed\openstack_DEPL" --epochs 20
python .\scripts\plot_openstack_results.py

. You should obtain near-perfect test metrics (since all labels are positive) and the characteristic calibration/confusion plots described in the report.

# 6. Notes and Troubleshooting

Virtualenv not active?
Make sure to run:
.\.venv\Scripts\activate

1. FileNotFoundError (parquet or .pt missing)?
Re-run the corresponding script in order:

make_synthetic_google2019.py

make_windows.py

make_tensors.py

2. Metrics look too low on synthetic data?
This is expected due to:

small dataset

severe class imbalance

hard 15-minute prediction horizon
Mentioned explicitly in the report.

3. OpenStack metrics are all 1.0 (perfect)?
This is correct: DEPL contains only failure samples, so classification degenerates into always predicting failure.
The purpose here is to demonstrate compatibility with real logs.
