#  AI-Powered Cloud Outage Prediction

This project predicts **cloud service outages** (AWS, Azure, GCP, or Kubernetes clusters) **10–15 minutes in advance** using server metric time-series data.  
It is built completely in **Python 3 + PyTorch**, following the academic requirement that prohibits use of high-level ML libraries (e.g., scikit-learn) for the main model.

---

## Problem Statement

Modern cloud infrastructures consist of thousands of servers streaming telemetry such as CPU usage, memory utilization, disk I/O, and network throughput.  
Unexpected outages in these environments can cause **massive financial losses** and **downtime**.  
Our goal is to build a data-driven model that learns these temporal degradation patterns and predicts failures **before they occur**.

---

##  Objectives

1. Build an end-to-end PyTorch pipeline that:
   - Parses large-scale cluster traces.
   - Constructs proxy labels for outages.
   - Generates fixed-length time-series windows for ML.
2. Train a **GRU-based neural network** to predict outage probability for the next 10–15 minutes.
3. Compare against **baseline models** (Logistic Regression / Random Forest).
4. Evaluate using PR-AUC, ROC-AUC, and F1 Score.

---

##  Project Structure

cloud-outage-predictor/
│
├── data/
│ ├── raw/ # Original trace files (Google2019/Azure/Alibaba)
│ ├── interim/ # Cleaned & merged parquet files
│ └── processed/ # Tensor datasets (.pt)
│
├── scripts/
│ ├── make_synthetic_google2019.py # Generate small synthetic dataset
│ ├── make_windows.py # Aggregation + labeling pipeline
│ └── make_tensors.py # Convert labeled data to torch tensors
│
├── src/
│ ├── labels/
│ │ └── google2019.py # Label builder logic
│ ├── models/
│ │ └── gru.py # GRU-based outage predictor
│ └── train.py # Training + evaluation loop
│
├── README.md
└── requirements.txt



---

##  Setup Instructions

### 1️ Environment
 
python -m venv .venv
.\.venv\Scripts\activate        # (Windows)
pip install -r requirements.txt

# basic dependecies
torch
pandas
numpy
pyarrow
scikit-learn



Workflow
Week 1 — Data Preprocessing

Generate or download raw cluster trace data.

Run:

python .\scripts\make_synthetic_google2019.py
python .\scripts\make_windows.py --provider google2019 --horizon 15 --T 30 --out ".\data\processed\google2019"


Outputs metrics_train.parquet under data/interim/google2019/.

Week 2 — Labeling & Tensor Generation

Build labels using:

from src.labels.google2019 import build_labels


Run tensor builder:

python .\scripts\make_tensors.py --provider google2019 --T 30 --outdir ".\data\processed\google2019"


Verify output:

data/processed/google2019/train.pt
data/processed/google2019/val.pt
data/processed/google2019/test.pt


Each .pt file contains:

X_metrics: [N, T, #metric_features]

X_events: [N, T, #event_features]

y: [N]

Week 3 — Model Training

Train the GRU model:

python -m src.train --data ".\data\processed\google2019" --epochs 5 --pos-weight 10


Example output:

[epoch 1] PR-AUC=0.12 ROC-AUC=0.63 F1=0.02
[epoch 5] PR-AUC=0.27 ROC-AUC=0.77 F1=0.15


Best model checkpoint:
data/processed/google2019/best_gru.pt

Week 4 — Baselines & Report

Implement Logistic Regression / Random Forest on aggregated features.

Compare PR-AUC, ROC-AUC, F1.

Write 1-page report covering:

Problem definition

Dataset & preprocessing pipeline

Proposed GRU model

Baseline comparison

Evaluation metrics & discussion

##  Evaluation Metrics
Metric	Description
PR-AUC	Precision-Recall Area — best for imbalanced outage data
ROC-AUC	Measures separability between positive and negative classes
F1-Score	Harmonic mean of precision & recall
Lead Time (min)	Average time difference between prediction and outage
## Model Architecture (GRU)
Input  →  GRU_metrics (2 layers, 64 h) → 
          GRU_events  (1 layer, 32 h)  →
          Concatenate →
          MLP (128→1, Sigmoid)


Loss: Weighted BCE
Optimizer: AdamW
Learning Rate: 1e-3
Window length (T): 30 minutes
Prediction horizon (H): 15 minutes

### Dataset Reference

You can later replace the synthetic dataset with real ones:

Provider	Dataset	Notes
Google Cluster Trace 2019	GitHub – google/cluster-data
	Richest metadata
Azure Public Dataset V2	GitHub – Azure Public Dataset
	Large VM telemetry
Alibaba Cluster Trace v2018	Alibaba Cluster Trace
	8-day metrics trace
