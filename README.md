# Privacy-Preserving Pneumonia Detection
### Using Federated Learning, Differential Privacy, and Reservoir Sampling

A Data Science student project implementing the pipeline described in:
> *Privacy-Preserving Deep Learning for Disease Diagnosis in Medical Imaging: A Systematic Review*
> Rahman, Khan & Nijim · IEEE Access · 2025

---

## Project Overview

| Part | Description |
|------|-------------|
| Disease Diagnosis | Binary classification: NORMAL vs PNEUMONIA from chest X-rays |
| Privacy Preservation | Federated Learning (FedAvg) + Differential Privacy (DP-SGD via Opacus) |
| DS Extension (Unit 5) | Reservoir Sampling to evaluate bounded-memory data selection |

---

## Experiments

| ID | Method | DP | Sampling |
|----|--------|----|----------|
| E1 | Centralized baseline | No | 100% |
| E2 | Federated (3 clients) | No | 100% |
| E3 | Federated + DP-SGD | Yes | 100% |
| E4a | Federated + DP + Reservoir | Yes | 50% |
| E4b | Federated + DP + Reservoir | Yes | 25% |

---

## Dataset

Chest X-Ray Images (Pneumonia) — Kermany et al. 2018  
Download from: https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia

Place the dataset so the folder structure looks like:
```
DS-Project/
  chest_xray/
    train/
      NORMAL/
      PNEUMONIA/
    test/
      NORMAL/
      PNEUMONIA/
```

The project creates a stratified validation split from `train/`; the
dataset's small `val/` split is not used. All scripts expect `train/` and
`test/` to be siblings directly inside `chest_xray/`.

---

## Setup

```bash
pip install -r requirements.txt
```

---

## How to Run

### Option A — Run everything at once
```bash
python src/run_all.py
```

### Option B — Run stage by stage

**Stage 1: EDA**
```bash
python notebooks/eda.py
```

**Stage 2: Centralized Baseline (E1)**
```bash
python src/train_centralized.py
```

**Stage 3–5: Federated / DP / Reservoir experiments (E2, E3, E4)**
```bash
python src/federated.py
```

**Generate final plots**
```bash
python src/visualize.py
```

---

## Project Structure

```
DS-Project/
├── chest_xray/              Dataset (download from Kaggle)
├── src/
│   ├── data_loader.py       Image loading, preprocessing, client partitioning
│   ├── model.py             MobileNetV2 with custom classifier head
│   ├── train_centralized.py E1 — centralized baseline
│   ├── federated.py         E2/E3/E4 — federated loop + FedAvg + DP-SGD
│   ├── reservoir.py         Algorithm R — reservoir sampling (Unit 5)
│   ├── evaluate.py          Metrics, confusion matrix, plot utilities
│   ├── visualize.py         Final comparison plots
│   └── run_all.py           Master script — runs all experiments
├── notebooks/
│   └── eda.py               Stage 1: exploratory data analysis
├── results/
│   ├── metrics.csv          All experiment results
│   └── plots/               Saved figures
├── requirements.txt
└── README.md
```

---

## DS Concepts Used

| Unit | Concept | Where |
|------|---------|-------|
| Unit 1 | Data Science Lifecycle | Full project structure |
| Unit 2 | Statistics & Evaluation | Class distribution, precision/recall/F1/AUC |
| Unit 3 | Distributed Processing | Federated clients = distributed nodes |
| Unit 5 | Data Stream Mining | Reservoir sampling on client training data |

---

## Key Results Location

- `results/metrics.csv` — all experiment metrics
- `results/plots/` — all visualizations including confusion matrices

## Experiment notes

- E1 uses the original MobileNetV2 BatchNorm architecture. E2, E3, E4a, and
  E4b use the same Opacus-compatible GroupNorm-converted architecture so that
  federated comparisons keep matching model state dictionaries.
- Federated DP accounting is cumulative per client across rounds. The reported
  experiment epsilon is the maximum client epsilon under the assumption that
  client training partitions contain disjoint image records. It is not a
  patient-level guarantee; if records for the same patient occur at multiple
  clients, a more conservative composition is required.
- DP-SGD does not provide complete privacy or confidentiality by itself.
  This project does not implement secure aggregation, and Opacus secure RNG
  mode is not enabled by default.
- Reservoir sampling is uniform, not class-aware, and may change each client's
  class balance. The scripts report sampled class counts. It operates on
  already-loaded training paths to simulate a stream, not real-time ingestion.
- Running `src/run_all.py` archives an existing `results/metrics.csv` before
  starting a fresh complete run. Running an individual experiment continues
  to append its result.
